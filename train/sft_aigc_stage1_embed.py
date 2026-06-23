#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_str, load_project_env

load_project_env(REPO_ROOT / ".env")

DEFAULT_INPUT_DIR = Path(env_str("NAVIGEN_SFT_INPUT_DIR", str(REPO_ROOT / "dataset")))
DEFAULT_SFT_DATA_DIR = SCRIPT_DIR / "sft_data_stage1"
DEFAULT_TRAIN_OUT_DIR = Path(env_str("NAVIGEN_STAGE1_OUTPUT_DIR", str(SCRIPT_DIR / "sft_output" / "qwen3_1p7b_sft_embed_stage1")))
DEFAULT_MODEL_DIR = Path(env_str("NAVIGEN_CID_MODEL_DIR", str(SCRIPT_DIR / "Qwen3-1.7B-cid-expanded-clean")))


SYSTEM_PROMPTS = {
    "cid2tid": (
        "You are an ID mapping assistant. "
        "Your task is to map the input cid to the corresponding tid (tid is the item's metadata). "
        "The final answer must be a JSON object only, with the field target_tid and no other fields."
    ),
    "tid2cid": (
        "You are an ID mapping assistant. "
        "Your task is to map the input tid (tid is the item's metadata) to the corresponding cid. "
        "The final answer must be a JSON object only, with the field target_cid and no other fields."
    ),
    "cid2cid": (
        "You are a personalized recommendation assistant. "
        "Predict the target cid interaction based on user history cid interaction. "
        "The final answer must be a JSON object only, with the field target_cid and no other fields."
    ),
    "cid2ins": (
        "You are a personalized recommendation and AIGC instruction generation assistant. "
        "Based on user history cid interaction, output user target tid interaction and the AIGC instruction . "
        "Think step by step, then answer. The final answer must be a JSON object only, with the fields target_tid and target_ins."
    ),
}


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    # numpy.ndarray compatibility without hard dependency on numpy
    if hasattr(v, "tolist"):
        lv = v.tolist()
        if isinstance(lv, list):
            return lv
        return [lv]
    return [v]


def _first(v: Any) -> Any:
    seq = _as_list(v)
    return seq[0] if seq else None


def _str_or_empty(v: Any) -> str:
    return "" if v is None else str(v)


def _string_list(v: Any) -> list[str]:
    return [_str_or_empty(item) for item in _as_list(v) if item is not None]


def _json_value(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _json_text(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _with_think(reasoning: str, answer_json_text: str) -> str:
    r = (reasoning or "").strip()
    if not r:
        return answer_json_text
    return f"<think>\n{r}\n</think>\n\n{answer_json_text}"


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _get_dist_info() -> tuple[int, int, int]:
    def _env_int(keys: list[str], default: int) -> int:
        for key in keys:
            raw = os.environ.get(key)
            if raw is None or raw == "":
                continue
            try:
                return int(raw)
            except ValueError:
                continue
        return default

    world_size = _env_int(["WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"], 1)
    rank = _env_int(["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"], 0)
    local_rank = _env_int(
        ["LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "PMI_LOCAL_RANK"],
        0,
    )

    world_size = 1 if world_size < 1 else world_size
    rank = 0 if rank < 0 else rank
    local_rank = 0 if local_rank < 0 else local_rank
    return rank, local_rank, world_size


def _is_main_process() -> bool:
    rank, _, _ = _get_dist_info()
    return rank == 0


def _maybe_init_distributed() -> tuple[int, int, int]:
    rank, local_rank, world_size = _get_dist_info()
    if world_size <= 1:
        return rank, local_rank, world_size

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return rank, local_rank, world_size


def _dist_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _maybe_destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _import_unsloth_fast_language_model():
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise RuntimeError(
            "Failed to import Unsloth. The current environment appears incompatible "
            f"with the installed Unsloth package: {exc}"
        ) from exc
    return FastLanguageModel


def _load_model_and_tokenizer(
    model_name: str,
    max_seq_len: int,
    dtype: Any,
    gradient_checkpointing_mode: str,
    loader: str,
):
    use_gc = gradient_checkpointing_mode != "off"
    if loader == "unsloth":
        FastLanguageModel = _import_unsloth_fast_language_model()
        model_gc_mode: bool | str = "unsloth" if gradient_checkpointing_mode == "unsloth" else use_gc
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            dtype=dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=True,
            full_finetuning=True,
            trust_remote_code=True,
            use_gradient_checkpointing=model_gc_mode,
        )
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if use_gc:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    return model, tokenizer


def _resolve_text_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _normalize_token_id(token_id: Any) -> int | None:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        return _normalize_token_id(token_id[0] if token_id else None)
    return int(token_id)


def _resolve_token_from_id(tokenizer: Any, token_id: Any) -> str | None:
    normalized = _normalize_token_id(token_id)
    if normalized is None:
        return None
    try:
        token = tokenizer.convert_ids_to_tokens(normalized)
    except Exception:
        return None
    if token is None:
        return None
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token is not None and token == unk_token:
        return None
    return str(token)


def _resolve_training_special_tokens(model: Any, tokenizer: Any) -> tuple[str | None, str | None]:
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token in (None, "<EOS_TOKEN>"):
        eos_token = _resolve_token_from_id(tokenizer, getattr(config, "eos_token_id", None))
    if eos_token in (None, "<EOS_TOKEN>"):
        eos_token = _resolve_token_from_id(tokenizer, getattr(generation_config, "eos_token_id", None))

    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token in (None, "<PAD_TOKEN>"):
        pad_token = _resolve_token_from_id(tokenizer, getattr(config, "pad_token_id", None))
    if pad_token in (None, "<PAD_TOKEN>"):
        pad_token = _resolve_token_from_id(tokenizer, getattr(generation_config, "pad_token_id", None))
    if pad_token in (None, "<PAD_TOKEN>"):
        pad_token = eos_token

    if eos_token is not None:
        tokenizer.eos_token = eos_token
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    return eos_token, pad_token


def _prepare_model_for_training(model: Any, gradient_checkpointing_mode: str) -> None:
    use_gc = gradient_checkpointing_mode != "off"

    if hasattr(model, "for_training"):
        try:
            model.for_training(use_gradient_checkpointing=use_gc)
        except TypeError:
            model.for_training()

    if hasattr(model, "gradient_checkpointing_enable") and hasattr(model, "gradient_checkpointing_disable"):
        if use_gc:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = use_gc


class _PackedDatasetCollator:
    def __init__(self, pad_token_id: int, fixed_length: int) -> None:
        self.pad_token_id = int(pad_token_id)
        self.fixed_length = int(fixed_length)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            input_ids = [int(x) for x in feature["input_ids"]]
            labels = [int(x) for x in feature["labels"]]
            if len(input_ids) > self.fixed_length:
                raise ValueError(
                    f"Packed sample length {len(input_ids)} exceeds fixed_length={self.fixed_length}"
                )
            pad_len = self.fixed_length - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _assert_cid_tokenizer_ready(tokenizer: Any, model_name: str) -> None:
    probe = "<|cid_begin|><s_a_1><s_b_2><s_c_3><|cid_end|>"
    expected = ["<|cid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|cid_end|>"]
    encoded = tokenizer(probe, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    if tokens != expected:
        raise ValueError(
            "The tokenizer loaded from "
            f"{model_name!r} does not recognize the CID vocabulary. "
            "Expected tokens "
            f"{expected}, but got {tokens}. "
            "Merge the updated tokenizer files into the model directory, or point "
            "--model_name to a full model checkpoint that already contains the new CID vocab."
        )


def _assert_assistant_masking_ready(tokenizer: Any, model_name: str) -> None:
    probe_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": '{"ok":true}'},
    ]
    try:
        processed = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
    except Exception as exc:
        raise ValueError(
            "The tokenizer loaded from "
            f"{model_name!r} could not produce assistant masks. "
            "Assistant-only loss requires a chat template with `{% generation %}` "
            f"support: {exc}"
        ) from exc

    assistant_masks = _as_list(processed.get("assistant_masks"))
    if not assistant_masks or 1 not in assistant_masks:
        raise ValueError(
            "The tokenizer loaded from "
            f"{model_name!r} did not return any assistant mask positions. "
            "Assistant-only loss requires a chat template with `{% generation %}` "
            "around assistant content."
        )


def _load_added_cid_token_ids(model_name: str, tokenizer: Any) -> list[int]:
    model_dir = Path(model_name)
    added_tokens_path = model_dir / "cid_tokens_added.txt"
    if not added_tokens_path.exists():
        return []

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: list[int] = []
    missing_tokens: list[str] = []

    with added_tokens_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            token = raw_line.strip()
            if not token:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == unk_token_id:
                missing_tokens.append(token)
                continue
            token_ids.append(int(token_id))

    if missing_tokens:
        preview = ", ".join(repr(tok) for tok in missing_tokens[:5])
        raise ValueError(
            f"Tokenizer from {model_name!r} is missing {len(missing_tokens)} CID tokens "
            f"listed in {added_tokens_path}. First few missing tokens: {preview}"
        )

    return sorted(set(token_ids))


def _build_packed_tokenized_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    tokenizer: Any,
    max_seq_len: int,
    split_name: str,
) -> Path:
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(input_jsonl), split="train")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    packed_rows = 0
    dropped_oversize = 0
    cur_input_ids: list[int] = []
    cur_assistant_masks: list[int] = []

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")

    def _flush_current(f) -> None:
        nonlocal packed_rows, cur_input_ids, cur_assistant_masks
        if not cur_input_ids:
            return
        labels = [token_id if mask == 1 else -100 for token_id, mask in zip(cur_input_ids, cur_assistant_masks)]
        f.write(
            json.dumps(
                {
                    "input_ids": cur_input_ids,
                    "assistant_masks": cur_assistant_masks,
                    "labels": labels,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        packed_rows += 1
        cur_input_ids = []
        cur_assistant_masks = []

    with tmp_path.open("w", encoding="utf-8") as f:
        for messages in ds["messages"]:
            processed = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
            )
            input_ids = [int(x) for x in _as_list(processed["input_ids"])]
            assistant_masks = [int(x) for x in _as_list(processed.get("assistant_masks"))]
            if assistant_masks and len(assistant_masks) != len(input_ids):
                raise ValueError(
                    f"Assistant mask length mismatch for {split_name}: "
                    f"{len(assistant_masks)} vs {len(input_ids)}"
                )
            if not assistant_masks:
                assistant_masks = [0] * len(input_ids)

            if eos_token_id is not None:
                input_ids = input_ids + [int(eos_token_id)]
                assistant_masks = assistant_masks + [0]

            piece_len = len(input_ids)
            if piece_len > max_seq_len:
                dropped_oversize += 1
                continue

            if len(cur_input_ids) + piece_len <= max_seq_len:
                cur_input_ids.extend(input_ids)
                cur_assistant_masks.extend(assistant_masks)
            else:
                _flush_current(f)
                cur_input_ids = list(input_ids)
                cur_assistant_masks = list(assistant_masks)

        _flush_current(f)

    tmp_path.replace(output_jsonl)
    print(
        f"[safe_packing:{split_name}] in_rows={len(ds)} "
        f"out_packs={packed_rows} dropped_oversize={dropped_oversize}"
    )
    return output_jsonl


def _iter_named_params_with_names(model: Any) -> list[tuple[str, Any]]:
    return list(model.named_parameters())


def _make_row_mask(num_rows: int, trainable_rows: list[int], device: Any, dtype: Any):
    import torch

    mask = torch.zeros(num_rows, dtype=dtype, device=device)
    if trainable_rows:
        mask[torch.tensor(trainable_rows, dtype=torch.long, device=device)] = 1
    return mask


def _register_row_mask_hook(param: Any, trainable_rows: list[int]) -> None:
    import torch

    num_rows = int(param.shape[0])
    trainable_idx_cpu = torch.tensor(trainable_rows, dtype=torch.long, device="cpu") if trainable_rows else None
    mask_cache: dict[tuple[str, str], Any] = {}

    def _get_mask(device: Any, dtype: Any):
        key = (str(device), str(dtype))
        cached = mask_cache.get(key)
        if cached is not None:
            return cached

        mask = torch.zeros(num_rows, dtype=dtype, device=device)
        if trainable_idx_cpu is not None and trainable_idx_cpu.numel() > 0:
            mask[trainable_idx_cpu.to(device=device)] = 1
        mask_cache[key] = mask
        return mask

    def _hook(grad):
        if grad is None:
            return grad
        mask = _get_mask(grad.device, grad.dtype)
        return grad * mask.unsqueeze(1)

    param.register_hook(_hook)


def _untie_output_embeddings_if_needed(model: Any) -> tuple[Any | None, bool]:
    import torch

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        input_embeddings is None
        or not hasattr(input_embeddings, "weight")
        or output_embeddings is None
        or not hasattr(output_embeddings, "weight")
    ):
        return output_embeddings, False

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    if output_weight is None or output_weight.data_ptr() != input_weight.data_ptr():
        return output_embeddings, False

    new_output_embeddings = torch.nn.Linear(
        in_features=output_embeddings.in_features,
        out_features=output_embeddings.out_features,
        bias=output_embeddings.bias is not None,
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    with torch.no_grad():
        new_output_embeddings.weight.copy_(output_weight.detach())
        if output_embeddings.bias is not None and new_output_embeddings.bias is not None:
            new_output_embeddings.bias.copy_(output_embeddings.bias.detach())

    model.set_output_embeddings(new_output_embeddings)
    if getattr(model, "config", None) is not None:
        model.config.tie_word_embeddings = False
    return model.get_output_embeddings(), True


def _configure_stage1_embedding_only_training(
    model: Any,
    trainable_token_ids: list[int],
    lm_head_training: str,
) -> dict[str, Any]:
    for _, param in _iter_named_params_with_names(model):
        param.requires_grad_(False)

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise RuntimeError("Model does not expose input embeddings.")
    input_weight = input_embeddings.weight
    input_weight.requires_grad_(True)
    _register_row_mask_hook(input_weight, trainable_token_ids)

    output_embeddings = model.get_output_embeddings()
    output_weight = None
    lm_head_is_tied = False
    lm_head_was_untied = False
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        try:
            model.tie_weights()
        except Exception:
            pass
        output_embeddings = model.get_output_embeddings()
        output_weight = getattr(output_embeddings, "weight", None)
        if output_weight is None:
            lm_head_is_tied = True
        else:
            lm_head_is_tied = output_weight.data_ptr() == input_weight.data_ptr()
        if lm_head_training == "full" and lm_head_is_tied:
            output_embeddings, lm_head_was_untied = _untie_output_embeddings_if_needed(model)
            output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
            lm_head_is_tied = False if output_weight is not None else lm_head_is_tied
        if lm_head_training == "full" and lm_head_is_tied:
            raise RuntimeError("Requested full lm_head training, but lm_head is still tied to input embeddings.")
        should_train_output = output_weight is not None and (lm_head_training == "full" or not lm_head_is_tied)
        if should_train_output:
            output_weight.requires_grad_(True)
            if lm_head_training == "cid_rows":
                _register_row_mask_hook(output_weight, trainable_token_ids)

    trainable_param_names: list[str] = []
    for name, param in _iter_named_params_with_names(model):
        if param.requires_grad:
            trainable_param_names.append(name)

    return {
        "input_weight": input_weight,
        "output_weight": output_weight,
        "lm_head_is_tied": lm_head_is_tied,
        "lm_head_was_untied": lm_head_was_untied,
        "lm_head_training": lm_head_training,
        "trainable_param_names": trainable_param_names,
    }


def _sample_cid2tid(row: dict[str, Any]) -> dict[str, Any] | None:
    cid = _first(row.get("sid"))
    tid = _string_list(row.get("tid"))
    if cid is None or not tid:
        return None
    user = (
        "Task: Map the given cid to the corresponding tid.\n"
        f"cid: {_str_or_empty(cid)}\n"
        "Output JSON only."
    )
    assistant = _json_text({"target_tid": tid})
    return {
        "task": "cid2tid",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["cid2tid"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _sample_tid2cid(row: dict[str, Any]) -> dict[str, Any] | None:
    tid = _string_list(row.get("tid"))
    cid = _first(row.get("sid"))
    if not tid or cid is None:
        return None
    user = (
        "Task: Map the given tid to the corresponding cid.\n"
        f"tid: {_json_value(tid)}\n"
        "Output JSON only."
    )
    assistant = _json_text({"target_cid": _str_or_empty(cid)})
    return {
        "task": "tid2cid",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["tid2cid"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _sample_cid2cid(row: dict[str, Any]) -> dict[str, Any] | None:
    hist_cid = _string_list(row.get("hist_sid"))
    target_cid = _first(row.get("target_sid"))
    if not hist_cid or target_cid is None:
        return None
    user = (
        "Task: Predict the target_cid based on the user's historical interactions in hist_cid.\n"
        f"hist_cid: {_json_value(hist_cid)}\n"
        "Output JSON only."
    )
    assistant = _json_text({"target_cid": _str_or_empty(target_cid)})
    return {
        "task": "cid2cid",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["cid2cid"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def _sample_cid2ins(row: dict[str, Any]) -> dict[str, Any] | None:
    hist_cid = _string_list(row.get("hist_sid"))
    target_tid = _string_list(row.get("target_tid"))
    target_ins = _str_or_empty(row.get("target_ins"))
    if not hist_cid or not target_tid or not target_ins:
        return None

    reasoning = _str_or_empty(row.get("reasoning"))
    user = (
        "Task: Generate the target recommendation result based on the user's historical interactions in hist_cid.\n"
        "The output must be JSON with the fields target_tid and target_ins.\n"
        f"hist_cid: {_json_value(hist_cid)}"
    )
    answer = _json_text(
        {
            "target_tid": target_tid,
            "target_ins": target_ins,
        }
    )
    assistant = _with_think(reasoning, answer)
    return {
        "task": "cid2ins",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["cid2ins"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


# def build_sft_jsonl(output_dir: Path, save_dir: Path) -> tuple[Path, Path, Path]:
#     save_dir.mkdir(parents=True, exist_ok=True)
#     out_paths = {
#         "train": save_dir / "train_qwen_sft.jsonl",
#         "valid": save_dir / "valid_qwen_sft.jsonl",
#         "test": save_dir / "test_qwen_sft.jsonl",
#     }

#     for split in ["train", "valid", "test"]:
#         cid2tid_path = output_dir / f"{split}_cid2tid.parquet"
#         tid2cid_path = output_dir / f"{split}_tid2cid.parquet"
#         cid2cid_path = output_dir / f"{split}_cid2cid.parquet"

#         required = [cid2tid_path, tid2cid_path, cid2cid_path]
#         for p in required:
#             if not p.exists():
#                 raise FileNotFoundError(f"Missing required file: {p}")

#         cid2tid_rows = _read_parquet_rows(cid2tid_path)
#         tid2cid_rows = _read_parquet_rows(tid2cid_path)
#         cid2cid_rows = _read_parquet_rows(cid2cid_path)
#         samples: list[dict[str, Any]] = []
#         dropped = {"cid2tid": 0, "tid2cid": 0, "cid2cid": 0}

#         for row in cid2tid_rows:
#             s = _sample_cid2tid(row)
#             if s is None:
#                 dropped["cid2tid"] += 1
#             else:
#                 samples.append(s)
#         for row in tid2cid_rows:
#             s = _sample_tid2cid(row)
#             if s is None:
#                 dropped["tid2cid"] += 1
#             else:
#                 samples.append(s)
#         for row in cid2cid_rows:
#             s = _sample_cid2cid(row)
#             if s is None:
#                 dropped["cid2cid"] += 1
#             else:
#                 samples.append(s)
#         out_path = out_paths[split]
#         with out_path.open("w", encoding="utf-8") as f:
#             for s in samples:
#                 f.write(json.dumps(s, ensure_ascii=False) + "\n")

#         print(f"[{split}] wrote {out_path} rows={len(samples)}")
#         print(
#             f"  dropped: cid2tid={dropped['cid2tid']}, "
#             f"tid2cid={dropped['tid2cid']}, "
#             f"cid2cid={dropped['cid2cid']}"
#         )

#     return out_paths["train"], out_paths["valid"], out_paths["test"]
def build_sft_jsonl(output_dir: Path, save_dir: Path) -> tuple[Path, Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "train": save_dir / "train_qwen_sft.jsonl",
        "valid": save_dir / "valid_qwen_sft.jsonl",
        "test": save_dir / "test_qwen_sft.jsonl",
    }

    for split in ["train", "valid", "test"]:
        cid2tid_path = output_dir / f"{split}_cid2tid.parquet"
        tid2cid_path = output_dir / f"{split}_tid2cid.parquet"
        cid2cid_path = output_dir / f"{split}_cid2cid.parquet"

        required = [cid2tid_path, tid2cid_path, cid2cid_path]
        for p in required:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        cid2tid_rows = _read_parquet_rows(cid2tid_path)
        tid2cid_rows = _read_parquet_rows(tid2cid_path)
        cid2cid_rows = _read_parquet_rows(cid2cid_path)
        samples: list[dict[str, Any]] = []
        dropped = {"cid2tid": 0, "tid2cid": 0, "cid2cid": 0}

        for row in cid2tid_rows:
            s = _sample_cid2tid(row)
            if s is None:
                dropped["cid2tid"] += 1
            else:
                samples.append(s)
        for row in tid2cid_rows:
            s = _sample_tid2cid(row)
            if s is None:
                dropped["tid2cid"] += 1
            else:
                samples.append(s)
        for row in cid2cid_rows:
            s = _sample_cid2cid(row)
            if s is None:
                dropped["cid2cid"] += 1
            else:
                samples.append(s)

        out_path = out_paths[split]
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"[{split}] wrote {out_path} rows={len(samples)}")
        print(
            f"  dropped: cid2tid={dropped['cid2tid']}, "
            f"tid2cid={dropped['tid2cid']}, "
            f"cid2cid={dropped['cid2cid']}"
        )

    return out_paths["train"], out_paths["valid"], out_paths["test"]

def train_qwen_sft(
    train_jsonl: Path,
    valid_jsonl: Path,
    model_name: str,
    out_dir: Path,
    max_seq_len: int,
    epochs: int,
    max_steps: int,
    lr: float,
    train_bs: int,
    eval_bs: int,
    grad_accum: int,
    loader: str,
    gradient_checkpointing_mode: str,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
    lm_head_training: str,
    resume_from_checkpoint: str | None = None,
) -> None:
    import torch
    from transformers import Trainer, TrainingArguments

    rank, local_rank, world_size = _get_dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    trainer_use_gc = gradient_checkpointing_mode == "hf"
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        max_seq_len=max_seq_len,
        dtype=dtype,
        gradient_checkpointing_mode=gradient_checkpointing_mode,
        loader=loader,
    )
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    _assert_cid_tokenizer_ready(text_tokenizer, model_name)
    _assert_assistant_masking_ready(text_tokenizer, model_name)
    trainable_cid_token_ids = _load_added_cid_token_ids(model_name, text_tokenizer)
    eos_token, pad_token = _resolve_training_special_tokens(model, text_tokenizer)
    _prepare_model_for_training(model, gradient_checkpointing_mode)
    if not trainable_cid_token_ids:
        raise ValueError(
            "No cid_tokens_added.txt found or no valid token ids were resolved. "
            "Stage-1 embedding training requires newly-added CID token ids."
        )
    stage1_info = _configure_stage1_embedding_only_training(
        model,
        trainable_cid_token_ids,
        lm_head_training=lm_head_training,
    )
    print(
        "Stage-1 embedding-only training enabled: "
        f"new_cid_tokens={len(trainable_cid_token_ids)} "
        f"lm_head_training={stage1_info['lm_head_training']} "
        f"lm_head_tied={stage1_info['lm_head_is_tied']} "
        f"lm_head_was_untied={stage1_info['lm_head_was_untied']} "
        f"trainable_params={stage1_info['trainable_param_names']}"
    )
    print(f"Model loader: {loader}")
    print(f"Gradient checkpointing mode: {gradient_checkpointing_mode}")
    print(f"Trainer gradient checkpointing enabled: {trainer_use_gc}")
    print(f"Resolved eos_token={eos_token!r} pad_token={pad_token!r}")
    print(
        "Manual safe packing enabled: samples are tokenized into input_ids + "
        "assistant_masks, labels are pre-masked to assistant-only, and a sample "
        "that does not fit the current pack is moved to the next pack instead of "
        "being dropped."
    )

    packed_dir = train_jsonl.parent / f"packed_tokenized_maxlen_{max_seq_len}"
    train_packed_jsonl = packed_dir / "train_qwen_sft_packed.jsonl"
    valid_packed_jsonl = packed_dir / "valid_qwen_sft_packed.jsonl"

    if _is_main_process():
        _build_packed_tokenized_jsonl(
            input_jsonl=train_jsonl,
            output_jsonl=train_packed_jsonl,
            tokenizer=text_tokenizer,
            max_seq_len=max_seq_len,
            split_name="train",
        )
        _build_packed_tokenized_jsonl(
            input_jsonl=valid_jsonl,
            output_jsonl=valid_packed_jsonl,
            tokenizer=text_tokenizer,
            max_seq_len=max_seq_len,
            split_name="valid",
        )
    _dist_barrier()

    from datasets import load_dataset

    train_ds = load_dataset("json", data_files=str(train_packed_jsonl), split="train")
    valid_ds = load_dataset("json", data_files=str(valid_packed_jsonl), split="train")
    model.config.use_cache = False
    pad_token_id = getattr(text_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(text_tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    model.config.pad_token_id = int(pad_token_id)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = int(pad_token_id)

    config = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=trainer_use_gc,
        report_to="none",
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        weight_decay=0.0,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=_PackedDatasetCollator(
            pad_token_id=pad_token_id,
            fixed_length=max_seq_len,
        ),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    _dist_barrier()
    if trainer.is_world_process_zero():
        print(f"Final eval metrics: {metrics}")
        trainer.save_model(str(out_dir / "final"))
        tokenizer.save_pretrained(str(out_dir / "final"))
        print(f"Training done. Saved to: {out_dir / 'final'}")


def eval_qwen_checkpoint(
    valid_jsonl: Path,
    model_name: str,
    checkpoint_dir: Path,
    max_seq_len: int,
    eval_bs: int,
    loader: str,
) -> dict[str, Any]:
    import torch
    from transformers import Trainer, TrainingArguments

    rank, local_rank, world_size = _get_dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        max_seq_len=max_seq_len,
        dtype=dtype,
        gradient_checkpointing_mode="off",
        loader=loader,
    )
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    _assert_cid_tokenizer_ready(text_tokenizer, model_name)
    _assert_assistant_masking_ready(text_tokenizer, model_name)
    _resolve_training_special_tokens(model, text_tokenizer)
    model.config.use_cache = False
    pad_token_id = getattr(text_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(text_tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    model.config.pad_token_id = int(pad_token_id)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = int(pad_token_id)

    packed_dir = valid_jsonl.parent / f"packed_tokenized_maxlen_{max_seq_len}"
    valid_packed_jsonl = packed_dir / "valid_qwen_sft_packed.jsonl"
    if _is_main_process():
        _build_packed_tokenized_jsonl(
            input_jsonl=valid_jsonl,
            output_jsonl=valid_packed_jsonl,
            tokenizer=text_tokenizer,
            max_seq_len=max_seq_len,
            split_name="valid",
        )
    _dist_barrier()

    from datasets import load_dataset

    valid_ds = load_dataset("json", data_files=str(valid_packed_jsonl), split="train")
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_eval_batch_size=eval_bs,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=valid_ds,
        data_collator=_PackedDatasetCollator(
            pad_token_id=pad_token_id,
            fixed_length=max_seq_len,
        ),
    )
    metrics = trainer.evaluate()
    _dist_barrier()
    return metrics


def _checkpoint_step_from_dir(checkpoint_dir: Path) -> int | None:
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        suffix = name[len("checkpoint-") :]
        try:
            return int(suffix)
        except ValueError:
            return None
    return None


def _append_eval_metrics_to_trainer_state(checkpoint_dir: Path, metrics: dict[str, Any]) -> None:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return

    log_history = state.get("log_history")
    if not isinstance(log_history, list):
        log_history = []
        state["log_history"] = log_history

    step = _checkpoint_step_from_dir(checkpoint_dir)
    if step is None:
        step = state.get("global_step")
    if isinstance(step, str) and step.isdigit():
        step = int(step)
    if not isinstance(step, int):
        step = None

    entry: dict[str, Any] = {"event": "eval_only"}
    if step is not None:
        entry["step"] = step
        state["global_step"] = step
    entry.update(metrics)
    log_history.append(entry)

    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build stage-1 SFT data from cid2tid/tid2cid/cid2cid and train only new CID embeddings."
    )
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Directory containing *_cid2*.parquet files")
    p.add_argument("--sft_data_dir", type=str, default=str(DEFAULT_SFT_DATA_DIR), help="Where to save train/valid/test jsonl")
    p.add_argument("--model_name", type=str, default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--train_out_dir", type=str, default=str(DEFAULT_TRAIN_OUT_DIR))
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override total optimizer steps. Use -1 to follow --epochs.",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--loader", choices=["transformers", "unsloth"], default="unsloth")
    p.add_argument("--gradient_checkpointing_mode", choices=["off", "hf", "unsloth"], default="off")
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument(
        "--lm_head_training",
        choices=["cid_rows", "full"],
        default="full",
        help=(
            "How to train lm_head during stage-1. "
            "'cid_rows' keeps the original behavior and only updates the added CID rows. "
            "'full' unties lm_head if needed and trains the full output matrix."
        ),
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint directory, e.g. output/qwen3_1p7b_sft/checkpoint-1200",
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation on --eval_checkpoint (or --resume_from_checkpoint if set), do not train.",
    )
    p.add_argument(
        "--eval_checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint directory to evaluate, e.g. output/.../checkpoint-400",
    )
    p.add_argument("--build_only", action="store_true", help="Only build jsonl, do not train")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _maybe_init_distributed()

    output_dir = Path(args.output_dir)
    sft_data_dir = Path(args.sft_data_dir)
    train_out_dir = Path(args.train_out_dir)
    train_jsonl = sft_data_dir / "train_qwen_sft.jsonl"
    valid_jsonl = sft_data_dir / "valid_qwen_sft.jsonl"
    test_jsonl = sft_data_dir / "test_qwen_sft.jsonl"

    try:
        if _is_main_process():
            train_jsonl, valid_jsonl, test_jsonl = build_sft_jsonl(output_dir=output_dir, save_dir=sft_data_dir)
            print(f"Built SFT files:\n  train={train_jsonl}\n  valid={valid_jsonl}\n  test={test_jsonl}")
        _dist_barrier()

        for path in [train_jsonl, valid_jsonl, test_jsonl]:
            if not path.exists():
                raise FileNotFoundError(f"Expected SFT file does not exist after build: {path}")

        if args.build_only:
            if _is_main_process():
                print("build_only=True, skip training.")
            return 0

        if args.eval_only:
            ckpt = args.eval_checkpoint or args.resume_from_checkpoint
            if not ckpt:
                raise ValueError("--eval_only requires --eval_checkpoint (or --resume_from_checkpoint).")
            eval_dir = Path(ckpt)
            if not eval_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {eval_dir}")
            metrics = eval_qwen_checkpoint(
                valid_jsonl=valid_jsonl,
                model_name=str(eval_dir),
                checkpoint_dir=eval_dir,
                max_seq_len=args.max_seq_len,
                eval_bs=args.eval_bs,
                loader=args.loader,
            )
            if _is_main_process():
                print(f"Eval metrics ({eval_dir}): {metrics}")
                (eval_dir / "eval_metrics.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                _append_eval_metrics_to_trainer_state(eval_dir, metrics)
            return 0

        train_qwen_sft(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            model_name=args.model_name,
            out_dir=train_out_dir,
            max_seq_len=args.max_seq_len,
            epochs=args.epochs,
            max_steps=args.max_steps,
            lr=args.lr,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            grad_accum=args.grad_accum,
            loader=args.loader,
            gradient_checkpointing_mode=args.gradient_checkpointing_mode,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            lm_head_training=args.lm_head_training,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        return 0
    finally:
        _maybe_destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
