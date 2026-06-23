#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_str, load_project_env

load_project_env(REPO_ROOT / ".env")

DEFAULT_INPUT_DIR = Path(env_str("NAVIGEN_INFER_INPUT_DIR", str(REPO_ROOT / "dataset")))
DEFAULT_MODEL_SEARCH_ROOT = Path(env_str("NAVIGEN_STAGE2_OUTPUT_DIR", str(SCRIPT_DIR / "sft_output" / "qwen3_1p7b_sft_fullft_stage2")))
DEFAULT_PRED_DIR = Path(env_str("NAVIGEN_CID2CID_PRED_DIR", str(SCRIPT_DIR / "sft_output" / "inference_outputs" / "stage2_final_cid2cid_constrained")))
DEFAULT_TASK = "hist_cid2cid"
TASK_FILE = "test_cid2cid.parquet"
TARGET_CID_JSON_PREFIX = '{"target_cid":"'
TWO_STAGE_CID2CID_PREFIX = '\n{"target_cid":"<|cid_begin|>'
CID_BEGIN_TOKEN_TEXT = "<|cid_begin|>"
GENERATION_MODE_DIRECT = "direct_json_prefix"
GENERATION_MODE_TWO_STAGE = "strict_two_stage"
PROMPT_MODE_TRAIN = "train"
PROMPT_MODE_SIMPLE_DIRECT = "simple_direct"
SYSTEM_PROMPT = (
    "You are a personalized recommendation assistant. "
    "Predict the target cid interaction based on user history cid interaction. "
    "Think step by step, then answer. "
    "The final answer must be exactly one JSON object with exactly one field: target_cid. "
    "Output schema: {\"target_cid\":\"<|cid_begin|><s_a_x><s_b_y><s_c_z><|cid_end|>\"}. "
    "target_cid must be a single JSON string, not a list. "
    "Do not output markdown, code fences, explanations, or any extra fields."
)
SIMPLE_DIRECT_SYSTEM_PROMPT = (
    "You are a personalized recommendation assistant. "
    "Predict the target cid interaction based on user history cid interaction. "
    "Think step by step, then answer. "
    "The final answer must be a JSON object only, with the field target_cid and no other fields."
)
CID_PATTERN = re.compile(r"^<\|cid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><\|cid_end\|>$")


def _get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _dist_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _maybe_destroy_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
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


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _normalize_target_cid(v: Any) -> str | None:
    first = _first(v)
    if first is None:
        return None
    text = _str_or_empty(first).strip()
    return text or None


def _build_infer_example(row: dict[str, Any], row_idx: int, prompt_mode: str = PROMPT_MODE_TRAIN) -> dict[str, Any] | None:
    hist_cid = [_str_or_empty(x) for x in _as_list(row.get("hist_sid")) if _str_or_empty(x)]
    target_cid = _normalize_target_cid(row.get("target_sid"))
    if not hist_cid or target_cid is None:
        return None

    if prompt_mode == PROMPT_MODE_TRAIN:
        system_prompt = SYSTEM_PROMPT
        user = (
            "Task: Predict the target_cid based on the user's historical interactions in hist_cid.\n"
            f"hist_cid: {json.dumps(hist_cid, ensure_ascii=False)}\n"
            "Output constraints:\n"
            '1. After reasoning, output JSON only: {"target_cid":"<|cid_begin|><s_a_x><s_b_y><s_c_z><|cid_end|>"}\n'
            "2. target_cid must be a single string, not a list.\n"
            "3. No extra text before or after the JSON object."
        )
    elif prompt_mode == PROMPT_MODE_SIMPLE_DIRECT:
        system_prompt = SIMPLE_DIRECT_SYSTEM_PROMPT
        user = (
            "Task: Predict the target_cid based on the user's historical interactions in hist_cid.\n"
            f"hist_cid: {json.dumps(hist_cid, ensure_ascii=False, separators=(',', ':'))}\n"
            "Think first, then output JSON only."
        )
    else:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")

    label = {"target_cid": target_cid}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return {
        "task": DEFAULT_TASK,
        "row_idx": row_idx,
        "prompt_mode": prompt_mode,
        "messages": messages,
        "label": label,
        "source_row": row,
    }


def _discover_model_dir(search_root: Path) -> Path:
    if not search_root.exists():
        raise FileNotFoundError(f"Model search root does not exist: {search_root}")

    candidates = sorted(
        {
            p.parent
            for p in search_root.rglob("model.safetensors")
            if (p.parent / "config.json").exists() and (p.parent / "tokenizer.json").exists()
        }
    )
    if not candidates:
        raise FileNotFoundError(
            f"No full model directory found under {search_root}. "
            "Pass --model_dir explicitly or wait for checkpoint/final to appear."
        )

    def score(path: Path) -> tuple[int, int, float]:
        if path.name == "final":
            return (2, 0, path.stat().st_mtime)
        if path.name.startswith("checkpoint-"):
            try:
                step = int(path.name.split("-", 1)[1])
            except ValueError:
                step = -1
            return (1, step, path.stat().st_mtime)
        return (0, -1, path.stat().st_mtime)

    return max(candidates, key=score)


def _is_full_model_dir(path: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    return all((path / name).exists() for name in required)


def _is_adapter_dir(path: Path) -> bool:
    if not (path / "adapter_config.json").exists():
        return False
    if (path / "adapter_model.safetensors").exists():
        return True
    if (path / "adapter_model.bin").exists():
        return True
    return True


def _load_adapter_base_model_path(adapter_dir: Path) -> Path:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model_name_or_path = str(config.get("base_model_name_or_path", "")).strip()
    if not base_model_name_or_path:
        raise ValueError(f"`base_model_name_or_path` missing in {config_path}")

    base_model_dir = Path(base_model_name_or_path).expanduser()
    if not base_model_dir.is_absolute():
        base_model_dir = (adapter_dir / base_model_dir).resolve()
    else:
        base_model_dir = base_model_dir.resolve()
    return base_model_dir


def _resolve_model_dir(model_dir: str | None, search_root: Path) -> tuple[str, Path, Path | None]:
    if model_dir:
        resolved = Path(model_dir).expanduser().resolve()
        if _is_full_model_dir(resolved):
            return "full", resolved, None
        if _is_adapter_dir(resolved):
            base_model_dir = _load_adapter_base_model_path(resolved)
            if not _is_full_model_dir(base_model_dir):
                raise FileNotFoundError(
                    f"Adapter base model resolved from {resolved / 'adapter_config.json'} "
                    f"to {base_model_dir}, but it is not a full model directory."
                )
            return "adapter", resolved, base_model_dir
        raise FileNotFoundError(
            f"{resolved} is neither a full model directory nor a PEFT adapter directory."
        )

    discovered = _discover_model_dir(search_root.resolve())
    return "full", discovered, None


def _sanitize_model_source_part(text: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_-")
    return sanitized or "unknown"


def _build_model_source(model_dir: Path, artifact_kind: str, base_model_dir: Path | None = None) -> str:
    parts: list[str] = []
    if artifact_kind == "adapter":
        if base_model_dir is not None:
            parts.extend([base_model_dir.parent.name, base_model_dir.name])
        parts.extend(["lora", model_dir.parent.name, model_dir.name])
    else:
        parts = [model_dir.parent.name, model_dir.name]
    return "__".join(_sanitize_model_source_part(part) for part in parts if part)


def _append_model_source_to_name(filename: str, model_source: str) -> str:
    path = Path(filename)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return f"{stem}.{model_source}{suffix}"


def _get_torch_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_model_and_tokenizer(artifact_kind: str, model_dir: Path, base_model_dir: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, local_rank, world_size = _get_dist_info()
    dtype = _get_torch_dtype()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if world_size > 1 else 0)
        device_map = {"": local_rank if world_size > 1 else 0}
    else:
        device_map = None

    tokenizer_source = base_model_dir if artifact_kind == "adapter" and base_model_dir is not None else model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    if tokenizer.eos_token is None and "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "left"

    if artifact_kind == "adapter":
        from peft import PeftModel
        import peft.utils.save_and_load as peft_save_load

        original_tp_shard = peft_save_load._maybe_shard_state_dict_for_tp

        def _skip_missing_transformers_tp_symbols(model, state_dict, adapter_name):
            try:
                from transformers.integrations.tensor_parallel import EmbeddingParallel  # noqa: F401
            except ImportError:
                return
            return original_tp_shard(model, state_dict, adapter_name)

        peft_save_load._maybe_shard_state_dict_for_tp = _skip_missing_transformers_tp_symbols

        if base_model_dir is None:
            raise ValueError("base_model_dir is required when loading a PEFT adapter.")
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_dir),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            str(model_dir),
            torch_dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def _infer_device(model) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def _parse_response_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate_text = _strip_fences(text)
    if "</think>" in candidate_text:
        candidate_text = candidate_text.rsplit("</think>", 1)[1].strip()

    starts = [idx for idx, ch in enumerate(candidate_text) if ch == "{"]
    last_error = "No JSON object found in model output."

    for start in reversed(starts):
        depth = 0
        for end in range(start, len(candidate_text)):
            ch = candidate_text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    fragment = candidate_text[start : end + 1]
                    try:
                        data = json.loads(fragment)
                        if isinstance(data, dict):
                            return data, None
                        last_error = "Parsed JSON is not an object."
                    except json.JSONDecodeError as exc:
                        last_error = str(exc)
                    break

    return None, last_error


def _get_gold_target_cid(source_row: dict[str, Any], label: dict[str, Any] | None = None) -> str | None:
    gold = _normalize_target_cid(source_row.get("target_sid"))
    if gold is not None:
        return gold
    if label:
        return _normalize_target_cid(label.get("target_cid"))
    return None


def _ndcg_at_k(predictions: list[str], gold_target_cid: str | None, k: int) -> float:
    if not gold_target_cid:
        return 0.0
    for rank, pred in enumerate(predictions[:k], start=1):
        if pred == gold_target_cid:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def _iter_cid_strings(value: Any):
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if CID_PATTERN.match(text):
            yield text
        return
    if isinstance(value, dict):
        for inner in value.values():
            yield from _iter_cid_strings(inner)
        return
    if isinstance(value, (list, tuple)):
        for inner in value:
            yield from _iter_cid_strings(inner)
        return
    if hasattr(value, "tolist"):
        yield from _iter_cid_strings(value.tolist())


def _collect_cid_catalog_from_parquet(path: Path) -> set[str]:
    rows = _read_parquet_rows(path)
    catalog: set[str] = set()
    for row in rows:
        for cid in _iter_cid_strings(row):
            catalog.add(cid)
    return catalog


def _discover_catalog_files(input_dir: Path, explicit_files: str | None) -> list[Path]:
    if explicit_files:
        files = [Path(part.strip()).expanduser().resolve() for part in explicit_files.split(",") if part.strip()]
    else:
        files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No catalog parquet files found under {input_dir}")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Catalog parquet files do not exist: {', '.join(missing)}")
    return files


def _single_token_id(tokenizer, text: str, label: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {label} to map to a single token, got ids={token_ids}")
    return token_ids[0]


def _build_allowed_cid_token_seqs(tokenizer, cid_catalog: set[str]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for cid in sorted(cid_catalog):
        token_ids = tokenizer.encode(cid, add_special_tokens=False)
        if not token_ids:
            continue
        sequences.append(token_ids)
    if not sequences:
        raise ValueError("No valid CID token sequences could be built from the catalog.")
    return sequences


def _build_trie(token_sequences: list[list[int]], close_token_id: int) -> dict[int, Any]:
    root: dict[int, Any] = {}
    for seq in token_sequences:
        node = root
        for token_id in seq:
            node = node.setdefault(token_id, {})
        node[close_token_id] = {}
    return root


def _follow_trie(trie: dict[int, Any], prefix: list[int]) -> dict[int, Any] | None:
    node = trie
    for token_id in prefix:
        node = node.get(token_id)
        if node is None:
            return None
    return node


def _build_continuation_trie(cid_trie: dict[int, Any], tokenizer, cid_prefill: str) -> dict[int, Any]:
    prefill_ids = tokenizer.encode(cid_prefill, add_special_tokens=False)
    node = _follow_trie(cid_trie, prefill_ids)
    if node is None:
        raise ValueError(f"CID trie does not contain prefill {cid_prefill!r} token ids={prefill_ids}")
    return node


def _make_stop_sequence_criteria(tokenizer, stop_text: str):
    import torch
    from transformers import StoppingCriteria

    stop_ids = tokenizer.encode(stop_text, add_special_tokens=False)
    if not stop_ids:
        return None

    class StopOnSequence(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
            if input_ids.shape[1] < len(stop_ids):
                return False
            suffix = input_ids[:, -len(stop_ids) :].tolist()
            return all(row == stop_ids for row in suffix)

    return StopOnSequence()


def _generate_with_trie(
    model,
    tokenizer,
    prompts_with_prefix: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_candidates: int,
    cid_trie: dict[int, Any],
    close_token_id: int,
    output_prefix: str,
) -> list[list[str]]:
    import torch

    if not prompts_with_prefix:
        return []

    device = _infer_device(model)
    encoded = tokenizer(
        prompts_with_prefix,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_token_lens = encoded["attention_mask"].sum(dim=1).tolist()
    prompt_width = encoded["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id

    def _prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        if pad_token_id is None:
            current_ids = input_ids.tolist()
        else:
            current_ids = [token_id for token_id in input_ids.tolist() if token_id != pad_token_id]

        generated_ids = current_ids[prompt_token_lens[batch_id] :]
        node = _follow_trie(cid_trie, generated_ids)
        if node is None:
            return [close_token_id]
        allowed = sorted(node.keys())
        return allowed or [close_token_id]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": close_token_id,
        "num_return_sequences": num_candidates,
        "prefix_allowed_tokens_fn": _prefix_allowed_tokens_fn,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    elif num_candidates > 1:
        generate_kwargs["num_beams"] = num_candidates
        generate_kwargs["early_stopping"] = True

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generate_kwargs)

    flat_texts: list[str] = []
    for seq in output_ids:
        generated_ids = seq[prompt_width:]
        cid_suffix_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        text = output_prefix + cid_suffix_text
        flat_texts.append(text)

    grouped_texts: list[list[str]] = []
    for start in range(0, len(flat_texts), num_candidates):
        grouped_texts.append(flat_texts[start : start + num_candidates])
    return grouped_texts


def _batch_generate_direct(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_candidates: int,
    cid_trie: dict[int, Any],
    close_token_id: int,
) -> list[list[str]]:
    prompts_with_prefix = [prompt + TARGET_CID_JSON_PREFIX for prompt in prompts]
    return _generate_with_trie(
        model=model,
        tokenizer=tokenizer,
        prompts_with_prefix=prompts_with_prefix,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_candidates=num_candidates,
        cid_trie=cid_trie,
        close_token_id=close_token_id,
        output_prefix=TARGET_CID_JSON_PREFIX,
    )


def _batch_generate_two_stage(
    model,
    tokenizer,
    prompts: list[str],
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_candidates: int,
    stage1_num_candidates: int,
    cid_trie: dict[int, Any],
    close_token_id: int,
    stage1_stop_text: str,
) -> list[list[str]]:
    import torch
    from transformers import StoppingCriteriaList

    if not prompts:
        return []

    device = _infer_device(model)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    stage1_kwargs = {
        "max_new_tokens": stage1_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": stage1_num_candidates,
    }
    if do_sample:
        stage1_kwargs["temperature"] = temperature
        stage1_kwargs["top_p"] = top_p
    elif stage1_num_candidates > 1:
        stage1_kwargs["num_beams"] = stage1_num_candidates
        stage1_kwargs["early_stopping"] = True

    stop_criteria = _make_stop_sequence_criteria(tokenizer, stage1_stop_text)
    if stop_criteria is not None:
        stage1_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])

    with torch.inference_mode():
        stage1_output_ids = model.generate(**encoded, **stage1_kwargs)

    stage1_texts: list[str] = []
    for seq in stage1_output_ids:
        generated_ids = seq[prompt_width:]
        stage1_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if stage1_stop_text and stage1_stop_text in stage1_text:
            stage1_text = stage1_text.split(stage1_stop_text, 1)[0] + stage1_stop_text
        stage1_texts.append(stage1_text)

    stage2_prompts = [
        prompt + stage1_text + TWO_STAGE_CID2CID_PREFIX
        for prompt, group in zip(prompts, _group_flat(stage1_texts, stage1_num_candidates))
        for stage1_text in group
    ]
    continuation_trie = _build_continuation_trie(cid_trie, tokenizer, CID_BEGIN_TOKEN_TEXT)
    stage2_candidate_groups = _generate_with_trie(
        model=model,
        tokenizer=tokenizer,
        prompts_with_prefix=stage2_prompts,
        max_new_tokens=stage2_max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        num_candidates=num_candidates,
        cid_trie=continuation_trie,
        close_token_id=close_token_id,
        output_prefix="",
    )

    flat_raw_texts: list[str] = []
    for stage1_text, generated_group in zip(stage1_texts, stage2_candidate_groups):
        for generated_text in generated_group:
            flat_raw_texts.append(stage1_text + TWO_STAGE_CID2CID_PREFIX + generated_text)

    per_prompt_groups: list[list[str]] = []
    per_prompt_width = stage1_num_candidates * num_candidates
    for group in _group_flat(flat_raw_texts, per_prompt_width):
        per_prompt_groups.append(group[:num_candidates])
    return per_prompt_groups


def _group_flat(items: list[str], group_size: int) -> list[list[str]]:
    return [items[start : start + group_size] for start in range(0, len(items), group_size)]


def _batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_candidates: int,
    cid_trie: dict[int, Any],
    close_token_id: int,
    generation_mode: str,
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    stage1_num_candidates: int,
    stage1_stop_text: str,
) -> list[list[str]]:
    if generation_mode == GENERATION_MODE_DIRECT:
        return _batch_generate_direct(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_candidates=num_candidates,
            cid_trie=cid_trie,
            close_token_id=close_token_id,
        )
    if generation_mode == GENERATION_MODE_TWO_STAGE:
        return _batch_generate_two_stage(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage2_max_new_tokens=stage2_max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_candidates=num_candidates,
            stage1_num_candidates=stage1_num_candidates,
            cid_trie=cid_trie,
            close_token_id=close_token_id,
            stage1_stop_text=stage1_stop_text,
        )
    raise ValueError(f"Unsupported generation_mode={generation_mode!r}")


def _iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _shard_examples(examples: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    if world_size <= 1:
        return examples
    return [example for idx, example in enumerate(examples) if idx % world_size == rank]


def _merge_rank_outputs(part_paths: list[Path], merged_path: Path) -> Path:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in part_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    records.sort(key=lambda record: record.get("row_idx", -1))
    with merged_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return merged_path


def _load_examples(input_dir: Path, max_rows: int | None, prompt_mode: str) -> tuple[Path, list[dict[str, Any]]]:
    path = input_dir / TASK_FILE
    rows = _read_parquet_rows(path)
    if max_rows is not None:
        rows = rows[:max_rows]

    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        example = _build_infer_example(row, idx, prompt_mode=prompt_mode)
        if example is not None:
            examples.append(example)
    return path, examples


def _summarize_metrics(out_path: Path, recall_ks: list[int]) -> dict[str, Any]:
    total = 0
    parsed_top1 = 0
    exact_match = 0
    recall_sums = {k: 0.0 for k in recall_ks}
    hit_counts = {k: 0 for k in recall_ks}
    ndcg_sums = {k: 0.0 for k in recall_ks}
    any_parsed = 0

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total += 1
            label = record.get("label") or {}
            source_row = record.get("source_row") or {}
            gold_target_cid = _get_gold_target_cid(source_row, label)
            candidates = record.get("candidates") or []

            candidate_target_cids: list[str] = []
            top1_target_cid: str | None = None
            top1_is_parsed = False
            parsed_any_this_example = False

            for idx, candidate in enumerate(candidates):
                parsed_json = candidate.get("parsed_json")
                if isinstance(parsed_json, dict):
                    parsed_any_this_example = True
                    parsed_target_cid = _normalize_target_cid(parsed_json.get("target_cid"))
                    if parsed_target_cid is not None:
                        candidate_target_cids.append(parsed_target_cid)
                    if idx == 0:
                        top1_is_parsed = True
                        top1_target_cid = parsed_target_cid

            unique_candidate_target_cids: list[str] = []
            seen_target_cids: set[str] = set()
            for target_cid in candidate_target_cids:
                if target_cid and target_cid not in seen_target_cids:
                    seen_target_cids.add(target_cid)
                    unique_candidate_target_cids.append(target_cid)

            if parsed_any_this_example:
                any_parsed += 1
            if top1_is_parsed:
                parsed_top1 += 1
            if top1_target_cid == gold_target_cid:
                exact_match += 1
            for k in recall_ks:
                topk_predictions = unique_candidate_target_cids[:k]
                hit = gold_target_cid in topk_predictions if gold_target_cid else False
                recall_sums[k] += 1.0 if hit else 0.0
                if hit:
                    hit_counts[k] += 1
                ndcg_sums[k] += _ndcg_at_k(unique_candidate_target_cids, gold_target_cid, k)

    sample_record: dict[str, Any] | None = None
    with out_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line:
            sample_record = json.loads(first_line)

    metrics = {
        "task": DEFAULT_TASK,
        "total_examples": total,
        "top1_parsed_examples": parsed_top1,
        "top1_json_parse_rate": (parsed_top1 / total) if total else 0.0,
        "any_candidate_parsed_examples": any_parsed,
        "any_candidate_parse_rate": (any_parsed / total) if total else 0.0,
        "exact_match_count": exact_match,
        "exact_match_accuracy": (exact_match / total) if total else 0.0,
        "exact_match_accuracy_on_top1_parsed": (exact_match / parsed_top1) if parsed_top1 else 0.0,
        "inference_model_source": (sample_record or {}).get("inference_model_source"),
        "model_artifact_kind": (sample_record or {}).get("model_artifact_kind"),
        "inference_model_dir": (sample_record or {}).get("model_dir"),
        "base_model_dir": (sample_record or {}).get("base_model_dir"),
        "catalog_size": (sample_record or {}).get("catalog_size"),
    }
    for k in recall_ks:
        metrics[f"recall@{k}"] = (recall_sums[k] / total) if total else 0.0
        metrics[f"hit_rate@{k}"] = (hit_counts[k] / total) if total else 0.0
        metrics[f"hit_count@{k}"] = hit_counts[k]
        metrics[f"ndcg@{k}"] = (ndcg_sums[k] / total) if total else 0.0

    metrics_filename = f"{Path(out_path.stem).stem}_metrics.json"
    metrics_path = out_path.with_name(
        _append_model_source_to_name(metrics_filename, metrics["inference_model_source"] or "unknown")
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def _run_task(
    parquet_path: Path,
    examples: list[dict[str, Any]],
    model,
    tokenizer,
    pred_dir: Path,
    artifact_kind: str,
    model_dir: Path,
    base_model_dir: Path | None,
    model_source: str,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_candidates: int,
    cid_trie: dict[int, Any],
    close_token_id: int,
    catalog_size: int,
    generation_mode: str,
    stage1_max_new_tokens: int,
    stage2_max_new_tokens: int,
    stage1_num_candidates: int,
    stage1_stop_text: str,
) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    rank, _, world_size = _get_dist_info()
    base_name = _append_model_source_to_name(f"{DEFAULT_TASK}_predictions.jsonl", model_source)
    if world_size > 1:
        out_path = pred_dir / base_name.replace(".jsonl", f".rank{rank}.jsonl")
    else:
        out_path = pred_dir / base_name

    total_batches = max(1, math.ceil(len(examples) / batch_size))
    with out_path.open("w", encoding="utf-8") as f:
        for batch_idx, (start, batch) in enumerate(_iter_batches(examples, batch_size), start=1):
            prompts = [
                tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for example in batch
            ]
            candidate_text_groups = _batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_candidates=num_candidates,
                cid_trie=cid_trie,
                close_token_id=close_token_id,
                generation_mode=generation_mode,
                stage1_max_new_tokens=stage1_max_new_tokens,
                stage2_max_new_tokens=stage2_max_new_tokens,
                stage1_num_candidates=stage1_num_candidates,
                stage1_stop_text=stage1_stop_text,
            )

            for example, prompt_text, candidate_texts in zip(batch, prompts, candidate_text_groups):
                candidates: list[dict[str, Any]] = []
                candidate_target_cids: list[str] = []
                for candidate_rank, raw_text in enumerate(candidate_texts, start=1):
                    parsed_json, parse_error = _parse_response_json(raw_text)
                    parsed_target_cid = None
                    if isinstance(parsed_json, dict):
                        parsed_target_cid = _normalize_target_cid(parsed_json.get("target_cid"))
                        if parsed_target_cid is not None:
                            candidate_target_cids.append(parsed_target_cid)
                    candidates.append(
                        {
                            "rank": candidate_rank,
                            "raw_text": raw_text,
                            "parsed_json": parsed_json,
                            "parse_error": parse_error,
                            "parsed_target_cid": parsed_target_cid,
                        }
                    )

                top_candidate = candidates[0] if candidates else None
                unique_candidate_target_cids: list[str] = []
                seen_target_cids: set[str] = set()
                for target_cid in candidate_target_cids:
                    if target_cid and target_cid not in seen_target_cids:
                        seen_target_cids.add(target_cid)
                        unique_candidate_target_cids.append(target_cid)

                record = {
                    "task": DEFAULT_TASK,
                    "row_idx": example["row_idx"],
                    "source_file": str(parquet_path),
                    "inference_model_source": model_source,
                    "model_artifact_kind": artifact_kind,
                    "model_dir": str(model_dir),
                    "base_model_dir": str(base_model_dir) if base_model_dir is not None else "",
                    "prompt_mode": example.get("prompt_mode", ""),
                    "prompt_text": prompt_text,
                    "num_candidates": num_candidates,
                    "catalog_size": catalog_size,
                    "top_candidate": top_candidate,
                    "gold_target_cid": _get_gold_target_cid(example["source_row"], example["label"]),
                    "candidate_target_cids": unique_candidate_target_cids,
                    "candidates": candidates,
                    "label": example["label"],
                    "source_row": example["source_row"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            end = start + len(batch)
            print(
                f"[{DEFAULT_TASK}][rank{rank}] batch {batch_idx}/{total_batches} "
                f"rows {start}:{end} -> {out_path}"
            )

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run stage2 constrained cid2cid inference with legal-CID trie decoding."
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--search_root", type=str, default=str(DEFAULT_MODEL_SEARCH_ROOT))
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Explicit full-model directory or PEFT adapter directory for inference",
    )
    p.add_argument("--pred_dir", type=str, default=str(DEFAULT_PRED_DIR))
    p.add_argument("--catalog_files", type=str, default=None, help="Comma-separated parquet paths used to build the legal CID catalog; defaults to all parquet files in input_dir")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap for quick smoke tests")
    p.add_argument(
        "--prompt_mode",
        choices=[PROMPT_MODE_TRAIN, PROMPT_MODE_SIMPLE_DIRECT],
        default=PROMPT_MODE_TRAIN,
        help="train uses the RL training prompt; simple_direct restores the original short prompt intended for direct_json_prefix inference.",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument(
        "--generation_mode",
        choices=[GENERATION_MODE_DIRECT, GENERATION_MODE_TWO_STAGE],
        default=GENERATION_MODE_DIRECT,
        help="direct_json_prefix keeps the current direct JSON-prefix trie decoding; strict_two_stage first generates reasoning, then appends the training-time CID JSON prefix and trie-decodes the CID continuation.",
    )
    p.add_argument("--stage1_max_new_tokens", type=int, default=768)
    p.add_argument("--stage2_max_new_tokens", type=int, default=32)
    p.add_argument(
        "--stage1_num_candidates",
        type=int,
        default=1,
        help="Number of reasoning candidates generated in strict_two_stage before CID trie decoding. Use 1 for fast inference; >1 explores multiple reasoning paths.",
    )
    p.add_argument("--stage1_stop_text", type=str, default="</think>")
    p.add_argument("--num_candidates", type=int, default=40, help="How many candidate generations to keep per example for ranking metrics")
    p.add_argument("--recall_ks", type=str, default="1,5,10,20,40", help="Comma-separated K values for Recall@K and NDCG@K")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _maybe_init_distributed()

    try:
        input_dir = Path(args.input_dir).resolve()
        search_root = Path(args.search_root).resolve()
        pred_dir = Path(args.pred_dir).resolve()
        recall_ks = sorted({int(v.strip()) for v in args.recall_ks.split(",") if v.strip()})
        if not recall_ks or any(k <= 0 for k in recall_ks):
            raise ValueError(f"Invalid --recall_ks: {args.recall_ks}")
        if args.num_candidates <= 0:
            raise ValueError(f"--num_candidates must be > 0, got {args.num_candidates}")

        effective_num_candidates = max(args.num_candidates, max(recall_ks))
        if effective_num_candidates != args.num_candidates and _is_main_process():
            print(
                f"[{DEFAULT_TASK}] num_candidates={args.num_candidates} is smaller than max K={max(recall_ks)}; "
                f"using num_candidates={effective_num_candidates} instead."
            )

        artifact_kind, model_dir, base_model_dir = _resolve_model_dir(args.model_dir, search_root)
        model_source = _build_model_source(model_dir, artifact_kind=artifact_kind, base_model_dir=base_model_dir)
        print(
            f"[{DEFAULT_TASK}][rank{rank}] using {artifact_kind} model artifact: {model_dir} "
            f"(base={base_model_dir or '-'}, source={model_source}, world_size={world_size}, local_rank={local_rank})"
        )

        model, tokenizer = _load_model_and_tokenizer(
            artifact_kind=artifact_kind,
            model_dir=model_dir,
            base_model_dir=base_model_dir,
        )
        close_token_id = _single_token_id(tokenizer, '"}', "JSON close token")

        catalog_files = _discover_catalog_files(input_dir=input_dir, explicit_files=args.catalog_files)
        cid_catalog: set[str] = set()
        for catalog_file in catalog_files:
            cid_catalog.update(_collect_cid_catalog_from_parquet(catalog_file))
        if not cid_catalog:
            raise ValueError("Collected empty CID catalog; check --catalog_files or input parquet contents.")

        cid_token_seqs = _build_allowed_cid_token_seqs(tokenizer, cid_catalog)
        cid_trie = _build_trie(cid_token_seqs, close_token_id=close_token_id)
        if _is_main_process():
            print(
                f"[{DEFAULT_TASK}] built legal CID trie from {len(catalog_files)} parquet files; "
                f"catalog_size={len(cid_catalog)} tokenized_paths={len(cid_token_seqs)}"
            )

        parquet_path, examples = _load_examples(input_dir=input_dir, max_rows=args.max_rows, prompt_mode=args.prompt_mode)
        if not examples:
            if _is_main_process():
                print(f"[{DEFAULT_TASK}] no valid examples in {parquet_path}, skip.")
            return 0

        sharded_examples = _shard_examples(examples, rank=rank, world_size=world_size)
        print(
            f"[{DEFAULT_TASK}][rank{rank}] loaded total={len(examples)} "
            f"local_shard={len(sharded_examples)} from {parquet_path}"
        )

        shard_out_path = _run_task(
            parquet_path=parquet_path,
            examples=sharded_examples,
            model=model,
            tokenizer=tokenizer,
            pred_dir=pred_dir,
            artifact_kind=artifact_kind,
            model_dir=model_dir,
            base_model_dir=base_model_dir,
            model_source=model_source,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_candidates=effective_num_candidates,
            cid_trie=cid_trie,
            close_token_id=close_token_id,
            catalog_size=len(cid_catalog),
            generation_mode=args.generation_mode,
            stage1_max_new_tokens=args.stage1_max_new_tokens,
            stage2_max_new_tokens=args.stage2_max_new_tokens,
            stage1_num_candidates=args.stage1_num_candidates,
            stage1_stop_text=args.stage1_stop_text,
        )

        _dist_barrier()

        if _is_main_process():
            if world_size > 1:
                final_name = _append_model_source_to_name(f"{DEFAULT_TASK}_predictions.jsonl", model_source)
                final_out_path = pred_dir / final_name
                part_paths = [
                    pred_dir / final_name.replace(".jsonl", f".rank{part_rank}.jsonl")
                    for part_rank in range(world_size)
                ]
                merged_path = _merge_rank_outputs(part_paths=part_paths, merged_path=final_out_path)
            else:
                merged_path = shard_out_path

            metrics = _summarize_metrics(merged_path, recall_ks=recall_ks)
            print(f"[{DEFAULT_TASK}] wrote predictions to {merged_path}")
            print(
                f"[{DEFAULT_TASK}] metrics: "
                f"top1_json_parse_rate={metrics['top1_json_parse_rate']:.4f}, "
                f"exact_match_accuracy={metrics['exact_match_accuracy']:.4f}, "
                + ", ".join(
                    f"recall@{k}={metrics[f'recall@{k}']:.4f}, ndcg@{k}={metrics[f'ndcg@{k}']:.4f}"
                    for k in recall_ks
                )
            )

        _dist_barrier()
        return 0
    finally:
        _maybe_destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
