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
DEFAULT_MODEL_SEARCH_ROOT = Path(env_str("NAVIGEN_STAGE2_CID2INS_OUTPUT_DIR", str(SCRIPT_DIR / "sft_output" / "qwen3_1p7b_sft_fullft_stage2_cid2ins")))
DEFAULT_PRED_DIR = Path(env_str("NAVIGEN_CID2INS_PRED_DIR", str(SCRIPT_DIR / "sft_output" / "inference_outputs" / "stage2_cid2ins")))
DEFAULT_TASK = "hist_cid2ins"
TASK_FILE = "test_cid2ins.parquet"
PROMPT_MODE_TRAIN = "train"
PROMPT_MODE_SFT = "sft"
TRAIN_SYSTEM_PROMPT = (
    "You are a personalized recommendation and AIGC instruction generation assistant. "
    "Based on user history cid interaction, output user target tid interaction and the AIGC instruction . "
    "Think step by step, then answer. "
    "The final answer must be exactly one JSON object with exactly two fields in this order: "
    "target_tid, target_ins. "
    "Output schema: {\"target_tid\":[\"tid_1\",\"tid_2\"],\"target_ins\":\"...\"}. "
    "target_tid must be a non-empty JSON array of strings. "
    "target_ins must be a non-empty JSON string. "
    "Do not output markdown, code fences, explanations, comments, trailing commas, or extra fields."
)
SFT_SYSTEM_PROMPT = (
    "You are a personalized recommendation and AIGC instruction generation assistant. "
    "Based on user history cid interaction, output user target tid interaction and the AIGC instruction. "
    "Think step by step, then answer. "
    "The final answer must be a JSON object only, with the fields target_tid and target_ins."
)
CID2INS_STAGE2_PREFIX = '\n{"target_tid":["'
CID2INS_STAGE2_REGEX = (
    r'^(?:[^"\\]|\\.)+"(?:\s*,\s*"(?:[^"\\]|\\.)+")*'
    r'\s*\]\s*,\s*"target_ins"\s*:\s*"(?:[^"\\]|\\.)+"\s*\}$'
)


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


def _str_or_empty(v: Any) -> str:
    return "" if v is None else str(v)


def _string_list(v: Any) -> list[str]:
    return [_str_or_empty(x) for x in _as_list(v) if _str_or_empty(x)]


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _build_infer_example(row: dict[str, Any], row_idx: int, prompt_mode: str = PROMPT_MODE_TRAIN) -> dict[str, Any] | None:
    hist_cid = _string_list(row.get("hist_sid"))
    target_tid = _string_list(row.get("target_tid"))
    target_ins = _str_or_empty(row.get("target_ins")).strip()
    if not hist_cid or not target_tid or not target_ins:
        return None

    if prompt_mode == PROMPT_MODE_TRAIN:
        system = TRAIN_SYSTEM_PROMPT
        user = (
            "Task: Generate the target recommendation result based on the user's historical interactions in hist_cid.\n"
            "Output constraints:\n"
            '1. After reasoning, output JSON only: {"target_tid":["tid_1","tid_2"],"target_ins":"..."}\n'
            "2. The JSON field order must be target_tid first, then target_ins.\n"
            "3. target_tid must be a non-empty array of strings.\n"
            "4. target_ins must be a non-empty string.\n"
            "5. No markdown, no code fence, no commentary, no extra fields.\n"
            f"hist_cid: {json.dumps(hist_cid, ensure_ascii=False)}"
        )
    elif prompt_mode == PROMPT_MODE_SFT:
        system = SFT_SYSTEM_PROMPT
        user = (
            "Task: Generate the target recommendation result based on the user's historical interactions in hist_cid.\n"
            "The output must be JSON with the fields target_tid and target_ins.\n"
            f"hist_cid: {json.dumps(hist_cid, ensure_ascii=False)}"
        )
    else:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")

    label = {
        "target_tid": target_tid,
        "target_ins": target_ins,
    }
    messages = [
        {"role": "system", "content": system},
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
    if tokenizer.eos_token is None and "</think>" in tokenizer.get_vocab():
        tokenizer.eos_token = "</think>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</think>"
    tokenizer.padding_side = "left"

    if artifact_kind == "adapter":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_dir if base_model_dir is not None else model_dir),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(model_dir))
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


def _strip_think_and_fence(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE)
    return _strip_fences(text)


def _extract_json_spans(text: str) -> list[tuple[int, int, Any]]:
    decoder = json.JSONDecoder()
    spans: list[tuple[int, int, Any]] = []
    cursor = 0
    while cursor < len(text):
        match = re.search(r"\{", text[cursor:])
        if not match:
            break
        start = cursor + match.start()
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        spans.append((start, start + end, value))
        cursor = start + max(end, 1)
    return spans


def _extract_json_field_value(text: str, field_name: str) -> Any:
    decoder = json.JSONDecoder()
    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*')
    for match in pattern.finditer(text):
        candidate = text[match.end() :].lstrip()
        if not candidate:
            continue
        try:
            value, _ = decoder.raw_decode(candidate)
            return value
        except json.JSONDecodeError:
            continue
    return None


def _extract_relaxed_string_field_values(text: str, field_name: str) -> list[str]:
    pattern = re.compile(
        rf'"?{re.escape(field_name)}"?\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)',
        flags=re.DOTALL,
    )
    values: list[str] = []
    for match in pattern.finditer(text):
        value = match.group("value").strip()
        if not value:
            continue
        try:
            value = json.loads(json.dumps(value))
        except Exception:
            pass
        if len(value) < 10:
            continue
        if "target_tid" in value or "target_ins" in value:
            continue
        values.append(value)
    return values


def _select_relaxed_target_ins(text: str) -> str:
    candidates = _extract_relaxed_string_field_values(text, "target_ins")
    if not candidates:
        return ""

    def score(candidate: str) -> tuple[int, int]:
        noisy = int("{" in candidate or "}" in candidate or "</think>" in candidate)
        return (-noisy, len(candidate))

    return max(candidates, key=score).strip()


def _has_required_cid2ins_fields(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    target_tid = _string_list(data.get("target_tid"))
    target_ins = _str_or_empty(data.get("target_ins")).strip()
    return bool(target_tid and target_ins)


def _recover_cid2ins_fields(text: str) -> dict[str, Any]:
    cleaned = _strip_think_and_fence(text)
    recovered: dict[str, Any] = {}

    target_tid = _extract_json_field_value(cleaned, "target_tid")
    target_tid_list = _string_list(target_tid)
    if target_tid_list:
        recovered["target_tid"] = target_tid_list

    target_ins = _extract_json_field_value(cleaned, "target_ins")
    target_ins_text = _str_or_empty(target_ins).strip()
    if not target_ins_text:
        target_ins_text = _select_relaxed_target_ins(cleaned)
    if target_ins_text:
        recovered["target_ins"] = target_ins_text

    return recovered


def _merge_recovered_cid2ins_fields(
    data: dict[str, Any] | None,
    recovered: dict[str, Any],
) -> dict[str, Any] | None:
    if data is None:
        return recovered or None
    if not recovered:
        return data

    merged = dict(data)
    if not _string_list(merged.get("target_tid")) and recovered.get("target_tid"):
        merged["target_tid"] = recovered["target_tid"]
    if not _str_or_empty(merged.get("target_ins")).strip() and recovered.get("target_ins"):
        merged["target_ins"] = recovered["target_ins"]
    return merged


def _parse_response_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_think_and_fence(text)
    recovered = _recover_cid2ins_fields(cleaned)
    spans = _extract_json_spans(cleaned)
    dicts = [value for _, _, value in spans if isinstance(value, dict)]

    for data in dicts:
        merged = _merge_recovered_cid2ins_fields(data, recovered)
        if _has_required_cid2ins_fields(merged):
            return merged, None

    best = _merge_recovered_cid2ins_fields(dicts[0], recovered) if dicts else (recovered or None)
    if best is None:
        return None, "No JSON object or recoverable target fields found in model output."
    if not _string_list(best.get("target_tid")):
        return best, "Missing or empty required field: target_tid."
    if not _str_or_empty(best.get("target_ins")).strip():
        return best, "Missing or empty required field: target_ins."
    return best, None
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


JSON_PREFIX = ''


def _truncate_to_json(text: str) -> str:
    """Truncate text to the first complete JSON object found within the text."""
    # Find the first '{' to start from
    start = text.find('{')
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            if not escape_next:
                in_string = not in_string
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
    return text


def _make_json_stop_criteria(tokenizer):
    """Stop generation when a complete JSON is followed by </think> or newline."""
    from transformers import StoppingCriteria

    stop_seqs = [
        tokenizer.encode("}\n", add_special_tokens=False),
        tokenizer.encode("}\n\n", add_special_tokens=False),
    ]
    stop_seqs = [s for s in stop_seqs if s]
    if not stop_seqs:
        return None

    class StopOnJSON(StoppingCriteria):
        def __init__(self):
            self.stopped = set()

        def __call__(self, input_ids, scores):
            for batch_id in range(input_ids.shape[0]):
                if batch_id in self.stopped:
                    continue
                ids = input_ids[batch_id].tolist()
                for stop_seq in stop_seqs:
                    if len(ids) >= len(stop_seq) and ids[-len(stop_seq):] == stop_seq:
                        self.stopped.add(batch_id)
                        break
            return len(self.stopped) == input_ids.shape[0]

    return StopOnJSON()


def _make_stop_on_text(tokenizer, stop_texts: list[str]):
    from transformers import StoppingCriteria

    stop_token_seqs = [
        tokenizer.encode(stop_text, add_special_tokens=False)
        for stop_text in stop_texts
        if stop_text
    ]
    stop_token_seqs = [seq for seq in stop_token_seqs if seq]
    if not stop_token_seqs:
        return None

    class StopOnText(StoppingCriteria):
        def __init__(self):
            self.stopped = set()

        def __call__(self, input_ids, scores):
            for batch_id in range(input_ids.shape[0]):
                if batch_id in self.stopped:
                    continue
                ids = input_ids[batch_id].tolist()
                for stop_seq in stop_token_seqs:
                    if len(ids) >= len(stop_seq) and ids[-len(stop_seq) :] == stop_seq:
                        self.stopped.add(batch_id)
                        break
            return len(self.stopped) == input_ids.shape[0]

    return StopOnText()


def _make_stop_on_regex_completion(tokenizer, prompt_width: int, regex_text: str):
    from transformers import StoppingCriteria

    pattern = re.compile(regex_text, flags=re.DOTALL)

    class StopOnRegexCompletion(StoppingCriteria):
        def __init__(self):
            self.stopped = set()

        def __call__(self, input_ids, scores):
            for batch_id in range(input_ids.shape[0]):
                if batch_id in self.stopped:
                    continue
                generated_ids = input_ids[batch_id][prompt_width:].tolist()
                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if pattern.fullmatch(text.strip()):
                    self.stopped.add(batch_id)
            return len(self.stopped) == input_ids.shape[0]

    return StopOnRegexCompletion()


def _build_regex_prefix_allowed_tokens_fn(tokenizer, regex_text: str, prompt_width: int):
    try:
        from lmformatenforcer import RegexParser, TokenEnforcer
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    except Exception:
        return None

    parser_regex = regex_text
    if parser_regex.startswith("^"):
        parser_regex = parser_regex[1:]
    if parser_regex.endswith("$"):
        parser_regex = parser_regex[:-1]
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    token_enforcer = TokenEnforcer(tokenizer_data, RegexParser(parser_regex))

    def _prefix_allowed_tokens_fn(batch_id: int, sent) -> list[int]:
        generated_token_ids = sent[prompt_width:].tolist()
        return token_enforcer.get_allowed_tokens(generated_token_ids).allowed_tokens

    return _prefix_allowed_tokens_fn


def _decode_generated(tokenizer, seq, prompt_width: int) -> str:
    generated_ids = seq[prompt_width:]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_mode: str,
    stage1_max_new_tokens: int | None,
    stage2_max_new_tokens: int | None,
    enforce_stage2_regex: bool,
) -> list[dict[str, str]]:
    import torch
    from transformers import StoppingCriteriaList

    if not prompts:
        return []

    device = _infer_device(model)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    base_generate_kwargs = {
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        base_generate_kwargs["temperature"] = temperature
        base_generate_kwargs["top_p"] = top_p

    if generation_mode == "two_stage":
        close_think = "</think>"
        stage1_limit = int(stage1_max_new_tokens or min(384, max_new_tokens))
        stage2_limit = int(stage2_max_new_tokens or min(256, max_new_tokens))
        stop_on_think = _make_stop_on_text(tokenizer, [close_think])
        stage1_kwargs = dict(base_generate_kwargs)
        stage1_kwargs["max_new_tokens"] = stage1_limit
        if stop_on_think is not None:
            stage1_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_on_think])

        with torch.inference_mode():
            stage1_output_ids = model.generate(**encoded, **stage1_kwargs)

        stage1_texts = [
            _decode_generated(tokenizer, seq, prompt_width)
            for seq in stage1_output_ids
        ]
        stage2_prompts: list[str] = []
        for prompt, stage1_text in zip(prompts, stage1_texts):
            if close_think not in stage1_text:
                stage1_text = f"{stage1_text.rstrip()}\n{close_think}".strip()
            stage2_prompts.append(prompt + stage1_text + CID2INS_STAGE2_PREFIX)

        stage2_encoded = tokenizer(
            stage2_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        stage2_encoded = {k: v.to(device) for k, v in stage2_encoded.items()}
        stage2_prompt_width = stage2_encoded["input_ids"].shape[1]
        stage2_kwargs = dict(base_generate_kwargs)
        stage2_kwargs["max_new_tokens"] = stage2_limit
        stop_on_fences = _make_stop_on_text(tokenizer, [close_think])
        stage2_stop_criteria = [stop_on_fences] if stop_on_fences is not None else []
        if enforce_stage2_regex:
            prefix_allowed = _build_regex_prefix_allowed_tokens_fn(
                tokenizer=tokenizer,
                regex_text=CID2INS_STAGE2_REGEX,
                prompt_width=stage2_prompt_width,
            )
            if prefix_allowed is not None:
                stage2_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed
            stage2_stop_criteria.append(
                _make_stop_on_regex_completion(tokenizer, stage2_prompt_width, CID2INS_STAGE2_REGEX)
            )
        if stage2_stop_criteria:
            stage2_kwargs["stopping_criteria"] = StoppingCriteriaList(stage2_stop_criteria)

        with torch.inference_mode():
            stage2_output_ids = model.generate(**stage2_encoded, **stage2_kwargs)

        results: list[dict[str, str]] = []
        for stage1_text, seq in zip(stage1_texts, stage2_output_ids):
            if close_think not in stage1_text:
                stage1_text = f"{stage1_text.rstrip()}\n{close_think}".strip()
            stage2_text = _decode_generated(tokenizer, seq, stage2_prompt_width)
            raw_completion_text = stage1_text + CID2INS_STAGE2_PREFIX + stage2_text
            results.append(
                {
                    "raw_completion_text": raw_completion_text,
                    "raw_text": _truncate_to_json(raw_completion_text),
                    "generation_mode": generation_mode,
                    "stage2_regex_enforced": str(bool(enforce_stage2_regex)),
                }
            )
        return results

    generate_kwargs = dict(base_generate_kwargs)
    generate_kwargs["max_new_tokens"] = max_new_tokens
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generate_kwargs)

    texts: list[dict[str, str]] = []
    for seq in output_ids:
        raw_text = _decode_generated(tokenizer, seq, prompt_width)
        # Truncate to first complete JSON object
        truncated = _truncate_to_json(raw_text)
        texts.append(
            {
                "raw_completion_text": raw_text,
                "raw_text": truncated,
                "generation_mode": generation_mode,
                "stage2_regex_enforced": "False",
            }
        )
    return texts

def _load_examples(input_dir: Path, max_rows: int | None, prompt_mode: str = PROMPT_MODE_TRAIN) -> tuple[Path, list[dict[str, Any]]]:
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


def _target_tid_overlap(pred_tid: list[str], gold_tid: list[str]) -> dict[str, float]:
    pred_set = set(pred_tid)
    gold_set = set(gold_tid)
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(pred_set & gold_set)
    precision = hit / len(pred_set)
    recall = hit / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _summarize_metrics(out_path: Path) -> dict[str, Any]:
    total = 0
    json_object_parsed = 0
    required_fields_parsed = 0
    target_tid_nonempty = 0
    target_ins_nonempty = 0
    partial_target_tid_only = 0
    target_tid_exact = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total += 1
            pred = record.get("prediction") or {}
            gold = record.get("label") or {}
            pred_tid = _string_list(pred.get("target_tid"))
            gold_tid = _string_list(gold.get("target_tid"))
            pred_ins = _str_or_empty(pred.get("target_ins")).strip()
            if isinstance(record.get("parsed_json"), dict):
                json_object_parsed += 1
            if pred_tid:
                target_tid_nonempty += 1
            if pred_ins:
                target_ins_nonempty += 1
            if pred_tid and not pred_ins:
                partial_target_tid_only += 1
            if record.get("parse_error") is None and _has_required_cid2ins_fields(record.get("parsed_json")):
                required_fields_parsed += 1
            if pred_tid == gold_tid:
                target_tid_exact += 1
            overlap = _target_tid_overlap(pred_tid, gold_tid)
            precision_sum += overlap["precision"]
            recall_sum += overlap["recall"]
            f1_sum += overlap["f1"]

    sample_record: dict[str, Any] | None = None
    with out_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line:
            sample_record = json.loads(first_line)

    metrics = {
        "task": DEFAULT_TASK,
        "total_examples": total,
        "json_parse_rate": (required_fields_parsed / total) if total else 0.0,
        "json_object_parse_rate": (json_object_parsed / total) if total else 0.0,
        "required_fields_parse_rate": (required_fields_parsed / total) if total else 0.0,
        "target_tid_nonempty_rate": (target_tid_nonempty / total) if total else 0.0,
        "target_ins_nonempty_rate": (target_ins_nonempty / total) if total else 0.0,
        "partial_target_tid_only_rate": (partial_target_tid_only / total) if total else 0.0,
        "target_tid_exact_match": (target_tid_exact / total) if total else 0.0,
        "target_tid_precision": (precision_sum / total) if total else 0.0,
        "target_tid_recall": (recall_sum / total) if total else 0.0,
        "target_tid_f1": (f1_sum / total) if total else 0.0,
        "generation_mode": (sample_record or {}).get("generation_mode"),
        "inference_model_source": (sample_record or {}).get("inference_model_source"),
        "inference_model_dir": (sample_record or {}).get("model_dir"),
    }
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
    model_dir: Path,
    model_source: str,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_mode: str,
    stage1_max_new_tokens: int | None,
    stage2_max_new_tokens: int | None,
    enforce_stage2_regex: bool,
    resume: bool = False,
    existing_out_path: Path | None = None,
) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    rank, _, world_size = _get_dist_info()
    base_name = _append_model_source_to_name(f"{DEFAULT_TASK}_predictions.jsonl", model_source)
    out_name = base_name.replace(".jsonl", f".rank{rank}.jsonl") if world_size > 1 else base_name
    out_path = pred_dir / out_name

    # Resume: append to existing file instead of overwriting
    if resume and existing_out_path is not None and existing_out_path.exists():
        # Load already-written row indices to avoid duplicates
        written_rows = set()
        with existing_out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    written_rows.add(rec.get("row_idx"))
        # Filter out examples already in the file
        pending = [ex for ex in examples if ex["row_idx"] not in written_rows]
        if not pending:
            return out_path
        examples = pending
        file_mode = "a"
        print(f"[{DEFAULT_TASK}][rank{rank}] resume mode: appending {len(examples)} pending rows")
    else:
        file_mode = "w"

    total_batches = max(1, math.ceil(len(examples) / batch_size))
    with out_path.open(file_mode, encoding="utf-8") as f:
        for batch_idx, (start, batch) in enumerate(_iter_batches(examples, batch_size), start=1):
            prompts = [
                tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for example in batch
            ]
            generated_outputs = _batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generation_mode=generation_mode,
                stage1_max_new_tokens=stage1_max_new_tokens,
                stage2_max_new_tokens=stage2_max_new_tokens,
                enforce_stage2_regex=enforce_stage2_regex,
            )

            for example, prompt_text, generated in zip(batch, prompts, generated_outputs):
                raw_completion_text = generated["raw_completion_text"]
                raw_text = generated["raw_text"]
                parsed_json, parse_error = _parse_response_json(raw_completion_text)
                clean_json_text = (
                    json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(parsed_json, dict)
                    else None
                )
                record = {
                    "task": DEFAULT_TASK,
                    "row_idx": example["row_idx"],
                    "source_file": str(parquet_path),
                    "inference_model_source": model_source,
                    "model_dir": str(model_dir),
                    "prompt_mode": example.get("prompt_mode"),
                    "generation_mode": generated.get("generation_mode", generation_mode),
                    "stage2_regex_enforced": generated.get("stage2_regex_enforced"),
                    "prompt_text": prompt_text,
                    "raw_completion_text": raw_completion_text,
                    "raw_text": raw_text,
                    "clean_json_text": clean_json_text,
                    "parsed_json": parsed_json,
                    "parse_error": parse_error,
                    "prediction": parsed_json if isinstance(parsed_json, dict) else {},
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
        description="Run stage2 cid2ins inference on dataset/test_cid2ins.parquet by default."
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--search_root", type=str, default=str(DEFAULT_MODEL_SEARCH_ROOT))
    p.add_argument("--model_dir", type=str, default=None, help="Explicit model directory (full model or PEFT adapter)")
    p.add_argument("--base_model_dir", type=str, default=None, help="Base model directory when --model_dir is a PEFT adapter")
    p.add_argument("--pred_dir", type=str, default=str(DEFAULT_PRED_DIR))
    p.add_argument("--max_rows", type=int, default=10, help="Default to 10 rows for quick manual inspection")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--prompt_mode",
        choices=[PROMPT_MODE_TRAIN, PROMPT_MODE_SFT],
        default=PROMPT_MODE_TRAIN,
        help="train uses the GRPO-style prompt with schema constraints; sft uses the simpler SFT training prompt.",
    )
    p.add_argument(
        "--generation_mode",
        choices=["two_stage", "direct"],
        default="two_stage",
        help="two_stage matches the GRPO rollout style: generate reasoning to </think>, then regex-constrained JSON.",
    )
    p.add_argument(
        "--stage1_max_new_tokens",
        type=int,
        default=None,
        help="Max tokens for the reasoning stage in --generation_mode two_stage. Defaults to min(384, --max_new_tokens).",
    )
    p.add_argument(
        "--stage2_max_new_tokens",
        type=int,
        default=None,
        help="Max tokens for the constrained JSON stage in --generation_mode two_stage. Defaults to min(256, --max_new_tokens).",
    )
    p.add_argument(
        "--enforce_stage2_regex",
        action="store_true",
        default=False,
        help=(
            "Use token-level regex constrained decoding for the two-stage JSON continuation. "
            "This is closest to vLLM guided decoding but can be slow with HF generate."
        ),
    )
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--resume", action="store_true", default=False,
                   help="Resume from existing output file, skip already-processed rows.")
    return p.parse_args()


def _load_completed_row_idx(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    completed = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                row_idx = record.get("row_idx")
                if row_idx is not None:
                    completed.add(row_idx)
    return completed


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _maybe_init_distributed()

    try:
        input_dir = Path(args.input_dir).resolve()
        search_root = Path(args.search_root).resolve()
        pred_dir = Path(args.pred_dir).resolve()

        artifact_kind, model_dir, base_model_dir = _resolve_model_dir(args.model_dir, search_root)
        model_source = _build_model_source(model_dir, artifact_kind, base_model_dir)
        print(
            f"[{DEFAULT_TASK}][rank{rank}] using model: {model_dir} "
            f"(source={model_source}, world_size={world_size}, local_rank={local_rank})"
        )

        model, tokenizer = _load_model_and_tokenizer(
            artifact_kind=artifact_kind, model_dir=model_dir, base_model_dir=base_model_dir
        )
        parquet_path, examples = _load_examples(input_dir=input_dir, max_rows=args.max_rows, prompt_mode=args.prompt_mode)
        if not examples:
            if _is_main_process():
                print(f"[{DEFAULT_TASK}] no valid examples in {parquet_path}, skip.")
            return 0

        sharded_examples = _shard_examples(examples, rank=rank, world_size=world_size)

        # Compute output path for resume check
        base_name = _append_model_source_to_name(f"{DEFAULT_TASK}_predictions.jsonl", model_source)
        out_name = base_name.replace(".jsonl", f".rank{rank}.jsonl") if world_size > 1 else base_name
        existing_out_path = pred_dir / out_name

        # Resume: load completed row indices and filter
        if args.resume and world_size <= 1:
            completed = _load_completed_row_idx(existing_out_path)
            if completed:
                print(f"[{DEFAULT_TASK}][rank{rank}] resume: {len(completed)} rows already done, skipping")
                sharded_examples = [e for e in sharded_examples if e["row_idx"] not in completed]
                if not sharded_examples:
                    if _is_main_process():
                        print(f"[{DEFAULT_TASK}] all {len(examples)} rows already done, nothing to resume.")
                    return 0
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
            model_dir=model_dir,
            model_source=model_source,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            generation_mode=args.generation_mode,
            stage1_max_new_tokens=args.stage1_max_new_tokens,
            stage2_max_new_tokens=args.stage2_max_new_tokens,
            enforce_stage2_regex=args.enforce_stage2_regex,
            resume=args.resume and world_size <= 1,
            existing_out_path=existing_out_path,
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

            metrics = _summarize_metrics(merged_path)
            print(f"[{DEFAULT_TASK}] wrote predictions to {merged_path}")
            print(
                f"[{DEFAULT_TASK}] metrics: "
                f"json_parse_rate={metrics['json_parse_rate']:.4f}, "
                f"json_object_parse_rate={metrics['json_object_parse_rate']:.4f}, "
                f"target_ins_nonempty_rate={metrics['target_ins_nonempty_rate']:.4f}, "
                f"target_tid_exact_match={metrics['target_tid_exact_match']:.4f}, "
                f"target_tid_f1={metrics['target_tid_f1']:.4f}"
            )

        _dist_barrier()
        return 0
    finally:
        _maybe_destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
