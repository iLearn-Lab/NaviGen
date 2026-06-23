"""
Generate Term IDs (TIDs) from item captions with qwen3.5-flash.

This script implements a simplified, single-item variant of the TID idea from
GRLM: use concise, semantically rich, standardized terms as item identifiers.

Compared with the paper's setting, this version changes:
1. The number of terms is not fixed. The model decides how many terms are
   needed, but the final TID is capped at 10 terms.
2. The input is a caption-like description, suitable for short videos, ads,
   products, and other cross-domain items.
3. The model remains qwen3.5-flash, using the existing DashScope-compatible
   client in teacher_qwen_client.py.

Expected input item schema:
{
  "id": "optional",
  "caption": "item caption text"
}

Output item schema:
{
  "id": "...",
  "tid": ["term_a", "term_b", ...]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_str, load_project_env

load_project_env(REPO_ROOT / ".env")

try:
    from dashscope_key_config import DASHSCOPE_API_KEY as FILE_DASHSCOPE_API_KEY
    from dashscope_key_config import DASHSCOPE_API_KEYS as FILE_DASHSCOPE_API_KEYS
except ImportError:
    FILE_DASHSCOPE_API_KEY = ""
    FILE_DASHSCOPE_API_KEYS: list[str] = []
from teacher_qwen_client import generate_text


INPUT_JSON_PATH = "products_user_pid2caption.json"
OUTPUT_JSON_PATH = "products_user_pid2tid.json"
MODEL_NAME = env_str("NAVIGEN_TEACHER_MODEL", "qwen3.5-flash")
MAX_TERMS = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
KEY_WORKERS = 80
ITEMS_PER_TASK = 1
CHECKPOINT_FORMAT = "incremental_jsonl_v1"


SYSTEM_PROMPT = f"""You are a cross-domain item identifier generator.
Your task is to compress the input caption into a Term ID (TID): a set of structured, standardized, semantically dense terms.

Goals:
1. The terms should cover the core semantics of the item.
2. Preserve key information while staying concise and avoiding redundant modifiers.
3. Prefer stable cross-domain signals such as:
   - subject or category
   - core scene or use case
   - key attributes or selling points
   - style, atmosphere, or audience only when truly important
4. Use standard, general, reusable wording. Avoid synonym stacking and avoid sentence-style expressions.
5. You decide how many terms are needed, but the result must be expressive enough and must not exceed {MAX_TERMS} terms.
6. Each term may be a single word or a very short phrase, but the overall result must stay compact.
7. Order the terms by importance, with the most essential term first.

Strict output requirements:
1. Output JSON only.
2. The format must be {{"terms": ["term1", "term2"]}}.
3. Do not output explanations, reasoning process, Markdown, or extra text.
4. All terms must be in English.
5. Do not use Chinese or any other non-English language in the terms."""


USER_PROMPT_TEMPLATE = """Generate a TID from the caption below.

caption:
{caption}

Output JSON only: {{"terms": ["term1", "term2"]}}
Important: every term must be written in English."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TIDs from captions.")
    parser.add_argument("--input", default=INPUT_JSON_PATH, help="Path to input JSON file.")
    parser.add_argument("--output", default=OUTPUT_JSON_PATH, help="Path to output JSON file.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name.")
    parser.add_argument(
        "--api-keys",
        default="",
        help=(
            "Comma-separated api keys. "
            "If empty, reads DASHSCOPE_API_KEYS (comma-separated) or DASHSCOPE_API_KEY."
        ),
    )
    parser.add_argument(
        "--key-workers",
        type=int,
        default=KEY_WORKERS,
        help="Concurrent worker count per api key.",
    )
    parser.add_argument(
        "--items-per-task",
        type=int,
        default=ITEMS_PER_TASK,
        help="Number of items handled by each parallel task.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file if it exists.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. Defaults to <output>.checkpoint.json.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Append checkpoint every N newly completed items.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N completed items.",
    )
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_checkpoint_path(output_path: str, checkpoint_path: str) -> str:
    if checkpoint_path:
        return checkpoint_path
    return f"{output_path}.checkpoint.json"


def init_incremental_checkpoint(path: str, total: int, done_outputs: dict[int, dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "meta", "format": CHECKPOINT_FORMAT, "total": total},
            f,
            ensure_ascii=False,
        )
        f.write("\n")
        for idx in sorted(done_outputs):
            json.dump(
                {"type": "item", "index": idx, "value": done_outputs[idx]},
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def append_checkpoint_updates(path: str, updates: dict[int, dict[str, Any]]) -> None:
    if not updates:
        return
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for idx in sorted(updates):
            json.dump(
                {"type": "item", "index": idx, "value": updates[idx]},
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def load_incremental_checkpoint_jsonl(
    path: str,
    expected_total: int,
) -> tuple[list[dict[str, Any] | None], dict[int, dict[str, Any]], str]:
    output_slots: list[dict[str, Any] | None] = [None] * expected_total
    done_outputs: dict[int, dict[str, Any]] = {}
    seen_meta = False

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: invalid checkpoint line ignored at {path}:{line_no}.")
                continue
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type")
            if entry_type == "meta":
                if entry.get("format") != CHECKPOINT_FORMAT:
                    continue
                seen_meta = True
                total = entry.get("total")
                if isinstance(total, int) and total != expected_total:
                    print(
                        f"Warning: checkpoint size mismatch (checkpoint={total}, expected={expected_total}). "
                        "Ignoring checkpoint."
                    )
                    return [None] * expected_total, {}, "mismatch"
                continue

            if entry_type != "item":
                continue

            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            value = entry.get("value")
            if 0 <= idx < expected_total and isinstance(value, dict):
                output_slots[idx] = value
                done_outputs[idx] = value

    return output_slots, done_outputs, CHECKPOINT_FORMAT if seen_meta else "jsonl_without_meta"


def load_checkpoint(
    path: str,
    expected_total: int,
) -> tuple[list[dict[str, Any] | None], dict[int, dict[str, Any]], str]:
    if not os.path.exists(path):
        return [None] * expected_total, {}, "missing"

    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return load_incremental_checkpoint_jsonl(path, expected_total)

    if not isinstance(data, dict):
        return [None] * expected_total, {}, "invalid"

    # Backward compatibility: old format {"total": N, "outputs": [..]}
    old_outputs = data.get("outputs")
    if isinstance(old_outputs, list):
        if len(old_outputs) != expected_total:
            print(
                f"Warning: checkpoint size mismatch (checkpoint={len(old_outputs)}, expected={expected_total}). "
                "Ignoring checkpoint."
            )
            return [None] * expected_total, {}, "mismatch"
        done_outputs_old: dict[int, dict[str, Any]] = {}
        for idx, value in enumerate(old_outputs):
            if isinstance(value, dict):
                done_outputs_old[idx] = value
        return old_outputs, done_outputs_old, "legacy_outputs"

    done_raw = data.get("done")
    if not isinstance(done_raw, dict):
        return [None] * expected_total, {}, "invalid"

    output_slots: list[dict[str, Any] | None] = [None] * expected_total
    done_outputs: dict[int, dict[str, Any]] = {}
    for key, value in done_raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < expected_total and isinstance(value, dict):
            output_slots[idx] = value
            done_outputs[idx] = value
    return output_slots, done_outputs, str(data.get("format") or "legacy_done")


def normalize_caption(sample: dict[str, Any]) -> str:
    for key in ("caption", "item_caption", "description", "text"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def build_messages(caption: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption)},
    ]


def parse_api_keys(cli_value: str) -> list[str]:
    source = cli_value.strip()
    if not source:
        source = os.getenv("DASHSCOPE_API_KEYS", "").strip()
    if source:
        keys = [k.strip() for k in source.split(",") if k.strip()]
        if keys:
            return keys

    if FILE_DASHSCOPE_API_KEYS:
        keys = [k.strip() for k in FILE_DASHSCOPE_API_KEYS if str(k).strip()]
        if keys:
            return keys

    single = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if single:
        return [single]
    if FILE_DASHSCOPE_API_KEY.strip():
        return [FILE_DASHSCOPE_API_KEY.strip()]
    raise ValueError(
        "No api key found. Set --api-keys or env DASHSCOPE_API_KEYS/DASHSCOPE_API_KEY "
        "or dashscope_key_config.py."
    )


def generate_with_retry(messages: list[dict[str, str]], model_name: str, api_key: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output = generate_text(messages=messages, model_name=model_name, api_key=api_key)
            if output and str(output).strip():
                return str(output).strip()
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    if last_error is not None:
        print(f"Warning: generate_text failed after {MAX_RETRIES} attempts: {last_error}")
    return ""


def clean_term(term: Any) -> str:
    text = str(term).strip()
    text = text.strip("[]{}()\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;|")


def is_english_term(term: str) -> bool:
    if not term:
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s&+/\-,'().]*", term):
        return False
    return bool(re.search(r"[A-Za-z]", term))


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for item in items:
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def parse_terms_from_output(output: str) -> list[str]:
    if not output:
        return []

    candidates: list[str] = []

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict) and isinstance(parsed.get("terms"), list):
            candidates = [clean_term(x) for x in parsed["terms"]]
        elif isinstance(parsed, list):
            candidates = [clean_term(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    if not candidates:
        match = re.search(r"\{[\s\S]*\"terms\"\s*:\s*\[(.*?)\][\s\S]*\}", output)
        if match:
            inner = match.group(1)
            parts = re.split(r'","|",\s*"|",\s*|\s*,\s*', inner.replace("\n", " "))
            candidates = [clean_term(x) for x in parts]

    if not candidates:
        bracket_match = re.search(r"\[(.*?)\]", output, flags=re.S)
        if bracket_match:
            inner = bracket_match.group(1)
            parts = re.split(r"\s*[,;|\n]\s*", inner)
            candidates = [clean_term(x) for x in parts]

    if not candidates:
        parts = re.split(r"\s*[,;|\n]\s*", output)
        candidates = [clean_term(x) for x in parts]

    candidates = [x for x in candidates if x and is_english_term(x)]
    candidates = dedupe_keep_order(candidates)
    return candidates[:MAX_TERMS]


def fallback_terms_from_caption(caption: str) -> list[str]:
    tokens = re.split("[\uFF0C,\u3002.;\uFF1B:\uFF1A/|()\\[\\]\n]+", caption)
    cleaned = [clean_term(token) for token in tokens]
    cleaned = [token for token in cleaned if token and len(token) <= 40 and is_english_term(token)]
    return dedupe_keep_order(cleaned)[:MAX_TERMS]


def generate_tid(caption: str, model_name: str, api_key: str) -> list[str]:
    messages = build_messages(caption)
    output = generate_with_retry(messages, model_name, api_key)
    terms = parse_terms_from_output(output)
    if terms:
        return terms
    return fallback_terms_from_caption(caption)


def transform_sample(sample: dict[str, Any], model_name: str, api_key: str) -> dict[str, Any]:
    caption = normalize_caption(sample)
    return {
        "id": sample.get("id"),
        "tid": generate_tid(caption, model_name, api_key) if caption else [],
    }


def is_pid_to_caption_mapping(data: dict[str, Any]) -> bool:
    if not data:
        return False
    # Heuristic: top-level dict where all values are caption-like strings.
    return all(isinstance(v, str) for v in data.values())


def transform_pid_caption_mapping(data: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for pid, caption in data.items():
        outputs.append({"id": pid, "caption": caption})
    return outputs


def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("--items-per-task must be > 0.")
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def process_chunk(
    chunk: list[int],
    samples: list[dict[str, Any]],
    model_name: str,
    api_key: str,
) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for index in chunk:
        sample = samples[index]
        out.append((index, transform_sample(sample, model_name, api_key)))
    return out


def process_list_samples_parallel(
    samples: list[dict[str, Any]],
    model_name: str,
    api_keys: list[str],
    key_workers: int,
    items_per_task: int,
    checkpoint_path: str = "",
    resume: bool = False,
    save_every: int = 100,
    progress_every: int = 20,
) -> list[dict[str, Any]]:
    if key_workers <= 0:
        raise ValueError("--key-workers must be > 0.")
    if save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if progress_every <= 0:
        raise ValueError("--progress-every must be > 0.")

    total = len(samples)
    if resume and checkpoint_path:
        output_slots, done_outputs, checkpoint_format = load_checkpoint(checkpoint_path, total)
    else:
        output_slots = [None] * total
        done_outputs = {}
        checkpoint_format = "missing"

    if checkpoint_path and (not resume or checkpoint_format != CHECKPOINT_FORMAT):
        init_incremental_checkpoint(checkpoint_path, total, done_outputs if resume else {})
        if resume and done_outputs:
            print(f"Checkpoint converted to append-only log: {checkpoint_path}")

    already_done = set(done_outputs.keys())
    valid_indices = [idx for idx, sample in enumerate(samples) if isinstance(sample, dict) and idx not in already_done]
    key_to_indices: dict[int, list[int]] = {k: [] for k in range(len(api_keys))}
    for pos, index in enumerate(valid_indices):
        key_to_indices[pos % len(api_keys)].append(index)

    all_futures = []
    executors: list[ThreadPoolExecutor] = []
    pending_checkpoint_updates: dict[int, dict[str, Any]] = {}
    completed = len(already_done)
    print(f"Progress: {completed}/{total} completed (resume={resume}).")

    try:
        for key_idx, api_key in enumerate(api_keys):
            executor = ThreadPoolExecutor(max_workers=key_workers)
            executors.append(executor)
            for chunk in chunk_indices(key_to_indices[key_idx], items_per_task):
                all_futures.append(executor.submit(process_chunk, chunk, samples, model_name, api_key))

        for future in as_completed(all_futures):
            changed = 0
            for index, transformed in future.result():
                output_slots[index] = transformed
                done_outputs[index] = transformed
                pending_checkpoint_updates[index] = transformed
                changed += 1
            completed += changed
            if completed % progress_every == 0 or completed == total:
                print(f"Progress: {completed}/{total} completed.")
            if checkpoint_path and pending_checkpoint_updates and (len(pending_checkpoint_updates) >= save_every or completed == total):
                flushed = len(pending_checkpoint_updates)
                append_checkpoint_updates(checkpoint_path, pending_checkpoint_updates)
                pending_checkpoint_updates.clear()
                print(f"Checkpoint appended: {checkpoint_path} (+{flushed}, {completed}/{total})")
    finally:
        if checkpoint_path and pending_checkpoint_updates:
            flushed = len(pending_checkpoint_updates)
            append_checkpoint_updates(checkpoint_path, pending_checkpoint_updates)
            pending_checkpoint_updates.clear()
            print(f"Checkpoint appended: {checkpoint_path} (+{flushed}, {completed}/{total})")
        for executor in executors:
            executor.shutdown(wait=True)

    outputs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        transformed = output_slots[idx]
        if transformed is not None:
            outputs.append(transformed)
            continue
        if isinstance(sample, dict):
            outputs.append({"id": sample.get("id"), "tid": []})
        else:
            outputs.append({"id": None, "tid": []})
    return outputs


def main() -> None:
    args = parse_args()
    if not args.input:
        raise ValueError("Input JSON path is empty. Set INPUT_JSON_PATH or pass --input.")
    if not args.output:
        raise ValueError("Output JSON path is empty. Set OUTPUT_JSON_PATH or pass --output.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    data = read_json(args.input)
    api_keys = parse_api_keys(args.api_keys)
    checkpoint_path = build_checkpoint_path(args.output, args.checkpoint)
    if isinstance(data, list):
        outputs = process_list_samples_parallel(
            samples=data,
            model_name=args.model,
            api_keys=api_keys,
            key_workers=args.key_workers,
            items_per_task=args.items_per_task,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            save_every=args.save_every,
            progress_every=args.progress_every,
        )
    elif isinstance(data, dict):
        if is_pid_to_caption_mapping(data):
            samples = transform_pid_caption_mapping(data)
            outputs = process_list_samples_parallel(
                samples=samples,
                model_name=args.model,
                api_keys=api_keys,
                key_workers=args.key_workers,
                items_per_task=args.items_per_task,
                checkpoint_path=checkpoint_path,
                resume=args.resume,
                save_every=args.save_every,
                progress_every=args.progress_every,
            )
        else:
            outputs = transform_sample(data, args.model, api_keys[0])
    else:
        raise TypeError("Input JSON must be an object or an array of objects.")

    write_json(args.output, outputs)
    print(f"Saved TID outputs to: {args.output}")


if __name__ == "__main__":
    main()
