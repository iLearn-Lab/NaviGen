"""
Step 2: Search for the best final AIGC prompt via evolutionary search.

Input JSON path is intentionally left empty by default. Update INPUT_JSON_PATH
before running, or pass --input/--output from CLI.

Expected input item schema:
{
  "sample_id": "optional",
  "history_tids": [["tid_a", "tid_b"], ["tid_c"]],
  "target_tid": ["target_term_a", "target_term_b"] | "target text"
}

Output item schema:
{
  "sample_id": "...",
  "history_tids": ...,
  "target_tid": ...,
  "step1_reasoning": "...",
  "founder_candidates": [...],
  "evolution_trace": [...],
  "evolution_summary": "...",
  "final_judge": {...},
  "final_mode": "...",
  "final_reasoning": "...",
  "final_prompt": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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


INPUT_JSON_PATH = "products_user_step1_output.json"
OUTPUT_JSON_PATH = "products_user_step2_output.json"
FAILED_OUTPUT_JSON_PATH = "products_user_step2_output_failed.json"
MODEL_NAME = env_str("NAVIGEN_TEACHER_MODEL", "qwen3.5-flash")
NUM_ROUNDS = 1
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
MAX_REASONING_CHARS = 200
KEY_WORKERS = 60
ITEMS_PER_TASK = 1
SAVE_EVERY = 200
PROGRESS_EVERY = 1
MILESTONE_EVERY = 1000
INTERNAL_SEARCH_WORKERS = 4
EMERGENCY_TIMEOUT_FAILURES = 10


SYSTEM_PROMPT = """You are an AIGC creative search teacher model.
Your task is to search for the best AIGC creative instruction for the target item.

Search principles:
1. The target item is the primary objective. Historical TIDs are only auxiliary clues.
2. Each candidate must implicitly point to the target item, but the creative instruction must not explicitly mention, copy, or quote the target item text or tokens.
3. Each candidate must also output a concise reasoning explaining why the prompt is suitable and what information it used.
4. Use multiple rounds of selection, crossover, and mutation to refine the candidate set.
5. Output only one JSON object with fields "prompt" and "reasoning"."""


BATCH_SYSTEM_PROMPT = """You are an AIGC creative search teacher model.
Your task is to search for the best AIGC creative instruction for the target item.

Search principles:
1. The target item is the primary objective. Historical TIDs are only auxiliary clues.
2. Every candidate must implicitly point to the target item, but the creative instruction must not explicitly mention, copy, or quote the target item text or tokens.
3. Every candidate must also output a concise reasoning explaining why the prompt is suitable and what information it used.
4. Follow every named task exactly once and keep the output aligned with the provided task names.
5. Output only one JSON object whose keys are the provided task names."""


SCORER_SYSTEM_PROMPT = """You are a strict AIGC prompt evaluation model.
Given the user's historical TIDs, the target TID, and a candidate prompt, score the candidate independently on the following four dimensions:
1. consistency: whether it is centered on the target item semantics, uses history only as supporting context, and avoids drifting or explicitly naming the target.
2. novelty: whether it introduces reasonable innovation without losing the intended direction.
3. aesthetic: whether it shows clear and layered visual composition ability, including subject, scene, style, camera language, lighting, mood, and key details.
4. executability: whether it is specific, clear, contradiction-free, and directly usable for AIGC generation.

Scoring requirements:
1. Output a floating-point score from 0 to 10 for each dimension.
2. The four dimensions must be judged independently; do not give all high scores just because you like the candidate overall.
3. The reasoning should briefly explain the main strengths and weaknesses and stay within 200 characters.
4. If the prompt explicitly mentions the target item, lower consistency and executability.
5. Output only one JSON object. Do not output Markdown and do not output any extra explanation.

Output format:
{
  "consistency": 0.0,
  "novelty": 0.0,
  "aesthetic": 0.0,
  "executability": 0.0,
  "reasoning": "..."
}"""


BATCH_SCORER_SYSTEM_PROMPT = """You are a strict AIGC prompt evaluation model.
Given the user's historical TIDs, the target TID, and multiple candidate prompts, score each candidate independently on the following four dimensions:
1. consistency: whether it is centered on the target item semantics, uses history only as supporting context, and avoids drifting or explicitly naming the target.
2. novelty: whether it introduces reasonable innovation without losing the intended direction.
3. aesthetic: whether it shows clear and layered visual composition ability, including subject, scene, style, camera language, lighting, mood, and key details.
4. executability: whether it is specific, clear, contradiction-free, and directly usable for AIGC generation.

Scoring requirements:
1. Output a floating-point score from 0 to 10 for each dimension.
2. Judge each candidate independently; do not let one candidate's quality inflate or suppress another.
3. The reasoning for each candidate should briefly explain the main strengths and weaknesses and stay within 200 characters.
4. If a prompt explicitly mentions the target item, lower consistency and executability.
5. Output only one JSON object whose keys are the provided candidate names and whose values are score objects."""


FOUNDER_DIRECTIONS = {
    "conservative": "Stay close to the latent target semantics with limited innovation, using history only as secondary support.",
    "balanced": "Center the prompt on the latent target while balancing target alignment, creativity, and stability.",
    "exploratory": "Stay anchored to the latent target, but innovate more boldly in scene, camera, lighting, or emotion.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search best final prompt via evolution.")
    parser.add_argument("--input", default=INPUT_JSON_PATH, help="Path to input JSON file.")
    parser.add_argument("--output", default=OUTPUT_JSON_PATH, help="Path to output JSON file.")
    parser.add_argument(
        "--failed-output",
        default=FAILED_OUTPUT_JSON_PATH,
        help="Path to failed-sample JSON file. Defaults to <output_stem>_failed.json.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Teacher model name placeholder.")
    parser.add_argument("--rounds", default=NUM_ROUNDS, type=int, help="Number of evolution rounds.")
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
        help="Checkpoint JSON path. Defaults to <output>.checkpoint.json.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help="Overwrite the main checkpoint every N completed samples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Print progress every N completed samples.",
    )
    parser.add_argument(
        "--milestone-every",
        type=int,
        default=MILESTONE_EVERY,
        help="Keep a non-overwriting milestone checkpoint every N completed samples.",
    )
    parser.add_argument(
        "--emergency-timeout-failures",
        type=int,
        default=EMERGENCY_TIMEOUT_FAILURES,
        help="Emergency-save and abort step2 after this many consecutive timeout-related sample failures.",
    )
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_failed_output_path(output_path: str, failed_output_path: str) -> str:
    if failed_output_path:
        return failed_output_path
    output = Path(output_path)
    return str(output.with_name(f"{output.stem}_failed.json"))


def build_checkpoint_path(output_path: str, checkpoint_path: str) -> str:
    if checkpoint_path:
        return checkpoint_path
    return f"{output_path}.checkpoint.json"


def build_milestone_checkpoint_path(checkpoint_path: str, milestone: int) -> str:
    path = Path(checkpoint_path)
    return str(path.with_name(f"{path.stem}.{milestone}{path.suffix}"))


def save_checkpoint(
    path: str,
    total: int,
    checkpoint_entries: dict[int, dict[str, Any]],
    done_count: int,
) -> None:
    if not checkpoint_entries:
        return
    checkpoint_path = Path(path)
    if checkpoint_path.parent and not checkpoint_path.parent.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "incremental_append_v1",
        "total": total,
        "done_count": done_count,
        "entries": {str(idx): value for idx, value in checkpoint_entries.items()},
    }
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def copy_checkpoint_milestone(source_path: str, milestone_path: str) -> None:
    source = Path(source_path)
    destination = Path(milestone_path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_emergency_snapshot_path(path: str) -> str:
    output = Path(path)
    return str(output.with_name(f"{output.stem}_emergency{output.suffix}"))


def is_timeout_error_text(text: str) -> bool:
    lowered = str(text or "").lower()
    timeout_markers = (
        "timeout",
        "timed out",
        "readtimeout",
        "connecttimeout",
        "apitimeouterror",
        "read timeout",
        "connect timeout",
        "connection timed out",
    )
    return any(marker in lowered for marker in timeout_markers)


def save_emergency_state(
    *,
    output_slots: list[dict[str, Any] | None],
    failures: list[dict[str, Any]],
    done_outputs: dict[int, dict[str, Any]],
    output_path: str,
    failed_output_path: str,
    checkpoint_path: str,
    total: int,
    completed: int,
) -> tuple[str, str]:
    if checkpoint_path:
        save_checkpoint(checkpoint_path, total, done_outputs, len(done_outputs))

    emergency_output_path = build_emergency_snapshot_path(output_path)
    emergency_failed_output_path = build_emergency_snapshot_path(failed_output_path)
    write_json(emergency_output_path, [x for x in output_slots if x is not None])
    write_json(emergency_failed_output_path, failures)
    print(f"Emergency save completed at progress {completed}/{total}.")
    print(f"Emergency output snapshot: {emergency_output_path}")
    print(f"Emergency failed snapshot: {emergency_failed_output_path}")
    if checkpoint_path:
        print(f"Emergency checkpoint updated: {checkpoint_path}")
    return emergency_output_path, emergency_failed_output_path


def apply_checkpoint_entries(
    output_slots: list[dict[str, Any] | None],
    done_outputs: dict[int, dict[str, Any]],
    entries: dict[str, Any],
    expected_total: int,
) -> None:
    for key, value in entries.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < expected_total and isinstance(value, dict):
            output_slots[idx] = value
            done_outputs[idx] = value


def parse_checkpoint_snapshot(data: dict[str, Any], expected_total: int) -> tuple[list[dict[str, Any] | None], dict[int, dict[str, Any]]]:
    output_slots: list[dict[str, Any] | None] = [None] * expected_total
    done_outputs: dict[int, dict[str, Any]] = {}

    old_outputs = data.get("outputs")
    if isinstance(old_outputs, list):
        if len(old_outputs) != expected_total:
            raise ValueError(
                f"checkpoint size mismatch (checkpoint={len(old_outputs)}, expected={expected_total})"
            )
        for idx, value in enumerate(old_outputs):
            if isinstance(value, dict):
                output_slots[idx] = value
                done_outputs[idx] = value
        return output_slots, done_outputs

    done_raw = data.get("done")
    if not isinstance(done_raw, dict):
        raise ValueError("checkpoint snapshot missing 'done' map")

    apply_checkpoint_entries(output_slots, done_outputs, done_raw, expected_total)
    return output_slots, done_outputs


def load_checkpoint(path: str, expected_total: int) -> tuple[list[dict[str, Any] | None], dict[int, dict[str, Any]]]:
    if not os.path.exists(path):
        return [None] * expected_total, {}

    try:
        data = read_json(path)
    except Exception:
        data = None

    if isinstance(data, dict):
        try:
            return parse_checkpoint_snapshot(data, expected_total)
        except Exception as exc:
            print(f"Warning: failed to parse checkpoint snapshot {path}: {exc}. Falling back to append log parsing.")

    output_slots: list[dict[str, Any] | None] = [None] * expected_total
    done_outputs: dict[int, dict[str, Any]] = {}
    saw_valid_record = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except Exception as exc:
                    print(f"Warning: skipping invalid checkpoint line {lineno} in {path}: {exc}")
                    continue
                if not isinstance(record, dict):
                    continue
                record_total = record.get("total")
                if isinstance(record_total, int) and record_total != expected_total:
                    print(
                        f"Warning: skipping checkpoint line {lineno} in {path} due to total mismatch "
                        f"(checkpoint={record_total}, expected={expected_total})."
                    )
                    continue
                if isinstance(record.get("entries"), dict):
                    apply_checkpoint_entries(output_slots, done_outputs, record["entries"], expected_total)
                    saw_valid_record = True
                elif isinstance(record.get("done"), dict):
                    apply_checkpoint_entries(output_slots, done_outputs, record["done"], expected_total)
                    saw_valid_record = True
    except Exception as exc:
        print(f"Warning: failed to read checkpoint {path}: {exc}. Ignoring checkpoint.")
        return [None] * expected_total, {}

    if saw_valid_record:
        return output_slots, done_outputs

    print(f"Warning: no valid checkpoint records found in {path}. Ignoring checkpoint.")
    return [None] * expected_total, {}


def normalize_history_tids(history_tids: Any) -> list[list[str]]:
    if not isinstance(history_tids, list):
        return []
    normalized: list[list[str]] = []
    for row in history_tids:
        if isinstance(row, list):
            normalized.append([str(x).strip() for x in row if str(x).strip()])
        elif row is not None:
            normalized.append([str(row).strip()])
    return normalized


def format_history_tids(history_tids: list[list[str]]) -> str:
    if not history_tids:
        return "[]"
    return "\n".join(f"{idx + 1}. {', '.join(row)}" for idx, row in enumerate(history_tids))


def normalize_target_tid(target_tid: Any) -> str:
    if isinstance(target_tid, list):
        return ", ".join(str(x).strip() for x in target_tid if str(x).strip())
    return str(target_tid or "").strip()


def normalize_target_terms(target_tid: Any) -> list[str]:
    if isinstance(target_tid, list):
        return [str(x).strip().lower() for x in target_tid if str(x).strip()]

    text = str(target_tid or "").strip().lower()
    if not text:
        return []

    parts = [part.strip() for part in re.split(r"[,/;\n]+", text) if part.strip()]
    return parts or [text]


def compact_text(text: str) -> str:
    return " ".join(str(text or "").split())


def trim_reasoning(text: str, max_chars: int = MAX_REASONING_CHARS) -> str:
    cleaned = compact_text(text)
    if len(cleaned) <= max_chars:
        return cleaned

    cut = cleaned[:max_chars].rstrip()
    for token in (". ", "; ", ", ", "\u3002", "\uFF1B", "\uFF0C"):
        idx = cut.rfind(token)
        if idx >= max_chars - 60:
            return cut[: idx + 1].rstrip()
    return cut.rstrip(" ,;\uFF0C\uFF1B\u3002") + "..."


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


def generate_with_retry(
    messages: list[dict[str, str]],
    model_name: str,
    api_key: str,
    *,
    enable_thinking: bool,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output = generate_text(
                messages=messages,
                model_name=model_name,
                api_key=api_key,
                enable_thinking=enable_thinking,
            )
            if output and str(output).strip():
                return str(output).strip()
            last_error = ValueError("Model returned empty output.")
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if last_error is None:
        last_error = RuntimeError("Unknown generate_text failure.")
    raise RuntimeError(f"generate_text failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def extract_json_payload(text: str) -> Any:
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty JSON response.")
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    object_start = content.find("{")
    array_start = content.find("[")
    starts = [idx for idx in (object_start, array_start) if idx != -1]
    if not starts:
        raise ValueError("Response does not contain JSON.")

    start = min(starts)
    opening = content[start]
    closing = "}" if opening == "{" else "]"
    end = content.rfind(closing)
    if end == -1 or end <= start:
        raise ValueError("Response does not contain a complete JSON payload.")
    return json.loads(content[start : end + 1])


def extract_json_object(text: str) -> dict[str, Any]:
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Response does not contain a JSON object.")
    return payload


def parse_candidate_payload(payload: Any, candidate_name: str = "candidate") -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{candidate_name} payload must be a JSON object.")
    prompt = str(payload.get("prompt", "")).strip()
    reasoning = trim_reasoning(str(payload.get("reasoning", "")).strip())
    if not prompt:
        raise ValueError(f"{candidate_name} prompt is empty.")
    if not reasoning:
        raise ValueError(f"{candidate_name} reasoning is empty.")
    return {"prompt": prompt, "reasoning": reasoning}


def parse_candidate_response(text: str) -> dict[str, str]:
    return parse_candidate_payload(extract_json_object(text))


def parse_named_candidate_map_response(text: str, required_names: list[str]) -> dict[str, dict[str, str]]:
    payload = extract_json_object(text)
    results: dict[str, dict[str, str]] = {}
    for name in required_names:
        if name not in payload:
            raise ValueError(f"Missing candidate '{name}' in batch response.")
        results[name] = parse_candidate_payload(payload[name], candidate_name=name)
    return results


def parse_score_payload(payload: Any, candidate_name: str = "candidate") -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{candidate_name} score payload must be a JSON object.")
    consistency = float(payload["consistency"])
    novelty = float(payload["novelty"])
    aesthetic = float(payload["aesthetic"])
    executability = float(payload["executability"])
    reasoning = trim_reasoning(str(payload.get("reasoning", "")).strip())

    overall = (
        0.35 * consistency
        + 0.20 * novelty
        + 0.25 * aesthetic
        + 0.20 * executability
    )
    if consistency < 5.0 or executability < 5.0:
        overall -= 1.0

    return {
        "consistency": round(consistency, 4),
        "novelty": round(novelty, 4),
        "aesthetic": round(aesthetic, 4),
        "executability": round(executability, 4),
        "reasoning": reasoning,
        "overall": round(overall, 4),
    }


def apply_target_penalty(judge: dict[str, Any], prompt: str, target_tid: Any) -> dict[str, Any]:
    consistency = float(judge["consistency"])
    novelty = float(judge["novelty"])
    aesthetic = float(judge["aesthetic"])
    executability = float(judge["executability"])
    reasoning = trim_reasoning(str(judge.get("reasoning", "")).strip())

    explicit_target_hits = find_explicit_target_hits(prompt, target_tid)
    if explicit_target_hits:
        consistency = max(0.0, consistency - 2.5)
        executability = max(0.0, executability - 2.0)
        penalty_note = "Explicitly mentions the target item, which breaks the concealment constraint."
        reasoning = trim_reasoning(f"{reasoning} {penalty_note}".strip())

    overall = (
        0.35 * consistency
        + 0.20 * novelty
        + 0.25 * aesthetic
        + 0.20 * executability
    )
    if consistency < 5.0 or executability < 5.0:
        overall -= 1.0
    if explicit_target_hits:
        overall -= 0.5

    return {
        "consistency": round(consistency, 4),
        "novelty": round(novelty, 4),
        "aesthetic": round(aesthetic, 4),
        "executability": round(executability, 4),
        "reasoning": reasoning,
        "overall": round(overall, 4),
    }


def generate_candidate(messages: list[dict[str, str]], model_name: str, api_key: str) -> dict[str, str]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate_text(
                messages=messages,
                model_name=model_name,
                api_key=api_key,
                enable_thinking=False,
            )
            if not result or not str(result).strip():
                raise ValueError("Model returned empty candidate.")
            return parse_candidate_response(str(result))
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if last_error is None:
        last_error = RuntimeError("Unknown candidate generation failure.")
    raise RuntimeError(f"Candidate generation failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def generate_named_candidates_batch(
    messages: list[dict[str, str]],
    model_name: str,
    api_key: str,
    required_names: list[str],
) -> dict[str, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = generate_text(
                messages=messages,
                model_name=model_name,
                api_key=api_key,
                enable_thinking=False,
            )
            if not result or not str(result).strip():
                raise ValueError("Model returned empty batch candidate output.")
            return parse_named_candidate_map_response(str(result), required_names)
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if last_error is None:
        last_error = RuntimeError("Unknown batch candidate generation failure.")
    raise RuntimeError(f"Batch candidate generation failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def score_candidates_batch(
    sample: dict[str, Any],
    candidates: list[tuple[str, str]],
    model_name: str,
    api_key: str,
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}

    history_tids = normalize_history_tids(sample.get("history_tids"))
    target_tid = normalize_target_tid(sample.get("target_tid"))
    candidate_lines = []
    for candidate_name, prompt in candidates:
        candidate_lines.append(f"{candidate_name}:\n{prompt}")
    user_prompt = f"""User historical TIDs:
{format_history_tids(history_tids)}

Target TID:
{target_tid}

Candidate prompts:
{chr(10).join(candidate_lines)}

Please score every candidate independently.
Output only one JSON object where each key is the candidate name and each value is:
{{"consistency": 0.0, "novelty": 0.0, "aesthetic": 0.0, "executability": 0.0, "reasoning": "..."}}"""

    result = generate_with_retry(
        messages=[
            {"role": "system", "content": BATCH_SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        api_key=api_key,
        enable_thinking=False,
    )

    payload = extract_json_object(result)
    judges: dict[str, dict[str, Any]] = {}
    for candidate_name, prompt in candidates:
        if candidate_name not in payload:
            raise ValueError(f"Missing score for candidate '{candidate_name}'.")
        raw_judge = parse_score_payload(payload[candidate_name], candidate_name=candidate_name)
        judges[candidate_name] = apply_target_penalty(raw_judge, prompt, sample.get("target_tid"))
    return judges


def run_parallel_tasks(
    task_builders: list[tuple[str, Any]],
    max_workers: int = INTERNAL_SEARCH_WORKERS,
) -> dict[str, Any]:
    if not task_builders:
        return {}

    results: dict[str, Any] = {}
    worker_count = max(1, min(max_workers, len(task_builders)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_name = {
            executor.submit(fn): name
            for name, fn in task_builders
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            results[name] = future.result()
    return results


def build_founder_specs() -> list[dict[str, str]]:
    return [
        {"mode": mode, "design_intent": direction}
        for mode, direction in FOUNDER_DIRECTIONS.items()
    ]


def build_context_block(sample: dict[str, Any]) -> str:
    history_tids = normalize_history_tids(sample.get("history_tids"))
    target_tid = normalize_target_tid(sample.get("target_tid"))
    return f"""User historical TIDs:
{format_history_tids(history_tids)}

Target item:
{target_tid}"""


def build_founder_prompt(sample: dict[str, Any], mode: str, direction: str) -> list[dict[str, str]]:
    user_prompt = f"""{build_context_block(sample)}

Please generate one candidate AIGC creative instruction using the {mode} strategy.
Strategy description: {direction}

Requirements:
1. The creative instruction must be centered on the target item, while treating historical interaction as auxiliary reference only.
2. The creative instruction must not explicitly mention, copy, or quote the target item text or tokens.
3. The instruction should be directly usable for AIGC generation and preferably specify subject, scene, style, camera language, atmosphere, and key details.
4. Also output one concise reasoning within 200 characters explaining why this candidate was generated and which information was used.
5. Output only one JSON object: {{"prompt": "...", "reasoning": "..."}}."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_founder_batch_prompt(sample: dict[str, Any], founder_specs: list[dict[str, str]]) -> list[dict[str, str]]:
    strategy_lines = [
        f'- "{founder["mode"]}": {founder["design_intent"]}'
        for founder in founder_specs
    ]
    user_prompt = f"""{build_context_block(sample)}

Please generate one candidate AIGC creative instruction for each strategy below:
{chr(10).join(strategy_lines)}

Requirements:
1. Generate exactly one candidate for each strategy name above.
2. Each candidate must be centered on the target item, while treating historical interaction as auxiliary reference only.
3. Each prompt must not explicitly mention, copy, or quote the target item text or tokens.
4. Each prompt should be directly usable for AIGC generation and preferably specify subject, scene, style, camera language, atmosphere, and key details.
5. Each candidate must also include one concise reasoning within 200 characters.
6. Output only one JSON object using the strategy names as keys, for example:
{{"conservative": {{"prompt": "...", "reasoning": "..."}}, "balanced": {{"prompt": "...", "reasoning": "..."}}, "exploratory": {{"prompt": "...", "reasoning": "..."}}}}"""
    return [
        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_crossover_prompt(
    sample: dict[str, Any],
    elite: dict[str, Any],
    mate: dict[str, Any],
) -> list[dict[str, str]]:
    user_prompt = f"""{build_context_block(sample)}

Please fuse the following two strong hidden candidates and generate a better candidate AIGC creative instruction.

Candidate A mode: {elite['mode']}
Candidate A prompt:
{elite['prompt']}
Candidate A reasoning:
{elite['reasoning']}

Candidate B mode: {mate['mode']}
Candidate B prompt:
{mate['prompt']}
Candidate B reasoning:
{mate['reasoning']}

Requirements:
1. Keep the target item as the core objective and use history only as supporting evidence.
2. Preserve the stronger target-facing anchor, then absorb the better scene organization, visual richness, camera language, or emotional tension.
3. Do not explicitly mention, copy, or quote the target item text or tokens in the prompt.
4. Also output one concise reasoning within 200 characters explaining what was inherited, fused, or discarded.
5. Output only one JSON object: {{"prompt": "...", "reasoning": "..."}}."""
    return [
        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_children_batch_prompt(
    sample: dict[str, Any],
    elite: dict[str, Any],
    mate: dict[str, Any],
) -> list[dict[str, str]]:
    user_prompt = f"""{build_context_block(sample)}

Please generate two new hidden candidates from the strong drafts below.

Elite mode: {elite['mode']}
Elite prompt:
{elite['prompt']}
Elite reasoning:
{elite['reasoning']}

Mate mode: {mate['mode']}
Mate prompt:
{mate['prompt']}
Mate reasoning:
{mate['reasoning']}

Task definitions:
1. "crossover": preserve the stronger target-facing anchor, then absorb the better scene organization, visual richness, camera language, or emotional tension from both drafts.
2. "mutation": keep the elite target-facing semantic anchor stable, but perform one controlled mutation in scene details, composition, camera language, lighting atmosphere, or emotional expression.

Shared requirements:
1. Keep the target item as the core objective and use history only as supporting evidence.
2. Do not explicitly mention, copy, or quote the target item text or tokens in the prompt.
3. Each candidate must include one concise reasoning within 200 characters.
4. Output only one JSON object:
{{"crossover": {{"prompt": "...", "reasoning": "..."}}, "mutation": {{"prompt": "...", "reasoning": "..."}}}}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_mutation_prompt(sample: dict[str, Any], elite: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = f"""{build_context_block(sample)}

Please perform one controlled mutation on the excellent hidden candidate below and generate a better candidate AIGC creative instruction.

Original mode: {elite['mode']}
Original prompt:
{elite['prompt']}
Original reasoning:
{elite['reasoning']}

Requirements:
1. Keep the target-facing semantic anchor stable and let history stay as auxiliary support.
2. Innovation is allowed in scene details, composition, camera language, lighting atmosphere, or emotional expression.
3. Do not explicitly mention, copy, or quote the target item text or tokens in the prompt.
4. Also output one concise reasoning within 200 characters explaining what was preserved and what was mutated.
5. Output only one JSON object: {{"prompt": "...", "reasoning": "..."}}."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def find_explicit_target_hits(prompt: str, target_tid: Any) -> list[str]:
    prompt_lower = str(prompt or "").lower()
    hits: list[str] = []
    for term in normalize_target_terms(target_tid):
        if len(term) < 2:
            continue
        if term in prompt_lower:
            hits.append(term)
    return hits


def score_candidate(sample: dict[str, Any], prompt: str, model_name: str, api_key: str) -> dict[str, Any]:
    history_tids = normalize_history_tids(sample.get("history_tids"))
    target_tid = normalize_target_tid(sample.get("target_tid"))
    user_prompt = f"""User historical TIDs:
{format_history_tids(history_tids)}

Target TID:
{target_tid}

Candidate prompt:
{prompt}"""
    result = generate_with_retry(
        messages=[
            {"role": "system", "content": SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        api_key=api_key,
        enable_thinking=False,
    )

    parsed = extract_json_object(result)
    raw_judge = parse_score_payload(parsed)
    return apply_target_penalty(raw_judge, prompt, sample.get("target_tid"))


def build_evolution_summary(
    founder_candidates: list[dict[str, Any]],
    evolution_trace: list[dict[str, Any]],
    final_candidate: dict[str, Any],
) -> str:
    mode_labels = {
        "conservative": "cautious draft",
        "balanced": "balanced draft",
        "exploratory": "bold draft",
        "crossover": "blended draft",
        "mutation": "refined draft",
    }

    parts: list[str] = []

    founder_lines = []
    for founder in founder_candidates:
        mode = str(founder.get("mode", "")).strip()
        design_intent = str(founder.get("design_intent", "")).strip()
        prompt = str(founder.get("prompt", "")).strip()
        candidate_reasoning = str(founder.get("reasoning", "")).strip()
        if mode:
            line = f"- Initial {mode_labels.get(mode, mode)}:"
            if design_intent:
                line += f" direction={design_intent}"
            if prompt:
                line += f' prompt="{prompt}"'
            if candidate_reasoning:
                line += f" why={candidate_reasoning}"
            founder_lines.append(line)
    if founder_lines:
        parts.append("Initial drafts:\n" + "\n".join(founder_lines))

    round_lines = []
    for trace in evolution_trace:
        round_idx = trace.get("round")
        elite_mode = str(trace.get("elite_mode", "")).strip() or "unknown"
        mate_mode = str(trace.get("mate_mode", "")).strip() or "unknown"
        elite_prompt = str(trace.get("elite_prompt", "")).strip()
        elite_reasoning = str(trace.get("elite_reasoning", "")).strip()
        mate_prompt = str(trace.get("mate_prompt", "")).strip()
        mate_reasoning = str(trace.get("mate_reasoning", "")).strip()
        crossover_prompt = str(trace.get("crossover_prompt", "")).strip()
        crossover_reasoning = str(trace.get("crossover_reasoning", "")).strip()
        mutation_prompt = str(trace.get("mutation_prompt", "")).strip()
        mutation_reasoning = str(trace.get("mutation_reasoning", "")).strip()
        elite_judge = trace.get("elite_judge", {})
        mate_judge = trace.get("mate_judge", {})

        elite_overall = elite_judge.get("overall")
        mate_overall = mate_judge.get("overall")
        elite_judge_reasoning = str(elite_judge.get("reasoning", "")).strip()
        mate_judge_reasoning = str(mate_judge.get("reasoning", "")).strip()

        line = f"- Round {round_idx}: kept draft={mode_labels.get(elite_mode, elite_mode)}"
        if elite_overall is not None:
            line += f" ({elite_overall})"
        if elite_prompt:
            line += f' prompt="{elite_prompt}"'
        line += f", compared draft={mode_labels.get(mate_mode, mate_mode)}"
        if mate_overall is not None:
            line += f" ({mate_overall})"
        if mate_prompt:
            line += f' prompt="{mate_prompt}"'
        line += "."
        if elite_reasoning:
            line += f" Kept draft why={elite_reasoning}"
        if elite_judge_reasoning:
            line += f" Kept draft evaluation={elite_judge_reasoning}"
        if mate_reasoning:
            line += f" Compared draft why={mate_reasoning}"
        if mate_judge_reasoning:
            line += f" Compared draft evaluation={mate_judge_reasoning}"
        if crossover_prompt:
            line += f' Blended draft prompt="{crossover_prompt}"'
        if crossover_reasoning:
            line += f" Blended draft why={crossover_reasoning}"
        if mutation_prompt:
            line += f' Refined draft prompt="{mutation_prompt}"'
        if mutation_reasoning:
            line += f" Refined draft why={mutation_reasoning}"
        round_lines.append(line)
    if round_lines:
        parts.append("Revision path:\n" + "\n".join(round_lines))

    final_mode = str(final_candidate.get("mode", "")).strip() or "unknown"
    final_prompt = str(final_candidate.get("prompt", "")).strip()
    final_reasoning = str(final_candidate.get("reasoning", "")).strip()
    final_judge = final_candidate.get("judge", {})
    final_items = []
    for key in ("consistency", "novelty", "aesthetic", "executability", "overall"):
        if key in final_judge:
            final_items.append(f"{key}={final_judge[key]}")
    final_judge_reasoning = str(final_judge.get("reasoning", "")).strip()
    if final_items or final_reasoning or final_judge_reasoning:
        final_line = f"Final result: source={mode_labels.get(final_mode, final_mode)}"
        if final_prompt:
            final_line += f' prompt="{final_prompt}"'
        if final_items:
            final_line += ", " + ", ".join(final_items)
        if final_reasoning:
            final_line += f". Final prompt why={final_reasoning}"
        if final_judge_reasoning:
            final_line += f" Final evaluation={final_judge_reasoning}"
        parts.append(final_line)

    if not parts:
        raise ValueError("No evolution summary available.")
    return "\n\n".join(parts)


def evolutionary_search(sample: dict[str, Any], model_name: str, rounds: int, api_key: str) -> dict[str, Any]:
    founder_specs = build_founder_specs()
    founder_results = generate_named_candidates_batch(
        messages=build_founder_batch_prompt(sample, founder_specs),
        model_name=model_name,
        api_key=api_key,
        required_names=[founder["mode"] for founder in founder_specs],
    )
    founders: list[dict[str, str]] = []
    for founder in founder_specs:
        generated = founder_results[founder["mode"]]
        founders.append(
            {
                "mode": founder["mode"],
                "design_intent": founder["design_intent"],
                "prompt": generated["prompt"],
                "reasoning": generated["reasoning"],
            }
        )

    population = founders[:]
    trace: list[dict[str, Any]] = []

    for round_idx in range(rounds):
        round_score_candidates = [
            (f"candidate_{idx}", cand["prompt"])
            for idx, cand in enumerate(population, start=1)
        ]
        round_score_results = score_candidates_batch(
            sample=sample,
            candidates=round_score_candidates,
            model_name=model_name,
            api_key=api_key,
        )
        scored = [
            {
                "mode": cand["mode"],
                "prompt": cand["prompt"],
                "reasoning": cand["reasoning"],
                "judge": round_score_results[f"candidate_{idx}"],
            }
            for idx, cand in enumerate(population, start=1)
        ]
        scored.sort(key=lambda x: x["judge"]["overall"], reverse=True)
        elite = scored[0]
        mate = scored[min(1, len(scored) - 1)]

        child_results = generate_named_candidates_batch(
            messages=build_children_batch_prompt(sample, elite, mate),
            model_name=model_name,
            api_key=api_key,
            required_names=["crossover", "mutation"],
        )
        crossover_candidate = child_results["crossover"]
        mutation_candidate = child_results["mutation"]

        trace.append(
            {
                "round": round_idx + 1,
                "elite_mode": elite["mode"],
                "mate_mode": mate["mode"],
                "elite_prompt": elite["prompt"],
                "elite_reasoning": elite["reasoning"],
                "elite_judge": elite["judge"],
                "mate_prompt": mate["prompt"],
                "mate_reasoning": mate["reasoning"],
                "mate_judge": mate["judge"],
                "crossover_prompt": crossover_candidate["prompt"],
                "crossover_reasoning": crossover_candidate["reasoning"],
                "mutation_prompt": mutation_candidate["prompt"],
                "mutation_reasoning": mutation_candidate["reasoning"],
            }
        )

        population = [
            {"mode": elite["mode"], "prompt": elite["prompt"], "reasoning": elite["reasoning"]},
            {"mode": "crossover", "prompt": crossover_candidate["prompt"], "reasoning": crossover_candidate["reasoning"]},
            {"mode": "mutation", "prompt": mutation_candidate["prompt"], "reasoning": mutation_candidate["reasoning"]},
        ]

    final_score_candidates = [
        (f"candidate_{idx}", cand["prompt"])
        for idx, cand in enumerate(population, start=1)
    ]
    final_score_results = score_candidates_batch(
        sample=sample,
        candidates=final_score_candidates,
        model_name=model_name,
        api_key=api_key,
    )
    final_population = [
        {
            "mode": cand["mode"],
            "prompt": cand["prompt"],
            "reasoning": cand["reasoning"],
            "judge": final_score_results[f"candidate_{idx}"],
        }
        for idx, cand in enumerate(population, start=1)
    ]
    final_population.sort(key=lambda x: x["judge"]["overall"], reverse=True)
    best_candidate = final_population[0]
    evolution_summary = build_evolution_summary(founders, trace, best_candidate)

    return {
        "founder_candidates": founders,
        "evolution_trace": trace,
        "evolution_summary": evolution_summary,
        "final_judge": best_candidate["judge"],
        "final_mode": best_candidate["mode"],
        "final_reasoning": best_candidate["reasoning"],
        "final_prompt": best_candidate["prompt"],
    }


def transform_sample(sample: dict[str, Any], model_name: str, rounds: int, api_key: str) -> dict[str, Any]:
    result = evolutionary_search(sample, model_name, rounds, api_key)
    step1_reasoning = ""
    for key in ("step1_reasoning", "identifier_reasoning", "history_to_target_reasoning", "reasoning"):
        value = str(sample.get(key, "")).strip()
        if value:
            step1_reasoning = value
            break
    return {
        "sample_id": sample.get("sample_id"),
        "history_tids": normalize_history_tids(sample.get("history_tids")),
        "target_tid": sample.get("target_tid"),
        "step1_reasoning": step1_reasoning,
        "founder_candidates": result["founder_candidates"],
        "evolution_trace": result["evolution_trace"],
        "evolution_summary": result["evolution_summary"],
        "final_judge": result["final_judge"],
        "final_mode": result["final_mode"],
        "final_reasoning": result["final_reasoning"],
        "final_prompt": result["final_prompt"],
    }


def build_failure_record(sample: Any, error: Exception, index: int) -> dict[str, Any]:
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    return {
        "index": index,
        "sample_id": sample_id,
        "error": str(error),
        "sample": sample,
    }


def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("--items-per-task must be > 0.")
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def process_chunk(
    chunk: list[int],
    samples: list[dict[str, Any]],
    model_name: str,
    rounds: int,
    api_key: str,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    success: list[tuple[int, dict[str, Any]]] = []
    failed: list[dict[str, Any]] = []
    for index in chunk:
        sample = samples[index]
        try:
            success.append((index, transform_sample(sample, model_name, rounds, api_key)))
        except Exception as exc:
            failed.append(build_failure_record(sample, exc, index))
    return success, failed


def process_list_samples_parallel(
    samples: list[dict[str, Any]],
    model_name: str,
    rounds: int,
    api_keys: list[str],
    key_workers: int,
    items_per_task: int,
    checkpoint_path: str = "",
    resume: bool = False,
    save_every: int = SAVE_EVERY,
    progress_every: int = PROGRESS_EVERY,
    milestone_every: int = MILESTONE_EVERY,
    output_path: str = "",
    failed_output_path: str = "",
    emergency_timeout_failures: int = EMERGENCY_TIMEOUT_FAILURES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if key_workers <= 0:
        raise ValueError("--key-workers must be > 0.")
    if save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if progress_every <= 0:
        raise ValueError("--progress-every must be > 0.")
    if milestone_every <= 0:
        raise ValueError("--milestone-every must be > 0.")
    if emergency_timeout_failures <= 0:
        raise ValueError("--emergency-timeout-failures must be > 0.")

    upfront_failures: list[dict[str, Any]] = []
    if resume and checkpoint_path:
        output_slots, done_outputs = load_checkpoint(checkpoint_path, len(samples))
    else:
        output_slots = [None] * len(samples)
        done_outputs = {}

    already_done = set(done_outputs.keys())
    valid_indices: list[int] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            upfront_failures.append(build_failure_record(sample, TypeError("Sample must be an object."), index))
        elif index not in already_done:
            valid_indices.append(index)

    key_to_indices: dict[int, list[int]] = {k: [] for k in range(len(api_keys))}
    for pos, index in enumerate(valid_indices):
        key_to_indices[pos % len(api_keys)].append(index)

    failures: list[dict[str, Any]] = list(upfront_failures)
    pending_checkpoint_outputs: dict[int, dict[str, Any]] = {}

    all_futures = []
    executors: list[ThreadPoolExecutor] = []
    total = len(samples)
    completed = len(already_done)
    last_milestone = completed // milestone_every
    consecutive_timeout_failures = 0
    print(f"Progress: {completed}/{total} completed (resume={resume}).")
    try:
        for key_idx, api_key in enumerate(api_keys):
            executor = ThreadPoolExecutor(max_workers=key_workers)
            executors.append(executor)
            for chunk in chunk_indices(key_to_indices[key_idx], items_per_task):
                all_futures.append(
                    executor.submit(process_chunk, chunk, samples, model_name, rounds, api_key)
                )

        for future in as_completed(all_futures):
            success, failed = future.result()
            for index, out in success:
                output_slots[index] = out
                done_outputs[index] = out
                pending_checkpoint_outputs[index] = out
            completed += len(success)
            if completed % progress_every == 0 or completed == total:
                print(f"Progress: {completed}/{total} completed.")
            if checkpoint_path and pending_checkpoint_outputs and (completed % save_every == 0 or completed == total):
                save_checkpoint(checkpoint_path, total, pending_checkpoint_outputs, len(done_outputs))
                pending_checkpoint_outputs = {}
                print(f"Checkpoint appended: {checkpoint_path} ({completed}/{total})")
            current_milestone = completed // milestone_every
            if checkpoint_path and current_milestone > last_milestone:
                if pending_checkpoint_outputs:
                    save_checkpoint(checkpoint_path, total, pending_checkpoint_outputs, len(done_outputs))
                    pending_checkpoint_outputs = {}
                    print(f"Checkpoint appended before milestone copy: {checkpoint_path} ({completed}/{total})")
                for milestone_idx in range(last_milestone + 1, current_milestone + 1):
                    milestone = milestone_idx * milestone_every
                    milestone_path = build_milestone_checkpoint_path(checkpoint_path, milestone)
                    if checkpoint_path and os.path.exists(checkpoint_path) and not os.path.exists(milestone_path):
                        copy_checkpoint_milestone(checkpoint_path, milestone_path)
                        print(f"Milestone checkpoint saved: {milestone_path} ({completed}/{total})")
                last_milestone = current_milestone
            failures.extend(failed)

            timeout_failures = sum(1 for item in failed if is_timeout_error_text(item.get("error", "")))
            if success:
                consecutive_timeout_failures = 0
            elif failed and timeout_failures == len(failed):
                consecutive_timeout_failures += timeout_failures
            elif failed:
                consecutive_timeout_failures = 0

            if consecutive_timeout_failures >= emergency_timeout_failures:
                save_emergency_state(
                    output_slots=output_slots,
                    failures=failures,
                    done_outputs=done_outputs,
                    output_path=output_path or OUTPUT_JSON_PATH,
                    failed_output_path=failed_output_path or FAILED_OUTPUT_JSON_PATH,
                    checkpoint_path=checkpoint_path,
                    total=total,
                    completed=completed,
                )
                raise RuntimeError(
                    f"Emergency stop: detected {consecutive_timeout_failures} consecutive timeout-related sample failures."
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    outputs = [x for x in output_slots if x is not None]
    return outputs, failures


def main() -> None:
    args = parse_args()
    if not args.input:
        raise ValueError("Input JSON path is empty. Set INPUT_JSON_PATH or pass --input.")
    if not args.output:
        raise ValueError("Output JSON path is empty. Set OUTPUT_JSON_PATH or pass --output.")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    failed_output_path = build_failed_output_path(args.output, args.failed_output)
    checkpoint_path = build_checkpoint_path(args.output, args.checkpoint)
    data = read_json(args.input)
    api_keys = parse_api_keys(args.api_keys)

    failures: list[dict[str, Any]] = []
    if isinstance(data, list):
        outputs, failures = process_list_samples_parallel(
            samples=data,
            model_name=args.model,
            rounds=args.rounds,
            api_keys=api_keys,
            key_workers=args.key_workers,
            items_per_task=args.items_per_task,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            save_every=args.save_every,
            progress_every=args.progress_every,
            milestone_every=args.milestone_every,
            output_path=args.output,
            failed_output_path=failed_output_path,
            emergency_timeout_failures=args.emergency_timeout_failures,
        )
    elif isinstance(data, dict):
        try:
            outputs = transform_sample(data, args.model, args.rounds, api_keys[0])
        except Exception as exc:
            outputs = None
            failures.append(build_failure_record(data, exc, 0))
    else:
        raise TypeError("Input JSON must be an object or an array of objects.")

    write_json(args.output, outputs)
    write_json(failed_output_path, failures)
    print(f"Saved step2 outputs to: {args.output}")
    print(f"Saved step2 failed samples to: {failed_output_path}")
    print(f"Step2 summary: success={0 if outputs is None else (len(outputs) if isinstance(outputs, list) else 1)}, failed={len(failures)}")


if __name__ == "__main__":
    main()
