"""
Step 3: Distill a final reasoning paragraph from step1 reasoning and step2 summary.

Input JSON path is intentionally left empty by default. Update INPUT_JSON_PATH
before running, or pass --input/--output from CLI.

Expected input item schema:
{
  "sample_id": "optional",
  "history_tids": [["tid_a", "tid_b"], ["tid_c"]],
  "target_tid": ["target_term_a", "target_term_b"] | "target text",
  "step1_reasoning": "reasoning from step1",
  "evolution_summary": "summary from step2"
}

Output item schema:
{
  "sample_id": "...",
  "history_tids": ...,
  "target_tid": ...,
  "reasoning": "..."
}
"""

from __future__ import annotations

import argparse
import json
import os
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


INPUT_JSON_PATH = "products_user_step2_output.json"
OUTPUT_JSON_PATH = "products_user_step3_output.json"
FAILED_OUTPUT_JSON_PATH = "products_user_step3_output_failed.json"
MODEL_NAME = env_str("NAVIGEN_TEACHER_MODEL", "")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
KEY_WORKERS = 60
ITEMS_PER_TASK = 20
SAVE_EVERY = 1000
PROGRESS_EVERY = 20
MILESTONE_EVERY = 5000


SYSTEM_PROMPT = """You are an AIGC creative planning teacher model.
Your task is to distill a final reasoning paragraph from the user's historical TIDs, the step1 reasoning, the target TID, and the step2 evolution summary.

Strict requirements:
1. The output must be one coherent paragraph of natural-language reasoning written as first-person internal thinking, with "I" as the subject.
2. I should first infer, from the history and the step1 reasoning, what kind of next item the user is likely to engage with, without naming the final target too early.
3. Then I should summarize how the prompt-writing route gradually moved closer to the right answer by comparing early drafts, keeping stronger parts, rewriting weaker parts, and refining the wording.
4. Do not use biological evolution terms such as selection, crossover, mutation, founder, or similar jargon. Use plain language instead.
5. Near the end of the paragraph, I may explicitly mention the final target TID and explain why the route converges there.
6. The paragraph should explain how the inferred target direction, the history clues, and the prompt revision route together support the final prompt-writing idea.
7. Do not output XML/HTML tags, Markdown code blocks, bullet points, titles, or template fields.
8. Do not write as a teacher note, analysis report, or explanation addressed to someone else. The paragraph must read like the student model's own think content."""


USER_PROMPT_TEMPLATE = """User historical TIDs:
{history_tids}

Step1 reasoning from history to the next target:
{step1_reasoning}

Target TID:
{target_tid}

Step2 prompt revision summary:
{evolution_summary}

Based on the information above, output one final reasoning paragraph.

Requirements:
1. Write in first person, using "I" as the subject throughout.
2. You may condense the step1 reasoning, but keep its core judgment about the likely next-item direction.
3. Use the step2 summary to explain in plain language how the drafts moved closer to the right final prompt.
4. Do not use technical or biological evolution wording such as selection, crossover, mutation, founder, elite, mate, or similar labels.
5. Near the end, explicitly mention the target TID and explain why the route finally points there.
6. End by summarizing how I would write the final prompt around that target direction, using history as support rather than as the main objective.
7. Output one complete paragraph only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill one-shot prompt reasoning.")
    parser.add_argument("--input", default=INPUT_JSON_PATH, help="Path to input JSON file.")
    parser.add_argument("--output", default=OUTPUT_JSON_PATH, help="Path to output JSON file.")
    parser.add_argument(
        "--failed-output",
        default=FAILED_OUTPUT_JSON_PATH,
        help="Path to failed-sample JSON file. Defaults to <output_stem>_failed.json.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Teacher model name placeholder.")
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


def extract_step1_reasoning(sample: dict[str, Any]) -> str:
    for key in ("step1_reasoning", "identifier_reasoning", "history_to_target_reasoning", "reasoning"):
        value = str(sample.get(key, "")).strip()
        if value:
            return value
    return "[]"


def extract_evolution_summary(sample: dict[str, Any]) -> str:
    summary = str(sample.get("evolution_summary", "")).strip()
    if not summary:
        raise ValueError("evolution_summary is required for step3.")
    return summary


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
            last_error = ValueError("Model returned empty reasoning.")
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    if last_error is None:
        last_error = RuntimeError("Unknown generate_text failure.")
    raise RuntimeError(f"generate_text failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def ensure_first_person_reasoning(reasoning: str, model_name: str, api_key: str) -> str:
    text = str(reasoning or "").strip()
    if not text:
        return text
    if "I" in text or "I'm" in text or "I'd" in text or "I'll" in text:
        return text

    rewrite_prompt = f"""Please rewrite the following reasoning into one coherent first-person think paragraph in English, using "I" as the subject.

Requirements:
1. Keep the original semantics and order of thought.
2. Keep it as one paragraph.
3. Do not add bullet points, titles, or explanations.
4. Do not use biological evolution terms such as selection, crossover, mutation, founder, elite, or mate.

Original reasoning:
{text}"""
    return generate_with_retry(
        messages=[
            {"role": "system", "content": "You rewrite reasoning into English first-person student-model think text."},
            {"role": "user", "content": rewrite_prompt},
        ],
        model_name=model_name,
        api_key=api_key,
        enable_thinking=False,
    )


def transform_sample(sample: dict[str, Any], model_name: str, api_key: str) -> dict[str, Any]:
    history_tids = normalize_history_tids(sample.get("history_tids"))
    history_text = format_history_tids(history_tids)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        history_tids=history_text,
        step1_reasoning=extract_step1_reasoning(sample),
        target_tid=normalize_target_tid(sample.get("target_tid")),
        evolution_summary=extract_evolution_summary(sample),
    )
    reasoning = generate_with_retry(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model_name=model_name,
        api_key=api_key,
        enable_thinking=False,
    )
    reasoning = ensure_first_person_reasoning(reasoning, model_name, api_key)
    return {
        "sample_id": sample.get("sample_id"),
        "history_tids": history_tids,
        "target_tid": sample.get("target_tid"),
        "reasoning": reasoning,
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
    api_key: str,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    success: list[tuple[int, dict[str, Any]]] = []
    failed: list[dict[str, Any]] = []
    for index in chunk:
        sample = samples[index]
        try:
            success.append((index, transform_sample(sample, model_name, api_key)))
        except Exception as exc:
            failed.append(build_failure_record(sample, exc, index))
    return success, failed


def process_list_samples_parallel(
    samples: list[dict[str, Any]],
    model_name: str,
    api_keys: list[str],
    key_workers: int,
    items_per_task: int,
    checkpoint_path: str = "",
    resume: bool = False,
    save_every: int = SAVE_EVERY,
    progress_every: int = PROGRESS_EVERY,
    milestone_every: int = MILESTONE_EVERY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if key_workers <= 0:
        raise ValueError("--key-workers must be > 0.")
    if save_every <= 0:
        raise ValueError("--save-every must be > 0.")
    if progress_every <= 0:
        raise ValueError("--progress-every must be > 0.")
    if milestone_every <= 0:
        raise ValueError("--milestone-every must be > 0.")

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
    print(f"Progress: {completed}/{total} completed (resume={resume}).")
    try:
        for key_idx, api_key in enumerate(api_keys):
            executor = ThreadPoolExecutor(max_workers=key_workers)
            executors.append(executor)
            for chunk in chunk_indices(key_to_indices[key_idx], items_per_task):
                all_futures.append(executor.submit(process_chunk, chunk, samples, model_name, api_key))

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
            api_keys=api_keys,
            key_workers=args.key_workers,
            items_per_task=args.items_per_task,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            save_every=args.save_every,
            progress_every=args.progress_every,
            milestone_every=args.milestone_every,
        )
    elif isinstance(data, dict):
        try:
            outputs = transform_sample(data, args.model, api_keys[0])
        except Exception as exc:
            outputs = None
            failures.append(build_failure_record(data, exc, 0))
    else:
        raise TypeError("Input JSON must be an object or an array of objects.")

    write_json(args.output, outputs)
    write_json(failed_output_path, failures)
    print(f"Saved step3 outputs to: {args.output}")
    print(f"Saved step3 failed samples to: {failed_output_path}")
    print(f"Step3 summary: success={0 if outputs is None else (len(outputs) if isinstance(outputs, list) else 1)}, failed={len(failures)}")


if __name__ == "__main__":
    main()
