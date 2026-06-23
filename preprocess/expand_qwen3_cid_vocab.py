#!/usr/bin/env python
"""Expand a Qwen3 tokenizer/model with CID tokens from a parquet file.

Default behavior:
1. Read CID values from the original `sid` column in `pid2cid2tid.parquet`.
2. Decompose each CID like
   `<|cid_begin|><s_a_3855><s_b_7257><s_c_3681><|cid_end|>`
   into atomic tokens.
3. Add the missing CID tokens into the tokenizer.
4. Save the updated tokenizer and a full `vocab.json` token-id dump.
5. Optionally resize and save the model weights to match the new vocab size.

Examples:
    python expand_qwen3_cid_vocab.py \
      --model_name_or_path ./Qwen3-1.7B \
      --parquet_path ./pid2cid2tid.parquet \
      --output_dir ./Qwen3-1.7B-cid-expanded

    python expand_qwen3_cid_vocab.py \
      --output_dir ./Qwen3-1.7B-cid-tokenizer \
      --tokenizer_only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_str, load_project_env

load_project_env(REPO_ROOT / ".env")


CID_ATOMIC_PATTERN = re.compile(r"<\|cid_begin\|>|<\|cid_end\|>|<s_[abc]_\d+>")
CID_NUMERIC_PATTERN = re.compile(r"<s_([abc])_(\d+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand Qwen3 tokenizer/model vocab from CID values in parquet."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=env_str("NAVIGEN_QWEN3_BASE_MODEL", "./Qwen3-1.7B"),
        help="Base Qwen3 model/tokenizer directory.",
    )
    parser.add_argument(
        "--parquet_path",
        default=env_str("NAVIGEN_PID2CID2TID_PATH", str(REPO_ROOT / "dataset" / "pid2cid2tid.parquet")),
        help="Parquet file containing the original `sid` column used as CID values.",
    )
    parser.add_argument(
        "--cid_column",
        default="sid",
        help="Column name used to read CID strings from parquet. Keep the original data key by default.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for the expanded tokenizer/model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Batch size used when streaming CID values from parquet.",
    )
    parser.add_argument(
        "--token_granularity",
        choices=["atomic", "whole_cid"],
        default="atomic",
        help=(
            "How to expand the vocab: `atomic` adds `<|cid_begin|>`, `<s_a_x>`, "
            "`<s_b_x>`, `<s_c_x>`, `<|cid_end|>`; `whole_cid` adds full CID strings."
        ),
    )
    parser.add_argument(
        "--tokenizer_only",
        action="store_true",
        help="Only save the expanded tokenizer and vocab dump, do not save model weights.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--allow_overwrite_model_dir",
        action="store_true",
        help="Allow output_dir to be the same as model_name_or_path. Use with care.",
    )
    return parser.parse_args()


def iter_cid_values(
    parquet_path: Path,
    cid_column: str,
    batch_size: int,
):
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema.names
    if cid_column not in schema_names:
        raise ValueError(
            f"Column '{cid_column}' not found in {parquet_path}. "
            f"Available columns: {schema_names}"
        )

    for batch in parquet_file.iter_batches(columns=[cid_column], batch_size=batch_size):
        for value in batch.column(0).to_pylist():
            if value is None:
                continue
            cid = str(value).strip()
            if cid:
                yield cid


def cid_token_sort_key(token: str) -> tuple[int, int | str]:
    if token == "<|cid_begin|>":
        return (0, 0)
    match = CID_NUMERIC_PATTERN.fullmatch(token)
    if match:
        bucket = {"a": 1, "b": 2, "c": 3}[match.group(1)]
        return (bucket, int(match.group(2)))
    if token == "<|cid_end|>":
        return (4, 0)
    return (5, token)


def collect_cid_tokens(
    parquet_path: Path,
    cid_column: str,
    batch_size: int,
    token_granularity: str,
) -> tuple[list[str], dict[str, int]]:
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    sample_invalid_cids: list[str] = []
    tokens: set[str] = set()

    for cid in iter_cid_values(parquet_path, cid_column=cid_column, batch_size=batch_size):
        total_rows += 1
        if token_granularity == "whole_cid":
            tokens.add(cid)
            valid_rows += 1
            continue

        pieces = CID_ATOMIC_PATTERN.findall(cid)
        if "".join(pieces) != cid:
            invalid_rows += 1
            if len(sample_invalid_cids) < 5:
                sample_invalid_cids.append(cid)
            continue

        tokens.update(pieces)
        valid_rows += 1

    if token_granularity == "atomic" and invalid_rows > 0:
        raise ValueError(
            "Found CID rows that do not fully match the expected atomic token format. "
            f"invalid_rows={invalid_rows}, sample_invalid_cids={sample_invalid_cids}"
        )

    sorted_tokens = sorted(tokens, key=cid_token_sort_key)
    stats = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "unique_cid_tokens": len(sorted_tokens),
    }
    return sorted_tokens, stats


def build_added_tokens(tokens: list[str]) -> list[AddedToken]:
    return [
        AddedToken(
            token,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for token in tokens
    ]


def dump_vocab_json(tokenizer, output_dir: Path) -> None:
    vocab = tokenizer.get_vocab()
    ordered_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_vocab, f, ensure_ascii=False)


def dump_token_list(tokens: list[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token)
            f.write("\n")


def ensure_output_dir_is_safe(
    model_dir: Path,
    output_dir: Path,
    allow_overwrite_model_dir: bool,
) -> None:
    if model_dir.resolve() == output_dir.resolve() and not allow_overwrite_model_dir:
        raise ValueError(
            "output_dir points to the original model directory. "
            "Refusing to overwrite in place. Pass --allow_overwrite_model_dir "
            "only if you explicitly want that."
        )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_name_or_path).expanduser().resolve()
    parquet_path = Path(args.parquet_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    ensure_output_dir_is_safe(
        model_dir=model_dir,
        output_dir=output_dir,
        allow_overwrite_model_dir=args.allow_overwrite_model_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cid_tokens, cid_stats = collect_cid_tokens(
        parquet_path=parquet_path,
        cid_column=args.cid_column,
        batch_size=args.batch_size,
        token_granularity=args.token_granularity,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    vocab_before = tokenizer.get_vocab()
    tokenizer_size_before = len(tokenizer)

    missing_cid_tokens = [token for token in cid_tokens if token not in vocab_before]
    added_count = tokenizer.add_tokens(build_added_tokens(missing_cid_tokens))
    tokenizer_size_after = len(tokenizer)

    tokenizer.save_pretrained(output_dir)
    dump_vocab_json(tokenizer, output_dir)
    dump_token_list(cid_tokens, output_dir / "cid_tokens_all.txt")
    dump_token_list(missing_cid_tokens, output_dir / "cid_tokens_added.txt")

    summary = {
        "model_name_or_path": str(model_dir),
        "parquet_path": str(parquet_path),
        "cid_column": args.cid_column,
        "token_granularity": args.token_granularity,
        "cid_stats": cid_stats,
        "tokenizer_vocab_size_before": tokenizer_size_before,
        "tokenizer_vocab_size_after": tokenizer_size_after,
        "cid_tokens_already_in_vocab": len(cid_tokens) - len(missing_cid_tokens),
        "cid_tokens_added_requested": len(missing_cid_tokens),
        "cid_tokens_added_actual": added_count,
        "tokenizer_only": args.tokenizer_only,
    }

    if args.tokenizer_only:
        summary["notes"] = (
            "Only tokenizer artifacts were saved. If you want to train or infer with "
            "the expanded vocab directly, also save a resized model."
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=args.trust_remote_code,
        )
        model_vocab_size_before = model.get_input_embeddings().num_embeddings
        model.resize_token_embeddings(tokenizer_size_after)
        model.config.vocab_size = tokenizer_size_after
        model.save_pretrained(output_dir, safe_serialization=True)

        summary["model_vocab_size_before"] = model_vocab_size_before
        summary["model_vocab_size_after"] = tokenizer_size_after
        summary["notes"] = (
            "New embedding rows for added CID tokens are freshly initialized and need "
            "subsequent finetuning."
        )

    summary_path = output_dir / "cid_vocab_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
