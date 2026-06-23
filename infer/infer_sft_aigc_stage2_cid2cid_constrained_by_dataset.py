#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import infer_sft_aigc_stage2_cid2cid_constrained as base


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "dataset"
DEFAULT_PRED_DIR = SCRIPT_DIR / "sft_output" / "inference_outputs" / "stage2_final_cid2cid_constrained_by_dataset"
CID2CID_GLOB = "*_test_cid2cid.parquet"


def _dataset_name(path: Path) -> str:
    return path.name.removesuffix(".parquet")


def _discover_cid2cid_files(input_dir: Path, datasets: str | None) -> list[Path]:
    if datasets:
        names = [part.strip() for part in datasets.split(",") if part.strip()]
        files: list[Path] = []
        for name in names:
            if name.endswith(".parquet"):
                path = Path(name).expanduser()
                if not path.is_absolute():
                    path = input_dir / path
            elif name.endswith("_test_cid2cid"):
                path = input_dir / f"{name}.parquet"
            else:
                path = input_dir / f"{name}_test_cid2cid.parquet"
            files.append(path.resolve())
    else:
        files = sorted(input_dir.glob(CID2CID_GLOB))
        if not files:
            single_file = input_dir / "test_cid2cid.parquet"
            if single_file.exists():
                files = [single_file]

    if not files:
        raise FileNotFoundError(f"No cid2cid parquet files found under {input_dir} with pattern {CID2CID_GLOB}")

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cid2cid parquet files do not exist: {', '.join(missing)}")
    return files


def _load_examples_from_file(path: Path, max_rows: int | None, prompt_mode: str) -> list[dict]:
    rows = base._read_parquet_rows(path)
    if max_rows is not None:
        rows = rows[:max_rows]

    examples: list[dict] = []
    for idx, row in enumerate(rows):
        example = base._build_infer_example(row, idx, prompt_mode=prompt_mode)
        if example is not None:
            example["dataset"] = _dataset_name(path)
            examples.append(example)
    return examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run stage2 constrained cid2cid inference on each domain cid2cid parquet under "
            "dataset/ by default."
        )
    )
    p.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--search_root", type=str, default=str(base.DEFAULT_MODEL_SEARCH_ROOT))
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Explicit full-model directory or PEFT adapter directory for inference",
    )
    p.add_argument("--pred_dir", type=str, default=str(DEFAULT_PRED_DIR))
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated domains/files to run, e.g. games,prods,video or "
            "games_test_cid2cid.parquet. Defaults to all *_test_cid2cid.parquet files."
        ),
    )
    p.add_argument(
        "--catalog_scope",
        choices=["all", "per_dataset"],
        default="all",
        help=(
            "Build the legal CID trie from all discovered cid2cid files, or rebuild it per dataset "
            "from only that dataset's cid2cid file."
        ),
    )
    p.add_argument("--max_rows", type=int, default=None, help="Optional per-dataset cap for quick smoke tests")
    p.add_argument(
        "--prompt_mode",
        choices=[base.PROMPT_MODE_TRAIN, base.PROMPT_MODE_SIMPLE_DIRECT],
        default=base.PROMPT_MODE_TRAIN,
        help="train uses the RL training prompt; simple_direct restores the original short prompt intended for direct_json_prefix inference.",
    )
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument(
        "--generation_mode",
        choices=[base.GENERATION_MODE_DIRECT, base.GENERATION_MODE_TWO_STAGE],
        default=base.GENERATION_MODE_DIRECT,
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
    p.add_argument(
        "--num_candidates",
        type=int,
        default=40,
        help="How many candidate generations to keep per example for ranking metrics",
    )
    p.add_argument(
        "--recall_ks",
        type=str,
        default="1,5,10,20,40",
        help="Comma-separated K values for Recall@K and NDCG@K",
    )
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    return p.parse_args()


def _build_cid_trie(tokenizer, parquet_files: list[Path], close_token_id: int) -> tuple[dict, int]:
    cid_catalog: set[str] = set()
    for parquet_file in parquet_files:
        cid_catalog.update(base._collect_cid_catalog_from_parquet(parquet_file))
    if not cid_catalog:
        raise ValueError("Collected empty CID catalog from cid2cid parquet files.")

    cid_token_seqs = base._build_allowed_cid_token_seqs(tokenizer, cid_catalog)
    cid_trie = base._build_trie(cid_token_seqs, close_token_id=close_token_id)
    return cid_trie, len(cid_catalog)


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = base._maybe_init_distributed()

    try:
        input_dir = Path(args.input_dir).resolve()
        search_root = Path(args.search_root).resolve()
        pred_dir = Path(args.pred_dir).resolve()
        cid2cid_files = _discover_cid2cid_files(input_dir=input_dir, datasets=args.datasets)

        recall_ks = sorted({int(v.strip()) for v in args.recall_ks.split(",") if v.strip()})
        if not recall_ks or any(k <= 0 for k in recall_ks):
            raise ValueError(f"Invalid --recall_ks: {args.recall_ks}")
        if args.num_candidates <= 0:
            raise ValueError(f"--num_candidates must be > 0, got {args.num_candidates}")

        effective_num_candidates = max(args.num_candidates, max(recall_ks))
        if effective_num_candidates != args.num_candidates and base._is_main_process():
            print(
                f"[{base.DEFAULT_TASK}] num_candidates={args.num_candidates} is smaller than max K={max(recall_ks)}; "
                f"using num_candidates={effective_num_candidates} instead."
            )

        artifact_kind, model_dir, base_model_dir = base._resolve_model_dir(args.model_dir, search_root)
        model_source = base._build_model_source(
            model_dir,
            artifact_kind=artifact_kind,
            base_model_dir=base_model_dir,
        )
        print(
            f"[{base.DEFAULT_TASK}][rank{rank}] using {artifact_kind} model artifact: {model_dir} "
            f"(base={base_model_dir or '-'}, source={model_source}, world_size={world_size}, local_rank={local_rank})"
        )
        if base._is_main_process():
            print(
                f"[{base.DEFAULT_TASK}] cid2cid datasets: "
                + ", ".join(path.name for path in cid2cid_files)
            )

        model, tokenizer = base._load_model_and_tokenizer(
            artifact_kind=artifact_kind,
            model_dir=model_dir,
            base_model_dir=base_model_dir,
        )
        close_token_id = base._single_token_id(tokenizer, '"}', "JSON close token")

        all_cid_trie: dict | None = None
        all_catalog_size: int | None = None
        if args.catalog_scope == "all":
            all_cid_trie, all_catalog_size = _build_cid_trie(
                tokenizer=tokenizer,
                parquet_files=cid2cid_files,
                close_token_id=close_token_id,
            )
            if base._is_main_process():
                print(
                    f"[{base.DEFAULT_TASK}] built legal CID trie from {len(cid2cid_files)} cid2cid files; "
                    f"catalog_size={all_catalog_size}"
                )

        summary: list[dict] = []
        for parquet_path in cid2cid_files:
            dataset = _dataset_name(parquet_path)
            examples = _load_examples_from_file(parquet_path, max_rows=args.max_rows, prompt_mode=args.prompt_mode)
            if not examples:
                if base._is_main_process():
                    print(f"[{base.DEFAULT_TASK}][{dataset}] no valid examples in {parquet_path}, skip.")
                continue

            if args.catalog_scope == "per_dataset":
                cid_trie, catalog_size = _build_cid_trie(
                    tokenizer=tokenizer,
                    parquet_files=[parquet_path],
                    close_token_id=close_token_id,
                )
                if base._is_main_process():
                    print(f"[{base.DEFAULT_TASK}][{dataset}] built per-dataset CID trie; catalog_size={catalog_size}")
            else:
                if all_cid_trie is None or all_catalog_size is None:
                    raise RuntimeError("Shared CID trie was not initialized.")
                cid_trie = all_cid_trie
                catalog_size = all_catalog_size

            sharded_examples = base._shard_examples(examples, rank=rank, world_size=world_size)
            print(
                f"[{base.DEFAULT_TASK}][{dataset}][rank{rank}] loaded total={len(examples)} "
                f"local_shard={len(sharded_examples)} from {parquet_path}"
            )

            dataset_pred_dir = pred_dir / dataset
            shard_out_path = base._run_task(
                parquet_path=parquet_path,
                examples=sharded_examples,
                model=model,
                tokenizer=tokenizer,
                pred_dir=dataset_pred_dir,
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
                catalog_size=catalog_size,
                generation_mode=args.generation_mode,
                stage1_max_new_tokens=args.stage1_max_new_tokens,
                stage2_max_new_tokens=args.stage2_max_new_tokens,
                stage1_num_candidates=args.stage1_num_candidates,
                stage1_stop_text=args.stage1_stop_text,
            )

            base._dist_barrier()

            if base._is_main_process():
                final_name = base._append_model_source_to_name(
                    f"{base.DEFAULT_TASK}_predictions.jsonl",
                    model_source,
                )
                if world_size > 1:
                    final_out_path = dataset_pred_dir / final_name
                    part_paths = [
                        dataset_pred_dir / final_name.replace(".jsonl", f".rank{part_rank}.jsonl")
                        for part_rank in range(world_size)
                    ]
                    merged_path = base._merge_rank_outputs(part_paths=part_paths, merged_path=final_out_path)
                else:
                    merged_path = shard_out_path

                metrics = base._summarize_metrics(merged_path, recall_ks=recall_ks)
                metrics["dataset"] = dataset
                metrics["source_file"] = str(parquet_path)
                summary.append(metrics)
                print(f"[{base.DEFAULT_TASK}][{dataset}] wrote predictions to {merged_path}")
                print(
                    f"[{base.DEFAULT_TASK}][{dataset}] metrics: "
                    f"top1_json_parse_rate={metrics['top1_json_parse_rate']:.4f}, "
                    f"exact_match_accuracy={metrics['exact_match_accuracy']:.4f}, "
                    + ", ".join(
                        f"recall@{k}={metrics[f'recall@{k}']:.4f}, ndcg@{k}={metrics[f'ndcg@{k}']:.4f}"
                        for k in recall_ks
                    )
                )

            base._dist_barrier()

        if base._is_main_process() and summary:
            summary_path = pred_dir / base._append_model_source_to_name(
                "cid2cid_by_dataset_metrics_summary.json",
                model_source,
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[{base.DEFAULT_TASK}] wrote dataset metrics summary to {summary_path}")

        return 0
    finally:
        base._maybe_destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
