#!/usr/bin/env python3
"""
Z-Image image generation — generates images from baseline instruction outputs.

Usage:
    python baselines/gen_image_zimage.py                     # uses sample_uids.json (300 image + 100 video)
    python baselines/gen_image_zimage.py --limit 10          # override: first 10 image-type
    python baselines/gen_image_zimage.py --limit 10 --use-gt # use ground truth ins
    python baselines/gen_image_zimage.py --baseline cipher

Args:
    --limit N          Number of samples (default: uses sample_uids.json stratified subset)
    --use-gt           Use ground_truth_ins instead of target_ins
    --baseline NAME    Baseline output to read from (default: oracle)
    --height H         Image height (default: 768)
    --width W          Image width (default: 768)
    --steps N          Inference steps (default: 8, Turbo)
    --guidance F       Guidance scale / CFG (default: 4.0)

Output:
    baselines/{baseline}_output/zimage_{baseline}_output/images/{uid}.png
    baselines/{baseline}_output/zimage_{baseline}_output/metadata.jsonl
"""
import sys
import json
import time
import argparse
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from project_env import env_str, load_project_env

load_project_env(BASE_DIR / ".env")

ZIMAGE_PATH = Path(env_str("NAVIGEN_ZIMAGE_PATH", str(BASE_DIR / "Z-Image-Turbo")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of samples (default: all image-type)")
    parser.add_argument("--use-gt", action="store_true", help="Use ground_truth_ins instead of target_ins")
    parser.add_argument("--baseline", type=str, default="oracle", help="Baseline name to read from")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--prompt-suffix", type=str, default="", help="Append quality/format cues to the prompt")
    args = parser.parse_args()

    # Input: prefer normalized if available, fall back to original
    baseline_dir = Path(__file__).resolve().parent / f"{args.baseline}_output"
    normalized_path = baseline_dir / "results_normalized.jsonl"
    original_path = baseline_dir / "results.jsonl"

    input_path = normalized_path if normalized_path.exists() else original_path
    if not input_path.exists():
        print(f"Error: {original_path} not found")
        sys.exit(1)

    use_normalized = (input_path == normalized_path)

    with open(input_path) as f:
        all_rows = [json.loads(line) for line in f]

    if args.use_gt:
        ins_key = "ground_truth_ins"
        ins_label = "GT"
    elif use_normalized and "normalized_target_ins" in all_rows[0]:
        ins_key = "normalized_target_ins"
        ins_label = "NORM"
    else:
        ins_key = "target_ins"
        ins_label = "GEN"

    # Filter to image type only
    image_rows = [r for r in all_rows if r.get("target_type") == "image"]

    # Apply sample UIDs filter (stratified subset from make_sample.py)
    sample_path = Path(__file__).resolve().parent / "sample_uids.json"
    if sample_path.exists() and not args.limit:
        with open(sample_path) as f:
            sample_data = json.load(f)
        allowed_uids = set(sample_data.get("image", []))
        image_rows = [r for r in image_rows if r["uid"] in allowed_uids]

    # Apply explicit limit if provided
    rows = image_rows[:args.limit] if args.limit else image_rows
    print(f"  Filtered to {len(rows)} image samples (from {len(all_rows)} total)")

    # Output: inside baseline dir, separate for normalized
    baseline_dir = Path(__file__).resolve().parent / f"{args.baseline}_output"
    suffix = "_norm" if use_normalized else ""
    out_dir = baseline_dir / f"zimage_{args.baseline}{suffix}_output"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Z-Image Generation ({ins_label} prompts from {args.baseline})")
    print("=" * 60)
    print(f"  {len(rows)} samples")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}, Guidance: {args.guidance}")
    print(f"  Output: {out_dir}")

    # Load model (once)
    device = f"cuda:{args.device}"
    print("Loading Z-Image pipeline...")
    started_load = time.time()
    pipe = ZImagePipeline.from_pretrained(
        str(ZIMAGE_PATH),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)

    # FP8 quantization (dynamic activation + weight) for ~1.2x speedup
    print("Quantizing transformer to FP8...")
    quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
    print(f"  Model loaded + quantized in {time.time()-started_load:.1f}s")

    # Generate
    print("Generating images...")
    started_gen = time.time()
    metadata = []
    for i, row in enumerate(rows):
        uid = row["uid"]
        prompt = row[ins_key]
        if args.prompt_suffix:
            prompt = prompt + args.prompt_suffix
        source_dataset = row.get("source_dataset", "")
        seed = sum(ord(c) for c in uid) % 100000
        out_path = img_dir / f"{uid}.png"

        try:
            started = time.time()
            image = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                cfg_normalization=False,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]
            image.save(str(out_path))
            elapsed = time.time() - started
        except Exception as e:
            elapsed = 0
            print(f"  [{i+1}/{len(rows)}] uid={uid[:20]}... ERROR: {e}")
            metadata.append({
                "uid": uid,
                "source_dataset": source_dataset,
                "target_type": "image",
                "prompt_source": ins_label,
                "prompt_text": prompt,
                "image_path": "",
                "generation_time": 0,
                "error": str(e),
            })
            continue

        metadata.append({
            "uid": uid,
            "source_dataset": source_dataset,
            "target_type": "image",
            "prompt_source": ins_label,
            "prompt_text": prompt,
            "image_path": str(out_path),
            "generation_time": round(elapsed, 2),
        })

        total_elapsed = time.time() - started_gen
        avg = total_elapsed / (i + 1)
        eta = avg * (len(rows) - i - 1)
        print(f"  [{i+1}/{len(rows)}] uid={uid[:20]}... type={row.get('target_type','?')} {elapsed:.1f}s (ETA {eta:.0f}s)")

    total_time = time.time() - started_gen
    gen_time = sum(e["generation_time"] for e in metadata if "generation_time" in e)
    print(f"\nDone. {len(metadata)} images in {total_time:.0f}s (avg {gen_time/len(metadata):.1f}s/image)")

    # Save metadata
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
