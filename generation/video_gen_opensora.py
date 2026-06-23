"""
Ours-only batch video generation across multiple GPUs.

It writes videos to:
    ./extra_output_videos/ours_output/<dataset>/<uid>.mp4

Edit CONFIG below to change parameters. CLI args can override individual fields.
Supports resume: re-running the script skips already-generated videos.

Usage:
    python scripts/video_gen.py
    python scripts/video_gen.py --resolution 480p
    python scripts/video_gen.py --num-gpus 1
    python scripts/video_gen.py --input-json /path/to/ours.jsonl --prompt-field prediction.target_ins
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_float, env_int, env_str, load_project_env

load_project_env(REPO_ROOT / ".env")

CONFIG = dict(
    # --- data ---
    # Output folder name under output_dir. Keep this as ours_output for ours runs.
    baseline="ours_output",
    # Input data directory. The script reads input_dir / baseline / metadata_filename.
    input_dir=env_str("NAVIGEN_VIDEO_INPUT_DIR", str(REPO_ROOT / "extra_output")),
    # Ours 100-video prediction file.
    metadata_filename="",
    # Optional: read one JSON/JSONL file instead of input_dir/baseline/metadata_filename.
    input_json="",
    # Field used as generation prompt when input_json is set.
    prompt_field="prediction.target_ins",
    # Output folder name when input_json is set; empty keeps baseline=ours_output.
    output_name="ours_output",
    # default dataset subdir for records without source_dataset
    default_dataset="video",
    # skip records marked parse_error=true when input_json is set
    skip_parse_error=True,
    # output root directory
    output_dir=env_str("NAVIGEN_VIDEO_OUTPUT_DIR", str(REPO_ROOT / "extra_output_videos")),
    # number of GPUs to use
    num_gpus=env_int("NAVIGEN_VIDEO_NUM_GPUS", 3),

    # --- model checkpoints ---
    t5_path=env_str("NAVIGEN_OPENSORA_T5_PATH", str(REPO_ROOT / "ckpts" / "t5-v1_1-xxl")),
    vae_path=env_str("NAVIGEN_OPENSORA_VAE_PATH", str(REPO_ROOT / "ckpts" / "OpenSora-VAE-v1.3")),
    dit_path=env_str("NAVIGEN_OPENSORA_DIT_PATH", str(REPO_ROOT / "ckpts" / "OpenSora-STDiT-v4")),

    # --- inference ---
    resolution=env_str("NAVIGEN_VIDEO_RESOLUTION", "720p"),
    aspect_ratio=env_str("NAVIGEN_VIDEO_ASPECT_RATIO", "9:16"),
    num_frames=env_str("NAVIGEN_VIDEO_NUM_FRAMES", "81"),
    num_sampling_steps=env_int("NAVIGEN_VIDEO_STEPS", 30),
    cfg_scale=env_float("NAVIGEN_VIDEO_CFG_SCALE", 4.0),
    fps=env_int("NAVIGEN_VIDEO_FPS", 24),
    seed=env_int("NAVIGEN_VIDEO_SEED", 42),
    dtype=env_str("NAVIGEN_VIDEO_DTYPE", "bf16"),
)

# mapping from source_dataset field to short name
DATASET_KEY_MAP = {
    "games_test_cid2cid": "games",
    "prods_test_cid2cid": "prods",
    "video_test_cid2cid": "video",
}

METADATA_FILENAME_FALLBACKS = (
    "cid2cid_extra_100video.jsonl",
    "cid2cid_extra_300image_100video.jsonl",
    "results_extra_100video.jsonl",
    "results_video.jsonl",
    "results.jsonl",
)

UID_FIELDS = ("uid", "source_row.uid", "source.uid", "row.uid")
PROMPT_FIELDS = (
    "prediction.target_ins",
    "parsed_json.target_ins",
    "pred_target_ins",
    "target_ins",
    "source_row.target_ins",
)
SOURCE_DATASET_FIELDS = ("source_dataset", "source_row.source_dataset")
TARGET_TYPE_FIELDS = (
    "llm_label",
    "target_type",
    "source_row.target_type",
    "source.target_type",
    "row.target_type",
    "label.target_type",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--metadata-filename", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--prompt-field", type=str, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--default-dataset", type=str, default=None)
    parser.add_argument("--include-parse-error", action="store_true")
    parser.add_argument("--resolution", type=str, default=None)
    parser.add_argument("--num-frames", type=str, default=None)
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--aspect-ratio", type=str, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    # internal: worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-file", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def merge_config(cli_args):
    """Merge CLI args into CONFIG; CLI wins when not None."""
    cfg = dict(CONFIG)
    overrides = ["baseline", "input_dir", "metadata_filename", "output_dir", "num_gpus",
                 "input_json", "prompt_field", "output_name",
                 "default_dataset", "resolution", "num_frames",
                 "num_sampling_steps", "cfg_scale", "seed", "aspect_ratio",
                 "fps", "dtype"]
    for k in overrides:
        v = getattr(cli_args, k.replace("-", "_"), None)
        if v is not None:
            cfg[k] = v
    if getattr(cli_args, "include_parse_error", False):
        cfg["skip_parse_error"] = False
    return cfg


def sanitize_field(value):
    """Keep chunk files as one TSV record per sample."""
    return " ".join(str(value).split())


def get_nested_value(rec, path):
    value = rec
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def first_nested_value(rec, fields):
    for field in fields:
        value = get_nested_value(rec, field)
        if value is not None and value != "":
            return value
    return None


def dataset_short_name(rec, default_dataset):
    source_dataset = first_nested_value(rec, SOURCE_DATASET_FIELDS)
    if source_dataset:
        return DATASET_KEY_MAP.get(source_dataset, "unknown")
    return default_dataset


def resolve_metadata_path(baseline, input_dir, metadata_filename):
    baseline_dir = os.path.join(input_dir, baseline)
    candidates = []
    if metadata_filename:
        candidates.append(metadata_filename)
    candidates.extend(x for x in METADATA_FILENAME_FALLBACKS if x not in candidates)
    for filename in candidates:
        path = os.path.join(baseline_dir, filename)
        if os.path.exists(path):
            return path
    return os.path.join(baseline_dir, metadata_filename or METADATA_FILENAME_FALLBACKS[0])


def baseline_has_metadata(baseline, input_dir, metadata_filename):
    return os.path.exists(resolve_metadata_path(baseline, input_dir, metadata_filename))


def load_all_samples(baseline, input_dir, metadata_filename):
    """Load baseline jsonl, return [(uid, prompt, dataset_short_name), ...]."""
    path = resolve_metadata_path(baseline, input_dir, metadata_filename)
    if not os.path.exists(path):
        print(f"ERROR: no supported metadata file found for {baseline} under {input_dir}")
        sys.exit(1)

    samples = []
    missing_prompt_uids = []
    skipped_non_video = 0
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            target_type = first_nested_value(rec, TARGET_TYPE_FIELDS)
            if target_type and str(target_type).lower() != "video":
                skipped_non_video += 1
                continue
            uid = first_nested_value(rec, UID_FIELDS) or f"sample_{idx:05d}"
            prompt = first_nested_value(rec, PROMPT_FIELDS)
            if not prompt:
                missing_prompt_uids.append(str(uid))
                continue
            ds = dataset_short_name(rec, "unknown")
            samples.append((sanitize_field(uid), sanitize_field(prompt), ds))

    print(f"Metadata file: {path}")
    if skipped_non_video:
        print(f"Skipped non-video records: {skipped_non_video}")
    if missing_prompt_uids:
        preview = ", ".join(missing_prompt_uids[:10])
        suffix = "" if len(missing_prompt_uids) <= 10 else " ..."
        print(f"Skipped records missing prompt: {len(missing_prompt_uids)} ({preview}{suffix})")

    return samples


def load_json_samples(input_json, prompt_field, default_dataset, skip_parse_error=True):
    """Load JSON/JSONL records, return [(uid, prompt, dataset_short_name), ...]."""
    if not os.path.exists(input_json):
        print(f"ERROR: {input_json} not found")
        sys.exit(1)

    records = []
    if input_json.endswith(".jsonl"):
        with open(input_json, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with open(input_json, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ("records", "results", "data", "items"):
                if isinstance(data.get(key), list):
                    records = data[key]
                    break
            else:
                print(f"ERROR: {input_json} is a dict, but no records/results/data/items list was found")
                sys.exit(1)
        else:
            print(f"ERROR: unsupported JSON root type in {input_json}: {type(data).__name__}")
            sys.exit(1)

    samples = []
    skipped_parse_error = 0
    missing_prompt_uids = []
    for idx, rec in enumerate(records):
        if skip_parse_error and rec.get("parse_error"):
            skipped_parse_error += 1
            continue

        prompt = first_nested_value(rec, (prompt_field,))
        if not prompt:
            missing_prompt_uids.append(str(rec.get("uid") or f"sample_{idx:05d}"))
            continue

        uid = rec.get("uid") or f"sample_{idx:05d}"
        ds = dataset_short_name(rec, default_dataset)
        samples.append((sanitize_field(uid), sanitize_field(prompt), sanitize_field(ds)))

    if skipped_parse_error:
        print(f"Skipped parse_error records: {skipped_parse_error}")
    if missing_prompt_uids:
        preview = ", ".join(missing_prompt_uids[:10])
        suffix = "" if len(missing_prompt_uids) <= 10 else " ..."
        print(f"Skipped records missing {prompt_field}: {len(missing_prompt_uids)} ({preview}{suffix})")

    return samples


def count_done(save_dir):
    """Count already-generated .mp4 files (non-zero size)."""
    if not os.path.isdir(save_dir):
        return set()
    done = set()
    for f in os.listdir(save_dir):
        if f.endswith(".mp4"):
            fpath = os.path.join(save_dir, f)
            if os.path.getsize(fpath) > 0:
                done.add(f.replace(".mp4", ""))
    return done


def split_chunks(samples, num_gpus):
    """Split samples evenly across num_gpus."""
    chunks = [[] for _ in range(num_gpus)]
    for i, s in enumerate(samples):
        chunks[i % num_gpus].append(s)
    return chunks


def run_worker(args):
    """Worker mode: load model once, generate videos with resume support."""
    import torch
    from opensora.datasets import save_sample
    from opensora.datasets.aspect import get_image_size, get_num_frames
    from opensora.models.text_encoder.t5 import text_preprocessing
    from opensora.registry import MODELS, SCHEDULERS, build_module
    from opensora.utils.misc import to_torch_dtype

    gpu = args.gpu
    baseline = args.baseline

    torch.set_grad_enabled(False)
    cfg = merge_config(args)

    # --- read chunk ---
    items = []
    with open(args.chunk_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, prompt, ds = line.split("\t", 2)
            items.append((uid, prompt, ds))

    print(f"[GPU {gpu}] {len(items)} samples assigned")

    # --- resume: find already-done uids ---
    # Check all dataset subdirs for existing files
    done_uids = set()
    for ds in set(d for _, _, d in items):
        sd = os.path.join(cfg["output_dir"], baseline, ds)
        done_uids |= count_done(sd)

    pending = [(uid, p, ds) for uid, p, ds in items if uid not in done_uids]
    skipped = len(items) - len(pending)
    if skipped:
        print(f"[GPU {gpu}] Resuming: {skipped} done, {len(pending)} pending")

    if not pending:
        print(f"[GPU {gpu}] All done, nothing to generate.")
        return

    # --- device & dtype ---
    # CUDA_VISIBLE_DEVICES is set per-worker, so the assigned GPU is always cuda:0
    device = "cuda:0"
    dtype = to_torch_dtype(cfg["dtype"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- build models ---
    print(f"[GPU {gpu}] Building text encoder...")
    text_encoder = build_module(
        dict(type="t5", from_pretrained=cfg["t5_path"], model_max_length=300),
        MODELS, device=device,
    )

    print(f"[GPU {gpu}] Building VAE...")
    vae = build_module(
        dict(
            type="OpenSoraVAE_V1_3",
            from_pretrained=cfg["vae_path"],
            z_channels=16,
            micro_batch_size=1,
            micro_batch_size_2d=4,
            micro_frame_size=17,
            use_tiled_conv3d=True,
            tile_size=4,
            normalization="video",
            temporal_overlap=True,
            force_huggingface=True,
        ),
        MODELS,
    ).to(device, dtype).eval()

    print(f"[GPU {gpu}] Building STDiT3-XL/2...")
    image_size = get_image_size(cfg["resolution"], cfg["aspect_ratio"])
    num_frames = get_num_frames(cfg["num_frames"])
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    print(f"[GPU {gpu}]   resolution={cfg['resolution']}, pixel={image_size}, latent={latent_size}")

    model = build_module(
        dict(
            type="STDiT3-XL/2",
            from_pretrained=cfg["dit_path"],
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=False,
            kernel_size=(8, 8, -1),
            use_spatial_rope=True,
            force_huggingface=True,
        ),
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
    ).to(device, dtype).eval()
    text_encoder.y_embedder = model.y_embedder

    print(f"[GPU {gpu}] Building scheduler...")
    scheduler = build_module(
        dict(
            type="rflow",
            use_timestep_transform=True,
            num_sampling_steps=cfg["num_sampling_steps"],
            cfg_scale=cfg["cfg_scale"],
            use_oscillation_guidance=False,
            use_flaw_fix=True,
        ),
        SCHEDULERS,
    )

    # --- prepare shared tensors ---
    fps = torch.tensor([cfg["fps"]], device=device, dtype=dtype)
    height = torch.tensor([image_size[0]], device=device, dtype=dtype)
    width = torch.tensor([image_size[1]], device=device, dtype=dtype)
    nf = torch.tensor([num_frames], device=device, dtype=dtype)
    ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype)
    model_args = dict(height=height, width=width, num_frames=nf, ar=ar, fps=fps)

    # --- run ---
    t_start = time.time()
    for idx, (uid, prompt, ds) in enumerate(pending):
        save_dir = os.path.join(cfg["output_dir"], baseline, ds)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{uid}.mp4")

        # double-check in case another worker wrote it
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"[GPU {gpu}] [{idx+1}/{len(pending)}] {uid[:12]}... skip (exists)")
            continue

        print(f"[GPU {gpu}] [{idx+1}/{len(pending)}] [{ds}] {uid[:12]}... {prompt[:80]}")
        t_item = time.time()

        prompt_clean = [text_preprocessing(prompt)]
        torch.manual_seed(cfg["seed"])
        z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)

        samples = scheduler.sample(
            model, text_encoder,
            z=z, prompts=prompt_clean, device=device,
            additional_args=model_args, progress=False,
        )

        video = vae.decode(samples.to(dtype), num_frames=num_frames).squeeze(0)
        if torch.isnan(video).any():
            print(f"[GPU {gpu}] [WARN] NaN in {uid}")

        save_sample(video, fps=cfg["fps"], save_path=save_path)
        print(f"[GPU {gpu}]   -> {save_path}  ({time.time()-t_item:.1f}s)")

    total = len(pending)
    elapsed = time.time() - t_start
    print(f"[GPU {gpu}] done: {total} videos in {elapsed:.0f}s ({elapsed/max(total,1):.1f}s/ea)")


def get_baseline_list(cfg):
    """Return the single ours output name to run."""
    if cfg["input_json"]:
        output_name = cfg["output_name"].strip()
        if not output_name:
            output_name = cfg["baseline"]
        return [sanitize_field(output_name)]

    if cfg["baseline"] == "all":
        raise ValueError("video_gen.py is ours-only. Use baseline_video_gen.py to run all baselines.")
    return [cfg["baseline"]]


def run_baseline(cfg, baseline):
    """Launch 3-GPU workers for one baseline, wait for completion."""
    num_gpus = cfg["num_gpus"]

    print(f"\n{'='*60}")
    print(f"Ours output name: {baseline}")
    print(f"Config: resolution={cfg['resolution']}, frames={cfg['num_frames']}, steps={cfg['num_sampling_steps']}, cfg={cfg['cfg_scale']}")
    if cfg["input_json"]:
        print(f"Input JSON: {cfg['input_json']}")
        print(f"Prompt field: {cfg['prompt_field']}")
    else:
        print(f"Input dir: {cfg['input_dir']}")
        print(f"Metadata filename: {cfg['metadata_filename']} (with fallbacks)")
    print(f"{'='*60}")

    # --- load all samples ---
    if cfg["input_json"]:
        samples = load_json_samples(
            cfg["input_json"],
            cfg["prompt_field"],
            cfg["default_dataset"],
            cfg["skip_parse_error"],
        )
    else:
        samples = load_all_samples(baseline, cfg["input_dir"], cfg["metadata_filename"])
    print(f"Total samples: {len(samples)}")

    # show per-dataset counts
    from collections import Counter
    ds_counts = Counter(s[2] for s in samples)
    for ds, n in ds_counts.items():
        done = count_done(os.path.join(cfg["output_dir"], baseline, ds))
        print(f"  {ds}: {n} total, {len(done)} done, {n - len(done)} pending")

    # --- split evenly across GPUs ---
    chunks = split_chunks(samples, num_gpus)
    print(f"\nSplit across {num_gpus} GPUs:")
    for i, chunk in enumerate(chunks):
        c = Counter(s[2] for s in chunk)
        detail = ", ".join(f"{ds}={n}" for ds, n in sorted(c.items()))
        print(f"  GPU {i}: {len(chunk)} samples ({detail})")

    # --- write chunk files ---
    prompt_dir = os.path.join(cfg["output_dir"], baseline)
    os.makedirs(prompt_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        fpath = os.path.join(prompt_dir, f"_chunk_gpu{i}.txt")
        with open(fpath, "w") as f:
            for uid, prompt, ds in chunk:
                f.write(f"{uid}\t{prompt}\t{ds}\n")

    # --- launch workers ---
    print(f"\nLaunching {num_gpus} GPU workers...\n")
    procs = []
    script = __file__
    for i in range(num_gpus):
        cmd = [
            sys.executable, script,
            "--worker",
            "--baseline", baseline,
            "--gpu", str(i),
            "--dataset", "all",
            "--chunk-file", os.path.join(prompt_dir, f"_chunk_gpu{i}.txt"),
            "--output-dir", cfg["output_dir"],
            "--resolution", cfg["resolution"],
            "--num-frames", cfg["num_frames"],
            "--num-sampling-steps", str(cfg["num_sampling_steps"]),
            "--cfg-scale", str(cfg["cfg_scale"]),
            "--seed", str(cfg["seed"]),
            "--aspect-ratio", cfg["aspect_ratio"],
            "--fps", str(cfg["fps"]),
            "--dtype", cfg["dtype"],
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        print(f"  [GPU {i}] {len(chunks[i])} samples")
        p = subprocess.Popen(cmd, env=env)
        procs.append((i, p))

    print("\nWaiting for workers... (Ctrl+C to stop)")
    for i, p in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  [GPU {i}] {status}")

    print(f"Ours generation {baseline} done.\n")


def main():
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    cfg = merge_config(args)
    baselines = get_baseline_list(cfg)

    print(f"Will run ours video generation: {baselines}")
    t_all = time.time()

    for idx, bl in enumerate(baselines):
        print(f"\n[{idx+1}/{len(baselines)}] Starting ours run {bl}...")
        run_baseline(cfg, bl)

    print(f"\nOurs video generation finished in {(time.time()-t_all)/3600:.1f} hours.")


if __name__ == "__main__":
    main()
