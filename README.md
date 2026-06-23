# NaviGen
NaviGen is a personalized AIGC generation framework built around a **dual item identifier**. Each item is represented with a collaborative code and a textual code in one token stream, using behavioral signals as the substrate and semantic terms as the bridge.

On this representation, NaviGen uses a two-stage **SFT + RL** pipeline. SFT distills preference reasoning and instruction writing from evolutionarily searched supervision, while RL aligns generation with user intent through hierarchical and self-consistent rewards. Across product, game, and short-video domains, NaviGen improves personalized image and video generation, strengthens next-item prediction, and yields more specific, relevant, and visually generatable instructions.

## 🧭 Concepts
Our trained GRPO step-600 model checkpoint is available in [`grpo-step600/`](grpo-step600/).

- `CID`: Collaborative code, a tokenized behavioral identifier such as `<|cid_begin|><s_a_3855><s_b_7257><s_c_3681><|cid_end|>`.
- `TID`: Textual code, a compact list of English semantic terms for an item.
- `Dual identifier`: the coupled CID + TID representation used by NaviGen.
- `cid2cid`: recommend the next CID from user history.
- `cid2ins`: predict target TID and generate an AIGC instruction.

## 🗂️ Layout

```text
preprocess/   TID generation, reasoning data, prompt search, CID vocab expansion
train/        Stage-1 SFT, Stage-2 SFT, GRPO training
infer/        constrained cid2cid and cid2ins inference
generation/   Z-Image image generation and OpenSora video generation
```

## ⚙️ Setup

Install the core stack:

```bash
pip install -r requirements.txt
```

For CUDA training machines, install PyTorch and vLLM builds that match your driver and CUDA version. Video generation also needs an OpenSora environment and local checkpoints. Image generation expects a local Z-Image model directory.

## 🔧 Configuration

Repository scripts load `.env` automatically through `project_env.py`. Keep real API keys local.

Key settings:

| Variable | Purpose |
| --- | --- |
| `DASHSCOPE_API_KEY` / `DASHSCOPE_API_KEYS` | Teacher LLM credentials. Multiple keys are comma-separated. |
| `NAVIGEN_TEACHER_MODEL` | Teacher model name for preprocessing. |
| `NAVIGEN_QWEN3_BASE_MODEL` | Base Qwen3 model directory. |
| `NAVIGEN_CID_MODEL_DIR` | Expanded CID model/tokenizer directory. |
| `NAVIGEN_SFT_INPUT_DIR` | SFT parquet input directory. Defaults to `./dataset`. |
| `NAVIGEN_INFER_INPUT_DIR` | Inference parquet input directory. Defaults to `./dataset`. |
| `NAVIGEN_PID2CID2TID_PATH` | Catalog parquet. Defaults to `./dataset/pid2cid2tid.parquet`. |
| `NAVIGEN_STAGE1_OUTPUT_DIR` / `NAVIGEN_STAGE2_OUTPUT_DIR` | SFT output roots. |
| `NAVIGEN_RL_OUTPUT_DIR` | GRPO output root. |
| `NAVIGEN_VLLM_HOST` / `NAVIGEN_VLLM_PORT` | Local vLLM rollout endpoint. |
| `NAVIGEN_ZIMAGE_PATH` | Z-Image model directory. |
| `NAVIGEN_OPENSORA_*` | OpenSora checkpoint paths and video defaults. |

`dashscope_key_config.py` is still supported as an optional legacy fallback, but it is no longer required.

## 📦 Data

Bundled parquet data lives under `dataset/`:

```text
train_cid2tid.parquet
valid_cid2tid.parquet
test_cid2tid.parquet
train_tid2cid.parquet
valid_tid2cid.parquet
test_tid2cid.parquet
train_cid2cid.parquet
valid_cid2cid.parquet
test_cid2cid.parquet
train_cid2ins.parquet
valid_cid2ins.parquet
test_cid2ins.parquet
pid2cid2tid.parquet
```

Compatibility mapping:

| Task | Input columns | Target columns |
| --- | --- | --- |
| `cid2tid` | `sid` | `tid` / `target_tid` |
| `tid2cid` | `tid` | `sid` |
| `cid2cid` | `hist_sid` | `target_sid` |
| `cid2ins` | `hist_sid` | `target_tid`, `target_ins` |

## 🚀 Commands

Generate TIDs:

```bash
python preprocess/step0_generate_tid_from_caption.py --input products_user_pid2caption.json --output products_user_pid2tid.json --resume
```

Expand Qwen3 with CID tokens:

```bash
python preprocess/expand_qwen3_cid_vocab.py --trust_remote_code
```

Run Stage-1 and Stage-2 SFT:

```bash
python train/sft_aigc_stage1_embed.py
python train/sft_aigc_stage2_full_ft.py
```

Run GRPO:

```bash
python train/rl_grpo_rec_aigc_constrained.py --nproc_per_node 4
```

Run inference:

```bash
python infer/infer_sft_aigc_stage2_cid2cid_constrained.py --num_candidates 40
python infer/infer_sft_aigc_stage2_cid2ins.py --generation_mode two_stage
```

Generate images or videos:

```bash
python generation/gen_image_zimage.py --baseline oracle --height 512 --width 512
python generation/video_gen_opensora.py --input-json outputs.jsonl --prompt-field prediction.target_ins --num-gpus 3
```
