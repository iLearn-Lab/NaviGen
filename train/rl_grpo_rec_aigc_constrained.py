"""
rl_grpo_rec_aigc.py

GRPO reinforcement learning for Qwen3-based recommendation and AIGC tasks.

1. cid2cid
   Predict the target CID from a user's historical CID sequence.
   - Input field: hist_cid
   - Target field: target_cid
   - The model generates JSON online: {"target_cid": "..."}
   - Reward has two parts:
     a) format reward: checks JSON parsing and the target_cid field
     b) CID reward: layer-wise scoring over the three residual CID tokens
        * s_a match defaults to 0.5
        * s_b match defaults to 0.3
        * s_c match defaults to 0.2
        * all three layers matched gives 1.0
   - If no layer matches but the generated CID is in the known CID catalog
     from pid2sid2tid.parquet, a small in-range zero-hit bonus is awarded.
   - If the generated CID is outside the known catalog, CID reward is 0.
   - Final cid2cid reward = cid_reward_weight * cid_reward + format_reward_weight * format_reward.
   - Training and evaluation both use online generation plus online reward scoring.

2. cid2ins
   Predict the target TID and an AIGC creative instruction from CID history.
   - Input field: hist_cid
   - Target fields: target_tid, target_ins
   - The model generates JSON online: {"target_tid": [...], "target_ins": "..."}
   - Reward has two parts:
     a) format reward: checks JSON parsing and required fields
     b) semantic reward: uses a multi-dimensional LLM judge
   - Judge dimensions include specificity, creativity, content quality,
     visual generatability, target alignment, and predicted-TID alignment.
   - The judge returns sub-scores and bands, then code aggregates them with
     fixed weights for transparent tuning.
   - If DashScope keys are unavailable, cid2ins judge falls back to heuristic scoring.
   - Final cid2ins reward = cid2ins_semantic_reward_weight * semantic_reward
     + format_reward_weight * format_reward

Training flow:
  - Mix cid2cid and cid2ins parquet samples into one dataset.
  - Render each row into a chat prompt.
  - GRPOTrainer generates num_generations completions for each prompt.
  - RewardOrchestrator scores every completion online.
  - GRPO updates the policy from relative rewards within each prompt group.
"""

from __future__ import annotations

import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import copy
import functools
import json
import logging
import importlib.util
import inspect
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import types
import warnings
import dataclasses
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from transformers import HfArgumentParser, LogitsProcessor, LogitsProcessorList, TrainerCallback
import transformers.trainer as hf_trainer

# transformers 5.2.x may return `(available, version)` tuples from
# `_is_package_available` even when `return_version=False`, while current TRL
# expects a plain bool. Normalize the return type before importing TRL so
# `vllm_ascend` is not falsely treated as available on NVIDIA environments.
import trl.import_utils as trl_import_utils

def _normalize_trl_package_flag(value):
    if isinstance(value, tuple):
        return bool(value[0])
    return bool(value)


for _name, _value in list(vars(trl_import_utils).items()):
    if _name.startswith("_") and _name.endswith("_available"):
        setattr(trl_import_utils, _name, _normalize_trl_package_flag(_value))

from trl import GRPOConfig, GRPOTrainer
import trl.models.utils as trl_model_utils
import trl.trainer.grpo_trainer as trl_grpo_trainer
import trl.extras.vllm_client as trl_vllm_client


os.environ.setdefault("OMP_NUM_THREADS", "8")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main = local_rank == 0
num_gpus = int(os.environ.get("WORLD_SIZE", 1))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_env import env_int, env_str, load_project_env

load_project_env(REPO_ROOT / ".env")

PROJECT_ROOT = SCRIPT_DIR
DEFAULT_MODEL_DIR = Path(env_str("NAVIGEN_STAGE2_FINAL_DIR", str(PROJECT_ROOT / "sft_output" / "qwen3_1p7b_sft_fullft_stage2_cid2ins" / "final")))
DEFAULT_INPUT_DIR = Path(env_str("NAVIGEN_SFT_INPUT_DIR", str(REPO_ROOT / "dataset")))
DEFAULT_OUTPUT_DIR = Path(env_str("NAVIGEN_RL_OUTPUT_DIR", str(PROJECT_ROOT / "rl_output" / "grpo_run")))
DEFAULT_RESUME_CHECKPOINT_DIR = env_str("NAVIGEN_RL_RESUME_CHECKPOINT_DIR", "")
DEFAULT_PID2CID2TID_PATH = Path(env_str("NAVIGEN_PID2CID2TID_PATH", str(REPO_ROOT / "dataset" / "pid2cid2tid.parquet")))
DEFAULT_JUDGE_API_KEYS: list[str] = [
]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _resolve_checkpoint_path(
    path: Path | None,
    *,
    prefer_batch_size: int | None = None,
    require_adapter: bool = False,
) -> Path | None:
    if path is None:
        return None
    if not path.exists() or not path.is_dir():
        return path

    required_file = "adapter_config.json" if require_adapter else "trainer_state.json"
    if (path / required_file).exists():
        return path

    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and child.name.startswith("checkpoint-")
        and (child / required_file).exists()
    ]
    if not candidates:
        return path

    if prefer_batch_size is not None:
        bs_tag = f"bs{int(prefer_batch_size)}"
        tagged = [child for child in candidates if bs_tag in child.name]
        if tagged:
            candidates = tagged

    candidates.sort(
        key=lambda child: (
            ".bak" not in child.name,
            _checkpoint_step(child),
            child.stat().st_mtime,
        ),
        reverse=True,
    )
    resolved = candidates[0]
    if resolved != path:
        logger.info("resolved checkpoint parent %s -> %s", path, resolved)
    return resolved

OFFICIAL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_grpo_config(**kwargs):
    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    unsupported = sorted(k for k in kwargs if k not in supported)
    if unsupported:
        logger.warning("Ignoring unsupported GRPOConfig kwargs for current TRL/Unsloth: %s", unsupported)
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in supported})


SYSTEM_PROMPTS = {
    "cid2cid": (
        "You are a personalized recommendation assistant. "
        "Predict the target cid interaction based on user history cid interaction. "
        "Think step by step, then answer. "
        "The final answer must be a JSON object only, with the field target_cid and no other fields."
    ),
    "cid2ins": (
        "You are a personalized recommendation and AIGC instruction generation assistant. "
        "Based on user history cid interaction, output user target tid interaction and the AIGC instruction. "
        "Think step by step, then answer. "
        "The final answer must be a JSON object only, with the fields target_tid and target_ins."
    ),
}


@functools.lru_cache(None)
def _warning_once_compat(self, msg, *args, **kwargs):
    if args and isinstance(args[0], type) and issubclass(args[0], Warning):
        args = args[1:]
    self.warning(msg, *args, **kwargs)


logging.Logger.warning_once = _warning_once_compat
warnings.filterwarnings(
    "ignore",
    message=r"Passing `generation_config` together with generation-related arguments.*",
)


def quote_for_display(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def sync_unsloth_mixed_precision_env(training_args) -> str:
    if getattr(training_args, "bf16", False):
        mp = "bf16"
    elif getattr(training_args, "fp16", False):
        mp = "fp16"
    else:
        mp = "no"

    os.environ["ACCELERATE_MIXED_PRECISION"] = mp
    if hasattr(training_args, "mixed_precision"):
        training_args.mixed_precision = mp
    if hasattr(training_args, "bf16_full_eval"):
        training_args.bf16_full_eval = (mp == "bf16")
    if hasattr(training_args, "fp16_full_eval"):
        training_args.fp16_full_eval = (mp == "fp16")
    return mp


def resolve_gradient_checkpointing_mode() -> str | bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size > 1:
        return False
    return "unsloth"


def normalize_eval_batch_size_for_grpo(args, world_size: int) -> None:
    if str(getattr(args, "eval_strategy", "no")).lower() == "no":
        return

    per_device_eval_batch_size = int(getattr(args, "per_device_eval_batch_size", 0) or 0)
    num_generations = int(getattr(args, "num_generations", 0) or 0)
    if world_size <= 0 or per_device_eval_batch_size <= 0 or num_generations <= 0:
        return

    global_eval_batch_size = per_device_eval_batch_size * world_size
    if global_eval_batch_size % num_generations == 0:
        return

    # TRL requires each eval batch to contain full prompt groups.
    per_device_step = num_generations // math.gcd(world_size, num_generations)
    lower_compatible = (per_device_eval_batch_size // per_device_step) * per_device_step
    upper_compatible = ((per_device_eval_batch_size + per_device_step - 1) // per_device_step) * per_device_step
    adjusted_eval_batch_size = lower_compatible if lower_compatible > 0 else upper_compatible

    logger.warning(
        "Adjusted per_device_eval_batch_size from %d to %d so global eval batch size "
        "(%d * %d = %d) is divisible by num_generations=%d.",
        per_device_eval_batch_size,
        adjusted_eval_batch_size,
        adjusted_eval_batch_size,
        world_size,
        adjusted_eval_batch_size * world_size,
        num_generations,
    )
    args.per_device_eval_batch_size = adjusted_eval_batch_size


def normalize_vllm_server_base_url(host: str, port: int, base_url: str = "") -> str:
    if base_url:
        url = base_url.strip().rstrip("/")
        if "://" not in url:
            url = "http://" + url
        return url
    return f"http://{host}:{port}"


def _flatten_token_ids(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        flattened: List[int] = []
        for item in value:
            flattened.extend(_flatten_token_ids(item))
        return flattened
    return []


def build_vllm_stop_generation_kwargs(tokenizer, model) -> Dict[str, Any]:
    eos_token_ids: List[int] = []
    eos_token_ids.extend(_flatten_token_ids(getattr(tokenizer, "eos_token_id", None)))

    model_config = getattr(model, "config", None)
    eos_token_ids.extend(_flatten_token_ids(getattr(model_config, "eos_token_id", None)))

    deduped_eos_token_ids: List[int] = []
    seen = set()
    for token_id in eos_token_ids:
        if not isinstance(token_id, int):
            continue
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped_eos_token_ids.append(token_id)

    generation_kwargs: Dict[str, Any] = {"ignore_eos": False}
    if deduped_eos_token_ids:
        generation_kwargs["stop_token_ids"] = deduped_eos_token_ids

    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str) and eos_token:
        generation_kwargs["stop"] = [eos_token]

    return generation_kwargs


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _tokenize_fixed_text(tokenizer, text: str, label: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenization for {label} is empty: {text!r}")
    return [int(token_id) for token_id in token_ids]


def _build_token_trie(token_sequences: list[list[int]]) -> dict[int, Any]:
    root: dict[int, Any] = {}
    for seq in token_sequences:
        if not seq:
            continue
        node = root
        for token_id in seq:
            node = node.setdefault(int(token_id), {})
    return root


def _follow_token_trie(trie: dict[int, Any], prefix: list[int]) -> dict[int, Any] | None:
    node = trie
    for token_id in prefix:
        node = node.get(int(token_id))
        if node is None:
            return None
    return node


@dataclass(frozen=True)
class CidGenerationConstraint:
    cid_token_trie: dict[int, Any]
    marker_token_seqs: tuple[tuple[int, ...], ...]
    max_marker_token_len: int
    eos_token_ids: tuple[int, ...]


def _build_cid_generation_constraint(tokenizer, known_cids: frozenset[str], eos_token_ids: list[int]) -> CidGenerationConstraint:
    marker_variants = _dedupe_preserve_order(
        [
            '{"target_cid":"',
            '{"target_cid": "',
            '\n{"target_cid":"',
            '\n{"target_cid": "',
        ]
    )
    marker_token_seqs = []
    for marker in marker_variants:
        marker_token_seqs.append(tuple(_tokenize_fixed_text(tokenizer, marker, f"cid marker {marker!r}")))
    marker_token_seqs = sorted(set(marker_token_seqs), key=len, reverse=True)
    if not marker_token_seqs:
        raise ValueError("Failed to build target_cid marker token sequences.")

    close_token_ids = _tokenize_fixed_text(tokenizer, '"}', 'cid JSON close token')
    cid_token_sequences: list[list[int]] = []
    for cid in sorted(known_cids):
        cid_text = _stringify(cid).strip()
        if not cid_text:
            continue
        cid_ids = tokenizer.encode(cid_text, add_special_tokens=False)
        if not cid_ids:
            continue
        cid_token_sequences.append([int(token_id) for token_id in cid_ids] + close_token_ids)
    if not cid_token_sequences:
        raise ValueError("No valid CID token sequences could be built from the known CID catalog.")

    eos_token_ids = tuple(int(token_id) for token_id in eos_token_ids if isinstance(token_id, int))
    return CidGenerationConstraint(
        cid_token_trie=_build_token_trie(cid_token_sequences),
        marker_token_seqs=tuple(marker_token_seqs),
        max_marker_token_len=max(len(seq) for seq in marker_token_seqs),
        eos_token_ids=eos_token_ids,
    )


def _extract_cid_prefill_from_stage2_prefix(stage2_prefix: str) -> str | None:
    text = _stringify(stage2_prefix)
    for marker in ('{"target_cid":"', '{"target_cid": "'):
        pos = text.rfind(marker)
        if pos >= 0:
            return text[pos + len(marker) :]
    return None


def _build_vllm_cid2cid_continuation_trie(
    tokenizer,
    known_cids: frozenset[str],
    stage2_prefix: str,
) -> dict[int, Any]:
    cid_prefill = _extract_cid_prefill_from_stage2_prefix(stage2_prefix)
    if cid_prefill is None:
        raise ValueError(f"Cannot infer target_cid prefill from stage2 prefix: {stage2_prefix!r}")

    continuation_token_sequences: list[list[int]] = []
    close_token_ids = _tokenize_fixed_text(tokenizer, '"}', 'vLLM cid2cid close token')
    for cid in sorted(known_cids):
        cid_text = _stringify(cid).strip()
        if not cid_text:
            continue
        if cid_prefill and not cid_text.startswith(cid_prefill):
            continue
        continuation_text = cid_text[len(cid_prefill) :] + '"}'
        token_ids = tokenizer.encode(continuation_text, add_special_tokens=False)
        if not token_ids:
            continue
        continuation_token_sequences.append([int(token_id) for token_id in token_ids])

    if not continuation_token_sequences:
        raise ValueError(
            f"No valid CID continuations remain after applying cid prefill {cid_prefill!r} from prefix {stage2_prefix!r}"
        )
    return _build_token_trie(continuation_token_sequences)


def _get_cached_vllm_cid2cid_continuation_trie(client, tokenizer, stage2_prefix: str) -> dict[int, Any] | None:
    known_cids = getattr(client, "_grpo_two_stage_known_cids", None)
    if not isinstance(known_cids, frozenset) or not known_cids:
        return None

    trie_cache = getattr(client, "_grpo_two_stage_cid2cid_trie_cache", None)
    if trie_cache is None:
        trie_cache = {}
        client._grpo_two_stage_cid2cid_trie_cache = trie_cache
    if stage2_prefix not in trie_cache:
        trie_cache[stage2_prefix] = _build_vllm_cid2cid_continuation_trie(
            tokenizer=tokenizer,
            known_cids=known_cids,
            stage2_prefix=stage2_prefix,
        )
    return trie_cache[stage2_prefix]


def _stepwise_vllm_trie_generate(
    client,
    original_generate_fn,
    tokenizer,
    prompts: list[str],
    images,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: int,
    base_generation_kwargs: dict[str, Any],
    token_trie: dict[int, Any],
) -> list[list[int]]:
    if not prompts:
        return []

    prompt_texts = list(prompts)
    generated_token_ids: list[list[int]] = [[] for _ in prompts]
    completed = [False] * len(prompts)
    token_decode_cache: dict[int, str] = {}

    for step_idx in range(int(max_tokens)):
        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        active_count = 0
        for idx in range(len(prompts)):
            if completed[idx]:
                continue
            node = _follow_token_trie(token_trie, generated_token_ids[idx])
            if node is None:
                completed[idx] = True
                continue
            if not node:
                completed[idx] = True
                continue
            allowed = tuple(sorted(int(token_id) for token_id in node.keys()))
            if not allowed:
                completed[idx] = True
                continue
            grouped_indices.setdefault(allowed, []).append(idx)
            active_count += 1

        if active_count == 0:
            break

        for allowed_token_ids, indices in grouped_indices.items():
            batch_prompts = [prompt_texts[idx] for idx in indices]
            if images is not None:
                batch_images = [images[idx] for idx in indices]
            else:
                batch_images = None

            step_generation_kwargs = dict(base_generation_kwargs)
            step_generation_kwargs.pop("structured_outputs", None)
            step_generation_kwargs["allowed_token_ids"] = list(allowed_token_ids)

            batch_outputs = original_generate_fn(
                client,
                prompts=batch_prompts,
                images=batch_images,
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=1,
                guided_decoding_regex=None,
                generation_kwargs=step_generation_kwargs,
            )
            for idx, output_ids in zip(indices, batch_outputs):
                if not output_ids:
                    completed[idx] = True
                    continue
                output_ids = [int(token_id) for token_id in output_ids]
                generated_token_ids[idx].extend(output_ids)
                for token_id in output_ids:
                    piece = token_decode_cache.get(token_id)
                    if piece is None:
                        piece = tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        token_decode_cache[token_id] = piece
                    prompt_texts[idx] += piece

                node = _follow_token_trie(token_trie, generated_token_ids[idx])
                if node is None or not node:
                    completed[idx] = True

        if all(completed):
            break
    else:
        unfinished = sum(1 for flag in completed if not flag)
        if unfinished:
            logger.warning(
                "stepwise vLLM CID trie decoding hit max_tokens=%d with %d unfinished samples",
                max_tokens,
                unfinished,
            )

    return generated_token_ids


class _CidConstraintRowState:
    __slots__ = ("generated_len", "mode", "marker_tail", "cid_prefix_ids")

    def __init__(self) -> None:
        self.generated_len = 0
        self.mode = "search"
        self.marker_tail: list[int] = []
        self.cid_prefix_ids: list[int] = []


class LegalCidTrieLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        constraint: CidGenerationConstraint,
        prompt_width: int,
        task_names: list[str],
    ) -> None:
        self.constraint = constraint
        self.prompt_width = int(prompt_width)
        self.task_names = tuple(_stringify(task_name) for task_name in task_names)
        self._states = [_CidConstraintRowState() for _ in self.task_names]

    def _reset_state(self, batch_idx: int) -> _CidConstraintRowState:
        state = _CidConstraintRowState()
        self._states[batch_idx] = state
        return state

    def _maybe_activate_marker(self, state: _CidConstraintRowState) -> None:
        for marker_token_seq in self.constraint.marker_token_seqs:
            marker_len = len(marker_token_seq)
            if marker_len == 0 or len(state.marker_tail) < marker_len:
                continue
            if tuple(state.marker_tail[-marker_len:]) == marker_token_seq:
                state.mode = "cid"
                state.cid_prefix_ids = []
                return

    def _update_state(self, batch_idx: int, generated_ids: list[int]) -> _CidConstraintRowState:
        state = self._states[batch_idx]
        if len(generated_ids) < state.generated_len:
            state = self._reset_state(batch_idx)

        new_tokens = generated_ids[state.generated_len :]
        if not new_tokens:
            return state

        for token_id in new_tokens:
            token_id = int(token_id)
            if state.mode == "search":
                state.marker_tail.append(token_id)
                if len(state.marker_tail) > self.constraint.max_marker_token_len:
                    state.marker_tail = state.marker_tail[-self.constraint.max_marker_token_len :]
                self._maybe_activate_marker(state)
            elif state.mode == "cid":
                state.cid_prefix_ids.append(token_id)
                node = _follow_token_trie(self.constraint.cid_token_trie, state.cid_prefix_ids)
                if node is None:
                    state.mode = "invalid"
                elif not node:
                    state.mode = "closed"
            state.generated_len += 1

        return state

    @staticmethod
    def _mask_scores_to_allowed(scores_row: torch.Tensor, allowed_token_ids: list[int]) -> None:
        if not allowed_token_ids:
            return
        allowed_scores = scores_row[allowed_token_ids].clone()
        scores_row.fill_(torch.finfo(scores_row.dtype).min)
        scores_row[allowed_token_ids] = allowed_scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        for batch_idx in range(batch_size):
            if batch_idx >= len(self.task_names) or self.task_names[batch_idx] != "cid2cid":
                continue

            generated_ids = input_ids[batch_idx, self.prompt_width :].tolist()
            state = self._update_state(batch_idx, generated_ids)

            if state.mode == "cid":
                node = _follow_token_trie(self.constraint.cid_token_trie, state.cid_prefix_ids)
                if node:
                    self._mask_scores_to_allowed(scores[batch_idx], sorted(node.keys()))
                elif self.constraint.eos_token_ids:
                    self._mask_scores_to_allowed(scores[batch_idx], list(self.constraint.eos_token_ids))
            elif state.mode in {"closed", "invalid"} and self.constraint.eos_token_ids:
                self._mask_scores_to_allowed(scores[batch_idx], list(self.constraint.eos_token_ids))

        return scores


def _infer_two_stage_prefix_from_prompt(
    prompt: str,
    default_prefix: str,
    cid2cid_prefix: str,
    cid2ins_prefix: str,
) -> str:
    text = prompt or ""
    lowered = text.lower()
    if "target_tid" in lowered and "target_ins" in lowered:
        return cid2ins_prefix
    if "target_cid" in lowered:
        return cid2cid_prefix
    return default_prefix


def _infer_two_stage_task_from_prompt(prompt: str) -> str:
    text = prompt or ""
    lowered = text.lower()
    if "target_tid" in lowered and "target_ins" in lowered:
        return "cid2ins"
    if "target_cid" in lowered:
        return "cid2cid"
    return ""


def patch_vllm_client_two_stage_generation() -> None:
    original = trl_vllm_client.VLLMClient.generate
    if getattr(original, "_grpo_rec_aigc_two_stage_patched", False):
        return

    def _patched_generate(
        self,
        prompts,
        images=None,
        n=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=16,
        guided_decoding_regex=None,
        generation_kwargs=None,
    ):
        base_generation_kwargs = dict(generation_kwargs or {})
        two_stage_enabled = bool(base_generation_kwargs.pop("grpo_two_stage_enabled", False))
        if not two_stage_enabled:
            return original(
                self,
                prompts=prompts,
                images=images,
                n=n,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                guided_decoding_regex=guided_decoding_regex,
                generation_kwargs=generation_kwargs,
            )

        tokenizer = getattr(self, "_grpo_two_stage_tokenizer", None)
        if tokenizer is None:
            logger.warning("two-stage vLLM generation requested but tokenizer is unavailable; falling back to single-stage")
            return original(
                self,
                prompts=prompts,
                images=images,
                n=n,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                guided_decoding_regex=guided_decoding_regex,
                generation_kwargs=base_generation_kwargs,
            )

        prompt_count = len(prompts)
        if prompt_count == 0:
            return []

        stage1_stop = base_generation_kwargs.pop("grpo_two_stage_stop", ["</think>"])
        if isinstance(stage1_stop, str):
            stage1_stop = [stage1_stop]
        elif not isinstance(stage1_stop, list):
            stage1_stop = ["</think>"]

        default_stage2_prefix = str(base_generation_kwargs.pop("grpo_two_stage_prefix", "\n{"))
        cid2cid_stage2_prefix = str(
            base_generation_kwargs.pop("grpo_two_stage_cid2cid_prefix", '\n{"target_cid":"')
        )
        cid2ins_stage2_prefix = str(
            base_generation_kwargs.pop("grpo_two_stage_cid2ins_prefix", '\n{"target_tid":["')
        )
        default_stage2_regex = base_generation_kwargs.pop("grpo_two_stage_regex", guided_decoding_regex)
        cid2cid_stage2_regex = base_generation_kwargs.pop(
            "grpo_two_stage_cid2cid_regex",
            r'<\|cid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><\|cid_end\|>"\s*\}',
        )
        cid2ins_stage2_regex = base_generation_kwargs.pop(
            "grpo_two_stage_cid2ins_regex",
            r'(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\]\s*,\s*"target_ins"\s*:\s*"(?:[^"\\]|\\.)*"\s*\}',
        )
        force_close_suffix = str(base_generation_kwargs.pop("grpo_two_stage_force_close_suffix", "</think>"))
        include_stop = bool(base_generation_kwargs.pop("grpo_two_stage_include_stop", True))
        configured_stage1_max = int(base_generation_kwargs.pop("grpo_two_stage_stage1_max_tokens", max_tokens))
        configured_stage2_max = int(base_generation_kwargs.pop("grpo_two_stage_stage2_max_tokens", min(256, max_tokens)))

        max_tokens = int(max_tokens)
        close_suffix_ids = tokenizer.encode(force_close_suffix, add_special_tokens=False) if force_close_suffix else []
        candidate_prefixes = [
            default_stage2_prefix,
            cid2cid_stage2_prefix,
            cid2ins_stage2_prefix,
        ]
        max_prefix_token_len = max(
            len(tokenizer.encode(prefix, add_special_tokens=False))
            for prefix in candidate_prefixes
        )
        reserved_tokens = max_prefix_token_len + len(close_suffix_ids)
        if reserved_tokens >= max_tokens:
            logger.warning(
                "two-stage vLLM reserved tokens (%d) exceed max_tokens (%d); falling back to single-stage",
                reserved_tokens,
                max_tokens,
            )
            return original(
                self,
                prompts=prompts,
                images=images,
                n=n,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                guided_decoding_regex=guided_decoding_regex,
                generation_kwargs=base_generation_kwargs,
            )

        stage2_max_tokens = min(max(1, configured_stage2_max), max_tokens - reserved_tokens - 1)
        stage1_max_tokens = min(max(1, configured_stage1_max), max_tokens - reserved_tokens - stage2_max_tokens)
        if stage1_max_tokens < 1 or stage2_max_tokens < 1:
            logger.warning(
                "two-stage vLLM token split invalid (stage1=%d, stage2=%d, max=%d); falling back to single-stage",
                stage1_max_tokens,
                stage2_max_tokens,
                max_tokens,
            )
            return original(
                self,
                prompts=prompts,
                images=images,
                n=n,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                guided_decoding_regex=guided_decoding_regex,
                generation_kwargs=base_generation_kwargs,
            )

        if not getattr(self, "_grpo_two_stage_logged", False):
            logger.info(
                "two-stage vLLM rollout enabled: stage1_max_tokens=%d stage2_max_tokens=%d stage1_stop=%s default_prefix=%r cid2cid_prefix=%r cid2ins_prefix=%r",
                stage1_max_tokens,
                stage2_max_tokens,
                stage1_stop,
                default_stage2_prefix,
                cid2cid_stage2_prefix,
                cid2ins_stage2_prefix,
            )
            self._grpo_two_stage_logged = True

        stage1_generation_kwargs = dict(base_generation_kwargs)
        stage1_generation_kwargs["stop"] = stage1_stop
        if include_stop:
            stage1_generation_kwargs["include_stop_str_in_output"] = True

        stage1_completion_ids = original(
            self,
            prompts=prompts,
            images=images,
            n=n,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=stage1_max_tokens,
            guided_decoding_regex=None,
            generation_kwargs=stage1_generation_kwargs,
        )
        stage1_texts = tokenizer.batch_decode(
            stage1_completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        repeated_images = None
        if images:
            repeated_images = []
        stage2_prompts = []
        stage1_prefix_ids = []
        stage2_regexes = []
        stage2_tasks = []
        for prompt_idx, prompt in enumerate(prompts):
            for sample_idx in range(n):
                flat_idx = prompt_idx * n + sample_idx
                stage1_ids = list(stage1_completion_ids[flat_idx])
                stage1_text = stage1_texts[flat_idx]
                task_name = _infer_two_stage_task_from_prompt(prompt)
                stage2_prefix = _infer_two_stage_prefix_from_prompt(
                    prompt,
                    default_stage2_prefix,
                    cid2cid_stage2_prefix,
                    cid2ins_stage2_prefix,
                )
                if task_name == "cid2cid":
                    stage2_regex = cid2cid_stage2_regex
                elif task_name == "cid2ins":
                    stage2_regex = cid2ins_stage2_regex
                else:
                    stage2_regex = default_stage2_regex
                prefix_ids = tokenizer.encode(stage2_prefix, add_special_tokens=False)
                has_think_close = any(stop_text in stage1_text for stop_text in stage1_stop)
                if not has_think_close and force_close_suffix:
                    stage1_text = stage1_text + force_close_suffix
                    stage1_ids = stage1_ids + close_suffix_ids
                stage2_prompts.append(prompt + stage1_text + stage2_prefix)
                stage1_prefix_ids.append(stage1_ids + prefix_ids)
                stage2_regexes.append(stage2_regex)
                stage2_tasks.append(task_name)
                if repeated_images is not None:
                    repeated_images.append(images[prompt_idx])

        stage2_completion_ids = [None] * len(stage2_prompts)
        try:
            cid2cid_trie = _get_cached_vllm_cid2cid_continuation_trie(self, tokenizer, cid2cid_stage2_prefix)
        except Exception as exc:
            logger.warning(
                "failed to build cached vLLM cid2cid continuation trie for prefix %r; falling back to regex: %s",
                cid2cid_stage2_prefix,
                exc,
            )
            cid2cid_trie = None

        cid2cid_indices = [idx for idx, task_name in enumerate(stage2_tasks) if task_name == "cid2cid"]
        if cid2cid_indices and cid2cid_trie is not None:
            cid2cid_prompts = [stage2_prompts[idx] for idx in cid2cid_indices]
            if repeated_images is not None:
                cid2cid_images = [repeated_images[idx] for idx in cid2cid_indices]
            else:
                cid2cid_images = None
            cid2cid_outputs = _stepwise_vllm_trie_generate(
                client=self,
                original_generate_fn=original,
                tokenizer=tokenizer,
                prompts=cid2cid_prompts,
                images=cid2cid_images,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=stage2_max_tokens,
                base_generation_kwargs=base_generation_kwargs,
                token_trie=cid2cid_trie,
            )
            for idx, output_ids in zip(cid2cid_indices, cid2cid_outputs):
                stage2_completion_ids[idx] = output_ids

        grouped_indices: dict[str | None, list[int]] = {}
        for idx, regex in enumerate(stage2_regexes):
            if stage2_completion_ids[idx] is not None:
                continue
            grouped_indices.setdefault(regex, []).append(idx)

        for stage2_regex, indices in grouped_indices.items():
            batch_prompts = [stage2_prompts[idx] for idx in indices]
            if repeated_images is not None:
                batch_images = [repeated_images[idx] for idx in indices]
            else:
                batch_images = None
            batch_outputs = original(
                self,
                prompts=batch_prompts,
                images=batch_images,
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=stage2_max_tokens,
                guided_decoding_regex=stage2_regex,
                generation_kwargs=base_generation_kwargs,
            )
            for idx, output_ids in zip(indices, batch_outputs):
                stage2_completion_ids[idx] = output_ids

        return [
            list(prefix_part) + list(stage2_part)
            for prefix_part, stage2_part in zip(stage1_prefix_ids, stage2_completion_ids)
        ]

    _patched_generate._grpo_rec_aigc_two_stage_patched = True
    trl_vllm_client.VLLMClient.generate = _patched_generate


class ConstrainedCidGRPOTrainer(GRPOTrainer):
    def _stage_timing_every_steps(self) -> int:
        return int(getattr(self.args, "stage_timing_every_steps", 0) or 0)

    def _stage_timing_step(self) -> int:
        return int(getattr(getattr(self, "state", None), "global_step", 0) or 0) + 1

    def _should_log_stage_timing(self) -> bool:
        every = self._stage_timing_every_steps()
        return every > 0 and self._stage_timing_step() % every == 0

    def _log_stage_timing(self, label: str, started_at: float, **extra: Any) -> None:
        if not self._should_log_stage_timing():
            return
        rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        suffix = " ".join(f"{key}={value}" for key, value in extra.items())
        if suffix:
            suffix = " " + suffix
        print(
            f"[stage timing] step={self._stage_timing_step()} rank={rank} "
            f"{label}={time.perf_counter() - started_at:.2f}s{suffix}",
            flush=True,
        )

    def _prepare_inputs(self, inputs):
        started_at = time.perf_counter()
        output = super()._prepare_inputs(inputs)
        self._log_stage_timing("prepare_inputs_total", started_at)
        return output

    def compute_loss(self, model, inputs, *args, **kwargs):
        started_at = time.perf_counter()
        output = super().compute_loss(model, inputs, *args, **kwargs)
        self._log_stage_timing("compute_loss_forward", started_at)
        return output

    def training_step(self, model, inputs, *args, **kwargs):
        started_at = time.perf_counter()
        output = super().training_step(model, inputs, *args, **kwargs)
        self._log_stage_timing("training_step_total", started_at)
        return output

    def _build_legal_cid_logits_processor(
        self,
        inputs: list[dict[str, Any]],
        prompt_width: int,
    ) -> LogitsProcessorList | None:
        constraint = getattr(self, "_legal_cid_constraint", None)
        if constraint is None:
            return None

        task_names = [_stringify(example.get("task")).strip() for example in inputs]
        if not any(task_name == "cid2cid" for task_name in task_names):
            return None

        return LogitsProcessorList(
            [
                LegalCidTrieLogitsProcessor(
                    constraint=constraint,
                    prompt_width=prompt_width,
                    task_names=task_names,
                )
            ]
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        total_started_at = time.perf_counter()
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        original_prompts = copy.deepcopy(prompts)

        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    trl_grpo_trainer.prepare_multimodal_messages(prompt, num_images=1)

        tokenize_started_at = time.perf_counter()
        prompts_text = [trl_grpo_trainer.maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        # GRPO/Unsloth override _prepare_inputs to trigger generation. Here we
        # only need the base Trainer tensor/device preparation for tokenized prompts.
        prompt_inputs = hf_trainer.Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        self._log_stage_timing("prompt_tokenize", tokenize_started_at, batch=len(inputs))

        if self.max_prompt_length is not None:
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = trl_grpo_trainer.truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                sync_started_at = time.perf_counter()
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
                self._log_stage_timing("move_model_to_vllm", sync_started_at)

            if self.vllm_mode == "server":
                vllm_started_at = time.perf_counter()
                all_prompts_text = trl_grpo_trainer.gather_object(prompts_text)
                if has_images:
                    all_images = trl_grpo_trainer.gather_object(images)

                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with trl_grpo_trainer.profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                completion_ids = trl_grpo_trainer.broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                self._log_stage_timing("vllm_server_roundtrip", vllm_started_at)

            elif self.vllm_mode == "colocate":
                vllm_started_at = time.perf_counter()
                if self.guided_decoding_regex:
                    guided_decoding = trl_grpo_trainer.GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = trl_grpo_trainer.SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with trl_grpo_trainer.profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                self._log_stage_timing("vllm_colocate_generate", vllm_started_at)

            postprocess_started_at = time.perf_counter()
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = trl_grpo_trainer.pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            self._log_stage_timing("completion_tensor_postprocess", postprocess_started_at)

        elif self.use_transformers_paged:
            generate_started_at = time.perf_counter()
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if trl_grpo_trainer.is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                trl_grpo_trainer.profiling_context(self, "transformers.generate_batch"),
                trl_grpo_trainer.unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                trl_grpo_trainer.FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = trl_grpo_trainer.pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = trl_grpo_trainer.pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            self.model_wrapped.config._attn_implementation = previous_attn
            self._log_stage_timing("transformers_paged_generate", generate_started_at)
        else:
            logits_processor = self._build_legal_cid_logits_processor(inputs, prompt_width=prompt_ids.size(1))
            generate_started_at = time.perf_counter()
            with (
                trl_grpo_trainer.profiling_context(self, "transformers.generate"),
                trl_grpo_trainer.unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                trl_grpo_trainer.FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                generate_kwargs = {
                    "generation_config": self.generation_config,
                    "disable_compile": True,
                }
                if logits_processor is not None:
                    generate_kwargs["logits_processor"] = logits_processor
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    **generate_kwargs,
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            self._log_stage_timing("transformers_generate", generate_started_at)

        mask_started_at = time.perf_counter()
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]

        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            logps_started_at = time.perf_counter()
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None
            self._log_stage_timing("old_ref_logps", logps_started_at)
        self._log_stage_timing("mask_and_logps_total", mask_started_at)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if trl_grpo_trainer.is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        reward_started_at = time.perf_counter()
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        self._log_stage_timing("calculate_rewards_total", reward_started_at)

        advantage_started_at = time.perf_counter()
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = trl_grpo_trainer.nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        if not bool(getattr(self.args, "skip_trl_text_logs", True)):
            text_log_started_at = time.perf_counter()
            self._logs["prompt"].extend(trl_grpo_trainer.gather_object(prompts_text))
            self._logs["completion"].extend(trl_grpo_trainer.gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

            if has_images:
                self._logs["image"].extend(trl_grpo_trainer.gather_object(images))
            self._log_stage_timing("trl_text_log_gather", text_log_started_at)

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        self._log_stage_timing("advantage_metrics_pack", advantage_started_at)
        self._log_stage_timing("generate_and_score_total", total_started_at)
        return output


def patch_peft_tensor_parallel_compat() -> None:
    try:
        from peft.utils import save_and_load as peft_save_and_load
    except Exception:
        return

    if getattr(peft_save_and_load, "_unsloth_tp_compat_patched", False):
        return

    original = peft_save_and_load._maybe_shard_state_dict_for_tp

    def _wrapped_maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
        try:
            return original(model, state_dict, adapter_name)
        except ImportError as exc:
            if "EmbeddingParallel" not in str(exc):
                raise
            logger.warning(
                "Skipping PEFT tensor-parallel adapter sharding because current transformers "
                "build lacks EmbeddingParallel; this is safe for this script's loading path."
            )
            return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = _wrapped_maybe_shard_state_dict_for_tp
    peft_save_and_load._unsloth_tp_compat_patched = True


def ensure_model_warnings_issued(model) -> None:
    shared = getattr(model, "warnings_issued", None)
    if not isinstance(shared, dict):
        shared = {}

    visited = set()
    current = model
    for _ in range(6):
        if current is None:
            break
        obj_id = id(current)
        if obj_id in visited:
            break
        visited.add(obj_id)
        if not isinstance(getattr(current, "warnings_issued", None), dict):
            try:
                current.warnings_issued = shared
            except Exception:
                pass
        current = getattr(current, "base_model", None) or getattr(current, "model", None)


def patch_trl_prepare_peft_model_compat() -> None:
    original = trl_model_utils.prepare_peft_model
    if getattr(original, "_grpo_rec_aigc_compat_patched", False):
        return

    def _wrapped_prepare_peft_model(model, peft_config, args):
        original_replace = dataclasses.replace
        is_existing_peft_model = hasattr(model, "peft_config") and hasattr(model, "set_adapter")

        def _compat_replace(obj, /, **changes):
            if isinstance(obj, GRPOConfig) and "gradient_checkpointing" in changes:
                if (
                    getattr(obj, "generation_batch_size", None) is not None
                    and getattr(obj, "steps_per_generation", None) is not None
                    and "steps_per_generation" not in changes
                    and "generation_batch_size" not in changes
                ):
                    changes = dict(changes)
                    changes["steps_per_generation"] = None
            return original_replace(obj, **changes)

        dataclasses.replace = _compat_replace
        try:
            prepared_model = original(model, peft_config, args)
        finally:
            dataclasses.replace = original_replace

        # TRL 0.22.2 + PEFT 0.19.1 may silently freeze LoRA params when the
        # input is already a PeftModel and `peft_config=None`. Restore the
        # active adapter to train mode so the optimizer can see LoRA weights.
        if is_existing_peft_model and peft_config is None and hasattr(prepared_model, "set_adapter"):
            active_adapter = getattr(prepared_model, "active_adapter", None)
            try:
                if active_adapter:
                    prepared_model.set_adapter(active_adapter, inference_mode=False)
            except Exception as exc:
                logger.warning("failed to restore trainable PEFT adapter state: %s", exc)

        return prepared_model

    _wrapped_prepare_peft_model._grpo_rec_aigc_compat_patched = True
    trl_model_utils.prepare_peft_model = _wrapped_prepare_peft_model
    trl_grpo_trainer.prepare_peft_model = _wrapped_prepare_peft_model


def _active_lora_adapter_name(module) -> str | None:
    active_adapters = getattr(module, "active_adapters", None)
    if isinstance(active_adapters, list) and active_adapters:
        return active_adapters[0]

    active_adapter = getattr(module, "active_adapter", None)
    if isinstance(active_adapter, list) and active_adapter:
        return active_adapter[0]
    if isinstance(active_adapter, str) and active_adapter:
        return active_adapter
    return None


def _validate_finite_tensor(tensor: torch.Tensor, tensor_name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite tensor detected during vLLM sync: {tensor_name}")


def patch_grpo_vllm_peft_sync_compat() -> None:
    original = trl_grpo_trainer.GRPOTrainer._move_model_to_vllm
    if getattr(original, "_grpo_rec_aigc_vllm_sync_patched", False):
        return

    def _patched_move_model_to_vllm(self):
        if not getattr(self, "use_vllm", False):
            return original(self)

        if not hasattr(self, "model") or not hasattr(self.model, "named_modules"):
            return original(self)

        if self.is_fsdp_enabled:
            return original(self)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            return original(self)

        # TRL's PEFT path merges/unmerges the adapter in-place before syncing
        # weights to vLLM. With the current TRL/PEFT/Unsloth stack this corrupts
        # later generations. For single-process LoRA training, directly compose
        # base weight + LoRA delta and push the merged tensor to vLLM instead.
        if hasattr(self.model, "peft_config") and hasattr(self.model, "named_modules"):
            synced = 0
            for module_name, module in self.model.named_modules():
                if not hasattr(module, "base_layer") or not hasattr(module, "get_delta_weight"):
                    continue
                if not hasattr(module, "lora_A") or not getattr(module, "lora_A", None):
                    continue

                adapter_name = _active_lora_adapter_name(module)
                if not adapter_name:
                    continue
                if adapter_name not in module.lora_A or adapter_name not in getattr(module, "lora_B", {}):
                    continue

                base_weight = module.base_layer.weight.data
                delta_weight = module.get_delta_weight(adapter_name).to(
                    device=base_weight.device,
                    dtype=base_weight.dtype,
                )
                merged_weight = base_weight + delta_weight

                _validate_finite_tensor(base_weight, f"{module_name}.base_weight")
                _validate_finite_tensor(delta_weight, f"{module_name}.delta_weight")
                _validate_finite_tensor(merged_weight, f"{module_name}.merged_weight")

                server_name = module_name.removeprefix("base_model.model.") + ".weight"
                server_name = server_name.replace(".base_layer", "")

                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(server_name, merged_weight)
                elif self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(server_name, merged_weight)])
                synced += 1

            if synced == 0:
                logger.warning("vLLM LoRA sync patch found no PEFT modules to sync; falling back to TRL path")
                return original(self)

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.reset_prefix_cache()
            elif self.vllm_mode == "colocate":
                self.llm.reset_prefix_cache()

            logger.info("vLLM sync patch pushed %d merged LoRA modules", synced)
            return

        return original(self)

    _patched_move_model_to_vllm._grpo_rec_aigc_vllm_sync_patched = True
    trl_grpo_trainer.GRPOTrainer._move_model_to_vllm = _patched_move_model_to_vllm


def patch_trainer_optimizer_nonfinite_guard() -> None:
    original = trl_grpo_trainer.GRPOTrainer.create_optimizer
    if getattr(original, "_grpo_rec_aigc_nonfinite_guard_patched", False):
        return

    def _patched_create_optimizer(self, *args, **kwargs):
        optimizer = original(self, *args, **kwargs)
        if optimizer is None or getattr(optimizer, "_grpo_rec_aigc_nonfinite_guard_wrapped", False):
            return optimizer

        original_step = optimizer.step

        def _guarded_step(_optimizer, *step_args, **step_kwargs):
            sanitized = 0
            for group in _optimizer.param_groups:
                for param in group.get("params", []):
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    finite_mask = torch.isfinite(grad)
                    if bool(finite_mask.all()):
                        continue
                    grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                    sanitized += 1

            if sanitized:
                logger.warning("Sanitized non-finite gradients in %d parameter tensors before optimizer.step()", sanitized)
            return original_step(*step_args, **step_kwargs)

        optimizer.step = types.MethodType(_guarded_step, optimizer)
        optimizer._grpo_rec_aigc_nonfinite_guard_wrapped = True
        return optimizer

    _patched_create_optimizer._grpo_rec_aigc_nonfinite_guard_patched = True
    trl_grpo_trainer.GRPOTrainer.create_optimizer = _patched_create_optimizer


def _sanitize_nonfinite_model_grads(model) -> int:
    sanitized = 0
    if model is None or not hasattr(model, "parameters"):
        return sanitized

    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        finite_mask = torch.isfinite(grad)
        if bool(finite_mask.all()):
            continue
        grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        sanitized += 1
    return sanitized


def patch_trainer_grad_norm_nonfinite_guard() -> None:
    original_clip = getattr(hf_trainer.Trainer, "_clip_grad_norm", None)
    original_get = getattr(hf_trainer.Trainer, "_get_grad_norm", None)
    if original_clip is None or original_get is None:
        logger.warning(
            "Skipping Trainer grad-norm non-finite guard because current transformers.Trainer "
            "does not expose _clip_grad_norm/_get_grad_norm."
        )
        return
    if getattr(original_clip, "_grpo_rec_aigc_nonfinite_guard_patched", False):
        return

    def _patched_clip_grad_norm(self, model):
        sanitized = _sanitize_nonfinite_model_grads(model)
        if sanitized:
            logger.warning(
                "Sanitized non-finite gradients in %d parameter tensors before grad norm clipping",
                sanitized,
            )
        grad_norm = original_clip(self, model)
        if hasattr(grad_norm, "item"):
            grad_norm_value = grad_norm.item()
        else:
            grad_norm_value = grad_norm
        if isinstance(grad_norm_value, float) and not math.isfinite(grad_norm_value):
            logger.warning("Non-finite grad_norm detected after clipping; forcing logged grad_norm to 0.0")
            return 0.0
        return grad_norm

    def _patched_get_grad_norm(self, model, grad_norm=None):
        sanitized = _sanitize_nonfinite_model_grads(model)
        if sanitized:
            logger.warning(
                "Sanitized non-finite gradients in %d parameter tensors before grad norm measurement",
                sanitized,
            )
        grad_norm = original_get(self, model, grad_norm=grad_norm)
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        if isinstance(grad_norm, float) and not math.isfinite(grad_norm):
            logger.warning("Non-finite grad_norm detected during logging; forcing logged grad_norm to 0.0")
            return 0.0
        return grad_norm

    _patched_clip_grad_norm._grpo_rec_aigc_nonfinite_guard_patched = True
    hf_trainer.Trainer._clip_grad_norm = _patched_clip_grad_norm
    hf_trainer.Trainer._get_grad_norm = _patched_get_grad_norm


def patch_unsloth_trainer_ddp_compat() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return
    if getattr(hf_trainer.Trainer, "_grpo_rec_aigc_unsloth_ddp_compat_patched", False):
        return

    trainer_py = Path(hf_trainer.__file__).resolve()
    module_name = "transformers._grpo_rec_aigc_vanilla_trainer"
    spec = importlib.util.spec_from_file_location(module_name, str(trainer_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load vanilla trainer module from {trainer_py}")
    vanilla_module = importlib.util.module_from_spec(spec)
    vanilla_module.__package__ = "transformers"
    sys.modules[module_name] = vanilla_module
    spec.loader.exec_module(vanilla_module)
    vanilla_trainer_cls = vanilla_module.Trainer

    hf_trainer.Trainer.training_step = vanilla_trainer_cls.training_step
    hf_trainer.Trainer._inner_training_loop = vanilla_trainer_cls._inner_training_loop
    hf_trainer.Trainer.compute_loss = vanilla_trainer_cls.compute_loss

    trl_grpo_trainer.Trainer.training_step = vanilla_trainer_cls.training_step
    trl_grpo_trainer.Trainer._inner_training_loop = vanilla_trainer_cls._inner_training_loop
    trl_grpo_trainer.Trainer.compute_loss = vanilla_trainer_cls.compute_loss

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    if not hasattr(ddp_cls, "config"):
        ddp_cls.config = property(lambda self: self.module.config)
    if not hasattr(ddp_cls, "generation_config"):
        ddp_cls.generation_config = property(lambda self: getattr(self.module, "generation_config", None))

    hf_trainer.Trainer._grpo_rec_aigc_unsloth_ddp_compat_patched = True
    logger.warning(
        "Detected distributed training (WORLD_SIZE=%d); restored vanilla Transformers Trainer methods to avoid "
        "Unsloth DDP autograd hook failures.",
        world_size,
    )


def maybe_relaunch_with_torchrun(args: "RLRecAIGCArgs") -> bool:
    world_size = args.nproc_per_node * args.nnodes
    if world_size <= 1 or "LOCAL_RANK" in os.environ:
        return False

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    logger.info("torchrun launch command: %s", quote_for_display(cmd))
    if args.dry_run:
        return True
    subprocess.run(cmd, check=True)
    return True


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if hasattr(v, "tolist"):
        lv = v.tolist()
        return lv if isinstance(lv, list) else [lv]
    return [v]


def _stringify(v: Any) -> str:
    return "" if v is None else str(v)


def _maybe_parse_list_string(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    text = v.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _normalize_cid_value(v: Any) -> list[str]:
    v = _maybe_parse_list_string(v)
    seq = _as_list(v)
    if len(seq) == 1 and isinstance(seq[0], list):
        seq = _as_list(seq[0])
    return [_stringify(x).strip() for x in seq if _stringify(x).strip()]


def _normalize_tid_value(v: Any) -> list[str]:
    v = _maybe_parse_list_string(v)
    seq = _as_list(v)
    if len(seq) == 1 and isinstance(seq[0], list):
        seq = _as_list(seq[0])
    return [_stringify(x).strip() for x in seq if _stringify(x).strip()]


def _canonical_json_array_str(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _format_hist_cid(hist_cid: Any) -> str:
    return json.dumps(_as_list(hist_cid), ensure_ascii=False)


def _append_assistant_prefill(prompt: str, prefill: str) -> str:
    if not prefill:
        return prompt
    return prompt + prefill


def _build_reward_input_completion(completion: str, assistant_prefill: str) -> str:
    completion = completion or ""
    if not assistant_prefill:
        return completion
    return assistant_prefill + completion


def _strip_think_and_fence(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_fence_only(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _truncate_for_log(value: Any, limit: int = 4000) -> str:
    text = _stringify(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _to_jsonable_metric_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_metric_value(v) for v in value]
    return _stringify(value)


class TrainMetricsJSONLCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.log_path = Path(output_dir) / "train_metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not logs:
            return

        record = {
            "timestamp": round(time.time(), 3),
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": _to_jsonable_metric_value(getattr(state, "epoch", None)),
        }
        for key, value in logs.items():
            record[str(key)] = _to_jsonable_metric_value(value)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EmergencySaveManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trainer = None
        self._tokenizer = None
        self._output_dir: Path | None = None
        self._pattern: re.Pattern[str] | None = None
        self._threshold = 0
        self._window_seconds = 0.0
        self._cooldown_seconds = 0.0
        self._failure_timestamps: list[float] = []
        self._last_save_time = 0.0
        self._last_saved_step = -1

    def configure(
        self,
        output_dir: str,
        pattern_text: str,
        threshold: int,
        window_minutes: float,
        cooldown_minutes: float,
    ) -> None:
        with self._lock:
            self._output_dir = Path(output_dir)
            self._threshold = max(0, int(threshold))
            self._window_seconds = max(0.0, float(window_minutes) * 60.0)
            self._cooldown_seconds = max(0.0, float(cooldown_minutes) * 60.0)
            self._failure_timestamps = []
            self._last_save_time = 0.0
            self._last_saved_step = -1
            self._pattern = (
                re.compile(pattern_text, re.IGNORECASE)
                if pattern_text and self._threshold > 0
                else None
            )

    def bind_runtime(self, trainer: Any, tokenizer: Any) -> None:
        with self._lock:
            self._trainer = trainer
            self._tokenizer = tokenizer

    def record_judge_failure(self, error_text: str, step: int) -> None:
        now = time.time()
        error_text = _stringify(error_text)

        with self._lock:
            if self._pattern is None or self._output_dir is None or self._threshold <= 0:
                return
            if not self._pattern.search(error_text):
                return

            self._failure_timestamps.append(now)
            if self._window_seconds > 0:
                cutoff = now - self._window_seconds
                self._failure_timestamps = [ts for ts in self._failure_timestamps if ts >= cutoff]

            failure_count = len(self._failure_timestamps)
            if failure_count < self._threshold:
                return
            if self._cooldown_seconds > 0 and (now - self._last_save_time) < self._cooldown_seconds:
                return
            if int(step) == self._last_saved_step:
                return

            trainer = self._trainer
            tokenizer = self._tokenizer
            output_dir = self._output_dir
            self._last_save_time = now
            self._last_saved_step = int(step)

        if trainer is None or output_dir is None:
            logger.warning(
                "Emergency save trigger matched but trainer/output_dir is unavailable; step=%s error=%s",
                step,
                _truncate_for_log(error_text, 400),
            )
            return

        try:
            if hasattr(trainer, "is_world_process_zero") and not trainer.is_world_process_zero():
                return

            save_dir = output_dir / (
                f"emergency_save_step_{int(step)}_"
                f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Emergency save triggered at step=%s after %s matched judge failures. "
                "Saving model to %s",
                step,
                failure_count,
                save_dir,
            )
            trainer.save_model(str(save_dir))
            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(save_dir))
            meta = {
                "timestamp": round(now, 3),
                "step": int(step),
                "matched_failure_count": int(failure_count),
                "error_preview": _truncate_for_log(error_text, 1000),
            }
            (save_dir / "emergency_save_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            logger.exception("Emergency save failed at step=%s: %s", step, exc)


_emergency_save_manager = EmergencySaveManager()


def _compact_whitespace_for_log(value: Any) -> str:
    text = _stringify(value).strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def _preview_text_for_log(value: Any, limit: int = 600) -> str:
    return _truncate_for_log(_compact_whitespace_for_log(value), limit=limit)


def _compact_judge_debug_for_log(judge_debug: dict[str, Any] | None) -> dict[str, Any]:
    judge_debug = judge_debug or {}
    compact: dict[str, Any] = {}
    for key in (
        "judge_source",
        "skip_reason",
        "judge_attempts",
        "judge_key_suffix",
    ):
        value = judge_debug.get(key)
        if value not in (None, "", []):
            compact[key] = value

    last_error = _stringify(judge_debug.get("judge_last_error")).strip()
    if last_error:
        compact["judge_last_error_preview"] = _truncate_for_log(last_error, limit=500)

    request_text = _stringify(judge_debug.get("judge_request_text")).strip()
    if request_text:
        compact["judge_request_preview"] = _preview_text_for_log(request_text, limit=500)

    response_text = _stringify(judge_debug.get("judge_raw_response_text")).strip()
    if response_text:
        compact["judge_response_preview"] = _preview_text_for_log(response_text, limit=500)

    return compact


def _parse_summary_for_log(parse_debug: dict[str, Any] | None) -> dict[str, Any]:
    parse_debug = parse_debug or {}
    return {
        "parsed_fields": parse_debug.get("parsed_fields", []),
        "recovered_fields": parse_debug.get("recovered_fields", []),
        "think_format_reward": parse_debug.get("think_format_reward", 0.0),
        "think_open_count": parse_debug.get("think_open_count", 0),
        "think_close_count": parse_debug.get("think_close_count", 0),
        "think_block_count": parse_debug.get("think_block_count", 0),
        "json_object_count": parse_debug.get("json_object_count", 0),
        "leading_non_json_text_len": parse_debug.get("leading_non_json_text_len", 0),
        "trailing_text_len": parse_debug.get("trailing_text_len", 0),
        "target_key_repeat_count": parse_debug.get("target_key_repeat_count", 0),
        "reasoning_format_penalty": parse_debug.get("reasoning_format_penalty", 0.0),
        "reasoning_format_violations": parse_debug.get("reasoning_format_violations", []),
        "completion_char_len": parse_debug.get("completion_char_len", 0),
        "raw_completion_char_len": parse_debug.get("raw_completion_char_len", 0),
        "assistant_prefill_applied": parse_debug.get("assistant_prefill_applied", False),
    }


def _build_completion_log_entry(
    *,
    step: int,
    split_name: str,
    uid_value: str,
    task_name: str,
    reward_value: float,
    prompt_text: str,
    hist_cid_items: list[Any],
    gold_target_cid_items: list[Any],
    gold_target_tid_items: list[Any],
    gold_target_ins_text: str,
    parsed_completion: dict[str, Any] | None,
    optimized_parsed_completion: dict[str, Any] | None,
    raw_parse_debug: dict[str, Any],
    optimized_parse_debug: dict[str, Any],
    assistant_prefill: str,
    reward_input_completion: str,
    optimized_completion: str,
    raw_completion: str,
    components: dict[str, Any],
) -> dict[str, Any]:
    log_components = dict(components or {})
    judge_debug = log_components.pop("judge_debug", None)

    return {
        "step": step,
        "split": split_name,
        "uid": uid_value,
        "task": task_name,
        "reward": round(float(reward_value), 4),
        "sample": {
            "hist_len": len(hist_cid_items),
            "hist_cid_tail": hist_cid_items[-3:],
            "gold_target_cid": gold_target_cid_items,
            "gold_target_tid": gold_target_tid_items,
            "gold_target_ins_preview": _preview_text_for_log(gold_target_ins_text, limit=400),
            "prompt_preview": _preview_text_for_log(prompt_text, limit=500),
        },
        "prediction": {
            "parsed_completion": parsed_completion or {},
            "optimized_parsed_completion": optimized_parsed_completion or {},
        },
        "reward_components": log_components,
        "judge": _compact_judge_debug_for_log(judge_debug),
        "parse": {
            "raw": _parse_summary_for_log(raw_parse_debug),
            "optimized": _parse_summary_for_log(optimized_parse_debug),
        },
        "text": {
            "assistant_prefill": assistant_prefill,
            "raw_completion": _stringify(raw_completion),
            "reward_input_completion": _stringify(reward_input_completion),
            "optimized_completion": _stringify(optimized_completion),
        },
    }


def _extract_first_json_value(text: str) -> Any:
    cleaned = _strip_think_and_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        start = match.start()
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _json_value_non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return value is not None


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


def _recover_json_target_fields(text: str) -> dict[str, Any]:
    cleaned = _strip_think_and_fence(text)
    recovered: dict[str, Any] = {}

    target_cid = _extract_json_field_value(cleaned, "target_cid")
    if _json_value_non_empty(target_cid):
        recovered["target_cid"] = target_cid

    target_tid = _extract_json_field_value(cleaned, "target_tid")
    if _normalize_tid_value(target_tid):
        recovered["target_tid"] = target_tid

    target_ins = _extract_json_field_value(cleaned, "target_ins")
    target_ins_text = _stringify(target_ins).strip()
    if target_ins_text:
        recovered["target_ins"] = target_ins_text

    return recovered


def _merge_recovered_json_fields(
    obj: dict[str, Any] | None,
    recovered: dict[str, Any],
) -> dict[str, Any] | None:
    if obj is None:
        return recovered or None
    if not recovered:
        return obj

    merged = dict(obj)
    for field, recovered_value in recovered.items():
        if not _json_value_non_empty(merged.get(field)) and _json_value_non_empty(recovered_value):
            merged[field] = recovered_value
    return merged


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_think_and_fence(text)
    recovered = _recover_json_target_fields(cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return _merge_recovered_json_fields(obj, recovered)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return _merge_recovered_json_fields(obj, recovered)
        except json.JSONDecodeError:
            continue
    return recovered or None


def _extract_think_and_answer_segments(text: str) -> tuple[str, str]:
    raw = _strip_fence_only(text)
    think_open = raw.find("<think>")
    think_close = raw.find("</think>")

    if think_open != -1 and think_close != -1 and think_close > think_open:
        thought = raw[think_open + len("<think>") : think_close].strip()
        answer = raw[think_close + len("</think>") :].strip()
        return thought, answer

    if think_close != -1:
        thought = raw[:think_close].replace("<think>", "").strip()
        answer = raw[think_close + len("</think>") :].strip()
        return thought, answer

    obj_spans = _extract_json_spans(raw)
    if obj_spans:
        start, _, _ = obj_spans[0]
        return raw[:start].strip(), raw[start:].strip()

    return raw.strip(), ""


def _canonicalize_task_json_fields(task_name: str, obj: dict[str, Any] | None) -> dict[str, Any]:
    obj = obj or {}
    if task_name == "cid2cid":
        target_cid = _normalize_cid_value(obj.get("target_cid"))
        if not target_cid:
            return {}
        return {"target_cid": target_cid[0] if len(target_cid) == 1 else target_cid}

    if task_name == "cid2ins":
        canonical: dict[str, Any] = {}
        target_tid = _normalize_tid_value(obj.get("target_tid"))
        target_ins = _stringify(obj.get("target_ins")).strip()
        if target_tid:
            canonical["target_tid"] = target_tid
        if target_ins:
            canonical["target_ins"] = target_ins
        return canonical

    return {}


def _postprocess_completion_for_scoring(text: str, task_name: str) -> str:
    raw = _strip_fence_only(text)
    parsed_obj = _extract_json_object(raw)
    canonical_obj = _canonicalize_task_json_fields(task_name, parsed_obj)
    thought, answer = _extract_think_and_answer_segments(raw)

    pieces: list[str] = []
    if thought:
        pieces.append(f"<think>\n{thought}\n</think>")

    if canonical_obj:
        pieces.append(json.dumps(canonical_obj, ensure_ascii=False, separators=(",", ":")))
    elif answer:
        pieces.append(answer)
    elif raw and not thought:
        pieces.append(raw)

    return "\n".join(piece for piece in pieces if piece).strip() or raw


def _build_parse_debug(text: str, parsed_obj: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _strip_think_and_fence(text)
    recovered = _recover_json_target_fields(cleaned)
    reasoning_format = _analyze_reasoning_output_format(text, ["target_cid", "target_tid", "target_ins"])
    think_format_reward = _binary_think_format_reward(text)
    return {
        "completion_char_len": len(_stringify(text)),
        "parsed_fields": sorted(parsed_obj.keys()) if isinstance(parsed_obj, dict) else [],
        "recovered_fields": sorted(recovered.keys()),
        "raw_has_target_cid_key": '"target_cid"' in cleaned,
        "raw_has_target_tid_key": '"target_tid"' in cleaned,
        "raw_has_target_ins_key": '"target_ins"' in cleaned,
        "think_format_reward": think_format_reward,
        "think_open_count": reasoning_format["think_open_count"],
        "think_close_count": reasoning_format["think_close_count"],
        "think_block_count": reasoning_format["think_block_count"],
        "json_object_count": reasoning_format["json_object_count"],
        "leading_non_json_text_len": reasoning_format["leading_non_json_text_len"],
        "trailing_text_len": reasoning_format["trailing_text_len"],
        "target_key_repeat_count": reasoning_format["target_key_repeat_count"],
        "reasoning_format_penalty": reasoning_format["penalty"],
        "reasoning_format_violations": reasoning_format["violations"],
    }


class JudgeChunkError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request_text: str = "",
        response_text: str = "",
        key_suffix: str = "",
        attempts: int = 0,
    ):
        super().__init__(message)
        self.request_text = request_text
        self.response_text = response_text
        self.key_suffix = key_suffix
        self.attempts = attempts


_STRICT_CID_PATTERN = re.compile(
    r"^<\|cid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><\|cid_end\|>$"
)


def _is_valid_cid_string(value: Any) -> bool:
    text = _stringify(value).strip()
    if not text:
        return False
    return bool(_STRICT_CID_PATTERN.fullmatch(text))


def _json_format_reward(
    task_name: str,
    obj: dict[str, Any] | None,
    reasoning_format: dict[str, Any],
) -> float:
    if obj is None:
        return 0.0
    if reasoning_format.get("violations"):
        return 0.0
    if reasoning_format.get("json_object_count") != 1:
        return 0.0

    if task_name == "cid2cid":
        target_cid = _normalize_cid_value(obj.get("target_cid"))
        if len(target_cid) != 1:
            return 0.0
        return 1.0 if _is_valid_cid_string(target_cid[0]) else 0.0

    if task_name == "cid2ins":
        target_tid = _normalize_tid_value(obj.get("target_tid"))
        target_ins = _stringify(obj.get("target_ins")).strip()
        return 1.0 if target_tid and target_ins else 0.0

    return 0.0


def _binary_think_format_reward(text: str, min_content_len: int = 10) -> float:
    raw = _strip_fence_only(text)
    if "<think>" not in raw or "</think>" not in raw:
        return 0.0

    start_idx = raw.find("<think>")
    end_idx = raw.find("</think>")
    if end_idx < start_idx:
        return 0.0

    content = raw[start_idx + len("<think>") : end_idx]
    content_stripped = re.sub(r"[\s\r\n\t]+", "", content)
    return 1.0 if len(content_stripped) > min_content_len else 0.0


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


def _analyze_reasoning_output_format(text: str, target_fields: list[str]) -> dict[str, Any]:
    raw = _strip_fence_only(text)
    think_open_count = len(re.findall(r"<think>", raw, flags=re.IGNORECASE))
    think_close_count = len(re.findall(r"</think>", raw, flags=re.IGNORECASE))
    think_blocks = list(re.finditer(r"<think>[\s\S]*?</think>", raw, flags=re.IGNORECASE))
    think_block_count = len(think_blocks)

    working = raw.lstrip()
    leading_think_match = re.match(r"<think>[\s\S]*?</think>\s*", working, flags=re.IGNORECASE)
    if leading_think_match:
        working = working[leading_think_match.end() :]

    json_spans = _extract_json_spans(working)
    json_object_count = sum(1 for _, _, value in json_spans if isinstance(value, dict))

    first_json_start = None
    first_json_end = None
    for start, end, value in json_spans:
        if isinstance(value, dict):
            first_json_start = start
            first_json_end = end
            break

    if first_json_start is None:
        leading_non_json_text = working.strip()
        trailing_text = ""
    else:
        leading_non_json_text = working[:first_json_start].strip()
        trailing_text = working[first_json_end:].strip()

    target_key_repeat_count = 0
    for field in target_fields:
        matches = len(re.findall(rf'"{re.escape(field)}"\s*:', raw))
        if matches > 1:
            target_key_repeat_count += matches - 1

    violations: list[str] = []
    penalty = 0.0

    if think_open_count != think_close_count:
        violations.append("unbalanced_think_tags")
        penalty += 0.35
    if think_block_count > 1:
        violations.append("multiple_think_blocks")
        penalty += 0.2
    if think_block_count > 0 and not raw.lstrip().lower().startswith("<think>"):
        violations.append("think_not_at_start")
        penalty += 0.1
    if leading_non_json_text:
        violations.append("text_before_json")
        penalty += 0.2
    if trailing_text:
        violations.append("text_after_json")
        penalty += 0.45
    if json_object_count > 1:
        violations.append("multiple_json_objects")
        penalty += 0.2
    if target_key_repeat_count > 0:
        violations.append("repeated_target_keys")
        penalty += min(0.3, 0.1 * target_key_repeat_count)

    return {
        "penalty": round(min(1.0, penalty), 4),
        "violations": violations,
        "think_open_count": think_open_count,
        "think_close_count": think_close_count,
        "think_block_count": think_block_count,
        "json_object_count": json_object_count,
        "leading_non_json_text_len": len(leading_non_json_text),
        "trailing_text_len": len(trailing_text),
        "target_key_repeat_count": target_key_repeat_count,
    }


def _bounded_float(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    denom = sum(max(w, 0.0) for w in weights)
    if denom <= 0:
        return 0.0
    return sum(v * max(w, 0.0) for v, w in zip(values, weights)) / denom


def _empty_cid2ins_judge(reason: str) -> dict[str, Any]:
    return {
        "scores": {
            "specificity": 0.0,
            "creativity": 0.0,
            "content_quality": 0.0,
            "visual_generatability": 0.0,
            "ins_vs_gt_tid": 0.0,
            "ins_vs_pred_tid": 0.0,
            "pred_tid_vs_gt_tid": 0.0,
        },
        "bands": {
            "ins_vs_gt_tid": "off",
            "ins_vs_pred_tid": "off",
            "pred_tid_vs_gt_tid": "off",
        },
        "judge_debug": {
            "judge_source": "skipped",
            "skip_reason": reason,
        },
    }


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def _parse_api_keys(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\s,]+", value.strip())
        return [part for part in parts if part]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_parse_api_keys(item))
        return out
    return []


_CID_COMPONENT_PATTERNS = {
    "s_a": re.compile(r"<s_a_(\d+)>"),
    "s_b": re.compile(r"<s_b_(\d+)>"),
    "s_c": re.compile(r"<s_c_(\d+)>"),
}


def _parse_cid_components(cid: str) -> dict[str, str]:
    text = _stringify(cid).strip()
    out: dict[str, str] = {}
    for name, pattern in _CID_COMPONENT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            out[name] = match.group(1)
    return out


def _cid_layer_reward(pred_cid: str, gold_cid: str, layer_weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    pred = _parse_cid_components(pred_cid)
    gold = _parse_cid_components(gold_cid)
    if not pred or not gold:
        return 0.0, {"s_a": 0.0, "s_b": 0.0, "s_c": 0.0}

    component_scores: dict[str, float] = {}
    total = 0.0
    for layer in ("s_a", "s_b", "s_c"):
        score = float(layer_weights.get(layer, 0.0)) if pred.get(layer) == gold.get(layer) else 0.0
        component_scores[layer] = score
        total += score

    return min(total, 1.0), component_scores


def _string_overlap_score(a_items: list[str], b_items: list[str]) -> float:
    a_norm = {_stringify(x).strip().lower() for x in a_items if _stringify(x).strip()}
    b_norm = {_stringify(x).strip().lower() for x in b_items if _stringify(x).strip()}
    if not a_norm or not b_norm:
        return 0.0
    common = len(a_norm & b_norm)
    precision = common / len(a_norm)
    recall = common / len(b_norm)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tid_coverage_in_instruction(target_tid: list[str], target_ins: str) -> float:
    text = _stringify(target_ins).strip().lower()
    toks = [_stringify(x).strip().lower() for x in target_tid if _stringify(x).strip()]
    if not text or not toks:
        return 0.0
    hits = 0
    for tok in toks:
        if tok and tok in text:
            hits += 1
    return hits / len(toks)


@functools.lru_cache(maxsize=4)
def _load_known_cid_set(parquet_path: str) -> frozenset[str]:
    rows = _read_parquet_rows(Path(parquet_path))
    known_cids: set[str] = set()
    for row in rows:
        cid = _stringify(row.get("sid")).strip()
        if cid:
            known_cids.add(cid)
    return frozenset(known_cids)


def _cid_in_range_reward(
    pred_cid: str,
    gold_cid: str,
    layer_weights: dict[str, float],
    known_cids: frozenset[str],
    zero_match_in_range_ratio: float,
) -> tuple[float, dict[str, float]]:
    base_reward, layer_scores = _cid_layer_reward(pred_cid, gold_cid, layer_weights)
    if base_reward > 0.0:
        return base_reward, layer_scores

    if pred_cid not in known_cids:
        return 0.0, layer_scores

    bonus = max(0.0, float(layer_weights.get("s_a", 0.0)) * zero_match_in_range_ratio)
    layer_scores["in_range_zero_hit_bonus"] = bonus
    return min(bonus, 1.0), layer_scores


_judge_cache: Dict[str, dict[str, Any]] = {}
_judge_cache_lock = threading.Lock()
_dashscope_import_warned = False

_CID2INS_JUDGE_SYSTEM = """\
You are an evaluator for recommendation quality and AIGC prompt quality. Score each sample only from the provided content.

Evaluate four aspects:
1. Instruction quality with 4 dimensions in the range 0~1:
   - specificity: concrete, informative, non-vague, avoids hollow psychological narration
   - creativity: not templated, has novelty or expressive value
   - content_quality: coherent, clear, complete, and well-written
   - visual_generatability: directly usable for image or ad generation
2. ins_vs_gt_tid: how well target_ins aligns with the ground-truth gt_target_tid
3. ins_vs_pred_tid: how well target_ins aligns with the predicted target_tid
4. pred_tid_vs_gt_tid: how close the predicted target_tid is to the ground-truth gt_target_tid

Suggested bands:
- exact: 1.0
- high: 0.85
- medium: 0.6
- low: 0.3
- off: 0.0

Output rules:
- Return JSON only. No markdown, no code fence, no prose.
- If the user provides N samples, you must return a JSON array with exactly N objects.
- Never return a single JSON object when N > 1.
- Preserve the sample order and use indices 1..N exactly once.

Each element must follow:
{
  "index": 1,
  "scores": {
    "specificity": 0.0,
    "creativity": 0.0,
    "content_quality": 0.0,
    "visual_generatability": 0.0,
    "ins_vs_gt_tid": 0.0,
    "ins_vs_pred_tid": 0.0,
    "pred_tid_vs_gt_tid": 0.0
  },
  "bands": {
    "ins_vs_gt_tid": "exact|high|medium|low|off",
    "ins_vs_pred_tid": "exact|high|medium|low|off",
    "pred_tid_vs_gt_tid": "exact|high|medium|low|off"
  }
}
"""


def _import_dashscope_or_warn():
    global _dashscope_import_warned
    try:
        import dashscope

        return dashscope
    except ModuleNotFoundError:
        if not _dashscope_import_warned:
            logger.warning(
                "dashscope is not installed. cid2ins LLM judge will fall back to heuristic scoring."
            )
            _dashscope_import_warned = True
        return None


def _heuristic_cid2ins_judge(item: dict[str, Any]) -> dict[str, Any]:
    pred_tid = _normalize_tid_value(item.get("pred_target_tid"))
    gold_tid = _normalize_tid_value(item.get("gold_target_tid"))
    pred_ins = _stringify(item.get("pred_target_ins")).strip()

    ins_vs_gold = _tid_coverage_in_instruction(gold_tid, pred_ins)
    ins_vs_pred = _tid_coverage_in_instruction(pred_tid, pred_ins)
    pred_tid_vs_gold = _string_overlap_score(pred_tid, gold_tid)

    if pred_ins:
        token_count = len(pred_ins.split())
        specificity = min(token_count / 25.0, 1.0)
        if not re.search(r"feel|feeling|emotion|mood|inner|soul|dream|memory|longing", pred_ins, re.IGNORECASE):
            specificity = min(1.0, specificity + 0.15)
        creativity = 0.6 if len(set(pred_ins.lower().split())) >= 8 else 0.4
        content_quality = min(max(len(pred_ins) / 160.0, 0.3), 0.9)
        visual_generatability = max(ins_vs_gold, ins_vs_pred)
    else:
        specificity = creativity = content_quality = visual_generatability = 0.0

    return {
        "scores": {
            "specificity": round(_bounded_float(specificity), 4),
            "creativity": round(_bounded_float(creativity), 4),
            "content_quality": round(_bounded_float(content_quality), 4),
            "visual_generatability": round(_bounded_float(visual_generatability), 4),
            "ins_vs_gt_tid": round(_bounded_float(ins_vs_gold), 4),
            "ins_vs_pred_tid": round(_bounded_float(ins_vs_pred), 4),
            "pred_tid_vs_gt_tid": round(_bounded_float(pred_tid_vs_gold), 4),
        },
        "bands": {},
        "judge_debug": {
            "judge_source": "heuristic",
            "judge_request_text": "",
            "judge_raw_response_text": "",
            "judge_attempts": 0,
            "judge_key_suffix": "",
            "judge_last_error": "",
        },
    }


def llm_judge_cid2ins(
    items: List[dict[str, Any]],
    api_keys: list[str],
    model: str,
    batch_size: int,
    max_workers: int,
    max_retries: int,
    retry_backoff_base: float,
    default_failure_reward: float,
    current_step: int = 0,
) -> List[dict[str, Any]]:
    if not items:
        return []
    api_keys = [key for key in api_keys if key]
    if not api_keys:
        return [_heuristic_cid2ins_judge(item) for item in items]

    dashscope = _import_dashscope_or_warn()
    if dashscope is None:
        return [_heuristic_cid2ins_judge(item) for item in items]

    results: list[dict[str, Any] | None] = [None] * len(items)
    cache_keys = [
        json.dumps(
            {
                "gold_target_tid": item["gold_target_tid"],
                "pred_target_tid": item["pred_target_tid"],
                "pred_target_ins": item["pred_target_ins"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in items
    ]

    miss_idx = []
    for i, key in enumerate(cache_keys):
        if key in _judge_cache:
            results[i] = _judge_cache[key]
        else:
            miss_idx.append(i)

    if not miss_idx:
        return [x if x is not None else _heuristic_cid2ins_judge(items[i]) for i, x in enumerate(results)]

    chunks = [miss_idx[s : s + batch_size] for s in range(0, len(miss_idx), batch_size)]

    def _call_one_chunk(chunk_idxs: List[int], preferred_key_idx: int) -> Dict[int, dict[str, Any]]:
        sections = []
        for seq_num, idx in enumerate(chunk_idxs, 1):
            item = items[idx]
            sections.append(
                "\n".join(
                    [
                        f"=== Sample {seq_num} ===",
                        f"hist_cid: {json.dumps(item['hist_cid'], ensure_ascii=False)}",
                        f"gt_target_tid: {json.dumps(item['gold_target_tid'], ensure_ascii=False)}",
                        f"predicted_target_tid: {json.dumps(item['pred_target_tid'], ensure_ascii=False)}",
                        f"predicted_target_ins: {item['pred_target_ins']}",
                    ]
                )
            )

        user_msg = (
            "Please score the following samples.\n"
            f"There are exactly {len(chunk_idxs)} samples.\n"
            "Return a JSON array only, with exactly one object per sample.\n"
            "The array length must equal the sample count.\n"
            "Use indices 1..N exactly once and do not omit any sample.\n\n"
            + "\n\n".join(sections)
        )

        last_error = None
        last_response_text = ""
        last_key_suffix = ""
        ordered_keys = api_keys[preferred_key_idx:] + api_keys[:preferred_key_idx]
        for attempt in range(max_retries):
            for api_key in ordered_keys:
                key_suffix = api_key[-6:]
                last_key_suffix = key_suffix
                try:
                    resp = dashscope.MultiModalConversation.call(
                        api_key=api_key,
                        model=model,
                        messages=[
                            {"role": "system", "content": [{"text": _CID2INS_JUDGE_SYSTEM}]},
                            {"role": "user", "content": [{"text": user_msg}]},
                        ],
                        enable_thinking=False,
                    )
                    if resp.status_code == 200:
                        content = resp.output.choices[0].message.content
                        if isinstance(content, list):
                            text = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            ).strip()
                        else:
                            text = str(content).strip()
                        last_response_text = text
                        text = _strip_think_and_fence(text)
                        parsed = _extract_first_json_value(text)
                        if isinstance(parsed, dict):
                            if isinstance(parsed.get("results"), list):
                                parsed = parsed["results"]
                            elif isinstance(parsed.get("items"), list):
                                parsed = parsed["items"]
                            elif len(chunk_idxs) == 1 and "scores" in parsed:
                                parsed = [{**parsed, "index": 1}]

                        if isinstance(parsed, list):
                            out: Dict[int, dict[str, Any]] = {}
                            for parsed_item in parsed:
                                if not isinstance(parsed_item, dict):
                                    continue
                                idx = int(parsed_item.get("index", 0))
                                if idx <= 0:
                                    continue
                                raw_scores = parsed_item.get("scores") or {}
                                raw_bands = parsed_item.get("bands") or {}
                                norm_scores = {
                                    "specificity": _bounded_float(raw_scores.get("specificity"), 0.0),
                                    "creativity": _bounded_float(raw_scores.get("creativity"), 0.0),
                                    "content_quality": _bounded_float(raw_scores.get("content_quality"), 0.0),
                                    "visual_generatability": _bounded_float(raw_scores.get("visual_generatability"), 0.0),
                                    "ins_vs_gt_tid": _bounded_float(raw_scores.get("ins_vs_gt_tid"), 0.0),
                                    "ins_vs_pred_tid": _bounded_float(raw_scores.get("ins_vs_pred_tid"), 0.0),
                                    "pred_tid_vs_gt_tid": _bounded_float(raw_scores.get("pred_tid_vs_gt_tid"), 0.0),
                                }
                                out[idx] = {
                                    "scores": norm_scores,
                                    "bands": {
                                        "ins_vs_gt_tid": _stringify(raw_bands.get("ins_vs_gt_tid", "")),
                                        "ins_vs_pred_tid": _stringify(raw_bands.get("ins_vs_pred_tid", "")),
                                        "pred_tid_vs_gt_tid": _stringify(raw_bands.get("pred_tid_vs_gt_tid", "")),
                                    },
                                    "judge_debug": {
                                        "judge_source": "api",
                                        "judge_request_text": _truncate_for_log(user_msg, 12000),
                                        "judge_raw_response_text": _truncate_for_log(text, 12000),
                                        "judge_attempts": attempt + 1,
                                        "judge_key_suffix": key_suffix,
                                        "judge_last_error": "",
                                    },
                                }
                            if out:
                                result_map: Dict[int, dict[str, Any]] = {}
                                for j, original_idx in enumerate(chunk_idxs, 1):
                                    if j not in out:
                                        raise RuntimeError(f"judge missing index={j}")
                                    result_map[original_idx] = out[j]
                                return result_map
                        last_error = f"judge parse failed with key_suffix={key_suffix}: {text[:200]}"
                    else:
                        last_error = f"judge api {resp.status_code} with key_suffix={key_suffix}: {resp.message}"
                except Exception as exc:
                    last_error = f"judge exception with key_suffix={key_suffix}: {exc}"
            time.sleep(min(retry_backoff_base ** (attempt + 1), 30.0))

        should_split = (
            len(chunk_idxs) > 1
            and isinstance(last_error, str)
            and "judge parse failed" in last_error.lower()
        )
        if should_split:
            mid = len(chunk_idxs) // 2
            left_chunk = chunk_idxs[:mid]
            right_chunk = chunk_idxs[mid:]
            logger.warning(
                "judge parse failed for chunk_size=%s with key_suffix=%s; splitting into %s and %s",
                len(chunk_idxs),
                last_key_suffix,
                len(left_chunk),
                len(right_chunk),
            )
            merged: Dict[int, dict[str, Any]] = {}
            if left_chunk:
                merged.update(_call_one_chunk(left_chunk, preferred_key_idx))
            if right_chunk:
                merged.update(_call_one_chunk(right_chunk, (preferred_key_idx + 1) % len(api_keys)))
            return merged

        raise JudgeChunkError(
            last_error or "judge failed",
            request_text=_truncate_for_log(user_msg, 12000),
            response_text=_truncate_for_log(last_response_text, 12000),
            key_suffix=last_key_suffix,
            attempts=max_retries,
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        future_to_chunk = {
            executor.submit(_call_one_chunk, chunk, chunk_idx % len(api_keys)): chunk
            for chunk_idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_chunk):
            try:
                chunk_result = future.result()
            except Exception as exc:
                chunk = future_to_chunk[future]
                logger.warning(
                    "cid2ins judge failed for chunk=%s, fallback to default low reward: %s",
                    chunk,
                    exc,
                )
                _emergency_save_manager.record_judge_failure(exc, current_step)
                request_text = ""
                response_text = ""
                key_suffix = ""
                attempts = max_retries
                last_error = _truncate_for_log(exc, 2000)
                if isinstance(exc, JudgeChunkError):
                    request_text = exc.request_text
                    response_text = exc.response_text
                    key_suffix = exc.key_suffix
                    attempts = exc.attempts
                chunk_result = {
                    idx: {
                        "scores": {
                            "specificity": 0.0,
                            "creativity": 0.0,
                            "content_quality": 0.0,
                            "visual_generatability": 0.0,
                            "ins_vs_gt_tid": 0.0,
                            "ins_vs_pred_tid": 0.0,
                            "pred_tid_vs_gt_tid": 0.0,
                        },
                        "bands": {},
                        "forced_total_reward": float(default_failure_reward),
                        "judge_debug": {
                            "judge_source": "default_low_reward_after_api_failure",
                            "judge_request_text": request_text,
                            "judge_raw_response_text": response_text,
                            "judge_attempts": attempts,
                            "judge_key_suffix": key_suffix,
                            "judge_last_error": last_error,
                        },
                    }
                    for idx in chunk
                }
            with _judge_cache_lock:
                for idx, score_obj in chunk_result.items():
                    results[idx] = score_obj
                    _judge_cache[cache_keys[idx]] = score_obj

    return [x if x is not None else _heuristic_cid2ins_judge(items[i]) for i, x in enumerate(results)]


def _build_cid2cid_prompt(hist_cid: Any) -> list[dict[str, str]]:
    user = (
        "Task: Predict the target_cid based on the user's historical interactions in hist_cid.\n"
        f"hist_cid: {_format_hist_cid(hist_cid)}\n"
        "Think first, then output JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["cid2cid"]},
        {"role": "user", "content": user},
    ]


def _build_cid2ins_prompt(hist_cid: Any) -> list[dict[str, str]]:
    user = (
        "Task: Generate the target recommendation result based on the user's historical interactions in hist_cid.\n"
        "The output must be JSON with the fields target_tid and target_ins.\n"
        f"hist_cid: {_format_hist_cid(hist_cid)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPTS["cid2ins"]},
        {"role": "user", "content": user},
    ]


def load_cid2cid_dataset(
    parquet_path: Path,
    tokenizer,
    split_name: str,
    assistant_prefill: str,
    prompt_enable_thinking: bool,
    max_samples: Optional[int] = None,
) -> Dataset:
    rows = _read_parquet_rows(parquet_path)
    samples = []
    for row in rows:
        hist_cid = _as_list(row.get("hist_sid"))
        target_cid = _normalize_cid_value(row.get("target_sid"))
        if not hist_cid or not target_cid:
            continue
        prompt = tokenizer.apply_chat_template(
            _build_cid2cid_prompt(hist_cid),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=prompt_enable_thinking,
        )
        prompt = _append_assistant_prefill(prompt, assistant_prefill)
        samples.append(
            {
                "prompt": prompt,
                "task": "cid2cid",
                "uid": _stringify(row.get("uid")),
                "split": split_name,
                "hist_cid": _format_hist_cid(hist_cid),
                "gold_target_cid": _canonical_json_array_str(target_cid),
                "gold_target_tid": "[]",
                "gold_target_ins": "",
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    logger.info("[%s] cid2cid samples=%d from %s", split_name, len(samples), parquet_path.name)
    return Dataset.from_list(samples)


def load_cid2ins_dataset(
    parquet_path: Path,
    tokenizer,
    split_name: str,
    assistant_prefill: str,
    prompt_enable_thinking: bool,
    max_samples: Optional[int] = None,
) -> Dataset:
    rows = _read_parquet_rows(parquet_path)
    samples = []
    for row in rows:
        hist_cid = _as_list(row.get("hist_sid"))
        target_tid = _normalize_tid_value(row.get("target_tid"))
        target_ins = _stringify(row.get("target_ins")).strip()
        if not hist_cid or not target_tid or not target_ins:
            continue
        prompt = tokenizer.apply_chat_template(
            _build_cid2ins_prompt(hist_cid),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=prompt_enable_thinking,
        )
        prompt = _append_assistant_prefill(prompt, assistant_prefill)
        samples.append(
            {
                "prompt": prompt,
                "task": "cid2ins",
                "uid": _stringify(row.get("uid")),
                "split": split_name,
                "hist_cid": _format_hist_cid(hist_cid),
                "gold_target_cid": "[]",
                "gold_target_tid": _canonical_json_array_str(target_tid),
                "gold_target_ins": target_ins,
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    logger.info("[%s] cid2ins samples=%d from %s", split_name, len(samples), parquet_path.name)
    return Dataset.from_list(samples)


class RewardOrchestrator:
    __name__ = "reward_orchestrator"

    def __init__(self, args: "RLRecAIGCArgs"):
        self.args = args
        self._step = 0
        self._judge_keys = _parse_api_keys(args.judge_api_keys)
        if not self._judge_keys:
            self._judge_keys = [key for key in DEFAULT_JUDGE_API_KEYS if key]
        if not self._judge_keys:
            self._judge_keys = _parse_api_keys(args.judge_api_key)
        if not self._judge_keys:
            self._judge_keys = _parse_api_keys(args.dashscope_api_key)
        self._known_cids = _load_known_cid_set(args.pid2cid2tid_path)

    def __call__(
        self,
        completions: List[str],
        prompt: List[str] | None = None,
        task: List[str] | None = None,
        uid: List[str] | None = None,
        split: List[str] | None = None,
        hist_cid: List[str] | None = None,
        gold_target_cid: List[str] | None = None,
        gold_target_tid: List[str] | None = None,
        gold_target_ins: List[str] | None = None,
        **kwargs,
    ) -> List[float]:
        self._step += 1
        prompt = prompt or []
        task = task or []
        uid = uid or []
        split = split or []
        hist_cid = hist_cid or []
        gold_target_cid = gold_target_cid or []
        gold_target_tid = gold_target_tid or []
        gold_target_ins = gold_target_ins or []

        reward_input_completions = [
            _build_reward_input_completion(c, self.args.assistant_prefill) for c in completions
        ]
        raw_parsed_objs = [_extract_json_object(c) for c in reward_input_completions]
        raw_parse_debugs = [_build_parse_debug(c, obj) for c, obj in zip(reward_input_completions, raw_parsed_objs)]
        raw_reasoning_format_penalties = [
            _analyze_reasoning_output_format(
                c,
                ["target_cid"] if (task[i] if i < len(task) else "") == "cid2cid" else ["target_tid", "target_ins"],
            )
            for i, c in enumerate(reward_input_completions)
        ]
        think_format_rewards = [
            _binary_think_format_reward(
                c,
                min_content_len=self.args.think_format_min_content_len,
            )
            for c in reward_input_completions
        ]
        optimized_completions = [
            _postprocess_completion_for_scoring(
                c,
                task[i] if i < len(task) else "",
            )
            for i, c in enumerate(reward_input_completions)
        ]
        optimized_parsed_objs = [_extract_json_object(c) for c in optimized_completions]
        optimized_parse_debugs = [
            _build_parse_debug(c, obj) for c, obj in zip(optimized_completions, optimized_parsed_objs)
        ]
        optimized_reasoning_format_penalties = [
            _analyze_reasoning_output_format(
                c,
                ["target_cid"] if (task[i] if i < len(task) else "") == "cid2cid" else ["target_tid", "target_ins"],
            )
            for i, c in enumerate(optimized_completions)
        ]
        strict_format_rewards = [
            _json_format_reward(
                task[i] if i < len(task) else "",
                optimized_parsed_objs[i],
                optimized_reasoning_format_penalties[i],
            )
            for i in range(len(optimized_completions))
        ]
        t0 = time.perf_counter()

        judge_inputs: list[dict[str, Any]] = []
        judge_positions: list[int] = []
        for i, obj in enumerate(raw_parsed_objs):
            if i >= len(task) or task[i] != "cid2ins":
                continue
            if strict_format_rewards[i] < 1.0:
                continue
            pred_tid = _normalize_tid_value((obj or {}).get("target_tid"))
            pred_ins = _stringify((obj or {}).get("target_ins")).strip()
            if not pred_ins or not pred_tid:
                continue
            gold_tid = _safe_json_loads(gold_target_tid[i], []) if i < len(gold_target_tid) else []
            hist = _safe_json_loads(hist_cid[i], []) if i < len(hist_cid) else []
            judge_positions.append(i)
            judge_inputs.append(
                {
                    "hist_cid": hist,
                    "gold_target_tid": gold_tid,
                    "pred_target_tid": pred_tid,
                    "pred_target_ins": pred_ins,
                }
            )

        judge_outputs = llm_judge_cid2ins(
            items=judge_inputs,
            api_keys=self._judge_keys,
            model=self.args.judge_model,
            batch_size=self.args.judge_batch,
            max_workers=self.args.judge_max_workers,
            max_retries=self.args.judge_api_retries,
            retry_backoff_base=self.args.judge_retry_backoff_base,
            default_failure_reward=self.args.judge_failure_default_reward,
            current_step=self._step,
        )
        judge_output_map = {pos: score for pos, score in zip(judge_positions, judge_outputs)}

        rewards: List[float] = []
        completion_log_every_steps = int(getattr(self.args, "completion_log_every_steps", 1) or 0)
        completion_log_max_entries = int(getattr(self.args, "completion_log_max_entries", 0) or 0)
        should_log_completions = (
            os.environ.get("LOCAL_RANK", "0") == "0"
            and completion_log_every_steps > 0
            and self._step % completion_log_every_steps == 0
        )
        log_entries: List[dict[str, Any]] = []

        for i, completion in enumerate(completions):
            cur_task = task[i] if i < len(task) else ""
            raw_completion = completion or ""
            reward_input_completion = reward_input_completions[i]
            optimized_completion = optimized_completions[i]
            obj = raw_parsed_objs[i]
            raw_reasoning_format_penalty = raw_reasoning_format_penalties[i]
            penalty_value = raw_reasoning_format_penalty["penalty"]
            think_format_reward = think_format_rewards[i]
            think_gate_factor = (
                self.args.think_format_gate_floor
                + (1.0 - self.args.think_format_gate_floor) * think_format_reward
            )
            reward_penalty_factor = (
                self.args.raw_format_penalty_floor
                + (1.0 - self.args.raw_format_penalty_floor) * max(0.0, 1.0 - penalty_value)
            )

            if cur_task == "cid2cid":
                fmt_reward = strict_format_rewards[i]
                pred_cid_list = _normalize_cid_value((obj or {}).get("target_cid"))
                gold_cid_list = _safe_json_loads(gold_target_cid[i], []) if i < len(gold_target_cid) else []
                pred_cid = pred_cid_list[0] if pred_cid_list else ""
                gold_cid = gold_cid_list[0] if gold_cid_list else ""
                cid_reward, cid_layers = _cid_in_range_reward(
                    pred_cid=pred_cid,
                    gold_cid=gold_cid,
                    layer_weights={
                        "s_a": self.args.cid_a_reward_weight,
                        "s_b": self.args.cid_b_reward_weight,
                        "s_c": self.args.cid_c_reward_weight,
                    },
                    known_cids=self._known_cids,
                    zero_match_in_range_ratio=self.args.cid_in_range_zero_hit_ratio,
                )
                main_reward = self.args.cid_reward_weight * cid_reward
                reward = (
                    main_reward * think_gate_factor + self.args.format_reward_weight * fmt_reward
                ) * reward_penalty_factor
                components = {
                    "cid_reward": round(cid_reward, 4),
                    "main_reward_before_think_gate": round(main_reward, 4),
                    "cid_a_reward": round(cid_layers["s_a"], 4),
                    "cid_b_reward": round(cid_layers["s_b"], 4),
                    "cid_c_reward": round(cid_layers["s_c"], 4),
                    "in_range_zero_hit_bonus": round(cid_layers.get("in_range_zero_hit_bonus", 0.0), 4),
                    "format_reward": round(fmt_reward, 4),
                    "think_format_reward": round(think_format_reward, 4),
                    "think_gate_factor": round(think_gate_factor, 4),
                    "reward_penalty_factor": round(reward_penalty_factor, 4),
                    "reasoning_format_penalty": round(penalty_value, 4),
                    "reasoning_format_violations": raw_reasoning_format_penalty["violations"],
                }
            elif cur_task == "cid2ins":
                fmt_reward = strict_format_rewards[i]
                pred_tid = _normalize_tid_value((obj or {}).get("target_tid"))
                gold_tid = _safe_json_loads(gold_target_tid[i], []) if i < len(gold_target_tid) else []
                pred_ins = _stringify((obj or {}).get("target_ins")).strip()
                if fmt_reward >= 1.0 and pred_tid and pred_ins:
                    judge_output = judge_output_map.get(
                        i,
                        _heuristic_cid2ins_judge(
                            {
                                "gold_target_tid": gold_tid,
                                "pred_target_tid": pred_tid,
                                "pred_target_ins": pred_ins,
                            }
                        ),
                    )
                else:
                    judge_output = _empty_cid2ins_judge("invalid_format_or_missing_required_fields")
                judge_scores = judge_output.get("scores", {})
                ins_quality_reward = _weighted_mean(
                    [
                        _bounded_float(judge_scores.get("specificity"), 0.0),
                        _bounded_float(judge_scores.get("creativity"), 0.0),
                        _bounded_float(judge_scores.get("content_quality"), 0.0),
                        _bounded_float(judge_scores.get("visual_generatability"), 0.0),
                    ],
                    [
                        self.args.ins_specificity_weight,
                        self.args.ins_creativity_weight,
                        self.args.ins_content_quality_weight,
                        self.args.ins_visual_generatability_weight,
                    ],
                )
                ins_vs_gt_reward = _bounded_float(judge_scores.get("ins_vs_gt_tid"), 0.0)
                ins_vs_pred_reward = _bounded_float(judge_scores.get("ins_vs_pred_tid"), 0.0)
                pred_tid_vs_gt_reward = _bounded_float(judge_scores.get("pred_tid_vs_gt_tid"), 0.0)
                semantic_reward = _weighted_mean(
                    [
                        ins_quality_reward,
                        ins_vs_gt_reward,
                        ins_vs_pred_reward,
                        pred_tid_vs_gt_reward,
                    ],
                    [
                        self.args.cid2ins_instruction_quality_weight,
                        self.args.cid2ins_ins_vs_gt_tid_weight,
                        self.args.cid2ins_ins_vs_pred_tid_weight,
                        self.args.cid2ins_pred_tid_vs_gt_tid_weight,
                    ],
                )
                forced_total_reward = judge_output.get("forced_total_reward")
                if forced_total_reward is not None:
                    main_reward = float(forced_total_reward)
                else:
                    main_reward = self.args.cid2ins_semantic_reward_weight * semantic_reward
                reward = (
                    main_reward * think_gate_factor + self.args.format_reward_weight * fmt_reward
                ) * reward_penalty_factor
                components = {
                    "cid2ins_semantic_reward": round(semantic_reward, 4),
                    "main_reward_before_think_gate": round(main_reward, 4),
                    "instruction_quality_reward": round(ins_quality_reward, 4),
                    "ins_vs_gt_tid_reward": round(ins_vs_gt_reward, 4),
                    "ins_vs_pred_tid_reward": round(ins_vs_pred_reward, 4),
                    "pred_tid_vs_gt_tid_reward": round(pred_tid_vs_gt_reward, 4),
                    "specificity": round(_bounded_float(judge_scores.get("specificity"), 0.0), 4),
                    "creativity": round(_bounded_float(judge_scores.get("creativity"), 0.0), 4),
                    "content_quality": round(_bounded_float(judge_scores.get("content_quality"), 0.0), 4),
                    "visual_generatability": round(_bounded_float(judge_scores.get("visual_generatability"), 0.0), 4),
                    "format_reward": round(fmt_reward, 4),
                    "think_format_reward": round(think_format_reward, 4),
                    "think_gate_factor": round(think_gate_factor, 4),
                    "reward_penalty_factor": round(reward_penalty_factor, 4),
                    "reasoning_format_penalty": round(penalty_value, 4),
                    "reasoning_format_violations": raw_reasoning_format_penalty["violations"],
                }
                if forced_total_reward is not None:
                    components["forced_total_reward"] = round(float(forced_total_reward), 4)
                if judge_output.get("bands"):
                    components["bands"] = judge_output["bands"]
                if judge_output.get("judge_debug"):
                    components["judge_debug"] = judge_output["judge_debug"]
            else:
                reward = 0.0
                components = {}

            rewards.append(float(reward))

            if should_log_completions and (
                completion_log_max_entries <= 0 or len(log_entries) < completion_log_max_entries
            ):
                log_entries.append(
                    _build_completion_log_entry(
                        step=self._step,
                        split_name=split[i] if i < len(split) else "",
                        uid_value=uid[i] if i < len(uid) else "",
                        task_name=cur_task,
                        reward_value=float(reward),
                        prompt_text=prompt[i] if i < len(prompt) else "",
                        hist_cid_items=_safe_json_loads(hist_cid[i], []) if i < len(hist_cid) else [],
                        gold_target_cid_items=(
                            _safe_json_loads(gold_target_cid[i], []) if i < len(gold_target_cid) else []
                        ),
                        gold_target_tid_items=(
                            _safe_json_loads(gold_target_tid[i], []) if i < len(gold_target_tid) else []
                        ),
                        gold_target_ins_text=gold_target_ins[i] if i < len(gold_target_ins) else "",
                        parsed_completion=obj,
                        optimized_parsed_completion=(
                            optimized_parsed_objs[i] if i < len(optimized_parsed_objs) else None
                        ),
                        raw_parse_debug={
                            **(raw_parse_debugs[i] if i < len(raw_parse_debugs) else {}),
                            "raw_completion_char_len": len(_stringify(completion)),
                            "assistant_prefill_applied": bool(self.args.assistant_prefill),
                        },
                        optimized_parse_debug={
                            **(optimized_parse_debugs[i] if i < len(optimized_parse_debugs) else {}),
                            "raw_completion_char_len": len(_stringify(completion)),
                            "assistant_prefill_applied": bool(self.args.assistant_prefill),
                        },
                        assistant_prefill=self.args.assistant_prefill,
                        reward_input_completion=reward_input_completion,
                        optimized_completion=optimized_completion,
                        raw_completion=raw_completion,
                        components=components,
                    )
                )

        if should_log_completions and log_entries:
            log_path = Path(self.args.output_dir) / "completions_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                for item in log_entries:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            reward_groups: dict[tuple[str, str, str], list[float]] = {}
            for item in log_entries:
                key = (
                    _stringify(item.get("split")),
                    _stringify(item.get("task")),
                    _stringify(item.get("uid")),
                )
                reward_groups.setdefault(key, []).append(float(item.get("reward", 0.0)))
            group_stds = []
            zero_var_groups = 0
            for values in reward_groups.values():
                if len(values) <= 1:
                    continue
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = variance ** 0.5
                group_stds.append(std)
                if std < 1e-8:
                    zero_var_groups += 1
            if group_stds:
                print(
                    f"[reward group stats] step={self._step} groups={len(group_stds)} "
                    f"zero_var_groups={zero_var_groups} "
                    f"mean_std={sum(group_stds) / len(group_stds):.4f} "
                    f"min_std={min(group_stds):.4f} max_std={max(group_stds):.4f}",
                    flush=True,
                )

        print(
            f"[reward timing] step={self._step} total={time.perf_counter() - t0:.2f}s "
            f"judge_calls={len(judge_inputs)}",
            flush=True,
        )
        return rewards


@dataclass
class RLRecAIGCArgs:
    nproc_per_node: int = field(default=4, metadata={"help": "Number of GPUs to launch on this node"})
    nnodes: int = field(default=1, metadata={"help": "Number of nodes"})
    node_rank: int = field(default=0, metadata={"help": "Rank of the current node"})
    master_addr: str = field(default=env_str("NAVIGEN_MASTER_ADDR", "127.0.0.1"), metadata={"help": "Torchrun master address"})
    master_port: int = field(default=env_int("NAVIGEN_MASTER_PORT", 29500), metadata={"help": "Torchrun master port"})
    dry_run: bool = field(default=False, metadata={"help": "Only print the torchrun command"})

    base_model_dir: str = field(default=str(DEFAULT_MODEL_DIR), metadata={"help": "Base Qwen3 model path"})
    sft_checkpoint_dir: str = field(
        default="",
        metadata={"help": "Optional SFT LoRA checkpoint; empty starts a new LoRA adapter"},
    )
    resume_from_checkpoint: str = field(
        default=str(DEFAULT_RESUME_CHECKPOINT_DIR),
        metadata={"help": "Optional Trainer checkpoint directory to resume optimizer/scheduler/trainer state from"},
    )
    pid2cid2tid_path: str = field(
        default=str(DEFAULT_PID2CID2TID_PATH),
        metadata={"help": "Parquet file used to build the known CID set"},
    )

    train_cid2cid_path: str = field(default=str(DEFAULT_INPUT_DIR / "train_cid2cid.parquet"))
    val_cid2cid_path: str = field(default=str(DEFAULT_INPUT_DIR / "valid_cid2cid.parquet"))
    train_cid2ins_path: str = field(default=str(DEFAULT_INPUT_DIR / "train_cid2ins.parquet"))
    val_cid2ins_path: str = field(default=str(DEFAULT_INPUT_DIR / "valid_cid2ins.parquet"))

    max_train_cid2cid_samples: Optional[int] = field(default=None)
    max_train_cid2ins_samples: Optional[int] = field(default=None)
    max_eval_cid2cid_samples: Optional[int] = field(default=None)
    max_eval_cid2ins_samples: Optional[int] = field(default=None)

    dashscope_api_key: str = field(default="", metadata={"help": "Fallback DashScope API key for the judge"})
    judge_api_key: str = field(default="", metadata={"help": "AIGC LLM Judge API key"})
    judge_api_keys: str = field(
        default="",
        metadata={"help": "Comma- or space-separated judge API keys for key-level parallelism"},
    )
    judge_model: str = field(default="qwen3.5-flash", metadata={"help": "AIGC LLM Judge Model"})
    judge_batch: int = field(default=1, metadata={"help": "Judge Batch Size"})
    judge_max_workers: int = field(default=20, metadata={"help": "Judge Concurrent Workers for API calls"})
    judge_api_retries: int = field(default=3, metadata={"help": "Max retries for judge API calls"})
    judge_retry_backoff_base: float = field(
        default=1.5, metadata={"help": "Exponential backoff base for judge API retries"}
    )
    judge_failure_default_reward: float = field(
        default=0.1,
        metadata={"help": "Direct cid2ins reward used when judge API still fails after all retries"},
    )
    completion_log_every_steps: int = field(
        default=1,
        metadata={"help": "Write completions_log.jsonl every N GRPO generation steps; set <=0 to disable"},
    )
    completion_log_max_entries: int = field(
        default=0,
        metadata={"help": "Maximum completion log entries to write per logged step; set <=0 for no cap"},
    )
    stage_timing_every_steps: int = field(
        default=0,
        metadata={"help": "Print per-stage GRPO timing every N optimizer steps; set <=0 to disable"},
    )
    skip_trl_text_logs: bool = field(
        default=True,
        metadata={"help": "Skip TRL internal prompt/completion text gather logs; does not affect rewards or gradients"},
    )
    judge_emergency_save_error_pattern: str = field(
        default=r"access denied|overdue-payment|good standing",
        metadata={"help": "Regex pattern for judge errors that should trigger emergency save after repeated hits"},
    )
    judge_emergency_save_threshold: int = field(
        default=5,
        metadata={"help": "Emergency-save once this many matching judge failures occur within the configured window"},
    )
    judge_emergency_save_window_minutes: float = field(
        default=10.0,
        metadata={"help": "Time window for counting matching judge failures before emergency save"},
    )
    judge_emergency_save_cooldown_minutes: float = field(
        default=30.0,
        metadata={"help": "Minimum cooldown between emergency saves triggered by repeated judge failures"},
    )
    reasoning_format_penalty_weight: float = field(
        default=0.0,
        metadata={"help": "Deprecated soft reasoning penalty weight; keep 0 when using binary think gate"},
    )
    think_format_gate_floor: float = field(
        default=0.2,
        metadata={"help": "Main reward gate floor when think format is invalid; gate = floor + (1-floor)*binary_think_reward"},
    )
    raw_format_penalty_floor: float = field(
        default=0.3,
        metadata={"help": "Reward penalty floor applied to raw completion format violations so training still gets signal while learning to serialize JSON"},
    )
    think_format_min_content_len: int = field(
        default=10,
        metadata={"help": "Minimum stripped content length between <think> and </think> for binary think format reward"},
    )
    assistant_prefill: str = field(
        default="",
        metadata={"help": "Assistant-side generation prefix appended after the chat generation prompt; keep empty when testing postprocess-only decoding"},
    )
    prompt_enable_thinking: bool = field(
        default=True,
        metadata={"help": "Whether to render the chat generation prompt with enable_thinking=True so the model generates its own <think> block"},
    )
    use_vllm: bool = field(default=True, metadata={"help": "Whether to use vLLM for completion generation"})
    vllm_mode: str = field(
        default="server",
        metadata={"help": "vLLM mode: server or colocate"},
    )
    vllm_server_host: str = field(
        default=env_str("NAVIGEN_VLLM_HOST", "127.0.0.1"),
        metadata={"help": "vLLM server host used when vllm_server_base_url is empty"},
    )
    vllm_server_port: int = field(
        default=env_int("NAVIGEN_VLLM_PORT", 18002),
        metadata={"help": "vLLM server port used when vllm_server_base_url is empty"},
    )
    vllm_server_base_url: str = field(
        default=env_str("NAVIGEN_VLLM_BASE_URL", ""),
        metadata={"help": "Full vLLM server base URL; if empty it will be built from host and port"},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Timeout in seconds while waiting for the vLLM server"},
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size used by vLLM"},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "GPU memory utilization reserved for vLLM model/KV cache"},
    )
    vllm_two_stage_enabled: bool = field(
        default=True,
        metadata={"help": "Use NAVIGEN-style two-stage rollout for vLLM server generation"},
    )
    vllm_stage1_max_tokens: int = field(
        default=768,
        metadata={"help": "Stage-1 max tokens for vLLM two-stage rollout (thinking until </think>)"},
    )
    vllm_stage2_max_tokens: int = field(
        default=192,
        metadata={"help": "Stage-2 max tokens for vLLM two-stage rollout (structured JSON answer)"},
    )
    vllm_stage2_prefix: str = field(
        default="",
        metadata={"help": "Optional Stage-2 fixed answer prefix appended after the thinking block; keep empty when testing postprocess-only decoding"},
    )
    vllm_stage2_cid2cid_prefix: str = field(
        default='\n{"target_cid":"<|cid_begin|>',
        metadata={"help": "Task-specific Stage-2 prefix for cid2cid in vLLM two-stage rollout; pre-fills the JSON field and CID begin token so stage-2 only needs to emit the CID body plus closing tokens"},
    )
    vllm_stage2_cid2ins_prefix: str = field(
        default='\n{"target_tid":["',
        metadata={"help": "Task-specific Stage-2 prefix for cid2ins in vLLM two-stage rollout; opens the first target_tid string to reduce JSON serialization burden"},
    )
    vllm_stage2_cid2cid_regex: str = field(
        default=r'^<s_a_\d+><s_b_\d+><s_c_\d+><\|cid_end\|>"\s*\}$',
        metadata={"help": "Regex constraint applied to cid2cid stage-2 continuation after the stage-2 prefix"},
    )
    vllm_stage2_cid2ins_regex: str = field(
        default=r'^(?:[^"\\]|\\.)+"(?:\s*,\s*"(?:[^"\\]|\\.)+")*\s*\]\s*,\s*"target_ins"\s*:\s*"(?:[^"\\]|\\.)+"\s*\}$',
        metadata={"help": "Regex constraint applied to cid2ins stage-2 continuation after the stage-2 prefix"},
    )
    generation_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Generation batch size override for GRPO completion generation; tuned default for the formal single-train-GPU + single-vLLM-GPU run profile"},
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={"help": "When resuming, do not iterate through already-seen dataloader batches before continuing"},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer name passed to GRPOConfig; default uses non-fused AdamW for stability"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping norm passed to GRPOConfig"},
    )

    cid_reward_weight: float = field(default=0.9, metadata={"help": "CID match reward weight for cid2cid"})
    format_reward_weight: float = field(default=0.1, metadata={"help": "JSON format reward weight"})
    cid_a_reward_weight: float = field(default=0.5, metadata={"help": "s_a match weight for cid2cid"})
    cid_b_reward_weight: float = field(default=0.3, metadata={"help": "s_b match weight for cid2cid"})
    cid_c_reward_weight: float = field(default=0.2, metadata={"help": "s_c match weight for cid2cid"})
    cid_in_range_zero_hit_ratio: float = field(
        default=0.3,
        metadata={"help": "When predicted CID is a known CID but matches none of s_a/s_b/s_c, award cid_a_reward_weight * this ratio"},
    )

    cid2ins_semantic_reward_weight: float = field(
        default=0.9, metadata={"help": "Total semantic reward weight for cid2ins"}
    )
    cid2ins_instruction_quality_weight: float = field(
        default=0.25, metadata={"help": "Instruction quality aggregate weight for cid2ins"}
    )
    cid2ins_ins_vs_gt_tid_weight: float = field(
        default=0.25, metadata={"help": "Instruction-vs-ground-truth TID weight for cid2ins"}
    )
    cid2ins_ins_vs_pred_tid_weight: float = field(
        default=0.25, metadata={"help": "Instruction-vs-predicted TID weight for cid2ins"}
    )
    cid2ins_pred_tid_vs_gt_tid_weight: float = field(
        default=0.25, metadata={"help": "Predicted-vs-ground-truth TID weight for cid2ins"}
    )

    ins_specificity_weight: float = field(default=0.25, metadata={"help": "Instruction specificity weight"})
    ins_creativity_weight: float = field(default=0.25, metadata={"help": "Instruction creativity weight"})
    ins_content_quality_weight: float = field(default=0.25, metadata={"help": "Instruction content quality weight"})
    ins_visual_generatability_weight: float = field(
        default=0.25, metadata={"help": "Instruction visual generatability weight"}
    )
    beta: float = field(
        default=0.01,
        metadata={"help": "KL coefficient for GRPO. Set >0 to enable reference-model KL regularization and KL logging."},
    )

    max_seq_length: int = field(default=2048)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=2)
    num_generations: int = field(default=8)
    max_completion_length: int = field(default=1536)
    learning_rate: float = field(default=3e-4)
    warmup_steps: float = field(
        default=0.0,
        metadata={"help": "Warmup steps if >=1, or warmup ratio if in [0,1). Kept for compatibility with current transformers semantics."},
    )
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "Opfftional explicit warmup ratio alias. If set, overrides warmup_steps."},
    )
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    eval_strategy: str = field(default="no")
    eval_steps: int = field(default=50)
    save_steps: int = field(default=50)
    save_total_limit: Optional[int] = field(default=None)
    seed: int = field(default=42)
    output_dir: str = field(default=str(DEFAULT_OUTPUT_DIR))


def build_mixed_datasets(args: RLRecAIGCArgs, tokenizer) -> tuple[Dataset, Optional[Dataset]]:
    train_parts = [
        load_cid2cid_dataset(
            Path(args.train_cid2cid_path),
            tokenizer,
            "train",
            args.assistant_prefill,
            args.prompt_enable_thinking,
            args.max_train_cid2cid_samples,
        ),
        load_cid2ins_dataset(
            Path(args.train_cid2ins_path),
            tokenizer,
            "train",
            args.assistant_prefill,
            args.prompt_enable_thinking,
            args.max_train_cid2ins_samples,
        ),
    ]
    train_dataset = concatenate_datasets(train_parts).shuffle(seed=args.seed)

    eval_dataset = None
    if str(args.eval_strategy).lower() != "no":
        eval_parts = [
            load_cid2cid_dataset(
                Path(args.val_cid2cid_path),
                tokenizer,
                "valid",
                args.assistant_prefill,
                args.prompt_enable_thinking,
                args.max_eval_cid2cid_samples,
            ),
            load_cid2ins_dataset(
                Path(args.val_cid2ins_path),
                tokenizer,
                "valid",
                args.assistant_prefill,
                args.prompt_enable_thinking,
                args.max_eval_cid2ins_samples,
            ),
        ]
        eval_dataset = concatenate_datasets(eval_parts).shuffle(seed=args.seed)

    logger.info("mixed train dataset size=%d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("mixed eval dataset size=%d", len(eval_dataset))
    return train_dataset, eval_dataset


def main():
    parser = HfArgumentParser(RLRecAIGCArgs)
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if maybe_relaunch_with_torchrun(args):
        return

    if not args.judge_api_key:
        args.judge_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not args.judge_api_keys:
        args.judge_api_keys = os.environ.get("DASHSCOPE_API_KEYS", "")
    if not args.dashscope_api_key:
        args.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    args.vllm_server_base_url = normalize_vllm_server_base_url(
        args.vllm_server_host,
        args.vllm_server_port,
        args.vllm_server_base_url,
    )
    if args.use_vllm and args.vllm_two_stage_enabled and not args.prompt_enable_thinking:
        logger.warning(
            "Disabling vLLM two-stage rollout because prompt_enable_thinking=False would prefill an empty "
            "<think> block in this tokenizer/chat template."
        )
        args.vllm_two_stage_enabled = False

    base_model_dir = Path(args.base_model_dir)
    sft_checkpoint_dir = Path(args.sft_checkpoint_dir) if args.sft_checkpoint_dir else None
    sft_checkpoint_dir = _resolve_checkpoint_path(
        sft_checkpoint_dir,
        prefer_batch_size=args.per_device_train_batch_size,
        require_adapter=True,
    )
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = str(
            _resolve_checkpoint_path(
                Path(args.resume_from_checkpoint),
                prefer_batch_size=args.per_device_train_batch_size,
                require_adapter=False,
            )
        )

    required_paths = [
        (base_model_dir, "base_model_dir"),
        (Path(args.pid2cid2tid_path), "pid2cid2tid_path"),
        (Path(args.train_cid2cid_path), "train_cid2cid_path"),
        (Path(args.train_cid2ins_path), "train_cid2ins_path"),
    ]
    if str(args.eval_strategy).lower() != "no":
        required_paths.extend(
            [
                (Path(args.val_cid2cid_path), "val_cid2cid_path"),
                (Path(args.val_cid2ins_path), "val_cid2ins_path"),
            ]
        )

    for path, name in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if sft_checkpoint_dir is not None and not sft_checkpoint_dir.exists():
        raise FileNotFoundError(f"sft_checkpoint_dir does not exist: {sft_checkpoint_dir}")

    use_warmup_ratio = args.warmup_ratio is not None
    resolved_warmup_ratio = float(args.warmup_ratio) if use_warmup_ratio else 0.0
    resolved_warmup_steps = 0 if use_warmup_ratio else int(args.warmup_steps)
    gradient_checkpointing_mode = resolve_gradient_checkpointing_mode()
    normalize_eval_batch_size_for_grpo(args, num_gpus)
    _emergency_save_manager.configure(
        output_dir=args.output_dir,
        pattern_text=args.judge_emergency_save_error_pattern,
        threshold=args.judge_emergency_save_threshold,
        window_minutes=args.judge_emergency_save_window_minutes,
        cooldown_minutes=args.judge_emergency_save_cooldown_minutes,
    )

    grpo_config = build_grpo_config(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        generation_batch_size=args.generation_batch_size,
        ignore_data_skip=args.ignore_data_skip,
        beta=args.beta,
        learning_rate=args.learning_rate,
        warmup_steps=resolved_warmup_steps,
        warmup_ratio=resolved_warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=False,
        ddp_static_graph=False,
        report_to="none",
    )
    grpo_config.stage_timing_every_steps = int(args.stage_timing_every_steps)
    grpo_config.skip_trl_text_logs = bool(args.skip_trl_text_logs)
    mixed_precision = sync_unsloth_mixed_precision_env(grpo_config)

    logger.info("=== Recommendation + AIGC RL (GRPO) ===")
    logger.info("base_model=%s", base_model_dir)
    logger.info("sft_checkpoint=%s", sft_checkpoint_dir or "<new lora>")
    logger.info("mixed_precision=%s", mixed_precision)
    if args.use_vllm:
        logger.info(
            "vLLM: enabled | mode=%s | server=%s | tp=%s | generation_batch_size=%s",
            args.vllm_mode,
            args.vllm_server_base_url,
            args.vllm_tensor_parallel_size,
            args.generation_batch_size,
        )
        logger.info(
            "vLLM two-stage: enabled=%s | stage1_max_tokens=%d | stage2_max_tokens=%d | default_prefix=%r | cid2cid_prefix=%r | cid2ins_prefix=%r | cid2cid_regex=%r | cid2ins_regex=%r",
            args.vllm_two_stage_enabled,
            args.vllm_stage1_max_tokens,
            args.vllm_stage2_max_tokens,
            args.vllm_stage2_prefix,
            args.vllm_stage2_cid2cid_prefix,
            args.vllm_stage2_cid2ins_prefix,
            args.vllm_stage2_cid2cid_regex,
            args.vllm_stage2_cid2ins_regex,
        )
    else:
        logger.info("vLLM: disabled")
    logger.info(
        "reward weights: cid=%.2f cid_layers=(%.2f,%.2f,%.2f) cid2ins_semantic=%.2f format=%.2f think_gate_floor=%.2f raw_format_penalty_floor=%.2f beta=%.4f",
        args.cid_reward_weight,
        args.cid_a_reward_weight,
        args.cid_b_reward_weight,
        args.cid_c_reward_weight,
        args.cid2ins_semantic_reward_weight,
        args.format_reward_weight,
        args.think_format_gate_floor,
        args.raw_format_penalty_floor,
        args.beta,
    )
    if use_warmup_ratio:
        logger.info(
            "lr schedule: lr=%g scheduler=%s warmup_ratio=%.4f",
            args.learning_rate,
            args.lr_scheduler_type,
            resolved_warmup_ratio,
        )
    else:
        logger.info(
            "lr schedule: lr=%g scheduler=%s warmup_steps=%d",
            args.learning_rate,
            args.lr_scheduler_type,
            resolved_warmup_steps,
        )
    logger.info("prompt_enable_thinking=%s", args.prompt_enable_thinking)
    logger.info("assistant_prefill=%r", args.assistant_prefill)
    logger.info("gradient_checkpointing_mode=%r", gradient_checkpointing_mode)
    logger.info("resume_from_checkpoint=%s", args.resume_from_checkpoint or "<none>")
    logger.info(
        "judge emergency save: pattern=%r threshold=%d window_minutes=%.1f cooldown_minutes=%.1f",
        args.judge_emergency_save_error_pattern,
        args.judge_emergency_save_threshold,
        args.judge_emergency_save_window_minutes,
        args.judge_emergency_save_cooldown_minutes,
    )
    logger.info("train metrics log path=%s", Path(args.output_dir) / "train_metrics.jsonl")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_model_dir),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
        full_finetuning=False,
        trust_remote_code=True,
        local_files_only=True,
    )

    patch_peft_tensor_parallel_compat()

    if sft_checkpoint_dir is not None:
        from peft import PeftModel

        logger.info("loading SFT LoRA checkpoint from %s", sft_checkpoint_dir)
        model = PeftModel.from_pretrained(
            model,
            str(sft_checkpoint_dir),
            is_trainable=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model = FastLanguageModel.patch_peft_model(
            model,
            use_gradient_checkpointing=gradient_checkpointing_mode,
        )
    else:
        logger.info("no SFT LoRA checkpoint provided, creating a new LoRA adapter")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=OFFICIAL_LORA_TARGET_MODULES,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=gradient_checkpointing_mode,
            random_state=args.seed,
            max_seq_length=args.max_seq_length,
            use_rslora=False,
            loftq_config=None,
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    known_cids = _load_known_cid_set(args.pid2cid2tid_path)
    eos_token_ids: list[int] = []
    eos_token_ids.extend(_flatten_token_ids(getattr(tokenizer, "eos_token_id", None)))
    eos_token_ids.extend(_flatten_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    cid_generation_constraint = _build_cid_generation_constraint(tokenizer, known_cids, eos_token_ids)
    logger.info(
        "built legal CID generation trie: catalog_size=%d marker_variants=%d eos_token_ids=%s",
        len(known_cids),
        len(cid_generation_constraint.marker_token_seqs),
        list(cid_generation_constraint.eos_token_ids),
    )

    if args.use_vllm:
        existing_generation_kwargs = dict(getattr(grpo_config, "generation_kwargs", None) or {})
        vllm_stop_generation_kwargs = build_vllm_stop_generation_kwargs(tokenizer, model)
        for key, value in vllm_stop_generation_kwargs.items():
            existing_generation_kwargs.setdefault(key, value)
        if args.vllm_mode == "server" and args.vllm_two_stage_enabled:
            existing_generation_kwargs["grpo_two_stage_enabled"] = True
            existing_generation_kwargs["grpo_two_stage_stage1_max_tokens"] = args.vllm_stage1_max_tokens
            existing_generation_kwargs["grpo_two_stage_stage2_max_tokens"] = args.vllm_stage2_max_tokens
            existing_generation_kwargs["grpo_two_stage_prefix"] = args.vllm_stage2_prefix
            existing_generation_kwargs["grpo_two_stage_cid2cid_prefix"] = args.vllm_stage2_cid2cid_prefix
            existing_generation_kwargs["grpo_two_stage_cid2ins_prefix"] = args.vllm_stage2_cid2ins_prefix
            existing_generation_kwargs["grpo_two_stage_cid2cid_regex"] = args.vllm_stage2_cid2cid_regex
            existing_generation_kwargs["grpo_two_stage_cid2ins_regex"] = args.vllm_stage2_cid2ins_regex
            existing_generation_kwargs.setdefault("grpo_two_stage_stop", ["</think>"])
            existing_generation_kwargs.setdefault("grpo_two_stage_include_stop", True)
            existing_generation_kwargs.setdefault("grpo_two_stage_force_close_suffix", "</think>")
        grpo_config.generation_kwargs = existing_generation_kwargs
        logger.info("vLLM stop alignment: %s", json.dumps(grpo_config.generation_kwargs, ensure_ascii=False))
        if args.vllm_mode == "server" and args.vllm_two_stage_enabled:
            logger.info("vLLM cid2cid will use stepwise trie-constrained decoding on stage-2 server generation.")
        else:
            logger.warning(
                "Legal CID hard constraint is enabled on the local transformers.generate path. "
                "Current vLLM mode/pipeline does not use the new stepwise trie constraint."
            )

    train_dataset, eval_dataset = build_mixed_datasets(args, tokenizer)
    reward_fn = RewardOrchestrator(args)
    ensure_model_warnings_issued(model)
    patch_peft_tensor_parallel_compat()
    patch_trl_prepare_peft_model_compat()
    patch_grpo_vllm_peft_sync_compat()
    patch_vllm_client_two_stage_generation()
    patch_unsloth_trainer_ddp_compat()
    patch_trainer_optimizer_nonfinite_guard()
    patch_trainer_grad_norm_nonfinite_guard()

    trainer = ConstrainedCidGRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[TrainMetricsJSONLCallback(args.output_dir)],
    )
    trainer._legal_cid_constraint = cid_generation_constraint
    _emergency_save_manager.bind_runtime(trainer=trainer, tokenizer=tokenizer)
    if args.use_vllm and args.vllm_mode == "server" and hasattr(trainer, "vllm_client"):
        trainer.vllm_client._grpo_two_stage_tokenizer = tokenizer
        trainer.vllm_client._grpo_two_stage_known_cids = known_cids
        trainer.vllm_client._grpo_two_stage_cid2cid_trie_cache = {}

    logger.info("start GRPO training ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    save_path = Path(args.output_dir) / "final_rl_lora"
    trainer.save_model(str(save_path))
    if is_main:
        tokenizer.save_pretrained(str(save_path))
        logger.info("saved RL model to %s", save_path)


if __name__ == "__main__":
    main()
