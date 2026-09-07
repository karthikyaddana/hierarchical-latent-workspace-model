"""Progressive correctness-first trainer for Qwen-backed HLWM Version 5.5.

Version 5.5 responds to the two preregistered Version 5.4 failures:

* policy-score overlap: the three candidate-level heads are additionally
  trained on-policy, on generated train-anchor emissions graded by the
  deterministic semantic graders, plus semantic and mechanical hard negatives,
  before thresholds are fitted on generated validation emissions;
* routing collapse: a router marginal-entropy regularizer joins the
  Switch-style balance term, and routing must later pass preregistered
  entropy, minimum-load, and intervention gates at evaluation.

The trainer also supports BF16 autocast for Ampere-class devices (including
A100 MIG slices) and asserts a device memory headroom after the real-Qwen
preflight so a 5 GB partition fails in minutes rather than mid-run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from .data import (
        HLWMCollator,
        RESPONSE_CUE_TEXT,
        Reasoning9000Dataset,
        V10AnchorCollator,
        encode_preserving_ends,
    )
    from .modeling_hlwm import (
        HLWMConfig,
        HLWMForConditionalGeneration,
        _masked_token_cross_entropy,
    )
    from .semantic_grading import grade_semantic_answer
except ImportError:  # Direct execution inside the Kaggle dataset directory.
    from data import (
        HLWMCollator,
        RESPONSE_CUE_TEXT,
        Reasoning9000Dataset,
        V10AnchorCollator,
        encode_preserving_ends,
    )
    from modeling_hlwm import (
        HLWMConfig,
        HLWMForConditionalGeneration,
        _masked_token_cross_entropy,
    )
    from semantic_grading import grade_semantic_answer


PINNED_MODEL = "Qwen/Qwen3-0.6B-Base"
PINNED_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

# Autocast dtype for every mixed-precision region.  ``main`` resolves this from
# ``--precision`` before any model work; helpers read the module global so the
# whole file switches consistently between FP16 (T4) and BF16 (A100/MIG).
AMP_DTYPE: torch.dtype = torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hlwm-output"))
    parser.add_argument("--model", default=PINNED_MODEL)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Legacy total-step override; when set, runs local denoising only.",
    )
    parser.add_argument("--overfit-steps", type=int, default=48)
    parser.add_argument("--local-steps", type=int, default=120)
    parser.add_argument("--joint-steps", type=int, default=80)
    parser.add_argument("--overfit-examples", type=int, default=24)
    parser.add_argument("--overfit-min-improvement", type=float, default=0.0)
    parser.add_argument("--max-runtime-hours", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--warmup-updates", "--warmup-steps", dest="warmup_updates", type=int, default=10
    )
    parser.add_argument("--initial-loss-scale", type=float, default=1024.0)
    parser.add_argument("--loss-scale-growth-interval", type=int, default=2000)
    parser.add_argument("--max-skipped-updates", type=int, default=4)
    parser.add_argument("--causal-ratio", type=float, default=0.15)
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=0.35,
        help="Deterministic fraction of post-overfit batches drawn from verified behavior anchors.",
    )
    parser.add_argument(
        "--unfreeze-tail-layers",
        type=int,
        default=0,
        help="Full Qwen blocks to tune; keep zero on T4 and use LoRA sidecars.",
    )
    # ---- Version 10.0 dense-supervision channel (zero = v9 behavior). ----
    parser.add_argument("--latent-thoughts", type=int, default=0)
    parser.add_argument(
        "--vocab-grounded-thoughts", action="store_true",
        help="Version 11.0: construct each thought inside the decoder's "
        "vocabulary basis (frozen lm_head softmax over frozen input "
        "embeddings). Off reproduces v10.0 exactly.",
    )
    parser.add_argument("--vocab-thought-tau", type=float, default=1.0)
    parser.add_argument(
        "--declared-routing", action="store_true",
        help="Version 12.0: the expert route is a literal <route:family> token "
        "the model emits and the harness parses back, instead of a gold label "
        "(training-only) or a detached probe. Off reproduces v10/v11 exactly.",
    )
    parser.add_argument(
        "--adapter", choices=("lora", "dora", "pissa"), default="lora",
        help="Version 11.1 shared-sidecar family: DoRA (arXiv:2402.09353) or "
        "PiSSA-init residual variant (arXiv:2404.02948). Routed family "
        "experts stay plain LoRA in every mode. Default reproduces v10.",
    )
    parser.add_argument(
        "--incontext-decode-weight", type=float, default=0.5,
        help="Weight of the in-context aligned decode CE on the student's "
        "segment logits (frozen lm_head over thought positions in the masked "
        "context). Zero reproduces v10.0 exactly.",
    )
    parser.add_argument("--kv-prefix-slots", type=int, default=16)
    parser.add_argument("--kv-prefix-rank", type=int, default=64)
    parser.add_argument("--prefix-attn-gate-init", type=float, default=0.08)
    parser.add_argument("--mlp-expert-count", type=int, default=0)
    parser.add_argument("--mlp-expert-rank", type=int, default=16)
    parser.add_argument("--warm-steps", type=int, default=600)
    parser.add_argument("--distill-gamma", type=float, default=10.0)
    parser.add_argument("--trace-tokens", type=int, default=96)
    parser.add_argument(
        "--masked-fraction-floor",
        type=float,
        default=0.5,
        help="Minimum masked share of main-phase rows (Amendment A1); the "
        "sampler oversamples masked anchors with replacement to reach it.",
    )
    parser.add_argument("--telemetry-every", type=int, default=100)
    parser.add_argument(
        "--warm-em-floor",
        type=float,
        default=0.50,
        help="Absolute digit-level reconstruction accuracy floor at the warm "
        "kill gate (Amendment A3; fixed before the run, never from the run).",
    )
    parser.add_argument("--gonogo-step", type=int, default=1200)
    parser.add_argument(
        "--gonogo-masked-numeric-em",
        type=float,
        default=0.05,
        help="Masked numeric-family exact-match floor at the go/no-go "
        "(guess-chance ~0 on numerics; abstention/ordering excluded).",
    )
    parser.add_argument("--wo-l1-branch-steps", type=int, default=300)
    parser.add_argument(
        "--v10-preflight-only",
        action="store_true",
        help="Run the blocking on-device Version 10.0 preflight (liveness, "
        "gate equivalence, decode parity, gamma calibration, throughput, "
        "memory gate), write preflight.json, and exit 0/3.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-tail-layers", type=int, default=4)
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--refinement-steps", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=96)
    parser.add_argument("--canvas-tokens", type=int, default=96)
    parser.add_argument("--brief-tokens", type=int, default=64)
    parser.add_argument("--causal-tokens", type=int, default=192)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=32)
    parser.add_argument(
        "--calibration-records",
        type=int,
        default=64,
        help="Maximum generated validation-anchor emissions used for policy calibration.",
    )
    parser.add_argument("--calibration-new-tokens", type=int, default=96)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "bf16"),
        default="auto",
        help="Autocast dtype; auto selects bf16 when the device supports it.",
    )
    parser.add_argument(
        "--router-entropy-weight",
        type=float,
        default=0.02,
        help="Weight of KL(mean routing probabilities || uniform); 0 disables.",
    )
    parser.add_argument(
        "--router-aux-weight",
        type=float,
        default=0.01,
        help="Weight of the Switch-style load-balance loss on hard assignments.",
    )
    parser.add_argument(
        "--expert-diversity-weight",
        type=float,
        default=0.0,
        help="Weight of the pairwise expert-output diversity penalty; 0 disables.",
    )
    parser.add_argument(
        "--policy-label-smoothing",
        type=float,
        default=0.0,
        help="Symmetric label smoothing for the on-policy head phase; 0 disables.",
    )
    parser.add_argument(
        "--expert-init-scale",
        type=float,
        default=0.0,
        help=(
            "Std of the expert up-projection init; 0 keeps the legacy "
            "zero-output init under which the diversity penalty starts inert."
        ),
    )
    parser.add_argument(
        "--causal-control",
        action="store_true",
        help=(
            "Matched plain-LoRA control: every step trains the causal objective "
            "only, HLWM-module parameters are frozen (LoRA sidecars train), and "
            "the policy phase, calibration, and HLWM preflight are skipped. "
            "This is the decisive dense-shared-trunk baseline from the paper's "
            "required comparison set."
        ),
    )
    parser.add_argument(
        "--memory-headroom-fraction",
        type=float,
        default=0.92,
        help="Maximum fraction of device memory the preflight may reserve.",
    )
    parser.add_argument(
        "--policy-records",
        type=int,
        default=96,
        help="Generated train-anchor emissions for the on-policy head phase.",
    )
    parser.add_argument("--policy-epochs", type=int, default=400)
    parser.add_argument("--policy-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--policy-new-tokens", type=int, default=96)
    parser.add_argument(
        "--policy-min-valid-per-family",
        type=int,
        default=0,
        help=(
            "Keep collecting head-phase emissions until every anchor family "
            "has at least this many grader-verified valid positives (0 = off)."
        ),
    )
    parser.add_argument(
        "--policy-max-attempts",
        type=int,
        default=0,
        help="Hard cap on head-phase emission attempts (0 = --policy-records).",
    )
    parser.add_argument(
        "--canary-anchors",
        type=int,
        default=0,
        help=(
            "Stratified train anchors generated at every evaluation boundary "
            "as a report-only mid-training canary (0 = off)."
        ),
    )
    parser.add_argument("--canary-new-tokens", type=int, default=96)
    parser.add_argument(
        "--candidate-temperatures",
        type=str,
        default="0.0",
        help=(
            "Comma-separated decode temperatures for the candidate pool; the "
            "first entry should be 0.0 (greedy) so single-candidate behavior "
            "is always among the candidates."
        ),
    )
    parser.add_argument(
        "--synthesis-kl-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the KL-to-causal register anchor on the synthesis "
            "answer span (Version 8.0; 0 disables)."
        ),
    )
    parser.add_argument(
        "--prefix-gate-init",
        type=float,
        default=0.05,
        help=(
            "Initial value of the learned scalar prefix gate (tanh applied); "
            "small-positive avoids the zero-init silent phase."
        ),
    )
    parser.add_argument(
        "--premise-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the latent-only premise probe loss on masked rows "
            "(Version 9.0; 0 disables the probe module)."
        ),
    )
    parser.add_argument(
        "--gist-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Token budget of the gist attribution-control prefix (Version "
            "9.0; must equal the workspace prefix budget when nonzero; 0 "
            "disables the module)."
        ),
    )
    parser.add_argument(
        "--harvest-max-hours",
        type=float,
        default=0.0,
        help=(
            "Wall-clock cap on the on-policy emission harvest (Version 9.0 "
            "decode-runaway guard; 0 disables)."
        ),
    )
    parser.add_argument(
        "--conformal-target-coverage",
        type=float,
        default=0.35,
        help=(
            "Nominal expected-coverage floor for the split-conformal publish "
            "threshold (SSBC-inflated above the audit's 0.25 gate floor)."
        ),
    )
    parser.add_argument(
        "--workspace-memory-windows",
        type=int,
        default=0,
        help=(
            "Windowed canvas hidden-state tokens per lane appended to the "
            "synthesis prefix (the Version 6.0 latent read-out memory; 0 "
            "preserves the narrow Version 5.x interface)."
        ),
    )
    parser.add_argument("--skip-policy-phase", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-architecture-smoke", action="store_true")
    parser.add_argument("--skip-real-qwen-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeterministicStepBatchSampler(Sampler[List[int]]):
    """Map every global microstep to a reproducible shuffled dataset batch.

    A resumed run receives exactly the same example order as an uninterrupted
    run without serializing a partially consumed DataLoader iterator.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        start_step: int,
        total_steps: int,
        seed: int,
        overfit_steps: int = 0,
        overfit_examples: Optional[int] = None,
        priority_indices: Optional[List[int]] = None,
        priority_ratio: float = 0.0,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if not 0 <= start_step <= total_steps:
            raise ValueError("start_step must be between zero and total_steps")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_steps = total_steps
        self.seed = seed
        self.overfit_steps = max(0, int(overfit_steps))
        self.overfit_examples = (
            dataset_size
            if overfit_examples is None
            else min(dataset_size, max(1, int(overfit_examples)))
        )
        self.priority_indices = sorted(set(priority_indices or []))
        if any(index < 0 or index >= dataset_size for index in self.priority_indices):
            raise ValueError("priority index is outside the dataset")
        if not 0.0 <= priority_ratio <= 1.0:
            raise ValueError("priority ratio must be between zero and one")
        self.priority_ratio = float(priority_ratio)
        priority_set = set(self.priority_indices)
        self.regular_indices = [
            index for index in range(dataset_size) if index not in priority_set
        ]

    def __len__(self) -> int:
        return self.total_steps - self.start_step

    def __iter__(self) -> Iterator[List[int]]:
        cached_key: tuple[int, int, int] | None = None
        order: List[int] = []
        pool_orders: Dict[tuple[str, int], List[int]] = {}

        def pool_value(pool_name: str, pool: List[int], position: int, seed: int) -> int:
            if not pool:
                pool = list(range(self.dataset_size))
            epoch = position // len(pool)
            key = (pool_name, epoch)
            if key not in pool_orders:
                generator = torch.Generator().manual_seed(seed + epoch)
                permutation = torch.randperm(len(pool), generator=generator).tolist()
                pool_orders[key] = [pool[item] for item in permutation]
            return pool_orders[key][position % len(pool)]

        for step in range(self.start_step, self.total_steps):
            in_overfit = step < self.overfit_steps
            effective_size = self.overfit_examples if in_overfit else self.dataset_size
            phase_step = step if in_overfit else step - self.overfit_steps
            phase_seed = self.seed if in_overfit else self.seed + 1_000_003
            indices: List[int] = []
            for offset in range(self.batch_size):
                position = phase_step * self.batch_size + offset
                if (
                    not in_overfit
                    and self.priority_indices
                    and self.priority_ratio > 0.0
                ):
                    selector = ((position + 1) * 2_654_435_761 + self.seed) % 10_000
                    use_priority = selector < int(round(self.priority_ratio * 10_000))
                    pool = self.priority_indices if use_priority else self.regular_indices
                    indices.append(
                        pool_value(
                            "priority" if use_priority else "regular",
                            pool,
                            position,
                            phase_seed + (31_337 if use_priority else 73_331),
                        )
                    )
                    continue
                epoch = position // effective_size
                key = (effective_size, phase_seed, epoch)
                if key != cached_key:
                    generator = torch.Generator().manual_seed(phase_seed + epoch)
                    order = torch.randperm(
                        effective_size, generator=generator
                    ).tolist()
                    cached_key = key
                indices.append(order[position % effective_size])
            yield indices


def architecture_smoke(device: torch.device) -> Dict[str, float]:
    """Exercise corruption, reverse posterior, lanes, barriers and backward."""

    config = HLWMConfig.tiny(
        num_lanes=2,
        max_refinement_steps=2,
        diffusion_steps=3,
        commitment_threshold=0.0,
    )
    model = HLWMForConditionalGeneration(config).to(device)
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(3, config.vocab_size, (2, 9), device=device)
    attention_mask = torch.ones_like(input_ids)
    target_ids = torch.randint(3, config.vocab_size, (2, 7), device=device)
    lane_targets = target_ids[:, None, :].expand(-1, config.num_lanes, -1).contiguous()
    output = model(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        lane_target_ids=lane_targets,
        target_attention_mask=torch.ones_like(target_ids),
        lane_target_attention_mask=torch.ones_like(lane_targets),
        negative_target_ids=torch.roll(target_ids, shifts=1, dims=-1),
        adaptive_halt=False,
    )
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("architecture smoke produced a non-finite loss")
    output.loss.backward()
    required = {
        "root": model.root_adapter.up.weight.grad,
        "router": model.expert_bank.router.weight.grad,
        "diffusion_conditioner": model.timestep_conditioner.projection[0].weight.grad,
        "slow_state": model.slow_candidate.weight.grad,
        "verification": model.verification_head.weight.grad,
        "commitment": model.commitment_head[-1].weight.grad,
    }
    missing = [name for name, gradient in required.items() if gradient is None]
    if missing:
        raise RuntimeError("missing smoke gradients: %s" % ", ".join(missing))
    metrics = {
        "loss": float(output.loss.detach().cpu()),
        "committed_fraction": float(output.commit_mask.float().mean().cpu()),
        "route_count": float(output.route_indices.unique().numel()),
    }
    del model, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def find_split(data_dir: Path, split: str) -> Path:
    candidates = (
        data_dir / "master" / (split + ".jsonl"),
        data_dir / (split + ".jsonl"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find %s split under %s" % (split, data_dir))


def move_model_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def cosine_learning_rate(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / max(1, warmup)
    progress = float(step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def causal_loss(model: HLWMForConditionalGeneration, batch: Mapping[str, Any]) -> Tensor:
    logits = model(
        batch["causal_input_ids"],
        batch["causal_attention_mask"],
        mode="causal",
    )
    labels = batch["causal_labels"][:, 1:].reshape(-1)
    active = labels != -100
    if not bool(active.any()):
        raise ValueError("causal batch contains no answer tokens")
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1])[active].float(),
        labels[active],
    )


def phase_for_step(step: int, args: argparse.Namespace) -> str:
    if not args.progressive:
        return "local"
    if step < args.overfit_steps:
        return "overfit"
    if step < args.overfit_steps + args.local_steps:
        return "local"
    return "joint"


def local_denoise_loss(
    model: HLWMForConditionalGeneration,
    batch: Mapping[str, Any],
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Dict[str, float]]:
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        mode="local_denoise",
        lane_target_ids=batch["lane_target_ids"],
        lane_target_attention_mask=batch["lane_target_attention_mask"],
        lane_brief_ids=batch["lane_brief_ids"],
        lane_brief_attention_mask=batch["lane_brief_attention_mask"],
        generator=generator,
    )
    metrics = {
        "denoise": float(output.denoise_loss.detach().float().cpu()),
        "brief_alignment": float(output.brief_loss.detach().float().cpu()),
        "router_balance": float(output.router_loss.detach().float().cpu()),
        "router_entropy": float(output.router_entropy_loss.detach().float().cpu()),
        "expert_diversity": float(output.expert_diversity_loss.detach().float().cpu()),
        "lane_diversity": float(output.lane_diversity_loss.detach().float().cpu()),
        "route_unique": float(output.route_indices.unique().numel()),
        "transition_count": float(output.transition_count),
        "corruption_fraction": float(
            (output.corrupted_ids != batch["lane_target_ids"])
            .float()
            .mean()
            .detach()
            .cpu()
        ),
    }
    return output.loss, metrics


def masked_binary_cross_entropy(
    logits: Tensor, targets: Tensor, eligible: Tensor
) -> Tensor:
    eligible = eligible.to(torch.bool)
    while eligible.ndim < logits.ndim:
        eligible = eligible.unsqueeze(-1)
    eligible = eligible.expand_as(logits)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if not bool(eligible.any()):
        return losses.new_zeros(())
    weights = eligible.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def hlwm_loss(
    model: HLWMForConditionalGeneration,
    batch: Mapping[str, Any],
    *,
    prefix_source: str = "workspace",
) -> tuple[Tensor, Dict[str, float]]:
    # Version 9.0 information asymmetry: a batch whose rows are all flagged
    # ``masked`` conditions the answer channel on the withheld-premise prompt,
    # skips the KL anchor (its causal reference cannot answer there), builds
    # its prefix from a pure-noise full reverse pass (the audit path; the
    # q_sample path leaves ~39% verbatim target tokens in the canvas at t=T),
    # and adds the latent-only premise probe loss. Production batch size is
    # one, so per-batch is per-row.
    masked_flags = batch.get("masked_rows")
    masked_batch = bool(masked_flags is not None and bool(masked_flags.all()))
    masked_kwargs: Dict[str, Any] = {}
    if masked_batch:
        lane_shape = batch["lane_target_ids"].shape
        masked_kwargs = {
            "answer_input_ids": batch["answer_input_ids"],
            "answer_attention_mask": batch["answer_attention_mask"],
            "skip_synthesis_kl": True,
            "premise_ids": batch.get("premise_ids"),
            "premise_attention_mask": batch.get("premise_attention_mask"),
            "premise_negative_ids": batch.get("premise_negative_ids"),
            "noisy_lane_ids": model.diffusion.sample_noise(
                tuple(lane_shape), batch["lane_target_ids"].device, None
            ),
            "full_reverse_schedule": True,
            "prefix_source": prefix_source,
        }
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        target_ids=batch["target_ids"],
        lane_target_ids=batch["lane_target_ids"],
        target_attention_mask=batch["target_attention_mask"],
        lane_target_attention_mask=batch["lane_target_attention_mask"],
        negative_target_ids=batch["negative_target_ids"],
        negative_target_attention_mask=batch["negative_target_attention_mask"],
        commitment_supervision_mask=batch["policy_supervision_mask"],
        lane_brief_ids=batch["lane_brief_ids"],
        lane_brief_attention_mask=batch["lane_brief_attention_mask"],
        adaptive_halt=False,
        sample_reverse=False,
        **masked_kwargs,
    )
    if output.loss is None:
        raise RuntimeError("HLWM forward returned no supervised loss")

    # Weak Reasoning9000 policy labels and clean/corrupt commitment pairing are
    # disabled unless a record was independently adjudicated or belongs to the
    # narrow programmatically verified behavior-anchor set.
    eligible = batch["policy_supervision_mask"]
    verifier_prediction = output.verification_logits.mean(dim=-1)
    external_verification = masked_binary_cross_entropy(
        verifier_prediction, batch["verification_targets"], eligible
    )
    external_halt = masked_binary_cross_entropy(
        output.halt_logits[..., -1], batch["halt_targets"], eligible
    )
    external_commitment = masked_binary_cross_entropy(
        output.commitment_logits[:, 0], batch["commitment_targets"], eligible
    )
    external_risk = masked_binary_cross_entropy(
        output.commitment_logits[:, 1], batch["risk_targets"], eligible
    )
    external_candidate_verifier = masked_binary_cross_entropy(
        output.commitment_logits[:, 2], batch["risk_targets"], eligible
    )
    total = (
        output.loss
        + 0.10 * external_verification
        + 0.05 * external_halt
        + 0.10 * external_commitment
        + 0.10 * external_risk
        + 0.10 * external_candidate_verifier
    )
    metrics = {name: float(value.detach().float().cpu()) for name, value in output.loss_components.items()}
    metrics.update(
        {
            "masked_row": float(masked_batch),
            "gist_batch": float(prefix_source == "gist"),
            "external_verification": float(external_verification.detach().float().cpu()),
            "external_halt": float(external_halt.detach().float().cpu()),
            "external_commitment": float(external_commitment.detach().float().cpu()),
            "external_risk": float(external_risk.detach().float().cpu()),
            "external_candidate_verifier": float(
                external_candidate_verifier.detach().float().cpu()
            ),
            "commit_rate": float(output.commit_mask.float().mean().detach().cpu()),
            "commit_probability": float(
                torch.sigmoid(output.commitment_logits[:, 0]).mean().detach().float().cpu()
            ),
            "risk_probability": float(
                torch.sigmoid(output.commitment_logits[:, 1]).mean().detach().float().cpu()
            ),
            "candidate_verifier_error_probability": float(
                torch.sigmoid(output.commitment_logits[:, 2])
                .mean()
                .detach()
                .float()
                .cpu()
            ),
            "negative_commit_probability": float(
                torch.sigmoid(output.negative_commitment_logits[:, 0])
                .mean()
                .detach()
                .float()
                .cpu()
            )
            if output.negative_commitment_logits is not None
            else 0.0,
            "negative_risk_probability": float(
                torch.sigmoid(output.negative_commitment_logits[:, 1])
                .mean()
                .detach()
                .float()
                .cpu()
            )
            if output.negative_commitment_logits is not None
            else 0.0,
            "negative_candidate_verifier_error_probability": float(
                torch.sigmoid(output.negative_commitment_logits[:, 2])
                .mean()
                .detach()
                .float()
                .cpu()
            )
            if output.negative_commitment_logits is not None
            else 0.0,
            "halt_probability": float(
                torch.sigmoid(output.halt_logits).mean().detach().float().cpu()
            ),
            "route_unique": float(output.route_indices.unique().numel()),
            "corruption_fraction": float(
                (output.corrupted_ids != batch["lane_target_ids"])
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "adjudicated_fraction": float(eligible.float().mean().detach().cpu()),
        }
    )
    route_load = F.one_hot(
        output.route_indices, model.config.num_experts
    ).float().reshape(-1, model.config.num_experts).mean(dim=0)
    for expert_index, load in enumerate(route_load):
        metrics["route_load_%d" % expert_index] = float(load.detach().cpu())
    if output.lane_summaries.shape[1] > 1:
        normalized_lanes = F.normalize(output.lane_summaries.float(), dim=-1)
        similarity = torch.matmul(normalized_lanes, normalized_lanes.transpose(1, 2))
        lane_count = similarity.shape[1]
        off_diagonal = ~torch.eye(
            lane_count, dtype=torch.bool, device=similarity.device
        )[None, :, :]
        metrics["lane_summary_cosine"] = float(
            similarity.masked_select(off_diagonal).mean().detach().cpu()
        )
    return total, metrics


@torch.no_grad()
def evaluate(
    model: HLWMForConditionalGeneration,
    loader: DataLoader,
    device: torch.device,
    phase: str = "joint",
    max_batches: int = 4,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_model_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            if phase in ("overfit", "local"):
                loss, metrics = local_denoise_loss(model, batch)
            else:
                loss, metrics = hlwm_loss(model, batch)
        totals["loss"] = totals.get("loss", 0.0) + float(loss.float().cpu())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
        if count >= max_batches:
            break
    if was_training:
        model.train()
    return {name: value / max(1, count) for name, value in totals.items()}


# Kept identical to evaluate_checkpoint.TURN_SCAFFOLD: a turn marker opening a line,
# anywhere in the answer. Version 6.0 emitted its scaffold mid-answer, so a
# start-of-string test scored those rows clean.
TURN_SCAFFOLD = re.compile(r"(?mi)^[\s>*#-]*(human|assistant|user|system)\s*:", re.MULTILINE)


def candidate_has_no_prompt_leak(candidate: str) -> bool:
    normalized = " ".join(candidate.lower().split())
    if TURN_SCAFFOLD.search(candidate):
        return False
    return not normalized.startswith(
        ("human:", "assistant:", "### instruction", "you are ")
    ) and not any(
        marker in normalized
        for marker in ("<|user_request|>", "<|hlwm_", "### response requirements")
    )


def _threshold_grid(values: List[float], maximum: int = 48) -> List[float]:
    """Return deterministic observed-value thresholds without an exhaustive grid."""

    unique = sorted(set([0.0, 1.0] + [min(1.0, max(0.0, value)) for value in values]))
    candidates = sorted(
        set(unique + [(left + right) / 2.0 for left, right in zip(unique, unique[1:])])
    )
    if len(candidates) <= maximum:
        return candidates
    indices = [round(index * (len(candidates) - 1) / (maximum - 1)) for index in range(maximum)]
    return [candidates[index] for index in sorted(set(indices))]


def _quantile(values: List[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * min(1.0, max(0.0, fraction)))
    return ordered[index]


def fit_policy_thresholds(
    positive_commit: List[float],
    positive_risk: List[float],
    negative_commit: List[float],
    negative_risk: List[float],
    positive_verifier: List[float],
    negative_verifier: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Fit the complete three-head gate to labeled emitted candidates.

    Positive rows are semantically correct validation emissions. Negative rows
    are incorrect emissions or mechanically corrupted snapshots. All three
    thresholds are searched jointly because a private-token-error quantile is
    not a calibrated proxy for final-answer correctness.
    """

    if negative_verifier is None:
        negative_verifier = list(positive_verifier)
    positive_lengths = {
        len(positive_commit), len(positive_risk), len(positive_verifier)
    }
    negative_lengths = {
        len(negative_commit), len(negative_risk), len(negative_verifier)
    }
    if (
        len(positive_lengths) != 1
        or len(negative_lengths) != 1
        or not positive_commit
        or not negative_commit
    ):
        raise ValueError("positive and negative calibration vectors must be aligned")

    commit_grid = _threshold_grid(positive_commit + negative_commit, maximum=36)
    risk_grid = _threshold_grid(positive_risk + negative_risk, maximum=36)
    verifier_grid = _threshold_grid(
        positive_verifier + negative_verifier, maximum=36
    )
    best: Optional[Dict[str, float]] = None
    for commit_threshold in commit_grid:
        for risk_threshold in risk_grid:
            for verifier_threshold in verifier_grid:
                positive_accept = sum(
                    commit >= commit_threshold
                    and risk <= risk_threshold
                    and error <= verifier_threshold
                    for commit, risk, error in zip(
                        positive_commit, positive_risk, positive_verifier
                    )
                ) / len(positive_commit)
                negative_accept = sum(
                    commit >= commit_threshold
                    and risk <= risk_threshold
                    and error <= verifier_threshold
                    for commit, risk, error in zip(
                        negative_commit, negative_risk, negative_verifier
                    )
                ) / len(negative_commit)
                negative_reject = 1.0 - negative_accept
                balanced_accuracy = 0.5 * (positive_accept + negative_reject)
                candidate = {
                    "commitment_threshold": commit_threshold,
                    "risk_threshold": risk_threshold,
                    "verifier_error_threshold": verifier_threshold,
                    "positive_accept_rate": positive_accept,
                    "negative_reject_rate": negative_reject,
                    # Compatibility aliases used by existing result readers.
                    "clean_accept_rate": positive_accept,
                    "corrupt_reject_rate": negative_reject,
                    "balanced_accuracy": balanced_accuracy,
                }
                key = (
                    positive_accept >= 0.70,
                    balanced_accuracy,
                    min(positive_accept, negative_reject),
                    negative_reject,
                    positive_accept,
                )
                if best is None or key > (
                    best["positive_accept_rate"] >= 0.70,
                    best["balanced_accuracy"],
                    min(best["positive_accept_rate"], best["negative_reject_rate"]),
                    best["negative_reject_rate"],
                    best["positive_accept_rate"],
                ):
                    best = candidate
    assert best is not None
    return best


def fit_publish_combiner(
    candidate_probabilities: Tensor, labels: Tensor
) -> Dict[str, Any]:
    """Fit the scalar publication rule on generated validation candidates.

    A logistic regression over the candidate features (three head
    probabilities plus, in Version 8.0, ``exp(causal mean logprob)`` — the
    fourth feature nests the max-logprob baseline in the rule — plus, in
    Version 9.0, the within-pool agreement fraction, so plurality voting is
    nested as the fifth feature) replaces the joint three-threshold sweep
    whose fitted commitment threshold degenerated to a rail in six separate
    seed-runs.  Class-balanced so the
    corrupt/invalid majority cannot buy accuracy by rejecting everything.
    The publication threshold is NOT chosen here: Version 8.0 selects it by
    a split-conformal order statistic (``conformal_publish_threshold``) so
    calibration can never strangle coverage; the 0.5-threshold statistics
    reported below are diagnostics only.
    """

    features = candidate_probabilities.detach().float().cpu()
    targets = labels.detach().float().cpu()
    if (
        features.ndim != 2
        or features.shape[-1] not in (3, 4, 5)
        or len(features) != len(targets)
    ):
        raise ValueError("combiner expects [N, 3|4|5] features and N labels")
    feature_count = int(features.shape[-1])
    positive_count = float(targets.sum())
    negative_count = float(len(targets)) - positive_count
    if positive_count < 8 or negative_count < 8:
        return {"fitted": False, "reason": "need eight candidates per class"}
    total = positive_count + negative_count
    sample_weights = torch.where(
        targets > 0.5,
        torch.full_like(targets, total / (2.0 * positive_count)),
        torch.full_like(targets, total / (2.0 * negative_count)),
    )
    with torch.enable_grad():
        weights = torch.zeros(feature_count, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        # L2 on the weights keeps the logistic calibrated (Platt scaling)
        # instead of saturating scores at the rails on separable data.
        optimizer = torch.optim.Adam([weights, bias], lr=0.1, weight_decay=1.0e-2)
        for _ in range(400):
            optimizer.zero_grad()
            logits = features @ weights + bias
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=sample_weights
            )
            loss.backward()
            optimizer.step()
    fitted_weights = weights.detach()
    fitted_bias = float(bias.detach())
    scores = torch.sigmoid(features @ fitted_weights + fitted_bias)
    accepted = scores >= 0.5
    accept_rate = float(accepted[targets > 0.5].float().mean())
    reject_rate = float((~accepted)[targets <= 0.5].float().mean())
    return {
        "fitted": True,
        "weights": [float(value) for value in fitted_weights.tolist()],
        "bias": fitted_bias,
        "feature_count": feature_count,
        "positive_accept_rate": accept_rate,
        "negative_reject_rate": reject_rate,
        "balanced_accuracy": 0.5 * (accept_rate + reject_rate),
    }


def conformal_publish_threshold(
    anchor_scores: Sequence[float],
    *,
    target_coverage: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """Split-conformal threshold with a structural coverage floor.

    Given one publish score per held-out calibration anchor, choose tau as
    the k-th smallest score with k = floor((n+1) * (1 - c)).  By
    exchangeability the expected audit coverage P(score >= tau) is at least
    c — the floor is an order statistic, not a fitted margin, so no fit can
    strangle coverage (the seed-29 pathology of Study 8).  If more than 20%
    of scores tie at the chosen order statistic, a seeded uniform jitter of
    1e-7 breaks the mass (the preregistered randomized tie rule).
    """

    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be inside (0, 1)")
    scores = [float(value) for value in anchor_scores]
    count = len(scores)
    if count < 16:
        return {"fitted": False, "reason": "need sixteen calibration anchors"}
    k = int(math.floor((count + 1) * (1.0 - target_coverage)))
    k = min(max(k, 1), count)
    ordered = sorted(scores)
    threshold = ordered[k - 1]
    tie_fraction = sum(
        1 for value in scores if abs(value - threshold) <= 1.0e-12
    ) / count
    jittered = False
    if tie_fraction > 0.20:
        generator = torch.Generator().manual_seed(seed)
        noise = (torch.rand(count, generator=generator) * 2.0 - 1.0) * 1.0e-7
        ordered = sorted(value + float(delta) for value, delta in zip(scores, noise))
        threshold = ordered[k - 1]
        jittered = True
    empirical_coverage = sum(1 for value in scores if value >= threshold) / count
    return {
        "fitted": True,
        "threshold": float(threshold),
        "calibration_anchors": count,
        "order_statistic_k": k,
        "target_coverage": float(target_coverage),
        "guaranteed_expected_coverage": (count + 1 - k) / (count + 1),
        "calibration_empirical_coverage": empirical_coverage,
        "tie_fraction": tie_fraction,
        "tie_jitter_applied": jittered,
    }


@torch.no_grad()
def calibrate_commitment_policy(
    model: HLWMForConditionalGeneration,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    *,
    max_records: int,
    max_new_tokens: int,
    seed: int,
    candidate_temperatures: Sequence[float] = (0.0,),
    target_coverage: float = 0.35,
) -> Dict[str, Any]:
    """Fit the scalar publication rule on generated validation candidates.

    Version 8.0: every validation anchor runs the audit's exact inference
    path — causal-channel candidates, workspace-head scores plus the causal
    mean-logprob feature per candidate.  Anchors alternate into split A
    (combiner weights fitted on candidate rows) and split B (publication
    threshold chosen as a split-conformal order statistic over one selected
    score per anchor, guaranteeing an expected-coverage floor).  Legacy
    triple thresholds are still fitted for cross-study comparability, and
    candidate selection quality (greedy / selected / oracle validity) is
    measured on the same anchors.  Test rows stay untouched.
    """

    if max_records <= 0 or max_new_tokens <= 0:
        raise ValueError("calibration record and token counts must be positive")
    was_training = model.training
    model.eval()
    positive_commit: List[float] = []
    positive_risk: List[float] = []
    positive_verifier: List[float] = []
    negative_commit: List[float] = []
    negative_risk: List[float] = []
    negative_verifier: List[float] = []
    probability_rows: List[List[float]] = []
    row_labels: List[float] = []
    row_splits: List[str] = []
    anchor_candidate_rows: List[List[int]] = []
    anchor_candidate_valid: List[List[bool]] = []
    anchor_splits: List[str] = []
    emitted = 0
    candidate_count = 0
    valid_emissions = 0
    invalid_emissions = 0
    grader_totals: Dict[str, int] = {}
    grader_correct: Dict[str, int] = {}
    for batch in loader:
        if emitted >= max_records:
            break
        batch = move_model_batch(batch, device)
        eligible = batch["policy_supervision_mask"] & batch["is_behavior_anchor"]
        if not bool(eligible.any()):
            continue
        for row_index in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            if emitted >= max_records:
                break
            generator = torch.Generator(device=device).manual_seed(seed + emitted)
            # Version 9.0 (RC-2 repair): calibrate on the unpadded audit
            # surface, not the training collator's fixed-length padding.
            prompt_ids = encode_preserving_ends(
                tokenizer,
                str(batch["public_prompts"][row_index]),
                int(batch["input_ids"].shape[-1]),
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                generation = model.generate_hlwm_nbest(
                    input_ids,
                    attention_mask,
                    canvas_length=int(batch["target_ids"].shape[-1]),
                    max_new_tokens=max_new_tokens,
                    candidate_temperatures=candidate_temperatures,
                    generator=generator,
                )
            answer_spec = batch["answer_specs"][row_index]
            # Version 9.0 three-way split, 3:2:3 by arrival order: combiner
            # weights on A, score-form selection on B1, conformal threshold on
            # B2 — disjoint, so no adopt-if-wins contaminates the threshold.
            split_slot = emitted % 8
            anchor_split = "A" if split_slot < 3 else ("B1" if split_slot < 5 else "B2")
            candidate_rows: List[int] = []
            candidate_valid: List[bool] = []
            for candidate_index, candidate_ids in enumerate(generation.candidate_ids):
                candidate = tokenizer.decode(
                    candidate_ids[0].detach().cpu(), skip_special_tokens=True
                ).strip()
                semantic = grade_semantic_answer(candidate, answer_spec)
                valid = bool(
                    semantic["correct"] and candidate_has_no_prompt_leak(candidate)
                )
                grader = str(semantic["grader"])
                grader_totals[grader] = grader_totals.get(grader, 0) + 1
                grader_correct[grader] = grader_correct.get(grader, 0) + int(valid)
                agreement_values = generation.candidate_agreements or [
                    1.0 / max(1, len(generation.candidate_ids))
                ] * len(generation.candidate_ids)
                row = [
                    float(generation.commit_probabilities[candidate_index]),
                    float(generation.risk_probabilities[candidate_index]),
                    float(generation.verifier_error_probabilities[candidate_index]),
                    float(
                        math.exp(generation.candidate_mean_logprobs[candidate_index])
                    ),
                    float(agreement_values[candidate_index]),
                ]
                candidate_rows.append(len(probability_rows))
                candidate_valid.append(valid)
                probability_rows.append(row)
                row_labels.append(float(valid))
                row_splits.append(anchor_split)
                candidate_count += 1
                if valid:
                    valid_emissions += 1
                    positive_commit.append(row[0])
                    positive_risk.append(row[1])
                    positive_verifier.append(row[2])
                else:
                    invalid_emissions += 1
                    negative_commit.append(row[0])
                    negative_risk.append(row[1])
                    negative_verifier.append(row[2])
            anchor_candidate_rows.append(candidate_rows)
            anchor_candidate_valid.append(candidate_valid)
            anchor_splits.append(anchor_split)

            greedy_ids = generation.candidate_ids[0]
            if greedy_ids.shape[1] > 1:
                corrupt_ids = torch.roll(greedy_ids, shifts=1, dims=-1)
            else:
                corrupt_ids = (greedy_ids + 1) % model.config.vocab_size
            corrupt_mask = torch.ones_like(corrupt_ids)
            private_error = torch.sigmoid(
                generation.workspace.verification_logits
            ).mean(dim=-1)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                corrupt_logits, corrupt_hidden = model._teacher_force_workspace(
                    input_ids,
                    attention_mask,
                    generation.workspace.workspace_prefix,
                    corrupt_ids,
                    corrupt_mask,
                )
                corrupt_policy = model._score_candidate(
                    corrupt_hidden,
                    corrupt_logits,
                    corrupt_mask,
                    generation.workspace.global_state,
                    private_error,
                )
            corrupt_scores = torch.sigmoid(corrupt_policy[0].float()).cpu().tolist()
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                corrupt_logprob = model.causal_answer_mean_logprob(
                    input_ids, attention_mask, corrupt_ids, corrupt_mask
                )
            negative_commit.append(float(corrupt_scores[0]))
            negative_risk.append(float(corrupt_scores[1]))
            negative_verifier.append(float(corrupt_scores[2]))
            probability_rows.append(
                [float(value) for value in corrupt_scores]
                + [
                    float(corrupt_logprob[0].exp().cpu()),
                    # A corrupt snapshot is outside the pool: it agrees only
                    # with itself.
                    1.0 / max(1, len(generation.candidate_ids)),
                ]
            )
            row_labels.append(0.0)
            row_splits.append(anchor_split)
            emitted += 1

    if was_training:
        model.train()
    if emitted < 8:
        raise RuntimeError(
            "need at least eight generated validation anchors for calibration; found %d"
            % emitted
        )

    enough_classes = len(positive_commit) >= 8 and len(negative_commit) >= 8
    if enough_classes:
        best = fit_policy_thresholds(
            positive_commit,
            positive_risk,
            negative_commit,
            negative_risk,
            positive_verifier,
            negative_verifier,
        )
    else:
        # A model that produces too few valid calibration emissions must not
        # manufacture apparent coverage from teacher-forced references.
        best = {
            "commitment_threshold": 1.0,
            "risk_threshold": 0.0,
            "verifier_error_threshold": 0.0,
            "positive_accept_rate": 0.0,
            "negative_reject_rate": 1.0,
            "clean_accept_rate": 0.0,
            "corrupt_reject_rate": 1.0,
            "balanced_accuracy": 0.5,
        }
    model.config.commitment_threshold = float(best["commitment_threshold"])
    model.config.risk_threshold = float(best["risk_threshold"])
    model.config.verifier_error_threshold = float(best["verifier_error_threshold"])

    # Split discipline (Version 9.0): weights on split A only, the score FORM
    # (four features vs five) selected on split B1 only, the threshold on
    # split B2 only. Fitting or selecting on the threshold split would break
    # the exchangeability argument behind the conformal coverage floor.
    fit_rows = [
        row for row, split in zip(probability_rows, row_splits) if split == "A"
    ]
    fit_labels = [
        label for label, split in zip(row_labels, row_splits) if split == "A"
    ]
    candidate_forms: Dict[int, Dict[str, Any]] = {}
    for feature_count in (4, 5):
        candidate_forms[feature_count] = fit_publish_combiner(
            torch.tensor(
                [row[:feature_count] for row in fit_rows], dtype=torch.float32
            ),
            torch.tensor(fit_labels, dtype=torch.float32),
        )

    def form_score(row: List[float], form: Mapping[str, Any]) -> float:
        if form.get("fitted"):
            logit = sum(
                weight * probability
                for weight, probability in zip(form["weights"], row)
            ) + form["bias"]
            return 1.0 / (1.0 + math.exp(-logit))
        return row[0] * (1.0 - row[1]) * (1.0 - row[2])

    def partial_augrc(pairs: List[tuple[float, bool]], low: float, high: float) -> float:
        """Area under the generalized risk-coverage curve restricted to a
        coverage band. Lower is better; degenerate inputs return 1.0."""

        if not pairs:
            return 1.0
        ordered = sorted(pairs, key=lambda item: -item[0])
        total = len(ordered)
        area = 0.0
        weight = 0.0
        errors = 0
        for position, (_, correct) in enumerate(ordered, start=1):
            errors += int(not correct)
            coverage = position / total
            if low <= coverage <= high:
                area += errors / total
                weight += 1.0
        return area / weight if weight else 1.0

    def anchor_selection_pairs(
        form: Mapping[str, Any], split_name: str
    ) -> List[tuple[float, bool]]:
        pairs: List[tuple[float, bool]] = []
        for rows, valids, split in zip(
            anchor_candidate_rows, anchor_candidate_valid, anchor_splits
        ):
            if split != split_name:
                continue
            scores = [form_score(probability_rows[index], form) for index in rows]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            pairs.append((scores[best_index], bool(valids[best_index])))
        return pairs

    form_selection: Dict[str, Any] = {"selected_feature_count": None}
    fitted_forms = {
        count: form for count, form in candidate_forms.items() if form.get("fitted")
    }
    if fitted_forms:
        b1_augrc = {
            count: partial_augrc(anchor_selection_pairs(form, "B1"), 0.05, 0.50)
            for count, form in fitted_forms.items()
        }
        # Lower partial AUGRC wins; ties prefer the simpler four-feature form.
        selected_count = min(sorted(b1_augrc), key=lambda count: (b1_augrc[count], count))
        combiner = fitted_forms[selected_count]
        form_selection = {
            "selected_feature_count": selected_count,
            "b1_partial_augrc": {str(k): float(v) for k, v in b1_augrc.items()},
            "selection_split": "B1",
        }
    else:
        combiner = candidate_forms[4]

    def scalar_score(row: List[float]) -> float:
        return form_score(row, combiner)

    conformal: Dict[str, Any] = {"fitted": False, "reason": "combiner not fitted"}
    if combiner.get("fitted"):
        # One score per held-out anchor: the score of the candidate the rule
        # itself selects (candidates within an anchor are dependent rows).
        holdout_scores = [
            max(scalar_score(probability_rows[index]) for index in rows)
            for rows, split in zip(anchor_candidate_rows, anchor_splits)
            if split == "B2"
        ]
        conformal = conformal_publish_threshold(
            holdout_scores, target_coverage=target_coverage, seed=seed
        )
    if combiner.get("fitted") and conformal.get("fitted"):
        model.config.publish_weights = tuple(combiner["weights"])
        model.config.publish_bias = float(combiner["bias"])
        model.config.publish_threshold = float(conformal["threshold"])
    else:
        # Without a fitted rule the triple thresholds above remain the
        # publication authority; selection falls back to the monotone blend.
        # There is deliberately no midpoint fallback — that road produced the
        # Study 8 seed-29 coverage strangulation.
        model.config.publish_weights = None

    greedy_valid = 0
    selected_valid = 0
    any_valid = 0
    for rows, valids in zip(anchor_candidate_rows, anchor_candidate_valid):
        greedy_valid += int(valids[0])
        any_valid += int(any(valids))
        scores = [scalar_score(probability_rows[index]) for index in rows]
        selected_valid += int(
            valids[max(range(len(scores)), key=scores.__getitem__)]
        )
    anchors = max(1, len(anchor_candidate_rows))

    accept_rate = combiner.get("positive_accept_rate", best["positive_accept_rate"])
    reject_rate = combiner.get("negative_reject_rate", best["negative_reject_rate"])
    balanced = combiner.get("balanced_accuracy", best["balanced_accuracy"])
    rule_fitted = bool(combiner.get("fitted") and conformal.get("fitted"))
    threshold_ok = bool(
        rule_fitted
        and 0.02 <= float(conformal["threshold"]) <= 0.98
        and (
            float(conformal.get("tie_fraction", 0.0)) <= 0.20
            or bool(conformal.get("tie_jitter_applied"))
        )
    )
    return {
        "method": "validation_generated_nbest_conformal_publish_v5",
        "form_selection": form_selection,
        "unpadded_prompt_surface": True,
        "source_split": "validation",
        "verified_anchor_records": emitted,
        "generated_candidate_records": candidate_count,
        "candidate_temperatures": [float(value) for value in candidate_temperatures],
        "valid_generated_candidates": valid_emissions,
        "invalid_generated_candidates": invalid_emissions,
        "positive_policy_records": len(positive_commit),
        "negative_policy_records": len(negative_commit),
        "generated_candidate_semantic_accuracy": valid_emissions
        / max(1, candidate_count),
        "accuracy_by_grader": {
            grader: grader_correct.get(grader, 0) / max(1, total)
            for grader, total in sorted(grader_totals.items())
        },
        "test_split_used": False,
        "split_sizes": {
            "fit_anchors": sum(1 for split in anchor_splits if split == "A"),
            "form_anchors": sum(1 for split in anchor_splits if split == "B1"),
            "holdout_anchors": sum(1 for split in anchor_splits if split == "B2"),
        },
        "thresholds": {
            "commitment": model.config.commitment_threshold,
            "risk": model.config.risk_threshold,
            "verifier_error": model.config.verifier_error_threshold,
        },
        "publish_rule": {
            "fitted": rule_fitted,
            "weights": combiner.get("weights"),
            "bias": combiner.get("bias"),
            "threshold": conformal.get("threshold"),
            "feature_count": combiner.get("feature_count"),
        },
        "conformal": conformal,
        "selection": {
            "anchors": anchors,
            "greedy_valid_rate": greedy_valid / anchors,
            "selected_valid_rate": selected_valid / anchors,
            "oracle_any_valid_rate": any_valid / anchors,
            "selection_gain": (selected_valid - greedy_valid) / anchors,
        },
        "positive_accept_rate": accept_rate,
        "negative_reject_rate": reject_rate,
        "clean_accept_rate": best["clean_accept_rate"],
        "corrupt_reject_rate": best["corrupt_reject_rate"],
        "balanced_accuracy": balanced,
        "passed_research_gate": bool(
            rule_fitted
            and accept_rate >= 0.70
            and reject_rate >= 0.70
            and balanced >= 0.70
        ),
        "publish_threshold_non_degenerate": threshold_ok,
        "commitment_threshold_non_degenerate": threshold_ok,
    }


@torch.no_grad()
def balanced_anchor_order(dataset: Reasoning9000Dataset, seed: int) -> List[int]:
    """Round-robin anchor positions across grader families, deterministically.

    Study 5's failing commit decisions were confident false rejections
    concentrated in one anchor family: a plain random draw of head-phase
    records can under-represent a family enough that the heads learn it as
    always-negative.  Interleaving families keeps every grader class present
    in the head-phase data at near-equal rates.
    """

    buckets: Dict[str, List[int]] = {}
    for position, index in enumerate(dataset.anchor_indices):
        family = str(
            (dataset.rows[index].get("answer_spec") or {}).get("type", "none")
        )
        buckets.setdefault(family, []).append(position)
    generator = torch.Generator().manual_seed(seed)
    for family, positions in buckets.items():
        permutation = torch.randperm(len(positions), generator=generator).tolist()
        buckets[family] = [positions[index] for index in permutation]
    families = sorted(buckets)
    order: List[int] = []
    cursor = 0
    while any(buckets.values()):
        family = families[cursor % len(families)]
        cursor += 1
        if buckets[family]:
            order.append(buckets[family].pop(0))
    return order


@torch.no_grad()
def collect_on_policy_policy_examples(
    model: HLWMForConditionalGeneration,
    dataset: Reasoning9000Dataset,
    collator: HLWMCollator,
    tokenizer: Any,
    device: torch.device,
    *,
    max_records: int,
    max_new_tokens: int,
    seed: int,
    min_valid_per_family: int = 0,
    max_attempts: int = 0,
    candidate_temperatures: Sequence[float] = (0.0,),
    max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate train-anchor emissions and cache graded policy features.

    Version 6.0: each anchor runs one private workspace and decodes
    ``len(candidate_temperatures)`` candidates from it (verified fan-in's
    training distribution).  Every candidate is graded and cached, sampled
    candidates recover valid positives on prompts where greedy fails, and
    within-anchor ranking pairs supervise exactly the selection judgment the
    fan-in makes at inference.  ``emitted`` counts candidates, so the
    validity floor and attempt cap operate on candidate counts.

    Version 5.6.2: balance must bind after validity filtering.  Study 6's
    request-level round-robin left valid positives at a 1:8 minority because
    two-thirds of emissions failed their graders; this revision keeps
    collecting past ``max_records`` until every anchor family holds at least
    ``min_valid_per_family`` grader-verified valid positives, or the
    ``max_attempts`` cap is reached.  A family that cannot reach the floor is
    reported (``validity_floor_met`` false) rather than aborting the session.

    Every emission is produced by the current frozen decision path, graded by
    the deterministic semantic graders, and converted into the exact
    commitment-head input features.  Three negative sources are used: invalid
    emissions themselves, the anchor's fluent wrong-value answer (semantic
    negative), and a mechanically corrupted snapshot of the emission.  Only
    train-split behavior anchors are consumed; validation stays reserved for
    threshold fitting and test remains untouched.
    """

    if max_records <= 0 or max_new_tokens <= 0:
        raise ValueError("policy record and token counts must be positive")
    was_training = model.training
    model.eval()
    anchor_indices = list(dataset.anchor_indices)
    order = balanced_anchor_order(dataset, seed)
    family_of_position = [
        str(
            (dataset.rows[anchor_indices[position]].get("answer_spec") or {}).get(
                "type", "none"
            )
        )
        for position in range(len(anchor_indices))
    ]
    families = sorted(set(family_of_position))
    family_valid: Dict[str, int] = {family: 0 for family in families}
    family_attempts: Dict[str, int] = {family: 0 for family in families}
    attempt_cap = max_attempts if max_attempts > 0 else max_records

    def floor_met() -> bool:
        return min_valid_per_family <= 0 or all(
            family_valid[family] >= min_valid_per_family for family in families
        )

    positive_features: List[Tensor] = []
    negative_features: List[Tensor] = []
    negative_kinds: List[str] = []
    ranking_pairs: List[tuple[int, int]] = []
    duplicate_candidates = 0
    anchors_consumed = 0
    harvest_started = time.time()
    wall_clock_capped = False
    padded_surface_rows = 0
    emitted = 0
    valid_emissions = 0
    invalid_emissions = 0
    semantic_negatives = 0
    grader_totals: Dict[str, int] = {}
    grader_correct: Dict[str, int] = {}

    def candidate_features(
        input_ids: Tensor,
        attention_mask: Tensor,
        workspace: Any,
        ids: Tensor,
        mask: Tensor,
        verification_error: Tensor,
    ) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits, hidden = model._teacher_force_workspace(
                input_ids, attention_mask, workspace.workspace_prefix, ids, mask
            )
            features = model.candidate_policy_features(
                hidden, logits, mask, workspace.global_state, verification_error
            )
        return features[0].float().cpu()

    for position in order:
        if emitted >= attempt_cap:
            break
        if emitted >= max_records and floor_met():
            break
        family = family_of_position[position]
        if (
            emitted >= max_records
            and family_valid[family] >= min_valid_per_family
        ):
            # Past the base budget, remaining attempts go only to families
            # still below their valid-positive floor.
            continue
        if max_seconds is not None and (time.time() - harvest_started) > max_seconds:
            wall_clock_capped = True
            break
        row = dataset[anchor_indices[position]]
        batch = move_model_batch(collator([row]), device)
        if not bool((batch["policy_supervision_mask"] & batch["is_behavior_anchor"])[0]):
            continue
        anchors_consumed += 1
        generator = torch.Generator(device=device).manual_seed(seed + emitted)
        # Version 9.0 (RC-2 repair): generate from the UNPADDED audit-surface
        # prompt via the shared encode primitive, never from the training
        # collator's fixed-length right-padded batch. Version 8.0's padded
        # harvest surface produced 1-3% candidate validity on generative
        # families against 0.82 causal validity on the audit surface.
        prompt_text = str(row.get("public_prompt", "")) if isinstance(row, Mapping) else ""
        context_budget = getattr(collator, "context_tokens", 0)
        if prompt_text and context_budget and hasattr(tokenizer, "encode"):
            prompt_ids = encode_preserving_ends(
                tokenizer, prompt_text, int(context_budget)
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
        else:
            # Offline stubs without prompts/tokenizers keep the padded path.
            padded_surface_rows += 1
            input_ids = batch["input_ids"][:1]
            attention_mask = batch["attention_mask"][:1]
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            generation = model.generate_hlwm_nbest(
                input_ids,
                attention_mask,
                canvas_length=int(batch["target_ids"].shape[-1]),
                max_new_tokens=max_new_tokens,
                candidate_temperatures=candidate_temperatures,
                generator=generator,
                return_features=True,
            )
        verification_error = torch.sigmoid(
            generation.workspace.verification_logits
        ).mean(dim=-1)

        # Every candidate from the shared workspace is graded and cached:
        # valid candidates are positives, invalid candidates are the
        # verifier-event negatives, and within-anchor pairs teach the heads
        # to rank inside one prompt's candidate set --- the exact judgment
        # verified fan-in asks of them at inference.
        anchor_positive_indices: List[int] = []
        anchor_negative_indices: List[int] = []
        seen_candidates: set = set()
        for candidate_index, candidate_ids in enumerate(generation.candidate_ids):
            candidate = tokenizer.decode(
                candidate_ids[0].detach().cpu(), skip_special_tokens=True
            ).strip()
            emitted += 1
            token_key = tuple(candidate_ids[0].tolist())
            if token_key in seen_candidates:
                duplicate_candidates += 1
                continue
            seen_candidates.add(token_key)
            semantic = grade_semantic_answer(candidate, batch["answer_specs"][0])
            valid = bool(
                semantic["correct"] and candidate_has_no_prompt_leak(candidate)
            )
            grader = str(semantic["grader"])
            grader_totals[grader] = grader_totals.get(grader, 0) + 1
            grader_correct[grader] = grader_correct.get(grader, 0) + int(valid)
            family_attempts[family] += 1
            emission = generation.candidate_features[candidate_index]
            if valid:
                family_valid[family] += 1
                valid_emissions += 1
                anchor_positive_indices.append(len(positive_features))
                positive_features.append(emission)
            else:
                invalid_emissions += 1
                anchor_negative_indices.append(len(negative_features))
                negative_features.append(emission)
                negative_kinds.append("invalid")

        semantic_negative_mask = batch["negative_target_attention_mask"][:1]
        if bool(semantic_negative_mask.any()):
            semantic_negatives += 1
            negative_features.append(
                candidate_features(
                    input_ids,
                    attention_mask,
                    generation.workspace,
                    batch["negative_target_ids"][:1],
                    semantic_negative_mask,
                    verification_error,
                )
            )
            negative_kinds.append("semantic")
            anchor_negative_indices.append(len(negative_features) - 1)

        greedy_ids = generation.candidate_ids[0]
        if greedy_ids.shape[1] > 1:
            corrupt_ids = torch.roll(greedy_ids, shifts=1, dims=-1)
        else:
            corrupt_ids = (greedy_ids + 1) % model.config.vocab_size
        negative_features.append(
            candidate_features(
                input_ids,
                attention_mask,
                generation.workspace,
                corrupt_ids,
                torch.ones_like(corrupt_ids),
                verification_error,
            )
        )
        negative_kinds.append("corrupt")
        anchor_negative_indices.append(len(negative_features) - 1)
        ranking_pairs.extend(
            (positive_index, negative_index)
            for positive_index in anchor_positive_indices
            for negative_index in anchor_negative_indices
        )
        if emitted % 16 == 0:
            print(
                json.dumps(
                    {
                        "policy_collection": {
                            "attempts": emitted,
                            "valid": valid_emissions,
                            "family_valid": family_valid,
                            "floor_met": floor_met(),
                        }
                    },
                    sort_keys=True,
                )
            )

    if was_training:
        model.train()
    return {
        "positive_features": (
            torch.stack(positive_features)
            if positive_features
            else torch.empty(0, 0)
        ),
        "negative_features": (
            torch.stack(negative_features)
            if negative_features
            else torch.empty(0, 0)
        ),
        "negative_kinds": negative_kinds,
        "ranking_pairs": ranking_pairs,
        "emitted": emitted,
        "valid_emissions": valid_emissions,
        "invalid_emissions": invalid_emissions,
        "semantic_negatives": semantic_negatives,
        "accuracy_by_grader": {
            grader: grader_correct.get(grader, 0) / max(1, total)
            for grader, total in sorted(grader_totals.items())
        },
        "family_valid_positives": dict(family_valid),
        "family_attempts": dict(family_attempts),
        "min_valid_per_family": int(min_valid_per_family),
        "attempt_cap": int(attempt_cap),
        "validity_floor_met": floor_met(),
        "candidate_temperatures": [float(value) for value in candidate_temperatures],
        "anchors_consumed": anchors_consumed,
        "duplicate_candidates": duplicate_candidates,
        "wall_clock_capped": wall_clock_capped,
        "harvest_seconds": round(time.time() - harvest_started, 1),
        "unpadded_prompt_surface": padded_surface_rows == 0,
        "source_split": "train",
    }


@torch.no_grad()
def domain_stratified_indices(
    dataset: Reasoning9000Dataset,
    count: int,
    seed: int,
    *,
    require_light_grader: bool = False,
) -> List[int]:
    """Round-robin dataset rows across domains for the generation canary.

    Study 7 showed an anchor-only canary is blind by construction: behavior
    anchors route and behave homogeneously while pathologies live in the
    other domains.  Sampling across domains makes the canary representative
    of the audit distribution.  ``require_light_grader`` restricts the pool
    to rows the in-training canary can actually grade, so every canary row
    produces a semantic verdict (Study 8's canary graded one row in eight).
    """

    buckets: Dict[str, List[int]] = {}
    for index, row in enumerate(dataset.rows):
        if require_light_grader:
            grader = str((row.get("answer_spec") or {}).get("type", "none")).lower()
            if grader not in CANARY_LIGHT_GRADERS:
                continue
        buckets.setdefault(str(row.get("domain", "unknown")), []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for domain, indices in buckets.items():
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        buckets[domain] = [indices[position] for position in permutation]
    domains = sorted(buckets)
    picked: List[int] = []
    cursor = 0
    while len(picked) < count and any(buckets.values()):
        domain = domains[cursor % len(domains)]
        cursor += 1
        if buckets[domain]:
            picked.append(buckets[domain].pop(0))
    return picked


# Graders safe to run inside the training loop; python_tests and io_tests
# execute generated code and stay confined to the fresh-process audit.
CANARY_LIGHT_GRADERS = (
    "numeric",
    "unit",
    "ordering",
    "abstention",
    "sql_exact",
)


@torch.no_grad()
def generation_canary(
    model: HLWMForConditionalGeneration,
    batches: List[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    *,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    """Report-only mid-training generation probe.

    Studies 5 and 6 both hit generation-time pathologies (prompt leakage,
    dialogue-marker drift, routing collapse and decay) that no training-time
    loss or validation statistic surfaced.  This probe generates a small
    stratified anchor set with the current weights at every evaluation
    boundary so those pathologies land in metrics.jsonl while the run can
    still be stopped cheaply.  It gates nothing in this revision.
    """

    was_training = model.training
    model.eval()
    markers = ("Human:", "Assistant:", "User:", "###")
    num_experts = int(model.config.num_experts)
    route_totals = torch.zeros(num_experts, dtype=torch.float64)
    leaks = 0
    marker_hits = 0
    commits = 0
    valid = 0
    graded = 0
    commit_probabilities: List[float] = []
    for index, batch in enumerate(batches):
        generator = torch.Generator(device=device).manual_seed(seed + index)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            generation = model.generate_hlwm(
                batch["input_ids"][:1],
                batch["attention_mask"][:1],
                canvas_length=int(batch["target_ids"].shape[-1]),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                generator=generator,
            )
        candidate = tokenizer.decode(
            generation.candidate_ids[0].detach().cpu(), skip_special_tokens=True
        ).strip()
        leaks += int(not candidate_has_no_prompt_leak(candidate))
        marker_hits += int(any(marker in candidate for marker in markers))
        commits += int(generation.decision == "publish")
        commit_probabilities.append(float(generation.commit_probability))
        spec = batch["answer_specs"][0]
        if str((spec or {}).get("type", "none")).lower() in CANARY_LIGHT_GRADERS:
            graded += 1
            valid += int(bool(grade_semantic_answer(candidate, spec)["correct"]))
        route_totals += (
            F.one_hot(generation.workspace.route_indices, num_experts)
            .float()
            .reshape(-1, num_experts)
            .sum(dim=0)
            .cpu()
            .double()
        )
    if was_training:
        model.train()
    count = max(1, len(batches))
    fractions = (route_totals / max(1.0, float(route_totals.sum()))).tolist()
    positive_loads = [value for value in fractions if value > 0.0]
    entropy = -sum(value * math.log(value) for value in positive_loads)
    return {
        "anchors": len(batches),
        "prefix_gate_tanh": float(torch.tanh(model.prefix_gate.detach()).cpu()),
        "prompt_leak_rate": leaks / count,
        "format_marker_rate": marker_hits / count,
        "graded": graded,
        "semantic_valid_rate": valid / max(1, graded),
        "commit_rate": commits / count,
        "mean_commit_probability": sum(commit_probabilities) / count,
        "route_load": fractions,
        "second_route_load": (
            sorted(fractions, reverse=True)[1] if len(fractions) > 1 else 0.0
        ),
        "route_entropy_normalized": entropy / math.log(max(2, num_experts)),
    }


def _policy_separation(
    head: torch.nn.Module, positive: Tensor, negative: Tensor
) -> Dict[str, float]:
    with torch.no_grad():
        positive_logits = head(positive)
        negative_logits = head(negative)
        commit_gap = positive_logits[:, None, 0] - negative_logits[None, :, 0]
        risk_gap = negative_logits[None, :, 1] - positive_logits[:, None, 1]
        verifier_gap = negative_logits[None, :, 2] - positive_logits[:, None, 2]
        return {
            "mean_positive_commit_probability": float(
                torch.sigmoid(positive_logits[:, 0]).mean().cpu()
            ),
            "mean_negative_commit_probability": float(
                torch.sigmoid(negative_logits[:, 0]).mean().cpu()
            ),
            "pairwise_commit_ranking_accuracy": float(
                (commit_gap > 0).float().mean().cpu()
            ),
            "pairwise_risk_ranking_accuracy": float(
                (risk_gap > 0).float().mean().cpu()
            ),
            "pairwise_verifier_ranking_accuracy": float(
                (verifier_gap > 0).float().mean().cpu()
            ),
        }


def train_policy_heads_on_policy(
    model: HLWMForConditionalGeneration,
    examples: Mapping[str, Any],
    *,
    epochs: int,
    learning_rate: float,
    label_smoothing: float = 0.0,
) -> Dict[str, Any]:
    """Fit only the three candidate-level heads on cached on-policy features.

    The generator, workspace, and every other module stay frozen: this phase
    changes how frozen candidates are scored, never what is generated.  If the
    graded emissions do not contain both classes the phase reports itself as
    skipped instead of manufacturing separation from teacher-forced text.
    """

    report: Dict[str, Any] = {
        "method": "on_policy_train_anchor_head_fit_v2_decorrelated",
        "emitted": int(examples["emitted"]),
        "valid_emissions": int(examples["valid_emissions"]),
        "invalid_emissions": int(examples["invalid_emissions"]),
        "semantic_negatives": int(examples["semantic_negatives"]),
        "accuracy_by_grader": dict(examples["accuracy_by_grader"]),
        "family_valid_positives": dict(examples.get("family_valid_positives", {})),
        "family_attempts": dict(examples.get("family_attempts", {})),
        "min_valid_per_family": int(examples.get("min_valid_per_family", 0)),
        "validity_floor_met": bool(examples.get("validity_floor_met", True)),
        "source_split": str(examples["source_split"]),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "label_smoothing": float(label_smoothing),
    }
    positive = examples["positive_features"]
    negative = examples["negative_features"]
    if positive.numel() == 0 or negative.numel() == 0 or len(positive) < 8 or len(negative) < 8:
        report.update(
            {
                "trained": False,
                "reason": "need at least eight graded positives and negatives",
                "positive_records": int(len(positive)) if positive.numel() else 0,
                "negative_records": int(len(negative)) if negative.numel() else 0,
            }
        )
        return report
    device = next(model.commitment_head.parameters()).device
    positive = positive.to(device=device, dtype=torch.float32)
    negative = negative.to(device=device, dtype=torch.float32)
    pairs = examples["ranking_pairs"]
    pair_positive = torch.tensor([pair[0] for pair in pairs], device=device)
    pair_negative = torch.tensor([pair[1] for pair in pairs], device=device)
    head = model.commitment_head
    was_training = model.training
    head.train()
    before = _policy_separation(head, positive, negative)
    # Version 5.5 fitted these heads to the rails (~5e-5 / 0.9999) after 400
    # epochs, which memorized the phase data and confounded the routing
    # causal-liveness measurement.  Smoothing keeps the calibrated score scale
    # unsaturated without changing which class each record belongs to.
    smoothing = min(max(float(label_smoothing), 0.0), 0.3)
    high = 1.0 - smoothing
    low = smoothing
    positive_labels = torch.tensor([high, low, low], device=device).expand(
        len(positive), -1
    )
    # Version 5.7 de-correlates the failure events (Known Limitations item 3):
    # the risk head trains on the mechanical-corruption event only and the
    # verifier-error head on the fluent-but-wrong event only, so the three
    # thresholds no longer describe a single valid/invalid axis.  "legacy"
    # (untagged callers) preserves the Version 5.6 anti-correlated labels.
    kinds = list(examples.get("negative_kinds", []))
    if len(kinds) != len(negative):
        kinds = ["legacy"] * len(negative)
    kind_labels = {
        "corrupt": (low, high, low),
        "invalid": (low, low, high),
        "semantic": (low, low, high),
        "legacy": (low, high, high),
    }
    negative_labels = torch.tensor(
        [kind_labels[kind] for kind in kinds], device=device
    )
    report["label_scheme"] = {
        "positive": [high, low, low],
        **{kind: list(kind_labels[kind]) for kind in sorted(set(kinds))},
    }
    report["negative_kind_counts"] = {
        kind: kinds.count(kind) for kind in sorted(set(kinds))
    }
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=0.01
    )
    first_loss = None
    last_loss = None
    for _ in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        positive_logits = head(positive)
        negative_logits = head(negative)
        loss = F.binary_cross_entropy_with_logits(
            positive_logits, positive_labels
        ) + F.binary_cross_entropy_with_logits(negative_logits, negative_labels)
        if len(pairs):
            paired_positive = positive_logits.index_select(0, pair_positive)
            paired_negative = negative_logits.index_select(0, pair_negative)
            # Rank each failure channel only against negatives of its own
            # kind; the commit channel ranks against every negative.
            pair_kinds = [kinds[pair[1]] for pair in pairs]
            risk_pairs = torch.tensor(
                [kind in ("corrupt", "legacy") for kind in pair_kinds],
                device=device,
                dtype=torch.float32,
            )
            verifier_pairs = torch.tensor(
                [kind in ("invalid", "semantic", "legacy") for kind in pair_kinds],
                device=device,
                dtype=torch.float32,
            )
            rank_loss = (
                F.relu(1.0 - paired_positive[:, 0] + paired_negative[:, 0])
                + F.relu(1.0 - paired_negative[:, 1] + paired_positive[:, 1])
                * risk_pairs
                + F.relu(1.0 - paired_negative[:, 2] + paired_positive[:, 2])
                * verifier_pairs
            ).mean()
            loss = loss + 0.5 * rank_loss
        if not torch.isfinite(loss):
            raise RuntimeError("on-policy policy-head loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().float().cpu())
        if first_loss is None:
            first_loss = last_loss
    head.eval()
    after = _policy_separation(head, positive, negative)
    if was_training:
        model.train()
    report.update(
        {
            "trained": True,
            "positive_records": int(len(positive)),
            "negative_records": int(len(negative)),
            "ranking_pairs": int(len(pairs)),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "separation_before": before,
            "separation_after": after,
        }
    )
    return report


@torch.no_grad()
def fixed_overfit_gate_loss(
    model: HLWMForConditionalGeneration,
    batch: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> float:
    """Measure the same examples, corruption RNG and timestep draw each time."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        _, metrics = local_denoise_loss(model, batch, generator=generator)
    if was_training:
        model.train()
    return float(metrics["denoise"])


def fixed_overfit_gate_suite(
    model: HLWMForConditionalGeneration,
    batches: List[Mapping[str, Any]],
    device: torch.device,
    seed: int,
) -> float:
    if not batches:
        raise ValueError("overfit gate suite cannot be empty")
    return sum(
        fixed_overfit_gate_loss(model, batch, device, seed + index)
        for index, batch in enumerate(batches)
    ) / len(batches)


def _floating_tensors(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating_tensors(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _floating_tensors(item)


def real_qwen_preflight(
    model: HLWMForConditionalGeneration,
    batch: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Exercise the real vocabulary forward/backward and name the first bad stage."""

    traces: List[Dict[str, Any]] = []
    calls: Dict[str, int] = {}
    handles = []

    def finite_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            call = calls.get(name, 0) + 1
            calls[name] = call
            tensors = list(_floating_tensors(output))
            for index, tensor in enumerate(tensors):
                finite = bool(torch.isfinite(tensor.detach()).all())
                if not finite:
                    raise FloatingPointError(
                        "real-Qwen preflight found a non-finite tensor at %s call %d output %d"
                        % (name, call, index)
                    )
            if tensors:
                maximum = max(
                    float(tensor.detach().float().abs().max().cpu())
                    for tensor in tensors
                )
                traces.append({"stage": name, "call": call, "max_abs": maximum})

        return hook

    for index, layer in enumerate(model.backbone.layers):
        handles.append(layer.register_forward_hook(finite_hook("decoder_layer_%02d" % index)))
    for name, module in (
        ("backbone_norm", model.backbone.norm),
        ("root_adapter", model.root_adapter),
        ("expert_bank", model.expert_bank),
        ("denoise_norm", model.denoise_norm),
        ("language_head", model.lm_head),
    ):
        handles.append(module.register_forward_hook(finite_hook(name)))

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            loss, metrics = local_denoise_loss(model, batch, generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "real-Qwen preflight local loss is non-finite: %r" % metrics
            )
        loss.backward()
        bad_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all())
        ]
        if bad_gradients:
            raise FloatingPointError(
                "real-Qwen preflight found non-finite gradients: %s"
                % ", ".join(bad_gradients[:12])
            )
        gradient_tensors = sum(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if gradient_tensors == 0:
            raise RuntimeError("real-Qwen preflight produced no trainable gradients")

        # The joint stage allocates the true training peak (multi-update
        # unroll, synthesis and paired negative scoring).  Exercising it here
        # makes the preflight an actual memory ceiling for small partitions.
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            joint_loss, joint_metrics = hlwm_loss(model, batch)
        if not torch.isfinite(joint_loss):
            raise FloatingPointError("real-Qwen preflight joint loss is non-finite")
        joint_loss.backward()

        total_memory_gb = (
            torch.cuda.get_device_properties(device).total_memory / 2**30
        )
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 2**30
        peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 2**30
        return {
            "passed": True,
            "loss": float(loss.detach().float().cpu()),
            "joint_loss": float(joint_loss.detach().float().cpu()),
            "metrics": metrics,
            "joint_denoise": joint_metrics.get("denoise"),
            "checked_stage_calls": len(traces),
            "gradient_tensors": gradient_tensors,
            "maximum_stage_abs": max(item["max_abs"] for item in traces),
            "device_total_memory_gb": total_memory_gb,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "amp_dtype": str(AMP_DTYPE),
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def trainable_state_dict(model: HLWMForConditionalGeneration) -> Dict[str, Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    buffer_names = {name for name, _ in model.named_buffers()}
    # Persistent buffers ride along with the trainables: the Version 10.0
    # EMA/std normalizers (``latent_embed_std``, ``distill_ema_std``) would
    # otherwise silently reinitialize on resume and rescale the distillation
    # loss by orders of magnitude mid-salvage — and Session H proved resume
    # is a load-bearing path, not a contingency.
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names or name in buffer_names
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    *,
    model: HLWMForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    optimizer_updates: int,
    skipped_optimizer_updates: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "hlwm-trainable-checkpoint-v5",
            "base_model": args.model,
            "base_revision": args.revision,
            "step": step,
            "trainer_state": {
                "microsteps": step,
                "optimizer_updates": optimizer_updates,
                "skipped_optimizer_updates": skipped_optimizer_updates,
            },
            "hlwm_config": vars(model.config),
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "args": serializable_args(args),
        },
        temporary,
    )
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        sha256(path) + "  " + path.name + "\n", encoding="utf-8"
    )


def load_resume(
    path: Path,
    model: HLWMForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    args: argparse.Namespace,
) -> Dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "hlwm-trainable-checkpoint-v5":
        raise RuntimeError("resume checkpoint is not an HLWM Version 5.4 checkpoint")
    if payload.get("base_model") != args.model or payload.get("base_revision") != args.revision:
        raise RuntimeError("resume checkpoint base model or revision does not match")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError("unexpected checkpoint keys: %s" % unexpected)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(name for name in missing if name in trainable)
    if missing_trainable:
        raise RuntimeError("resume checkpoint is missing trainable tensors: %s" % missing_trainable)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    rng = payload.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])
    trainer_state = payload.get("trainer_state") or {}
    step = int(trainer_state.get("microsteps", payload.get("step", 0)))
    return {
        "microsteps": step,
        "optimizer_updates": int(
            trainer_state.get("optimizer_updates", math.ceil(step / args.gradient_accumulation))
        ),
        "skipped_optimizer_updates": int(
            trainer_state.get("skipped_optimizer_updates", 0)
        ),
    }


def export_adapter(
    path: Path,
    model: HLWMForConditionalGeneration,
    args: argparse.Namespace,
    step: int,
) -> Dict[str, Any]:
    from safetensors.torch import save_file

    state = {
        name: value.contiguous()
        for name, value in trainable_state_dict(model).items()
    }
    metadata = {
        "format": "hlwm-adapter-v5",
        "base_model": args.model,
        "base_revision": args.revision,
        "step": str(step),
        "unfreeze_tail_layers": str(args.unfreeze_tail_layers),
        "hlwm_config": json.dumps(vars(model.config), sort_keys=True),
    }
    save_file(state, str(path), metadata=metadata)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tensor_count": len(state),
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    global AMP_DTYPE
    args = parse_args()
    if args.latent_thoughts > 0 and args.steps is None:
        raise ValueError("Version 10.0 requires an explicit --steps (warm + main)")
    args.progressive = args.steps is None
    if args.progressive:
        args.steps = args.overfit_steps + args.local_steps + args.joint_steps
    else:
        args.overfit_steps = 0
        args.local_steps = int(args.steps)
        args.joint_steps = 0
    if not 0.0 <= args.causal_ratio <= 1.0:
        raise ValueError("causal-ratio must be between zero and one")
    if not 0.0 <= args.anchor_ratio <= 1.0:
        raise ValueError("anchor-ratio must be between zero and one")
    if args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps and gradient-accumulation must be positive")
    if min(args.overfit_steps, args.local_steps, args.joint_steps) < 0:
        raise ValueError("progressive phase lengths must be nonnegative")
    if args.progressive and args.overfit_steps <= 0:
        raise ValueError("progressive training requires a positive overfit phase")
    if args.overfit_steps % args.gradient_accumulation:
        raise ValueError("overfit-steps must end on a gradient-accumulation boundary")
    if args.max_runtime_hours <= 0:
        raise ValueError("max-runtime-hours must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.calibration_records <= 0 or args.calibration_new_tokens <= 0:
        raise ValueError("calibration record and token counts must be positive")
    if args.policy_records <= 0 or args.policy_epochs <= 0 or args.policy_new_tokens <= 0:
        raise ValueError("policy phase record, epoch and token counts must be positive")
    if args.policy_learning_rate <= 0:
        raise ValueError("policy learning rate must be positive")
    if args.policy_min_valid_per_family < 0 or args.policy_max_attempts < 0:
        raise ValueError("policy validity floor and attempt cap cannot be negative")
    if args.policy_max_attempts and args.policy_max_attempts < args.policy_records:
        raise ValueError("policy attempt cap cannot be below --policy-records")
    if args.canary_anchors < 0 or args.canary_new_tokens <= 0:
        raise ValueError("canary anchor count cannot be negative and token budget must be positive")
    args.candidate_temperature_list = [
        float(part) for part in str(args.candidate_temperatures).split(",") if part.strip()
    ]
    if not args.candidate_temperature_list or any(
        value < 0 for value in args.candidate_temperature_list
    ):
        raise ValueError("candidate temperatures must be a nonempty list of nonnegative floats")
    if args.candidate_temperature_list[0] != 0.0:
        raise ValueError("the first candidate temperature must be 0.0 (greedy)")
    if args.workspace_memory_windows < 0:
        raise ValueError("workspace memory windows cannot be negative")
    if args.router_entropy_weight < 0:
        raise ValueError("router entropy weight cannot be negative")
    if args.router_aux_weight < 0 or args.expert_diversity_weight < 0:
        raise ValueError("router aux and expert diversity weights cannot be negative")
    if not 0.0 <= args.expert_init_scale <= 0.1:
        raise ValueError("expert init scale must lie in [0, 0.1]")
    if not 0.0 <= args.policy_label_smoothing <= 0.3:
        raise ValueError("policy label smoothing must lie in [0, 0.3]")
    if not 0.5 <= args.memory_headroom_fraction <= 1.0:
        raise ValueError("memory headroom fraction must be between 0.5 and 1.0")
    if args.initial_loss_scale <= 0 or args.max_skipped_updates < 0:
        raise ValueError("loss scale must be positive and max skipped updates nonnegative")
    if args.unfreeze_tail_layers < 0 or args.lora_rank < 0 or args.lora_tail_layers < 0:
        raise ValueError("language adaptation depths/ranks cannot be negative")
    if args.lora_alpha <= 0 or not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("LoRA alpha/dropout are invalid")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            "output directory is not empty; choose a new directory or pass --resume: %s"
            % args.output_dir
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this training entrypoint expects a CUDA training session")
    torch.set_float32_matmul_precision("high")

    if args.precision == "auto":
        args.resolved_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        args.resolved_precision = args.precision
    if args.resolved_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this device does not support it")
    AMP_DTYPE = torch.bfloat16 if args.resolved_precision == "bf16" else torch.float16
    base_dtype = AMP_DTYPE

    environment = {
        "torch": torch.__version__,
        "precision": args.resolved_precision,
        "device": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "base_model": args.model,
        "base_revision": args.revision,
    }
    print(json.dumps({"environment": environment}, indent=2))
    if not args.skip_architecture_smoke:
        print(json.dumps({"architecture_smoke": architecture_smoke(device)}, indent=2))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = Reasoning9000Dataset(
        find_split(args.data_dir, "train"), args.num_lanes, args.max_train
    )
    validation_data = Reasoning9000Dataset(
        find_split(args.data_dir, "validation"), args.num_lanes, args.max_validation
    )
    if args.latent_thoughts > 0:
        collator: Any = V10AnchorCollator(
            tokenizer,
            latent_thoughts=args.latent_thoughts,
            trace_tokens=args.trace_tokens,
            declared_routing=bool(getattr(args, "declared_routing", False)),
            num_lanes=args.num_lanes,
            context_tokens=args.context_tokens,
            canvas_tokens=args.canvas_tokens,
            brief_tokens=args.brief_tokens,
            causal_tokens=args.causal_tokens,
        )
    else:
        collator = HLWMCollator(
            tokenizer,
            num_lanes=args.num_lanes,
            context_tokens=args.context_tokens,
            canvas_tokens=args.canvas_tokens,
            brief_tokens=args.brief_tokens,
            causal_tokens=args.causal_tokens,
        )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    overrides = {
        "num_lanes": args.num_lanes,
        "max_refinement_steps": args.refinement_steps,
        "slow_update_every": 1,
        "min_halt_steps": min(2, args.refinement_steps),
        "diffusion_steps": args.diffusion_steps,
        "num_experts": args.num_experts,
        "expert_top_level": min(2, args.num_experts),
        "expert_bottleneck": 64,
        "synthesis_prefix_tokens": 8,
        "lane_diversity_weight": 0.25,
        "commitment_loss_weight": 0.75,
        "router_entropy_weight": args.router_entropy_weight,
        "router_aux_weight": args.router_aux_weight,
        "expert_diversity_weight": args.expert_diversity_weight,
        "expert_init_scale": args.expert_init_scale,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_tail_layers": args.lora_tail_layers,
        "workspace_memory_windows": args.workspace_memory_windows,
        "synthesis_kl_weight": args.synthesis_kl_weight,
        "prefix_gate_init": args.prefix_gate_init,
        "premise_aux_weight": args.premise_aux_weight,
        "gist_prefix_tokens": args.gist_prefix_tokens,
        # Version 10.0 dense-supervision channel (all zero-default = v9).
        "latent_thoughts": args.latent_thoughts,
        "vocab_grounded_thoughts": bool(getattr(args, "vocab_grounded_thoughts", False)),
        "adapter_mode": str(getattr(args, "adapter", "lora")),
        "vocab_thought_tau": float(getattr(args, "vocab_thought_tau", 1.0)),
        "declared_routing": bool(getattr(args, "declared_routing", False)),
        "kv_prefix_slots": args.kv_prefix_slots if args.latent_thoughts > 0 else 0,
        "kv_prefix_rank": args.kv_prefix_rank,
        "prefix_attn_gate_init": args.prefix_attn_gate_init,
        "mlp_expert_count": args.mlp_expert_count,
        "mlp_expert_rank": args.mlp_expert_rank,
        # Version 8.0 channel layout: the model appends the hard response cue
        # after the (optional) workspace prefix, so both channels always
        # continue generation from real text tokens.
        "response_cue_ids": tuple(
            tokenizer.encode(RESPONSE_CUE_TEXT, add_special_tokens=False)
        ),
    }
    print("Loading pinned Qwen base and transplanting language weights...")
    model = HLWMForConditionalGeneration.from_pretrained(
        args.model,
        hlwm_overrides=overrides,
        revision=args.revision,
        dtype=base_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    gc.collect()
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    model.unfreeze_language_tail(args.unfreeze_tail_layers)
    if args.causal_control:
        # Plain-LoRA control: only the LoRA sidecars (and any requested tail
        # layers) may train; every HLWM-specific module stays frozen so the
        # run is the dense-shared-trunk baseline, not a damaged full model.
        frozen_hlwm = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "_lora." not in name and not name.startswith(
                "backbone.layers."
            ):
                parameter.requires_grad_(False)
                frozen_hlwm += 1
        print(
            json.dumps(
                {"causal_control": {"frozen_non_lora_parameter_tensors": frozen_hlwm}}
            )
        )
    model.gradient_checkpointing_enable()
    model.to(device=device, dtype=base_dtype)
    # Optimizer-owned gradients must be FP32 (GradScaler requires it for FP16
    # and it is numerically preferable under BF16 too).  Retain every
    # pretrained Qwen tensor in the half-precision base dtype; only new HLWM
    # and LoRA sidecar parameters become FP32 optimizer parameters.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.train()
    parameter_summary = model.trainable_parameter_summary()
    print(json.dumps({"parameters": parameter_summary}, indent=2))

    preflight: Optional[Dict[str, Any]] = None
    if args.causal_control and not args.skip_real_qwen_preflight:
        # The HLWM preflight asserts gradients on router/head parameters,
        # which the control intentionally freezes.
        print("causal-control run: skipping the HLWM-module preflight")
        args.skip_real_qwen_preflight = True
    if not args.skip_real_qwen_preflight:
        preflight_batch = move_model_batch(collator([train_data[0]]), device)
        preflight = real_qwen_preflight(
            model, preflight_batch, device, args.seed + 10_003
        )
        print(json.dumps({"real_qwen_preflight": preflight}, indent=2, sort_keys=True))
        allowed_reserved = (
            args.memory_headroom_fraction * preflight["device_total_memory_gb"]
        )
        if preflight["peak_reserved_gb"] > allowed_reserved:
            raise RuntimeError(
                "preflight reserved %.2f GB but the headroom limit is %.2f GB of "
                "%.2f GB total; reduce context/canvas/causal token budgets before "
                "spending the session"
                % (
                    preflight["peak_reserved_gb"],
                    allowed_reserved,
                    preflight["device_total_memory_gb"],
                )
            )
        del preflight_batch
        gc.collect()
        torch.cuda.empty_cache()
    if args.preflight_only:
        if preflight is None:
            raise ValueError("--preflight-only cannot be combined with --skip-real-qwen-preflight")
        return

    gate_batches: Optional[List[Dict[str, Any]]] = None
    overfit_gate: Optional[Dict[str, Any]] = None

    if args.latent_thoughts > 0:
        # Version 10.0: the prefix attention gates are exempt from weight
        # decay (decay is a constant close-the-channel force on a gate
        # scalar with no opposing gradient until the channel is useful).
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, args.weight_decay),
            lr=cosine_learning_rate(
                0,
                math.ceil(args.steps / args.gradient_accumulation),
                args.warmup_updates,
                args.learning_rate,
            ),
            betas=(0.9, 0.95),
        )
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cosine_learning_rate(
                0,
                math.ceil(args.steps / args.gradient_accumulation),
                args.warmup_updates,
                args.learning_rate,
            ),
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    # BF16 shares the FP32 exponent range, so gradient scaling is unnecessary
    # and the scaler runs disabled (scale fixed at 1.0, no skipped updates).
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.resolved_precision == "fp16",
        init_scale=args.initial_loss_scale,
        growth_interval=args.loss_scale_growth_interval,
    )
    start_step = 0
    optimizer_updates = 0
    skipped_optimizer_updates = 0
    if args.resume:
        trainer_state = load_resume(args.resume, model, optimizer, scaler, args)
        start_step = trainer_state["microsteps"]
        optimizer_updates = trainer_state["optimizer_updates"]
        skipped_optimizer_updates = trainer_state["skipped_optimizer_updates"]
        print("Resumed from step", start_step)
    if start_step > args.steps:
        raise ValueError("resume checkpoint exceeded --steps")
    if start_step == args.steps:
        print(
            "Resume checkpoint already completed every training step; skipping "
            "the loop and rerunning the policy, calibration, and export phases."
        )
    if args.latent_thoughts > 0:
        if args.v10_preflight_only:
            report = v10_preflight(
                args=args,
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                validation_data=validation_data,
                collator=collator,
                device=device,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "preflight.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps({"v10_preflight": report}, sort_keys=True))
            raise SystemExit(0 if report["passed"] else 3)
        verdict = run_v10_training(
            args=args,
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            validation_data=validation_data,
            collator=collator,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            start_step=start_step,
            optimizer_updates=optimizer_updates,
            skipped_optimizer_updates=skipped_optimizer_updates,
        )
        print(json.dumps({"v10_training": verdict}, sort_keys=True))
        return

    if args.progressive:
        if start_step < args.overfit_steps:
            gate_count = min(4, args.overfit_examples, len(train_data))
            gate_batches = [
                move_model_batch(collator([train_data[index]]), device)
                for index in range(gate_count)
            ]
            gate_before = fixed_overfit_gate_suite(
                model, gate_batches, device, args.seed + 91
            )
            overfit_gate = {
                "before": gate_before,
                "after": None,
                "passed": None,
                "fixed_examples": gate_count,
            }
        else:
            overfit_gate = {
                "passed": True,
                "resumed_after_gate": True,
                "before": None,
                "after": None,
            }
        print(json.dumps({"overfit_gate": overfit_gate}, sort_keys=True))

    train_loader = DataLoader(
        train_data,
        batch_sampler=DeterministicStepBatchSampler(
            len(train_data),
            args.batch_size,
            start_step,
            args.steps,
            args.seed,
            overfit_steps=args.overfit_steps if args.progressive else 0,
            overfit_examples=args.overfit_examples,
            priority_indices=train_data.anchor_indices,
            priority_ratio=args.anchor_ratio,
        ),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    canary_batches: List[Dict[str, Any]] = []
    if args.canary_anchors > 0 and not args.causal_control:
        canary_indices = domain_stratified_indices(
            train_data,
            args.canary_anchors,
            args.seed + 811,
            require_light_grader=True,
        )
        canary_batches = [
            move_model_batch(collator([train_data[index]]), device)
            for index in canary_indices
        ]
        print(
            json.dumps(
                {
                    "canary_domains": [
                        str(train_data.rows[index].get("domain", "unknown"))
                        for index in canary_indices
                    ]
                },
                sort_keys=True,
            )
        )

    log_path = args.output_dir / "metrics.jsonl"
    if start_step >= args.steps:
        # Deliverable packaging copies metrics.jsonl even when a completed
        # checkpoint means no new microsteps run.
        append_jsonl(log_path, {"step": start_step, "resumed_at_completion": True})
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    accumulation_target = min(args.gradient_accumulation, args.steps - start_step)
    running_loss = 0.0
    wall_start = time.time()
    total_optimizer_updates = math.ceil(args.steps / args.gradient_accumulation)
    completed_steps = start_step
    time_budget_reached = False
    mode_counts: Dict[str, int] = {"causal": 0, "local_denoise": 0, "joint_hlwm": 0}

    for step, batch in zip(range(start_step, args.steps), train_loader):
        batch = move_model_batch(batch, device)
        phase = phase_for_step(step, args)
        is_causal = args.causal_control or (
            phase != "overfit" and random.random() < args.causal_ratio
        )
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            if is_causal:
                loss = causal_loss(model, batch)
                components = {"causal": float(loss.detach().float().cpu())}
                mode = "causal"
            elif phase in ("overfit", "local"):
                loss, components = local_denoise_loss(model, batch)
                mode = "local_denoise"
            else:
                # Version 9.0 attribution control: masked batches alternate
                # between the workspace prefix and the gist prefix (when the
                # gist module is enabled) so the audit's canvas-vs-gist
                # comparison is between two equally trained arms.
                masked_flags = batch.get("masked_rows")
                use_gist = bool(
                    args.gist_prefix_tokens > 0
                    and masked_flags is not None
                    and bool(masked_flags.all())
                    and step % 2 == 1
                )
                loss, components = hlwm_loss(
                    model, batch, prefix_source="gist" if use_gist else "workspace"
                )
                mode = "joint_hlwm"
            components["anchor_fraction"] = float(
                batch["is_behavior_anchor"].float().mean().detach().cpu()
            )
            scaled_loss = loss / accumulation_target
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at step %d" % (step + 1))
        scaler.scale(scaled_loss).backward()
        accumulated += 1
        running_loss += float(loss.detach().float().cpu())
        completed_steps = step + 1
        mode_counts[mode] += 1

        optimizer_boundary = accumulated >= accumulation_target or step + 1 == args.steps
        optimizer_step_applied: Optional[bool] = None
        if optimizer_boundary:
            lr = cosine_learning_rate(
                optimizer_updates,
                total_optimizer_updates,
                args.warmup_updates,
                args.learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            scale_before = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer_step_applied = scale_after >= scale_before
            if optimizer_step_applied:
                optimizer_updates += 1
            else:
                skipped_optimizer_updates += 1
                if skipped_optimizer_updates > args.max_skipped_updates:
                    raise RuntimeError(
                        "loss scaling skipped %d optimizer updates; refusing unstable training"
                        % skipped_optimizer_updates
                    )
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            accumulation_target = min(
                args.gradient_accumulation, args.steps - (step + 1)
            )
        else:
            gradient_norm = None
            lr = optimizer.param_groups[0]["lr"]

        record = {
            "step": step + 1,
            "phase": phase,
            "mode": mode,
            "loss": float(loss.detach().float().cpu()),
            "running_loss": running_loss / (step - start_step + 1),
            "learning_rate": lr,
            "gradient_norm": (
                float(gradient_norm.detach().float().cpu())
                if gradient_norm is not None and bool(torch.isfinite(gradient_norm))
                else None
            ),
            "gradient_norm_finite": (
                bool(torch.isfinite(gradient_norm))
                if gradient_norm is not None
                else None
            ),
            "optimizer_boundary": optimizer_boundary,
            "optimizer_step_applied": optimizer_step_applied,
            "optimizer_updates": optimizer_updates,
            "skipped_optimizer_updates": skipped_optimizer_updates,
            "loss_scale": float(scaler.get_scale()),
            "elapsed_seconds": time.time() - wall_start,
            "gpu_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "components": components,
        }
        append_jsonl(log_path, record)
        print(json.dumps(record, sort_keys=True))

        if args.progressive and step + 1 == args.overfit_steps:
            assert gate_batches is not None and overfit_gate is not None
            gate_after = fixed_overfit_gate_suite(
                model, gate_batches, device, args.seed + 91
            )
            relative_improvement = (
                float(overfit_gate["before"]) - gate_after
            ) / max(1.0e-12, float(overfit_gate["before"]))
            overfit_gate.update(
                {
                    "after": gate_after,
                    "relative_improvement": relative_improvement,
                    "required_improvement": args.overfit_min_improvement,
                    "passed": relative_improvement > args.overfit_min_improvement,
                }
            )
            append_jsonl(log_path, {"step": step + 1, "overfit_gate": overfit_gate})
            print(json.dumps({"overfit_gate": overfit_gate}, sort_keys=True))
            if not overfit_gate["passed"]:
                save_checkpoint(
                    args.output_dir / ("checkpoint-gate-failed-%06d.pt" % (step + 1)),
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step + 1,
                    optimizer_updates=optimizer_updates,
                    skipped_optimizer_updates=skipped_optimizer_updates,
                    args=args,
                )
                raise RuntimeError(
                    "tiny overfit gate failed; refusing to spend the remaining Kaggle budget"
                )

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = evaluate(model, validation_loader, device, phase=phase)
            evaluation = {"step": step + 1, "phase": phase, "validation": metrics}
            append_jsonl(log_path, evaluation)
            print(json.dumps(evaluation, sort_keys=True))
            if canary_batches:
                canary_record = {
                    "step": step + 1,
                    "phase": phase,
                    "canary": generation_canary(
                        model,
                        canary_batches,
                        tokenizer,
                        device,
                        max_new_tokens=args.canary_new_tokens,
                        seed=args.seed + 90_001 + step,
                    ),
                }
                append_jsonl(log_path, canary_record)
                print(json.dumps(canary_record, sort_keys=True))
        if ((step + 1) % args.save_every == 0 or step + 1 == args.steps) and accumulated == 0:
            save_checkpoint(
                args.output_dir / ("checkpoint-step-%06d.pt" % (step + 1)),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step + 1,
                optimizer_updates=optimizer_updates,
                skipped_optimizer_updates=skipped_optimizer_updates,
                args=args,
            )
        if optimizer_boundary and time.time() - wall_start >= args.max_runtime_hours * 3600:
            time_budget_reached = True
            save_checkpoint(
                args.output_dir / ("checkpoint-time-budget-%06d.pt" % (step + 1)),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step + 1,
                optimizer_updates=optimizer_updates,
                skipped_optimizer_updates=skipped_optimizer_updates,
                args=args,
            )
            print("Runtime safety budget reached; saved a resumable checkpoint.")
            break

    policy_report: Optional[Dict[str, Any]] = None
    if not args.skip_policy_phase and not args.causal_control:
        gc.collect()
        torch.cuda.empty_cache()
        print("Collecting on-policy train-anchor emissions for the policy heads...")
        policy_examples = collect_on_policy_policy_examples(
            model,
            train_data,
            collator,
            tokenizer,
            device,
            max_records=args.policy_records,
            max_new_tokens=args.policy_new_tokens,
            seed=args.seed + 70_003,
            min_valid_per_family=args.policy_min_valid_per_family,
            max_attempts=args.policy_max_attempts,
            candidate_temperatures=args.candidate_temperature_list,
            max_seconds=(
                args.harvest_max_hours * 3600.0 if args.harvest_max_hours > 0 else None
            ),
        )
        policy_report = train_policy_heads_on_policy(
            model,
            policy_examples,
            epochs=args.policy_epochs,
            learning_rate=args.policy_learning_rate,
            label_smoothing=args.policy_label_smoothing,
        )
        (args.output_dir / "policy-head-training.json").write_text(
            json.dumps(policy_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"on_policy_policy_head": policy_report}, sort_keys=True))

    if args.causal_control:
        calibration = {
            "method": "skipped_causal_control",
            "skipped": True,
            "reason": (
                "the plain-LoRA control has no commitment machinery to calibrate; "
                "it publishes every generation"
            ),
            "test_split_used": False,
        }
    else:
        calibration = calibrate_commitment_policy(
            model,
            validation_loader,
            tokenizer,
            device,
            max_records=args.calibration_records,
            max_new_tokens=args.calibration_new_tokens,
            seed=args.seed + 50_003,
            candidate_temperatures=args.candidate_temperature_list,
            target_coverage=args.conformal_target_coverage,
        )
    calibration["calibrated_at_step"] = completed_steps
    calibration["policy_heads_trained_on_policy"] = bool(
        policy_report is not None and policy_report.get("trained")
    )
    (args.output_dir / "commitment-calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"commitment_calibration": calibration}, sort_keys=True))

    final_checkpoint_path = args.output_dir / (
        ("checkpoint-time-budget-%06d.pt" % completed_steps)
        if time_budget_reached
        else ("checkpoint-step-%06d.pt" % completed_steps)
    )
    # Re-save after calibration so reload and adapter inference use only
    # validation-fitted thresholds embedded in the model configuration.
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=completed_steps,
        optimizer_updates=optimizer_updates,
        skipped_optimizer_updates=skipped_optimizer_updates,
        args=args,
    )
    adapter = export_adapter(
        args.output_dir / ("hlwm-adapter-step-%06d.safetensors" % completed_steps),
        model,
        args,
        completed_steps,
    )
    final_summary = {
        "status": (
            "time_budget_checkpoint_saved"
            if time_budget_reached
            else
            "stable_prototype_training_complete"
            if skipped_optimizer_updates == 0
            else "prototype_training_complete_with_skipped_updates"
        ),
        "steps": completed_steps,
        "planned_steps": args.steps,
        "overfit_gate": overfit_gate,
        "optimizer_updates": optimizer_updates,
        "skipped_optimizer_updates": skipped_optimizer_updates,
        "final_loss_scale": float(scaler.get_scale()),
        "elapsed_seconds": time.time() - wall_start,
        "parameters": parameter_summary,
        "real_qwen_preflight": preflight,
        "precision": args.resolved_precision,
        "on_policy_policy_head": policy_report,
        "commitment_calibration": calibration,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "prefix_gate": {
            "alpha": float(model.prefix_gate.detach().cpu()),
            "tanh": float(torch.tanh(model.prefix_gate.detach()).cpu()),
        },
        "response_cue_ids": list(model.config.response_cue_ids or ()),
        "adapter": adapter,
        "checkpoint": str(final_checkpoint_path),
        "args": serializable_args(args),
        "causal_control": bool(args.causal_control),
        "mode_counts": mode_counts,
        "behavior_anchor_count": len(train_data.anchor_indices),
        "warning": (
            "Reasoning9000 remains unreviewed. Programmatically verified behavior anchors "
            "exercise formatting and abstention but do not establish production capability."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))





# ======================================================================
# Version 10.0 dense-supervision training engine.
#
# Batch contract for anchor rows (the V10 collator in data.py produces
# exactly these keys; the builder guarantees every trace has at least
# ``latent_thoughts`` tokens):
#   input_ids / attention_mask          full prompt (premise visible)
#   student_input_ids / student_attention_mask
#                                       masked prompt on masked rows,
#                                       the full prompt otherwise
#   target_ids / target_attention_mask  answer tokens
#   trace_input_ids / trace_attention_mask
#                                       gold trace tokens, right-padded
#   teacher_supervised_mask             1 on trace tokens excluding the
#                                       final answer-producing step
#   trace_window_targets                [B, K, W] trace ids per window,
#                                       -100 padded
#   corrupt_trace_input_ids / corrupt_trace_attention_mask / corrupt_step_index
#   family_index                        [B] 0..3 (numeric/unit/ordering/abst.)
#   route_index                         [B] expert id from the family map
#   masked_rows                         [B] bool


def optimizer_parameter_groups(
    model: HLWMForConditionalGeneration, weight_decay: float
) -> List[Dict[str, Any]]:
    """Weight-decay groups with the prefix attention gates exempted.

    Decay on a gate scalar is a constant close-the-channel force with no
    opposing gradient until the channel is useful — 16% closure over a v10
    run at wd 0.1, measured in the red-team's arithmetic.  Norms and biases
    are exempted alongside, the standard practice this codebase inherited.
    """

    decay: List[Tensor] = []
    no_decay: List[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            "prefix_attn_gate" in name
            or parameter.ndim <= 1
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _segment_cross_entropy(
    segment_logits: Tensor, segment_ids: Tensor, supervised_mask: Tensor
) -> Tensor:
    if not bool(supervised_mask.any()):
        # Single-step traces exclude their only (answer-producing) step from
        # teacher supervision, leaving a legitimately empty mask; the term
        # contributes nothing rather than crashing the step.  Group-level
        # gradient liveness is certified separately by the grad-probe
        # preflight, so an unexpectedly dead mask cannot hide here.
        return segment_logits.new_zeros(())
    losses = _masked_token_cross_entropy(
        segment_logits, segment_ids.clamp_min(0), supervised_mask
    )
    return losses.sum() / supervised_mask.to(losses.dtype).sum().clamp_min(1.0)


def compressed_gold_thoughts(
    model: HLWMForConditionalGeneration,
    trace_input_ids: Tensor,
    trace_attention_mask: Tensor,
    windows: int,
) -> Tensor:
    """CoLaR compression targets: per-window sum/sqrt(c) of trace embeddings.

    Sum-pooling scaled by sqrt(c) preserves the embedding scale statistics
    (mean-pooling shrinks them, measured at 2-3 points in CoLaR); the result
    is grounded to the embedding std like every other latent.  Padded
    positions never enter a window: the window boundaries are computed over
    each row's VALID length (the builder also asserts no pad/eos inside any
    window at generation time).
    """

    batch, length = trace_input_ids.shape
    embeddings = model.backbone.embed_tokens(trace_input_ids.clamp_min(0))
    pooled = embeddings.new_zeros(batch, windows, embeddings.shape[-1])
    lengths = trace_attention_mask.long().sum(dim=1)
    for row in range(batch):
        valid = int(lengths[row])
        if valid < windows:
            raise ValueError(
                "trace row has %d valid tokens for %d windows; the builder must pad the trace"
                % (valid, windows)
            )
        boundaries = [
            (valid * index) // windows for index in range(windows + 1)
        ]
        for window in range(windows):
            start, stop = boundaries[window], boundaries[window + 1]
            span = embeddings[row, start:stop]
            pooled[row, window] = span.sum(dim=0) / float(max(1, stop - start)) ** 0.5
    return model.ground_latents(pooled)


def _update_distill_ema(
    model: HLWMForConditionalGeneration,
    teacher_states: List[Tensor],
    momentum: float = 0.99,
) -> None:
    with torch.no_grad():
        for layer, states in enumerate(teacher_states):
            std = states.detach().float().std().clamp_min(1.0e-6)
            current = model.distill_ema_std[layer]
            if float(current) <= 0.0:
                model.distill_ema_std[layer] = std
            else:
                model.distill_ema_std[layer] = momentum * current + (1.0 - momentum) * std


def _distill_smooth_l1(
    student_states: List[Tensor],
    teacher_states: List[Tensor],
    ema_std: Tensor,
) -> Tensor:
    total = student_states[0].new_zeros(())
    for layer, (student, teacher) in enumerate(zip(student_states, teacher_states)):
        scale = ema_std[layer].clamp_min(1.0e-6)
        total = total + F.smooth_l1_loss(
            student.float() / scale, teacher.detach().float() / scale
        )
    return total / float(len(student_states))


def v10_training_step(
    model: HLWMForConditionalGeneration,
    batch: Mapping[str, Any],
    *,
    step: int,
    gamma: float,
    accumulation_scale: float,
    warm_phase: bool = False,
    backward: bool = True,
    colar_weight: float = 0.5,
    recon_weight: float = 0.3,
    producer_weight: float = 0.5,
    step_decode_weight: float = 0.5,
    incontext_decode_weight: float = 0.5,
    router_probe_weight: float = 0.1,
    grad_probe: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """One Version 10.0 step: grouped passes with per-group backward.

    Group A (teacher): CE over the gold trace (final step excluded) plus the
    answer, through the plain channel; its pre-answer layer states are
    stashed detached as the distillation targets and update the EMA
    normalizer.  Group B (student): self-produced thoughts feed the masked
    student channel — CE_student + gamma-weighted all-layer smooth-L1 to the
    teacher states + producer MSE to compressed-gold + per-latent decode CE
    + (even steps) reconstruction, all sharing one producer graph, backward
    together.  Group C (odd steps): CoLaR teacher-forced compressed-gold
    thoughts — per-window segment CE plus answer CE.  Freeing each group's
    graph before the next pass bounds T4 memory (Session B's OOM class);
    the odd/even CoLaR/reconstruction alternation halves their cost at
    dense-supervision-every-2-steps, per the amended ledger.

    ``backward=False`` computes every group without touching gradients (the
    telemetry path measures isolated per-term norms itself).
    """

    metrics: Dict[str, float] = {}
    windows = model.config.latent_thoughts
    # Block-of-4 parity (v10.5 item 1, FATAL fix): the sampler rotates the
    # four families with period 4 at batch size 1, so a raw step parity
    # aliases with family identity — the shipped odd/even schedule would
    # have trained reconstruction ONLY on two families and CoLaR only on
    # the other two.  A 4-step block covers every family before the
    # alternation flips, decorrelating loss schedule from family.
    block_parity = (step // 4) % 2
    colar_step = block_parity == 1
    recon_step = warm_phase or block_parity == 0

    def run_backward(loss: Tensor, group: str = "loss") -> None:
        if not backward:
            return
        (loss * accumulation_scale).backward()
        if grad_probe is not None:
            # Isolated per-group gradient norms (blocking preflight): a
            # silently dead group — e.g. a no_grad-wrapped teacher pass —
            # degrades dense supervision back to the v9 sparse regime with
            # no crash, so each group must prove a nonzero norm on its own.
            grad_probe[group] = float(
                sum(
                    parameter.grad.abs().sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            model.zero_grad(set_to_none=True)

    if warm_phase:
        thought_embeds, thought_states = model.produce_latent_thoughts(
            batch["input_ids"], batch["attention_mask"]
        )
        recon_targets = torch.cat(
            (batch["trace_input_ids"], batch["target_ids"]), dim=1
        )
        recon_mask = torch.cat(
            (batch["trace_attention_mask"], batch["target_attention_mask"]), dim=1
        )
        recon = model.reconstruct_from_thoughts(thought_embeds, recon_targets, recon_mask)
        producer_targets = compressed_gold_thoughts(
            model, batch["trace_input_ids"], batch["trace_attention_mask"], windows
        )
        producer = F.mse_loss(
            model.ground_latents(thought_embeds).float(),
            producer_targets.detach().float(),
        )
        warm_loss = recon + producer_weight * producer
        run_backward(warm_loss, "warm")
        metrics["recon"] = float(recon.detach())
        metrics["producer_mse"] = float(producer.detach())
        with torch.no_grad():
            cosines = F.cosine_similarity(
                thought_states.unsqueeze(2), thought_states.unsqueeze(1), dim=-1
            )
            off_diagonal = cosines[
                :, ~torch.eye(windows, dtype=torch.bool, device=cosines.device)
            ]
            metrics["inter_latent_cosine"] = float(off_diagonal.mean())
        return metrics

    # ---- Group A: teacher --------------------------------------------------
    trace_embeds = model.backbone.embed_tokens(
        batch["trace_input_ids"].clamp_min(0)
    ) + model.mode_embedding.weight[model.MODE_CAUSAL].view(1, 1, -1)
    teacher = model.student_channel_teacher_force(
        batch["input_ids"],
        batch["attention_mask"],
        trace_embeds,
        None,
        batch["target_ids"],
        batch["target_attention_mask"],
        use_prefix_slots=False,
        collect_hidden_states=True,
        thought_attention_mask=batch["trace_attention_mask"],
        route_index=batch.get("route_index"),
    )
    teacher_trace_ce = _segment_cross_entropy(
        teacher["segment_logits"],
        batch["trace_input_ids"],
        batch["teacher_supervised_mask"],
    )
    teacher_answer_ce = _segment_cross_entropy(
        teacher["logits"], batch["target_ids"], batch["target_attention_mask"]
    )
    teacher_loss = teacher_trace_ce + teacher_answer_ce
    teacher_states = [state.detach() for state in teacher["pre_answer_layer_states"]]
    _update_distill_ema(model, teacher_states)
    run_backward(teacher_loss, "teacher")
    metrics["teacher_trace_ce"] = float(teacher_trace_ce.detach())
    metrics["teacher_answer_ce"] = float(teacher_answer_ce.detach())
    del teacher

    # ---- Group B: student + producer + distill (+ reconstruction) ---------
    thought_embeds, thought_states = model.produce_latent_thoughts(
        batch["input_ids"], batch["attention_mask"]
    )
    student = model.student_channel_teacher_force(
        batch["student_input_ids"],
        batch["student_attention_mask"],
        thought_embeds,
        thought_states,
        batch["target_ids"],
        batch["target_attention_mask"],
        use_prefix_slots=True,
        collect_hidden_states=True,
        route_index=batch.get("route_index"),
    )
    student_ce = _segment_cross_entropy(
        student["logits"], batch["target_ids"], batch["target_attention_mask"]
    )
    distill = _distill_smooth_l1(
        student["pre_answer_layer_states"], teacher_states, model.distill_ema_std
    )
    producer_targets = compressed_gold_thoughts(
        model, batch["trace_input_ids"], batch["trace_attention_mask"], windows
    )
    producer = F.mse_loss(
        model.ground_latents(thought_embeds).float(), producer_targets.detach().float()
    )
    step_logits = model.latent_step_logits(thought_states)
    window_targets = batch["trace_window_targets"]
    window_width = window_targets.shape[-1]
    step_losses = F.cross_entropy(
        step_logits.unsqueeze(2)
        .expand(-1, -1, window_width, -1)
        .reshape(-1, step_logits.shape[-1]),
        window_targets.reshape(-1),
        ignore_index=-100,
    )
    # Version 11.0 in-context aligned decode: the student's segment_logits are
    # the FROZEN lm_head read over the thought positions inside the masked
    # deployment context.  Supervising the trace windows THERE puts the premise
    # into coordinates the decoder's own output map reads at the place it
    # failed (v10.0's only premise supervision was a separate trained probe
    # head -- the readable-but-not-usable design error the J-space audit
    # measured).  Weight zero reproduces v10.0 exactly.
    incontext_step_ce = F.cross_entropy(
        student["segment_logits"]
        .unsqueeze(2)
        .expand(-1, -1, window_width, -1)
        .reshape(-1, student["segment_logits"].shape[-1]),
        window_targets.reshape(-1),
        ignore_index=-100,
    )
    group_b = (
        student_ce
        + gamma * distill
        + producer_weight * producer
        + step_decode_weight * step_losses
        + incontext_decode_weight * incontext_step_ce
    )
    if recon_step:
        recon_targets = torch.cat(
            (batch["trace_input_ids"], batch["target_ids"]), dim=1
        )
        recon_mask = torch.cat(
            (batch["trace_attention_mask"], batch["target_attention_mask"]), dim=1
        )
        recon = model.reconstruct_from_thoughts(thought_embeds, recon_targets, recon_mask)
        group_b = group_b + recon_weight * recon
        metrics["recon"] = float(recon.detach())
    if "family_index" in batch and model.config.mlp_expert_count > 0:
        router_ce = F.cross_entropy(
            model.route_family_logits(thought_states), batch["family_index"]
        )
        group_b = group_b + router_probe_weight * router_ce
        metrics["router_probe_ce"] = float(router_ce.detach())
        with torch.no_grad():
            metrics["router_probe_accuracy"] = float(
                (
                    model.route_family_logits(thought_states).argmax(dim=-1)
                    == batch["family_index"]
                ).float().mean()
            )
    run_backward(group_b, "student")
    metrics["student_ce"] = float(student_ce.detach())
    metrics["incontext_step_ce"] = float(incontext_step_ce.detach())
    metrics["distill_l1"] = float(distill.detach())
    metrics["producer_mse"] = float(producer.detach())
    metrics["latent_step_ce"] = float(step_losses.detach())
    with torch.no_grad():
        cosines = F.cosine_similarity(
            thought_states.unsqueeze(2), thought_states.unsqueeze(1), dim=-1
        )
        off_diagonal = cosines[
            :, ~torch.eye(windows, dtype=torch.bool, device=cosines.device)
        ]
        metrics["inter_latent_cosine"] = float(off_diagonal.mean())
        gates = [
            float(torch.tanh(layer.self_attn.prefix_attn_gate.float()).abs().median())
            for layer in model.backbone.layers
            if layer.self_attn.prefix_attn_gate is not None
        ]
        if gates:
            metrics["prefix_gate_median"] = float(
                torch.tensor(gates).median()
            )
    del student, thought_embeds, thought_states

    # ---- Group C: CoLaR teacher-forced consumer pass (odd steps) ----------
    if colar_step:
        colar_thoughts = compressed_gold_thoughts(
            model, batch["trace_input_ids"], batch["trace_attention_mask"], windows
        )
        colar = model.student_channel_teacher_force(
            batch["student_input_ids"],
            batch["student_attention_mask"],
            colar_thoughts,
            colar_thoughts,
            batch["target_ids"],
            batch["target_attention_mask"],
            use_prefix_slots=True,
            route_index=batch.get("route_index"),
        )
        window_first = window_targets[:, :, 0]
        colar_segment = _segment_cross_entropy(
            colar["segment_logits"],
            window_first,
            (window_first != -100).long(),
        )
        colar_answer = _segment_cross_entropy(
            colar["logits"], batch["target_ids"], batch["target_attention_mask"]
        )
        colar_loss = colar_weight * (colar_segment + colar_answer)
        run_backward(colar_loss, "colar")
        metrics["colar_segment_ce"] = float(colar_segment.detach())
        metrics["colar_answer_ce"] = float(colar_answer.detach())
    return metrics


# ----------------------------------------------------------------------
# Version 10.0 run orchestration.


def v10_anchor_indices(dataset: Any) -> Dict[str, List[int]]:
    """Anchor indices bucketed by family and masked flag.

    Only rows carrying a ``v10`` payload participate in Version 10.0
    training; the remaining corpus rows are unused this study (disclosed:
    presentations bind before rows at this budget, and repeated anchor
    epochs are the regime where the small-scale literature's synthetic-task
    wins live).
    """

    buckets: Dict[str, List[int]] = {}
    for index, row in enumerate(dataset.rows):
        payload = row.get("v10") if isinstance(row, Mapping) else None
        if not payload:
            continue
        family = str(payload.get("family", "numeric"))
        # Normalized rows carry a TOP-LEVEL masked flag; the nested
        # 'evaluation' read matched nothing, silently disabling the
        # Amendment A1 masked-oversampling floor for all of session I-5
        # (masked share 0.37 sampled vs the preregistered >= 0.5).
        masked = bool(row.get("masked"))
        buckets.setdefault("%s|%s" % (family, "m" if masked else "u"), []).append(index)
    if not buckets:
        raise ValueError("no v10 anchor rows found; run the Version 10.0 builder")
    return buckets


def v10_sample_indices(
    buckets: Mapping[str, List[int]],
    *,
    count: int,
    masked_floor: float,
    seed: int,
) -> List[int]:
    """Deterministic family-cycling sample with masked oversampling.

    Families rotate so every gradient-accumulation window is family-balanced
    (the experts' load balance is a dataloader property, never a loss);
    within maskable families, masked rows are drawn with the probability
    that lifts the OVERALL masked share to ``masked_floor`` (Amendment A1),
    sampling with replacement — the generator's ~49%-of-maskable masked
    supply is below the floor, so oversampling is the designed mechanism,
    not an accident.
    """

    families = sorted({key.split("|")[0] for key in buckets})
    maskable = [
        family for family in families if "%s|m" % family in buckets
    ]
    if maskable:
        # overall_masked = (len(maskable)/len(families)) * p  =>  solve for p.
        needed = masked_floor * len(families) / max(1, len(maskable))
        masked_probability = min(1.0, max(0.0, needed))
    else:
        masked_probability = 0.0
    generator = torch.Generator().manual_seed(seed)
    indices: List[int] = []
    for position in range(count):
        family = families[position % len(families)]
        use_masked = (
            "%s|m" % family in buckets
            and float(torch.rand((), generator=generator)) < masked_probability
        )
        key = "%s|%s" % (family, "m" if use_masked else "u")
        if key not in buckets:
            key = "%s|%s" % (family, "u" if use_masked else "m")
        pool = buckets[key]
        indices.append(pool[int(torch.randint(len(pool), (), generator=generator))])
    return indices


def v10_recon_digit_accuracy(
    model: HLWMForConditionalGeneration,
    batches: Sequence[Mapping[str, Any]],
    digit_ids: Sequence[int],
) -> float:
    """Held-out digit-level reconstruction accuracy (the warm-gate metric).

    Teacher-forced argmax over the reconstruction targets, restricted to
    digit-token positions: the premise and trace values are digit strings,
    so digit positions carry exactly the withheld content.
    """

    digit_set = torch.tensor(sorted(set(int(d) for d in digit_ids)))
    correct = 0
    total = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            thoughts, _ = model.produce_latent_thoughts(
                batch["input_ids"], batch["attention_mask"]
            )
            targets = torch.cat((batch["trace_input_ids"], batch["target_ids"]), dim=1)
            target_mask = torch.cat(
                (batch["trace_attention_mask"], batch["target_attention_mask"]), dim=1
            )
            cue = model.recon_cue_embedding.view(1, 1, -1).expand(
                thoughts.shape[0], 1, -1
            )
            causal_mode = model.mode_embedding.weight[model.MODE_CAUSAL].view(1, 1, -1)
            target_embeds = model.backbone.embed_tokens(targets.clamp_min(0)) + causal_mode
            combined = torch.cat(
                (thoughts, cue.to(thoughts.dtype), target_embeds.to(thoughts.dtype)), dim=1
            )
            ones = torch.ones(
                thoughts.shape[0],
                thoughts.shape[1] + 1,
                dtype=target_mask.dtype,
                device=target_mask.device,
            )
            hidden = model.backbone(
                inputs_embeds=combined,
                attention_mask=torch.cat((ones, target_mask), dim=1),
                attention_mode="causal",
            )
            hidden = model._apply_root(hidden)
            length = targets.shape[1]
            predicted = model.lm_head(hidden[:, -length - 1 : -1]).argmax(dim=-1)
            digit_positions = (
                torch.isin(targets, digit_set.to(targets.device)) & target_mask.bool()
            )
            correct += int((predicted[digit_positions] == targets[digit_positions]).sum())
            total += int(digit_positions.sum())
    if was_training:
        model.train()
    return correct / max(1, total)


def v10_masked_numeric_indices(dataset: Any, *, rows: int = 32) -> List[int]:
    """Row indices the go/no-go tripwire will actually decode.

    Split out from ``v10_masked_numeric_em`` so the preflight can certify
    that this population is non-empty BEFORE training: the tripwire's
    selector, not just its threshold, is now an audited instrument.
    """

    picked: List[int] = []
    for index, row in enumerate(dataset.rows):
        payload = row.get("v10") if isinstance(row, Mapping) else None
        if not payload or payload.get("family") != "numeric":
            continue
        # Normalized rows carry a TOP-LEVEL masked flag; the nested
        # 'evaluation' read matched nothing (session I-5).
        if not bool(row.get("masked")):
            continue
        # The grader is substring containment, so a row whose answer already
        # occurs inside its own masked prompt (typically nested in a
        # distractor: expected 546 inside the reference value 69546) can be
        # scored a hit by copying rather than by transfer. 5 of the corpus's
        # 2,212 masked rows collide this way; excluding them costs nothing
        # and keeps every tripwire hit attributable to the channel.
        expected = (row.get("answer_spec") or {}).get("expected")
        if expected is not None and str(expected) in str(row.get("masked_prompt") or ""):
            continue
        picked.append(index)
        if len(picked) >= max(1, int(rows)):
            break
    return picked


def v10_masked_numeric_em(
    model: HLWMForConditionalGeneration,
    dataset: Any,
    collator: Any,
    device: torch.device,
    *,
    rows: int = 32,
) -> float:
    """Masked NUMERIC exact-match through the live channel (the go/no-go).

    Guess-chance is ~0 on numerics; abstention and ordering are excluded so
    a dead channel cannot ride their guess floors past the gate
    (Amendment A3).
    """

    picked = v10_masked_numeric_indices(dataset, rows=rows)
    if not picked:
        # An empty population is an INSTRUMENT fault, never evidence. In
        # session I-5 this path returned 0.0 and both seeds aborted at the
        # go/no-go without a single row ever being decoded — a working
        # channel would have aborted identically. Fail loudly instead; the
        # preflight now certifies this population before any GPU hours.
        raise ValueError(
            "go/no-go instrument is vacuous: no masked numeric anchor rows "
            "in the dataset (check the row schema, not the model)"
        )
    hits = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for index in picked:
            row = dataset.rows[index]
            batch = move_model_batch(collator([row]), device)
            thoughts, states = model.produce_latent_thoughts(
                batch["input_ids"], batch["attention_mask"]
            )
            decoded = model.decode_candidate_v10(
                batch["student_input_ids"],
                batch["student_attention_mask"],
                thoughts,
                states,
                max_new_tokens=24,
                temperature=0.0,
                route_index=batch.get("route_index"),
            )
            text = collator.tokenizer.decode(
                decoded[0].tolist(), skip_special_tokens=True
            )
            expected = (row.get("answer_spec") or {}).get("expected")
            if expected is not None and str(expected) in text:
                hits += 1
    if was_training:
        model.train()
    return hits / max(1, len(picked))


def warm_gate_passed(em_200: float, em_600: float, floor: float) -> bool:
    """Warm reconstruction gate (plan section 13 amendment, session I-4).

    The 2x growth clause exists to reject a channel that starts low and
    flat-lines; it applies only when the step-200 EM sits below the floor.
    A channel already above the floor at step 200 cannot double a value
    above 0.5, so the unamended conjunction rejected exactly the strongest
    starts (both session I-4 seeds aborted at em_200 0.56-0.60 against
    floor 0.50).
    """

    if em_600 < floor:
        return False
    if em_200 >= floor:
        return True
    return em_600 >= 2.0 * em_200


def run_v10_training(
    *,
    args: argparse.Namespace,
    model: HLWMForConditionalGeneration,
    tokenizer: Any,
    train_data: Any,
    validation_data: Any,
    collator: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    start_step: int,
    optimizer_updates: int,
    skipped_optimizer_updates: int,
) -> Dict[str, Any]:
    """Warm phase, gated main phase, and the w/o-L1 branch (Version 10.0).

    Rungs are decided by the audit, but two in-run tripwires abort into the
    preregistered failure branch early: the warm reconstruction gate
    (absolute digit-EM floor fixed before the run plus a 2x growth
    requirement) and the masked-numeric go/no-go.  The wall clock, not the
    step counter, bounds the main phase (the amended ledger's rule), with a
    1,200-step validity floor.
    """

    log_path = args.output_dir / "metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(args.steps)
    warm_steps = int(args.warm_steps)
    digit_ids = [
        identifier
        for digit in "0123456789"
        for identifier in tokenizer.encode(digit, add_special_tokens=False)
    ]
    buckets = v10_anchor_indices(train_data)
    order = v10_sample_indices(
        buckets,
        count=total_steps * args.gradient_accumulation + args.gradient_accumulation,
        masked_floor=args.masked_fraction_floor,
        seed=args.seed + 17,
    )
    validation_buckets = v10_anchor_indices(validation_data)
    held_out = [
        validation_data.rows[index]
        for indices in validation_buckets.values()
        for index in indices[:4]
    ][:16]
    recon_eval_batches = [
        move_model_batch(collator([row]), device) for row in held_out
    ]
    warm_curve: Dict[int, float] = {}
    wall_start = time.time()
    accumulated = 0
    masked_seen = 0
    rows_seen = 0
    cursor = 0
    verdict: Dict[str, Any] = {"mode": "v10", "warm_steps": warm_steps}
    if start_step >= warm_steps and getattr(args, "resume", None):
        # A resume past the warm phase cannot re-run the warm gate; carry
        # the recorded result from the resumed run's own metrics (they sit
        # beside the checkpoint in the attached prior output) so the rung
        # readout cannot misread a live warm-gate pass as a rung-1 failure
        # (session I-5 salvage: binding_rung reads warm_gate_passed).
        sibling_metrics = Path(str(args.resume)).parent / "metrics.jsonl"
        if sibling_metrics.exists():
            for line in sibling_metrics.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "warm_gate" in record:
                    carried = dict(record["warm_gate"])
                    carried["carried_from_resume"] = True
                    verdict["warm_gate"] = carried
    if int(start_step) == int(args.gonogo_step) and start_step < total_steps:
        # Scheduled saves land on save_every multiples, so a resume landing
        # exactly on the go/no-go step means the prior session aborted there
        # (session I-5). Re-evaluate the tripwire on the restored weights
        # instead of silently training past a binding abort; a genuine pass
        # simply continues.
        em = v10_masked_numeric_em(model, validation_data, collator, device)
        verdict["gonogo_masked_numeric_em"] = em
        verdict["resumed_at_gonogo"] = int(start_step)
        append_jsonl(log_path, {"step": start_step, "gonogo_masked_numeric_em": em})
        print(json.dumps({"gonogo_masked_numeric_em": em, "resumed_at_gonogo": True}))
        if em < args.gonogo_masked_numeric_em:
            save_checkpoint(
                args.output_dir / ("checkpoint-step-%06d.pt" % start_step),
                model=model, optimizer=optimizer, scaler=scaler,
                step=start_step, optimizer_updates=optimizer_updates,
                skipped_optimizer_updates=skipped_optimizer_updates, args=args,
            )
            verdict["aborted"] = "gonogo"
            append_jsonl(log_path, {"v10_verdict": verdict})
            return verdict
    for step in range(start_step, total_steps):
        row = train_data.rows[order[cursor % len(order)]]
        cursor += 1
        batch = move_model_batch(collator([row]), device)
        rows_seen += 1
        masked_seen += int(bool(batch["masked_rows"].any()))
        warm_phase = step < warm_steps
        telemetry_step = (
            not warm_phase and args.telemetry_every > 0 and step % args.telemetry_every == 0
        )
        if telemetry_step:
            probe: Dict[str, float] = {}
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                v10_training_step(
                    model, batch, step=step, gamma=args.distill_gamma,
                    accumulation_scale=1.0, grad_probe=probe,
                )
            append_jsonl(log_path, {"step": step, "v10_grad_probe": probe})
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            metrics = v10_training_step(
                model,
                batch,
                step=step,
                gamma=args.distill_gamma,
                accumulation_scale=1.0 / args.gradient_accumulation,
                warm_phase=warm_phase,
                incontext_decode_weight=float(
                    getattr(args, "incontext_decode_weight", 0.5)
                ),
            )
        accumulated += 1
        if accumulated == args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            update_index = optimizer_updates
            for group in optimizer.param_groups:
                group["lr"] = cosine_learning_rate(
                    update_index,
                    math.ceil(total_steps / args.gradient_accumulation),
                    args.warmup_updates,
                    args.learning_rate,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
            accumulated = 0
        if step % 50 == 0 or step + 1 == total_steps:
            record = {"step": step, "phase": "warm" if warm_phase else "main"}
            # The 50-step cadence aliases with the period-4 family rotation
            # (50 = 2 mod 4), so every logged row was abstention or ordering
            # in session I-5 and the series was misread as a masked/unmasked
            # split. Stamp the row identity so telemetry is deconfoundable.
            record["family_index"] = int(batch["family_index"].reshape(-1)[0])
            record["masked_row"] = bool(batch["masked_rows"].reshape(-1)[0])
            record.update({key: round(value, 6) for key, value in metrics.items()})
            append_jsonl(log_path, record)
        if warm_phase and step + 1 in (200, 400, warm_steps):
            accuracy = v10_recon_digit_accuracy(model, recon_eval_batches, digit_ids)
            warm_curve[step + 1] = accuracy
            append_jsonl(log_path, {"step": step + 1, "warm_recon_digit_em": accuracy})
            print(json.dumps({"warm_recon_digit_em": {str(step + 1): accuracy}}))
        if step + 1 == warm_steps and start_step < warm_steps:
            em_600 = warm_curve.get(warm_steps, 0.0)
            em_200 = warm_curve.get(200, 0.0)
            gate_pass = warm_gate_passed(em_200, em_600, args.warm_em_floor)
            verdict["warm_gate"] = {
                "em_200": em_200,
                "em_600": em_600,
                "floor": args.warm_em_floor,
                "passed": bool(gate_pass),
            }
            append_jsonl(log_path, {"step": step + 1, "warm_gate": verdict["warm_gate"]})
            if not gate_pass:
                save_checkpoint(
                    args.output_dir / ("checkpoint-step-%06d.pt" % (step + 1)),
                    model=model, optimizer=optimizer, scaler=scaler,
                    step=step + 1, optimizer_updates=optimizer_updates,
                    skipped_optimizer_updates=skipped_optimizer_updates, args=args,
                )
                verdict["aborted"] = "warm_gate"
                # The notebook reads the abort status from metrics.jsonl;
                # session I-4 exited here without it and the audit ran a
                # warm-only checkpoint as if training had completed.
                append_jsonl(log_path, {"v10_verdict": verdict})
                return verdict
        if step + 1 == args.gonogo_step:
            em = v10_masked_numeric_em(model, validation_data, collator, device)
            verdict["gonogo_masked_numeric_em"] = em
            append_jsonl(log_path, {"step": step + 1, "gonogo_masked_numeric_em": em})
            print(json.dumps({"gonogo_masked_numeric_em": em}))
            if em < args.gonogo_masked_numeric_em:
                save_checkpoint(
                    args.output_dir / ("checkpoint-step-%06d.pt" % (step + 1)),
                    model=model, optimizer=optimizer, scaler=scaler,
                    step=step + 1, optimizer_updates=optimizer_updates,
                    skipped_optimizer_updates=skipped_optimizer_updates, args=args,
                )
                verdict["aborted"] = "gonogo"
                append_jsonl(log_path, {"v10_verdict": verdict})
                return verdict
        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            if accumulated == 0:
                save_checkpoint(
                    args.output_dir / ("checkpoint-step-%06d.pt" % (step + 1)),
                    model=model, optimizer=optimizer, scaler=scaler,
                    step=step + 1, optimizer_updates=optimizer_updates,
                    skipped_optimizer_updates=skipped_optimizer_updates, args=args,
                )
        if (
            accumulated == 0
            and not warm_phase
            and time.time() - wall_start >= args.max_runtime_hours * 3600
        ):
            if step + 1 < 1200:
                verdict["wall_clock_below_validity_floor"] = step + 1
            verdict["wall_clock_stop_step"] = step + 1
            save_checkpoint(
                args.output_dir / ("checkpoint-step-%06d.pt" % (step + 1)),
                model=model, optimizer=optimizer, scaler=scaler,
                step=step + 1, optimizer_updates=optimizer_updates,
                skipped_optimizer_updates=skipped_optimizer_updates, args=args,
            )
            break
    verdict["masked_fraction_seen"] = masked_seen / max(1, rows_seen)
    verdict["completed_steps"] = min(total_steps, step + 1)

    # ---- Final full checkpoint + recorded hash, then the w/o-L1 branch. ----
    final_path = args.output_dir / "checkpoint-v10-full.pt"
    save_checkpoint(
        final_path, model=model, optimizer=optimizer, scaler=scaler,
        step=verdict["completed_steps"], optimizer_updates=optimizer_updates,
        skipped_optimizer_updates=skipped_optimizer_updates, args=args,
    )
    verdict["full_checkpoint_sha256"] = sha256(final_path)
    try:
        try:
            from .evaluate_v10 import state_dict_sha256
        except ImportError:
            from evaluate_v10 import state_dict_sha256
        # The audit's w/o-L1 contamination guard compares STATE-DICT hashes
        # (not file hashes); record the training-time value so the audit can
        # refuse a wrong or contaminated full arm.
        verdict["full_state_sha256"] = state_dict_sha256(model)
    except Exception as error:  # diagnostic-only; never fail the run for it
        verdict["full_state_sha256_error"] = str(error)
    branch_lr = float(optimizer.param_groups[0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = branch_lr  # pinned constant: no scheduler past its horizon
    branch_steps = int(args.wo_l1_branch_steps)
    for branch_step in range(branch_steps):
        row = train_data.rows[order[cursor % len(order)]]
        cursor += 1
        batch = move_model_batch(collator([row]), device)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            v10_training_step(
                model, batch,
                step=verdict["completed_steps"] + branch_step,
                gamma=0.0,  # the ablated term, everything else unchanged
                accumulation_scale=1.0 / args.gradient_accumulation,
            )
        if (branch_step + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    branch_path = args.output_dir / "checkpoint-v10-wo-l1-branch.pt"
    save_checkpoint(
        branch_path, model=model, optimizer=optimizer, scaler=scaler,
        step=verdict["completed_steps"] + branch_steps,
        optimizer_updates=optimizer_updates,
        skipped_optimizer_updates=skipped_optimizer_updates, args=args,
    )
    verdict["wo_l1_branch_sha256"] = sha256(branch_path)
    if verdict["wo_l1_branch_sha256"] == verdict["full_checkpoint_sha256"]:
        raise RuntimeError(
            "w/o-L1 branch checkpoint is byte-identical to the full arm; "
            "the branch did not train (in-place contamination guard)"
        )
    append_jsonl(log_path, {"v10_verdict": verdict})
    return verdict


def v10_preflight(
    *,
    args: argparse.Namespace,
    model: HLWMForConditionalGeneration,
    tokenizer: Any,
    train_data: Any,
    validation_data: Any,
    collator: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Blocking on-device preflight (Version 10.0): no GPU-hour commitment
    until every check passes on the REAL model on the REAL device.

    This cell carries the CUDA-class validation (autocast, the two-SDPA gate
    path, memory peak, throughput) that unit tests on CPU cannot see — the
    defect classes that ended sessions B, D, and G.  It also calibrates the
    distillation weight gamma from measured loss magnitudes, per the plan's
    smoke phase.  Any failed check returns ``passed: False`` and the
    notebook aborts before training.
    """

    report: Dict[str, Any] = {"passed": False, "checks": {}}
    checks = report["checks"]
    buckets = v10_anchor_indices(train_data)
    sample_rows = [train_data.rows[indices[0]] for indices in buckets.values()][:4]
    batches = [move_model_batch(collator([row]), device) for row in sample_rows]
    batch = batches[0]
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # (1) Per-group isolated gradient liveness (odd step so CoLaR runs too).
    probe: Dict[str, float] = {}
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=cuda):
        v10_training_step(
            model, batch, step=1, gamma=10.0, accumulation_scale=1.0, grad_probe=probe,
        )
    checks["group_grad_norms"] = probe
    checks["groups_live"] = bool(probe) and all(value > 1.0e-8 for value in probe.values())

    # (2) Module-level liveness: gates, slot projector, producer each get
    # nonzero gradient from one ordinary step (the three-part test).
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=cuda):
        v10_training_step(model, batch, step=0, gamma=10.0, accumulation_scale=1.0)
    gate_norm = sum(
        float(layer.self_attn.prefix_attn_gate.grad.abs().sum())
        for layer in model.backbone.layers
        if layer.self_attn.prefix_attn_gate is not None
        and layer.self_attn.prefix_attn_gate.grad is not None
    )
    projector_norm = sum(
        float(p.grad.abs().sum())
        for p in model.kv_prefix_projector.parameters()
        if p.grad is not None
    )
    producer_norm = sum(
        float(p.grad.abs().sum())
        for p in model.latent_projection.parameters()
        if p.grad is not None
    )
    model.zero_grad(set_to_none=True)
    checks["gate_grad_norm"] = gate_norm
    checks["projector_grad_norm"] = projector_norm
    checks["producer_grad_norm"] = producer_norm
    checks["liveness_three_part"] = gate_norm > 0 and projector_norm > 0 and producer_norm > 0

    # (3) Gate-closed equivalence on-device: zero every gate, compare against
    # the slotless path (catches two-SDPA double-counting exactly), restore.
    model.eval()
    with torch.no_grad():
        thoughts, states = model.produce_latent_thoughts(
            batch["input_ids"], batch["attention_mask"]
        )
        saved_gates = [
            layer.self_attn.prefix_attn_gate.detach().clone()
            for layer in model.backbone.layers
            if layer.self_attn.prefix_attn_gate is not None
        ]
        for layer in model.backbone.layers:
            if layer.self_attn.prefix_attn_gate is not None:
                layer.self_attn.prefix_attn_gate.zero_()
        gated = model.student_channel_teacher_force(
            batch["student_input_ids"], batch["student_attention_mask"],
            thoughts, states, batch["target_ids"], batch["target_attention_mask"],
            use_prefix_slots=True,
        )["logits"]
        plain = model.student_channel_teacher_force(
            batch["student_input_ids"], batch["student_attention_mask"],
            thoughts, None, batch["target_ids"], batch["target_attention_mask"],
            use_prefix_slots=False,
        )["logits"]
        for layer, saved in zip(
            [l for l in model.backbone.layers if l.self_attn.prefix_attn_gate is not None],
            saved_gates,
        ):
            layer.self_attn.prefix_attn_gate.copy_(saved)
        gate_closed_delta = float((gated - plain).abs().max())
        checks["gate_closed_max_delta"] = gate_closed_delta
        checks["gate_closed_equivalent"] = gate_closed_delta < 5.0e-3

        # (4) Gate-perturbation sensitivity: nudging g must move logits.
        for layer in model.backbone.layers:
            if layer.self_attn.prefix_attn_gate is not None:
                layer.self_attn.prefix_attn_gate.add_(0.5)
        moved = model.student_channel_teacher_force(
            batch["student_input_ids"], batch["student_attention_mask"],
            thoughts, states, batch["target_ids"], batch["target_attention_mask"],
            use_prefix_slots=True,
        )["logits"]
        for layer, saved in zip(
            [l for l in model.backbone.layers if l.self_attn.prefix_attn_gate is not None],
            saved_gates,
        ):
            layer.self_attn.prefix_attn_gate.copy_(saved)
        checks["gate_perturbation_delta"] = float((moved - gated).abs().max())
        checks["gate_perturbation_live"] = checks["gate_perturbation_delta"] > 1.0e-6

        # (5) Golden decode parity on-device through the KV cache.
        decoded = model.decode_candidate_v10(
            batch["student_input_ids"], batch["student_attention_mask"],
            thoughts, states, max_new_tokens=4, temperature=0.0,
        )
        forced = model.student_channel_teacher_force(
            batch["student_input_ids"], batch["student_attention_mask"],
            thoughts, states, decoded, torch.ones_like(decoded),
        )["logits"]
        checks["decode_parity"] = bool(torch.equal(forced.argmax(dim=-1), decoded))
    model.train()

    # (6) Gamma calibration: pick the sweep value whose scaled distillation
    # magnitude best matches the student CE (measured, not assumed).
    gamma_report: Dict[str, Dict[str, float]] = {}
    best_gamma, best_gap = None, None
    for gamma in (5.0, 10.0, 20.0):
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=cuda):
            metrics = v10_training_step(
                model, batches[min(1, len(batches) - 1)], step=0, gamma=gamma,
                accumulation_scale=1.0, backward=False,
            )
        scaled = gamma * metrics["distill_l1"]
        gap = abs(scaled - metrics["student_ce"])
        gamma_report[str(gamma)] = {
            "distill_l1": metrics["distill_l1"],
            "scaled": scaled,
            "student_ce": metrics["student_ce"],
        }
        if best_gap is None or gap < best_gap:
            best_gamma, best_gap = gamma, gap
    checks["gamma_sweep"] = gamma_report
    report["calibrated_gamma"] = best_gamma

    # (7) Throughput: median s/step over 12 real steps with updates.
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay), lr=args.learning_rate
    )
    timings: List[float] = []
    for index in range(12):
        started = time.time()
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=cuda):
            v10_training_step(
                model, batches[index % len(batches)], step=index, gamma=10.0,
                accumulation_scale=1.0,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize(device)
        timings.append(time.time() - started)
    timings_sorted = sorted(timings[2:])  # discard warmup iterations
    seconds_per_step = timings_sorted[len(timings_sorted) // 2]
    checks["seconds_per_step"] = seconds_per_step
    projected_hours = (args.steps * seconds_per_step + args.wo_l1_branch_steps * seconds_per_step) / 3600.0
    checks["projected_training_hours"] = projected_hours

    # (8) Memory gate (CUDA only): peak <= 13 GB against the 15.3 usable.
    if cuda:
        peak_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
        checks["peak_memory_gb"] = peak_gb
        checks["memory_within_gate"] = peak_gb <= 13.0
    else:
        checks["memory_within_gate"] = True

    # (9) Teacher ceiling per family (report-only; Amendment A5): the
    # distillation target is bounded by teacher-with-premise accuracy.
    ceilings: Dict[str, float] = {}
    validation_buckets = v10_anchor_indices(validation_data)
    model.eval()
    with torch.no_grad():
        for key, indices in sorted(validation_buckets.items()):
            family = key.split("|")[0]
            if family in ceilings:
                continue
            hits, seen = 0, 0
            for index in indices[:8]:
                row = validation_data.rows[index]
                sample = move_model_batch(collator([row]), device)
                trace_embeds = model.backbone.embed_tokens(
                    sample["trace_input_ids"].clamp_min(0)
                ) + model.mode_embedding.weight[model.MODE_CAUSAL].view(1, 1, -1)
                out = model.student_channel_teacher_force(
                    sample["input_ids"], sample["attention_mask"], trace_embeds, None,
                    sample["target_ids"], sample["target_attention_mask"],
                    use_prefix_slots=False,
                    thought_attention_mask=sample["trace_attention_mask"],
                )
                predicted = out["logits"].argmax(dim=-1)
                mask = sample["target_attention_mask"].bool()
                hits += int((predicted[mask] == sample["target_ids"][mask]).all())
                seen += 1
            ceilings[family] = hits / max(1, seen)
    model.train()
    checks["teacher_ceiling_by_family"] = ceilings

    # (10) Tripwire POPULATIONS, not just thresholds (the session I-5
    # lesson): both binding aborts read a selector over the row schema, and
    # a selector that matches nothing manufactures a scientific verdict out
    # of nothing. Certify them here, before any GPU-hour commitment.
    # Both tripwires read VALIDATION rows, so validation is what gets
    # certified; the training buckets are checked too because the masked
    # oversampling floor (A1) reads them.
    masked_train = sorted(key for key in buckets if key.endswith("|m"))
    masked_validation = sorted(
        key for key in validation_buckets if key.endswith("|m")
    )
    checks["masked_anchor_buckets_train"] = masked_train
    checks["masked_anchor_buckets_validation"] = masked_validation
    checks["masked_numeric_rows"] = len(
        v10_masked_numeric_indices(validation_data, rows=64)
    )
    checks["tripwire_population_live"] = (
        bool(masked_train)
        and bool(masked_validation)
        and checks["masked_numeric_rows"] > 0
    )

    required = (
        "groups_live", "liveness_three_part", "gate_closed_equivalent",
        "gate_perturbation_live", "decode_parity", "memory_within_gate",
        "tripwire_population_live",
    )
    report["passed"] = all(bool(checks.get(name)) for name in required)
    report["failed_checks"] = [name for name in required if not bool(checks.get(name))]
    return report


# The entry point stays at the very end of the module: main() resolves names
# at call time, and the Session I launch crashed on a NameError because the
# Version 10.0 engine was appended below the old invocation point.
if __name__ == "__main__":
    main()
