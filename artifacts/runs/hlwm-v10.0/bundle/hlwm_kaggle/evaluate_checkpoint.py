"""Reload an HLWM checkpoint and audit the three decoupled Version 8.0 claims.

Claim S (selection): workspace-informed heads versus self-consistency and
max-logprob on the SAME causal-channel candidate pool, with a difficulty-
stratified headline (verifier-weighted SC vs plain SC, McNemar mid-p).
Claim A (abstention): the conformally calibrated publish rule versus a
mean-logprob abstention baseline on the risk-coverage curve (AUGRC).
Claim G (generation): the repaired workspace channel versus the causal
channel — non-inferiority, zero scaffold leak, live latent read-out.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.nn import functional as F

try:
    from .data import (
        RESPONSE_CUE_TEXT,
        Reasoning9000Dataset,
        encode_preserving_ends as shared_encode_preserving_ends,
    )
    from .modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration
    from .semantic_grading import grade_semantic_answer
except ImportError:  # Direct execution inside the Kaggle dataset directory.
    from data import (
        RESPONSE_CUE_TEXT,
        Reasoning9000Dataset,
        encode_preserving_ends as shared_encode_preserving_ends,
    )
    from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration
    from semantic_grading import grade_semantic_answer


SAFE_FALLBACK = "I cannot provide a sufficiently supported answer."

# Resolved once in ``main``: BF16 on devices that support it, otherwise FP16.
AMP_DTYPE = torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--canvas-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--candidate-temperatures",
        type=str,
        default="0.0",
        help="Comma-separated fan-in decode temperatures; first must be 0.0.",
    )
    parser.add_argument(
        "--anchor-ids-file",
        type=Path,
        default=None,
        help=(
            "JSON list of anchor episode ids selected by the in-session "
            "difficulty banding (Version 9.0); when present, exactly these "
            "anchors are audited in list order before the SQL tail."
        ),
    )
    parser.add_argument(
        "--max-sql-rows",
        type=int,
        default=64,
        help="Cap on non-anchor (text-to-SQL) audit rows (Version 9.0).",
    )
    parser.add_argument(
        "--max-audit-hours",
        type=float,
        default=0.0,
        help=(
            "Wall-clock guard: stop adding rows past this budget and report "
            "the truncation (row priority already puts anchors first; 0 "
            "disables)."
        ),
    )
    return parser.parse_args()


# Graders cheap enough to score every candidate and to vote over sampled
# causal baselines; code-executing graders still grade the published output.
LIGHT_GRADERS = ("numeric", "unit", "ordering", "abstention", "sql_exact")


def answer_signature(text: str, spec: Mapping[str, Any]) -> str:
    """Equivalence key for self-consistency voting, matched to the graders."""

    semantic = grade_semantic_answer(text, spec or {})
    grader = str(semantic.get("grader", "none"))
    numbers = semantic.get("numbers") or []
    if grader in ("numeric", "unit"):
        return "%s:%.6g" % (grader, numbers[-1]) if numbers else grader + ":none"
    if grader == "ordering":
        expected = (spec or {}).get("expected") or []
        tail = numbers[-len(expected) :] if expected else numbers
        return "ordering:" + ",".join("%.6g" % value for value in tail)
    return grader + ":" + str(semantic.get("normalized", ""))[:80]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_split(data_dir: Path, split: str) -> Path:
    for candidate in (
        data_dir / "master" / (split + ".jsonl"),
        data_dir / (split + ".jsonl"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find %s under %s" % (split, data_dir))


def normalized_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalized_tokens(prediction)
    target = normalized_tokens(reference)
    if not predicted and not target:
        return 1.0
    if not predicted or not target:
        return 0.0
    target_counts: Dict[str, int] = {}
    for token in target:
        target_counts[token] = target_counts.get(token, 0) + 1
    overlap = 0
    for token in predicted:
        available = target_counts.get(token, 0)
        if available:
            overlap += 1
            target_counts[token] = available - 1
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2.0 * precision * recall / max(1.0e-12, precision + recall)


def decode_new_tokens(tokenizer: Any, generated: torch.Tensor, prompt_length: int) -> str:
    return tokenizer.decode(
        generated[0, prompt_length:].detach().cpu(), skip_special_tokens=True
    ).strip()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation without SciPy (average ranks for ties)."""

    def average_ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        cursor = 0
        while cursor < len(order):
            tail = cursor
            while (
                tail + 1 < len(order)
                and values[order[tail + 1]] == values[order[cursor]]
            ):
                tail += 1
            rank = (cursor + tail) / 2.0 + 1.0
            for position in range(cursor, tail + 1):
                ranks[order[position]] = rank
            cursor = tail + 1
        return ranks

    if len(first) != len(second) or len(first) < 3:
        return 0.0
    ranks_a = average_ranks(list(first))
    ranks_b = average_ranks(list(second))
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    covariance = sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b)
    )
    variance_a = sum((a - mean_a) ** 2 for a in ranks_a)
    variance_b = sum((b - mean_b) ** 2 for b in ranks_b)
    denominator = math.sqrt(variance_a * variance_b)
    return covariance / denominator if denominator > 0 else 0.0


def mcnemar_mid_p(wins: int, losses: int) -> float:
    """One-sided mid-p McNemar test on discordant pairs.

    ``wins`` counts rows the treatment got right and the baseline wrong;
    ``losses`` the reverse.  Mid-p is preferred over the exact test, which is
    over-conservative at small n (Fagerland et al. 2013).
    """

    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(wins + 1, total + 1)) / 2.0**total
    at_observed = math.comb(total, wins) / 2.0**total
    return min(1.0, tail + 0.5 * at_observed)


def augrc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the generalized risk-coverage curve (Traub et al. 2024).

    Generalized risk at coverage c is P(error AND accepted); averaging it
    over the coverage sweep is robust to the few-high-confidence-failures
    distortion that affects plain AURC.  Lower is better.
    """

    count = len(confidences)
    if count == 0:
        return 0.0
    order = sorted(range(count), key=lambda i: confidences[i], reverse=True)
    area = 0.0
    accepted_errors = 0
    for accepted, index in enumerate(order, start=1):
        accepted_errors += int(not correct[index])
        area += accepted_errors / count
    return area / count


def partial_augrc(
    confidences: Sequence[float],
    correct: Sequence[bool],
    low: float = 0.05,
    high: float = 0.50,
) -> float:
    """AUGRC restricted to a coverage band (Version 9.0 headline statistic).

    Version 8.0's full-curve AUGRC on a 59%-easy pool measured the baseline's
    saturation region, not abstention quality; the deployable region is the
    low-coverage band where abstention actually operates. Lower is better.
    """

    count = len(confidences)
    if count == 0:
        return 0.0
    order = sorted(range(count), key=lambda i: confidences[i], reverse=True)
    area = 0.0
    weight = 0
    accepted_errors = 0
    for accepted, index in enumerate(order, start=1):
        accepted_errors += int(not correct[index])
        coverage = accepted / count
        if low <= coverage <= high:
            area += accepted_errors / count
            weight += 1
    return area / weight if weight else 0.0


def clopper_pearson_upper(errors: int, count: int, alpha: float = 0.05) -> float:
    """95% upper confidence bound on a binomial rate via beta inversion."""

    if count == 0:
        return 1.0
    if errors >= count:
        return 1.0
    # Bisection on the regularized incomplete beta (no SciPy on Kaggle base).
    def beta_cdf(x: float, a: float, b: float, steps: int = 4096) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        total = 0.0
        log_norm = (
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        )
        for step in range(steps):
            t = (step + 0.5) / steps * x
            total += math.exp(
                log_norm + (a - 1.0) * math.log(t) + (b - 1.0) * math.log(1.0 - t)
            )
        return total * x / steps

    low, high = errors / count, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if beta_cdf(mid, errors + 1.0, float(count - errors)) < 1.0 - alpha:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def difficulty_bin(pass_rate: float) -> str:
    """Preregistered difficulty bins from the same-pool pass@1 estimate."""

    if pass_rate > 0.625:
        return "easy"
    if pass_rate >= 0.25:
        return "medium"
    return "hard"


def encode_preserving_ends(
    tokenizer: Any, text: str, max_length: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Tensorized wrapper over the ONE shared encode primitive (Version 9.0).

    Harvest, calibration and audit all route through
    ``data.encode_preserving_ends`` so the three generation surfaces are
    byte-identical (the Version 8.0 harvest generated from the padded
    training-collator surface instead, which starved the validity floor).
    """

    ids = shared_encode_preserving_ends(tokenizer, text, max_length)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


# A conversational turn marker opening a line, anywhere in the answer. Anchored to
# line starts so ordinary prose ("the assistant: ..." mid-sentence) is not a leak.
TURN_SCAFFOLD = re.compile(
    r"(?mi)^[\s>*#-]*(human|assistant|user|system)\s*:", re.MULTILINE
)


def output_quality(text: str, row: Mapping[str, Any], ended_with_eos: bool) -> Dict[str, Any]:
    """Reject prompt copying and grade narrow probes by meaning, not wording."""

    normalized = " ".join(text.lower().split())
    tokens = normalized_tokens(text)
    leaked_prefix = normalized.startswith(
        ("human:", "assistant:", "### instruction", "### relevant context", "you are ")
    )
    marker_leak = any(
        marker in normalized
        for marker in ("<|user_request|>", "<|hlwm_", "### response requirements")
    )
    # Version 6.0's scaffold appeared *mid*-answer, after the eos-as-BOS boundary
    # opened a new turn, so a start-of-string test would have scored those rows
    # clean. This gate certifies the R1 repair, so it has to see a turn marker
    # wherever it lands.
    marker_leak = marker_leak or bool(TURN_SCAFFOLD.search(text))
    checks = row.get("quality_checks") or {}
    required = [str(value).lower() for value in checks.get("required_phrases", [])]
    forbidden = [str(value).lower() for value in checks.get("forbidden_phrases", [])]
    required_ok = all(value in normalized for value in required)
    forbidden_ok = all(value not in normalized for value in forbidden)
    semantic = grade_semantic_answer(text, row.get("answer_spec") or {})
    nonempty = len(tokens) >= 4
    no_prompt_leak = not leaked_prefix and not marker_leak
    complete = bool(ended_with_eos or (text.strip() and text.rstrip()[-1] in ".!?`"))
    return {
        "nonempty": nonempty,
        "no_prompt_leak": no_prompt_leak,
        "complete": complete,
        "required_phrases_ok": required_ok,
        "forbidden_phrases_ok": forbidden_ok,
        "semantic_correct": bool(semantic["correct"]),
        "semantic_grade": semantic,
        "passed": bool(
            nonempty
            and no_prompt_leak
            and required_ok
            and forbidden_ok
            and semantic["correct"]
        ),
    }


def load_hlwm(
    payload: Dict[str, Any], device: torch.device
) -> HLWMForConditionalGeneration:
    allowed = {item.name for item in fields(HLWMConfig)}
    overrides = {
        name: value
        for name, value in payload["hlwm_config"].items()
        if name in allowed
    }
    model = HLWMForConditionalGeneration.from_pretrained(
        payload["base_model"],
        revision=payload["base_revision"],
        hlwm_overrides=overrides,
        dtype=AMP_DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    unfreeze_tail = int((payload.get("args") or {}).get("unfreeze_tail_layers", 0))
    model.unfreeze_language_tail(unfreeze_tail)
    if bool((payload.get("args") or {}).get("causal_control")):
        # Plain-LoRA control checkpoints store only the LoRA (and tail)
        # tensors; mirror the trainer's freeze so the missing-tensor check
        # below inspects exactly the tensors the control actually trained.
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "_lora." not in name and not name.startswith(
                "backbone.layers."
            ):
                parameter.requires_grad_(False)
    model.to(device=device, dtype=AMP_DTYPE)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if unexpected:
        raise RuntimeError("unexpected checkpoint tensors: %s" % unexpected)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(name for name in missing if name in trainable)
    if missing_trainable:
        raise RuntimeError("checkpoint is missing trainable tensors: %s" % missing_trainable)
    return model.eval()


def main() -> None:
    global AMP_DTYPE
    args = parse_args()
    if (
        args.samples <= 0
        or args.context_tokens <= 0
        or args.canvas_tokens <= 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("sample and token counts are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation expects a CUDA device")
    device = torch.device("cuda")
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = sha256(args.checkpoint)
    sidecar = args.checkpoint.with_suffix(args.checkpoint.suffix + ".sha256")
    if sidecar.exists():
        expected_hash = sidecar.read_text(encoding="utf-8").split()[0]
        if expected_hash != checkpoint_hash:
            raise RuntimeError("checkpoint checksum mismatch")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "hlwm-trainable-checkpoint-v5":
        raise RuntimeError("unsupported checkpoint format: %r" % payload.get("format"))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        payload["base_model"],
        revision=payload["base_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_lanes = int(payload["hlwm_config"]["num_lanes"])
    test_data = Reasoning9000Dataset(
        find_split(args.data_dir, "test"), num_lanes=num_lanes, limit=None
    )
    # Version 8.0 audit population: every light-gradable test row, behavior
    # anchors first so the preregistered truncation order (P11) drops the
    # SQL tail, never anchors.  All selection/abstention arms need graded
    # verdicts, so ungraded rows no longer spend audit budget.
    light_rows = [
        row
        for row in test_data.rows
        if str((row.get("answer_spec") or {}).get("type", "none")).lower()
        in LIGHT_GRADERS
    ]
    anchors = [row for row in light_rows if row["is_behavior_anchor"]]
    others = [row for row in light_rows if not row["is_behavior_anchor"]]
    banded_selection = False
    if args.anchor_ids_file is not None and args.anchor_ids_file.exists():
        # Version 9.0 difficulty banding: the in-session banding cell freezes
        # the audited anchor ids before training; the audit consumes them
        # verbatim so the strata cannot drift with the audited model.
        selected_ids = [
            str(value)
            for value in json.loads(args.anchor_ids_file.read_text(encoding="utf-8"))
        ]
        by_id = {str(row["episode_id"]): row for row in anchors}
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise RuntimeError(
                "banded anchor ids missing from the test split: %s" % missing[:5]
            )
        anchors = [by_id[value] for value in selected_ids]
        banded_selection = True
    others = others[: max(0, args.max_sql_rows)]
    rows = (anchors + others)[: args.samples]
    masked_row_count = sum(1 for row in rows if bool(row.get("masked")))
    prompts = [str(row["public_prompt"]) for row in rows]
    print(
        json.dumps(
            {
                "audit_rows": len(rows),
                "behavior_anchors": sum(
                    1 for row in rows if row["is_behavior_anchor"]
                ),
                "masked_rows": masked_row_count,
                "banded_selection": banded_selection,
            }
        )
    )

    print("Generating pinned base-model controls...")
    base_model = AutoModelForCausalLM.from_pretrained(
        payload["base_model"],
        revision=payload["base_revision"],
        dtype=AMP_DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device).eval()
    base_outputs: List[str] = []
    cue_ids = torch.tensor(
        [tokenizer.encode(RESPONSE_CUE_TEXT, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        for prompt in prompts:
            encoded = encode_preserving_ends(
                tokenizer, prompt, args.context_tokens, device
            )
            # Format-matched control: the base model receives the same hard
            # response cue the HLWM channels append internally.
            cued_ids = torch.cat((encoded["input_ids"], cue_ids), dim=1)
            cued_mask = torch.ones_like(cued_ids)
            generated = base_model.generate(
                input_ids=cued_ids,
                attention_mask=cued_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            base_outputs.append(
                decode_new_tokens(tokenizer, generated, cued_ids.shape[1])
            )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    candidate_temperatures = [
        float(part)
        for part in str(args.candidate_temperatures).split(",")
        if part.strip()
    ]
    if not candidate_temperatures or candidate_temperatures[0] != 0.0:
        raise ValueError("candidate temperatures must start with 0.0 (greedy)")
    print("Reloading the HLWM checkpoint...")
    model = load_hlwm(payload, device)
    candidate_count = len(candidate_temperatures)
    records: List[Dict[str, Any]] = []
    route_totals = torch.zeros(model.config.num_experts, dtype=torch.float64)
    route_observations = 0
    audit_started = time.time()
    with torch.no_grad():
        for index, (row, prompt, base_text) in enumerate(zip(rows, prompts, base_outputs)):
            context_encoded = encode_preserving_ends(
                tokenizer, prompt, args.context_tokens, device
            )
            spec = row.get("answer_spec") or {}
            reference = str(row["public_target"])

            # ----- Claim S/A surface: causal-channel pool, workspace-scored.
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                generation = model.generate_hlwm_nbest(
                    context_encoded["input_ids"],
                    context_encoded["attention_mask"],
                    canvas_length=args.canvas_tokens,
                    max_new_tokens=args.max_new_tokens,
                    candidate_temperatures=candidate_temperatures,
                    generator=generator,
                )
            output = generation.workspace
            selected_index = generation.selected_index
            committed = generation.decision == "publish"
            candidate_texts = [
                tokenizer.decode(ids[0].detach().cpu(), skip_special_tokens=True).strip()
                for ids in generation.candidate_ids
            ]
            candidate_valid_flags = [
                bool(grade_semantic_answer(text, spec)["correct"])
                for text in candidate_texts
            ]
            candidate_lengths = [
                int(ids.shape[1]) for ids in generation.candidate_ids
            ]
            published = candidate_texts[selected_index] if committed else SAFE_FALLBACK
            pass_rate = mean(float(flag) for flag in candidate_valid_flags)
            bin_name = difficulty_bin(pass_rate)

            # Same-pool baselines.  Signature clustering marginalizes over
            # wording; the greedy candidate votes exactly once like any other.
            signatures = [answer_signature(text, spec) for text in candidate_texts]
            clusters: Dict[str, List[int]] = {}
            for candidate_index, signature in enumerate(signatures):
                clusters.setdefault(signature, []).append(candidate_index)

            def cluster_representative(members: List[int]) -> int:
                return max(
                    members,
                    key=lambda i: generation.candidate_mean_logprobs[i],
                )

            # Plain SC: plurality, verifier-free tie-break (max member mean
            # logprob, then first occurrence) — disclosed in the plan.
            sc_signature = max(
                clusters,
                key=lambda s: (
                    len(clusters[s]),
                    max(generation.candidate_mean_logprobs[i] for i in clusters[s]),
                    -min(clusters[s]),
                ),
            )
            sc_index = cluster_representative(clusters[sc_signature])
            sc_tied = (
                sum(
                    1
                    for members in clusters.values()
                    if len(members) == len(clusters[sc_signature])
                )
                > 1
            )
            # Verifier-weighted SC: cluster weight is the summed publish score.
            weighted_signature = max(
                clusters,
                key=lambda s: (
                    sum(generation.publish_scores[i] for i in clusters[s]),
                    max(generation.candidate_mean_logprobs[i] for i in clusters[s]),
                    -min(clusters[s]),
                ),
            )
            weighted_index = cluster_representative(clusters[weighted_signature])
            bon_index = max(
                range(candidate_count),
                key=lambda i: generation.candidate_mean_logprobs[i],
            )
            arms = {
                "greedy": candidate_valid_flags[0],
                "sc_vote": candidate_valid_flags[sc_index],
                "weighted_sc": candidate_valid_flags[weighted_index],
                "bon_logprob": candidate_valid_flags[bon_index],
                "heads_argmax": candidate_valid_flags[selected_index],
                "oracle_any": bool(any(candidate_valid_flags)),
            }

            # ----- Claim G surface: workspace-channel greedy plus ablations.
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                workspace_generation = model.generate_hlwm_nbest(
                    context_encoded["input_ids"],
                    context_encoded["attention_mask"],
                    canvas_length=args.canvas_tokens,
                    max_new_tokens=args.max_new_tokens,
                    candidate_temperatures=(0.0,),
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + index
                    ),
                    candidate_channel="workspace",
                )
                memory_ablated_generation = model.generate_hlwm_nbest(
                    context_encoded["input_ids"],
                    context_encoded["attention_mask"],
                    canvas_length=args.canvas_tokens,
                    max_new_tokens=args.max_new_tokens,
                    candidate_temperatures=(0.0,),
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + index
                    ),
                    candidate_channel="workspace",
                    disable_workspace_memory=True,
                )
            workspace_ids = workspace_generation.candidate_ids[0]
            workspace_text = tokenizer.decode(
                workspace_ids[0].detach().cpu(), skip_special_tokens=True
            ).strip()
            workspace_ended_with_eos = bool(
                workspace_ids.shape[1]
                and int(workspace_ids[0, -1].item()) == tokenizer.eos_token_id
            )
            ablated_text = tokenizer.decode(
                memory_ablated_generation.candidate_ids[0][0].detach().cpu(),
                skip_special_tokens=True,
            ).strip()
            workspace_valid = bool(grade_semantic_answer(workspace_text, spec)["correct"])
            ablated_valid = bool(grade_semantic_answer(ablated_text, spec)["correct"])
            # Full-prefix ablation (Version 9.0, the RC-1 instrument fix): on
            # unmasked rows the workspace arm minus its whole prefix IS the
            # causal greedy decode on identical inputs — greedy is
            # deterministic, so the value is read off the pool instead of
            # recomputed. Masked rows recompute it below as the floor arm.
            full_ablation_valid = bool(candidate_valid_flags[0])
            # Claim G quality battery runs on the workspace channel: the leak
            # gate and pass rate measure the repaired generation surface.
            quality = output_quality(workspace_text, row, workspace_ended_with_eos)

            # ----- Claim L surface (Version 9.0): information-asymmetric arms
            # on masked rows. The workspace reads the full context; the answer
            # channel reads the withheld-premise prompt, so the latent prefix
            # is the only path from the operands to the answer.
            row_masked = bool(row.get("masked"))
            masked_arms: Optional[Dict[str, Any]] = None
            if row_masked:
                masked_prompt_text = str(row.get("masked_prompt", prompt))
                withheld = [str(item) for item in row.get("withheld_literals", [])]
                masked_leak = any(
                    literal and literal in masked_prompt_text for literal in withheld
                )
                masked_encoded = encode_preserving_ends(
                    tokenizer, masked_prompt_text, args.context_tokens, device
                )
                masked_seed = torch.Generator(device=device).manual_seed(
                    args.seed + 50_000 + index
                )
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    masked_workspace = model.generate_hlwm_nbest(
                        context_encoded["input_ids"],
                        context_encoded["attention_mask"],
                        canvas_length=args.canvas_tokens,
                        max_new_tokens=args.max_new_tokens,
                        candidate_temperatures=(0.0,),
                        generator=masked_seed,
                        candidate_channel="workspace",
                        answer_input_ids=masked_encoded["input_ids"],
                        answer_attention_mask=masked_encoded["attention_mask"],
                    )
                    masked_floor = model.generate_hlwm_nbest(
                        context_encoded["input_ids"],
                        context_encoded["attention_mask"],
                        canvas_length=args.canvas_tokens,
                        max_new_tokens=args.max_new_tokens,
                        candidate_temperatures=(0.0,),
                        generator=torch.Generator(device=device).manual_seed(
                            args.seed + 50_000 + index
                        ),
                        candidate_channel="workspace",
                        answer_input_ids=masked_encoded["input_ids"],
                        answer_attention_mask=masked_encoded["attention_mask"],
                        disable_workspace_prefix=True,
                    )
                    masked_gist = (
                        model.generate_hlwm_nbest(
                            context_encoded["input_ids"],
                            context_encoded["attention_mask"],
                            canvas_length=args.canvas_tokens,
                            max_new_tokens=args.max_new_tokens,
                            candidate_temperatures=(0.0,),
                            generator=torch.Generator(device=device).manual_seed(
                                args.seed + 50_000 + index
                            ),
                            candidate_channel="workspace",
                            answer_input_ids=masked_encoded["input_ids"],
                            answer_attention_mask=masked_encoded["attention_mask"],
                            prefix_source="gist",
                        )
                        if model.config.gist_prefix_tokens > 0
                        else None
                    )
                masked_texts = {
                    "workspace": tokenizer.decode(
                        masked_workspace.candidate_ids[0][0].detach().cpu(),
                        skip_special_tokens=True,
                    ).strip(),
                    "floor": tokenizer.decode(
                        masked_floor.candidate_ids[0][0].detach().cpu(),
                        skip_special_tokens=True,
                    ).strip(),
                }
                if masked_gist is not None:
                    masked_texts["gist"] = tokenizer.decode(
                        masked_gist.candidate_ids[0][0].detach().cpu(),
                        skip_special_tokens=True,
                    ).strip()
                # Latent-only premise probe accuracy (report-only diagnostic).
                premise_accuracy: Optional[float] = None
                if getattr(model.config, "premise_aux_weight", 0.0) > 0 and withheld:
                    premise_token_ids: List[int] = []
                    for literal in withheld:
                        premise_token_ids.extend(
                            tokenizer.encode(" " + literal, add_special_tokens=False)
                        )
                    premise_token_ids = premise_token_ids[:24]
                    if premise_token_ids:
                        premise_tensor = torch.tensor(
                            [premise_token_ids], dtype=torch.long, device=device
                        )
                        premise_mask_tensor = torch.ones_like(premise_tensor)
                        negative_pool = [
                            token
                            for token in masked_encoded["input_ids"][0].tolist()
                            if token not in premise_token_ids
                        ] or [int(tokenizer.eos_token_id)]
                        negatives = torch.tensor(
                            [[negative_pool[i % len(negative_pool)] for i in range(32)]],
                            dtype=torch.long,
                            device=device,
                        )
                        probe = model._premise_probe_loss(
                            masked_workspace.workspace.workspace_prefix,
                            premise_tensor,
                            premise_mask_tensor,
                            negatives,
                        )
                        if probe is not None:
                            premise_accuracy = float(probe[1].cpu())
                masked_arms = {
                    "leak": bool(masked_leak),
                    "workspace_valid": bool(
                        grade_semantic_answer(masked_texts["workspace"], spec)["correct"]
                    ),
                    "floor_valid": bool(
                        grade_semantic_answer(masked_texts["floor"], spec)["correct"]
                    ),
                    "gist_valid": (
                        bool(grade_semantic_answer(masked_texts["gist"], spec)["correct"])
                        if "gist" in masked_texts
                        else None
                    ),
                    "workspace_text": masked_texts["workspace"][:400],
                    "floor_text": masked_texts["floor"][:400],
                    "gist_text": masked_texts.get("gist", "")[:400],
                    "premise_probe_accuracy": premise_accuracy,
                }
                full_ablation_valid = masked_arms["floor_valid"]

            # ----- Corrupt scoring: heads must reject a shuffled candidate.
            corrupt_source = workspace_ids
            if corrupt_source.shape[1] > 1:
                corrupt_ids = torch.roll(corrupt_source, shifts=1, dims=-1)
            else:
                corrupt_ids = (corrupt_source + 1) % model.config.vocab_size
            corrupt_mask = torch.ones_like(corrupt_ids)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                corrupt_logits, corrupt_hidden = model._teacher_force_workspace(
                    context_encoded["input_ids"],
                    context_encoded["attention_mask"],
                    output.workspace_prefix,
                    corrupt_ids,
                    corrupt_mask,
                )
                verification_error = torch.sigmoid(output.verification_logits).mean(dim=-1)
                corrupt_commitment = model._score_candidate(
                    corrupt_hidden,
                    corrupt_logits,
                    corrupt_mask,
                    output.global_state,
                    verification_error,
                )
                corrupt_logprob_feature = model.causal_answer_mean_logprob(
                    context_encoded["input_ids"],
                    context_encoded["attention_mask"],
                    corrupt_ids,
                    corrupt_mask,
                ).exp()
            corrupt_probabilities = torch.sigmoid(corrupt_commitment[0].float()).cpu()
            corrupt_rejected = not bool(
                model._commit_mask(corrupt_commitment, corrupt_logprob_feature)[0].item()
            )

            route_load = F.one_hot(
                output.route_indices, model.config.num_experts
            ).float().reshape(-1, model.config.num_experts).sum(dim=0).cpu()
            route_totals += route_load.double()
            route_observations += int(route_load.sum().item())
            normalized_lanes = F.normalize(output.lane_summaries.float(), dim=-1)
            lane_similarity = torch.matmul(
                normalized_lanes, normalized_lanes.transpose(1, 2)
            )
            if model.config.num_lanes > 1:
                lane_mask = ~torch.eye(
                    model.config.num_lanes,
                    dtype=torch.bool,
                    device=lane_similarity.device,
                )[None, :, :]
                lane_cosine = float(lane_similarity.masked_select(lane_mask).mean().cpu())
            else:
                lane_cosine = 1.0

            expected_commit = bool(row["expected_commit"])
            candidate_valid = candidate_valid_flags[selected_index]
            policy_target = candidate_valid if row["is_behavior_anchor"] else expected_commit
            records.append(
                {
                    "episode_id": row["episode_id"],
                    "domain": row["domain"],
                    "reference": reference,
                    "is_behavior_anchor": bool(row["is_behavior_anchor"]),
                    "expected_commit": expected_commit,
                    "expected_action": row["expected_action"],
                    "answer_spec": row["answer_spec"],
                    "base_causal": base_text,
                    "hlwm_causal": candidate_texts[0],
                    "hlwm_published": published,
                    "committed": committed,
                    "candidate_valid": candidate_valid,
                    "commit_correct": committed == policy_target,
                    "quality": quality,
                    "difficulty_bin": bin_name,
                    "pass_rate_estimate": pass_rate,
                    "commit_probability": generation.commit_probabilities[selected_index],
                    "risk_probability": generation.risk_probabilities[selected_index],
                    "verifier_error_probability": (
                        generation.verifier_error_probabilities[selected_index]
                    ),
                    "private_verifier_error_probability": (
                        generation.private_verifier_error_probability
                    ),
                    "pool": {
                        "temperatures": generation.temperatures,
                        "publish_scores": generation.publish_scores,
                        "candidate_agreements": generation.candidate_agreements,
                        "candidate_mean_logprobs": generation.candidate_mean_logprobs,
                        "candidate_valid_flags": candidate_valid_flags,
                        "candidate_lengths": candidate_lengths,
                        "selected_index": selected_index,
                        "selected_publish_score": generation.publish_scores[selected_index],
                        "selected_mean_logprob": generation.candidate_mean_logprobs[
                            selected_index
                        ],
                        "sc_index": sc_index,
                        "sc_tied": sc_tied,
                        "weighted_index": weighted_index,
                        "bon_index": bon_index,
                    },
                    "arms": arms,
                    "workspace_greedy": workspace_text,
                    "workspace_greedy_valid": workspace_valid,
                    "workspace_greedy_token_f1": token_f1(workspace_text, reference),
                    "ablated_candidate": ablated_text,
                    "ablated_valid": ablated_valid,
                    "ablated_token_f1": token_f1(ablated_text, reference),
                    "full_ablation_valid": bool(full_ablation_valid),
                    "masked": row_masked,
                    "masked_arms": masked_arms,
                    "corrupt_commit_probability": float(corrupt_probabilities[0]),
                    "corrupt_risk_probability": float(corrupt_probabilities[1]),
                    "corrupt_verifier_error_probability": float(corrupt_probabilities[2]),
                    "corrupt_rejected": corrupt_rejected,
                    "reverse_timesteps": output.reverse_timesteps[0].cpu().tolist(),
                    "lane_summary_cosine": lane_cosine,
                    "base_token_f1": token_f1(base_text, reference),
                    "hlwm_causal_token_f1": token_f1(candidate_texts[0], reference),
                }
            )
            if (index + 1) % 32 == 0:
                elapsed = time.time() - audit_started
                print(
                    json.dumps(
                        {
                            "audit_progress": index + 1,
                            "elapsed_seconds": round(elapsed, 1),
                            "seconds_per_row": round(elapsed / (index + 1), 2),
                        }
                    ),
                    flush=True,
                )
            if (
                args.max_audit_hours > 0
                and (time.time() - audit_started) > args.max_audit_hours * 3600.0
                and index + 1 < len(rows)
            ):
                print(
                    json.dumps(
                        {
                            "audit_truncated_at_row": index + 1,
                            "rows_dropped": len(rows) - (index + 1),
                        }
                    ),
                    flush=True,
                )
                break

    route_load_fractions = (route_totals / max(1, route_observations)).tolist()
    positive_loads = [value for value in route_load_fractions if value > 0.0]
    route_entropy = -sum(value * math.log(value) for value in positive_loads)
    route_entropy_normalized = route_entropy / math.log(
        max(2, model.config.num_experts)
    )
    second_route_load = sorted(route_load_fractions, reverse=True)[1] if len(
        route_load_fractions
    ) > 1 else 0.0

    aggregate: Dict[str, Any] = {
        "samples": len(records),
        "commit_coverage": mean(float(record["committed"]) for record in records),
        "commit_accuracy": mean(float(record["commit_correct"]) for record in records),
        "quality_pass_rate": mean(float(record["quality"]["passed"]) for record in records),
        "prompt_leak_rate": mean(
            float(not record["quality"]["no_prompt_leak"]) for record in records
        ),
        "complete_answer_rate": mean(
            float(record["quality"]["complete"]) for record in records
        ),
        "probe_content_accuracy": mean(
            float(record["workspace_greedy_valid"])
            for record in records
            if record["is_behavior_anchor"]
        ),
        "safe_abstention_accuracy": mean(
            float(record["candidate_valid"])
            for record in records
            if record["is_behavior_anchor"] and record["expected_action"] == "abstain"
        ),
        "safe_abstention_commit_rate": mean(
            float(record["candidate_valid"] and record["committed"])
            for record in records
            if record["is_behavior_anchor"] and record["expected_action"] == "abstain"
        ),
        "probe_commit_accuracy": mean(
            float(record["commit_correct"])
            for record in records
            if record["is_behavior_anchor"]
        ),
        "mean_commit_probability": mean(record["commit_probability"] for record in records),
        "mean_risk_probability": mean(record["risk_probability"] for record in records),
        "mean_verifier_error_probability": mean(
            record["verifier_error_probability"] for record in records
        ),
        "mean_private_verifier_error_probability": mean(
            record["private_verifier_error_probability"] for record in records
        ),
        "corrupt_rejection_rate": mean(
            float(record["corrupt_rejected"]) for record in records
        ),
        "mean_corrupt_commit_probability": mean(
            record["corrupt_commit_probability"] for record in records
        ),
        "mean_corrupt_risk_probability": mean(
            record["corrupt_risk_probability"] for record in records
        ),
        "mean_corrupt_verifier_error_probability": mean(
            record["corrupt_verifier_error_probability"] for record in records
        ),
        "mean_commit_margin": mean(
            record["commit_probability"] - record["corrupt_commit_probability"]
            for record in records
        ),
        "mean_risk_margin": mean(
            record["corrupt_risk_probability"] - record["risk_probability"]
            for record in records
        ),
        "mean_verifier_error_margin": mean(
            record["corrupt_verifier_error_probability"]
            - record["verifier_error_probability"]
            for record in records
        ),
        "mean_lane_summary_cosine": mean(
            record["lane_summary_cosine"] for record in records
        ),
        "mean_base_token_f1": mean(record["base_token_f1"] for record in records),
        "mean_hlwm_causal_token_f1": mean(
            record["hlwm_causal_token_f1"] for record in records
        ),
        "mean_workspace_greedy_token_f1": mean(
            record["workspace_greedy_token_f1"] for record in records
        ),
        "route_load": route_load_fractions,
        "route_entropy_normalized": route_entropy_normalized,
        "second_route_load": second_route_load,
        "prefix_gate_tanh": float(torch.tanh(model.prefix_gate.detach()).cpu()),
    }

    # ------------------------------------------------------------------
    # Claim G: channel parity and read-out liveness.
    causal_greedy_accuracy = mean(
        float(record["arms"]["greedy"]) for record in records
    )
    workspace_greedy_accuracy = mean(
        float(record["workspace_greedy_valid"]) for record in records
    )
    causal_f1 = aggregate["mean_hlwm_causal_token_f1"]
    workspace_f1 = aggregate["mean_workspace_greedy_token_f1"]
    aggregate["channel_parity"] = {
        "causal_greedy_accuracy": causal_greedy_accuracy,
        "workspace_greedy_accuracy": workspace_greedy_accuracy,
        "parity_gap": causal_greedy_accuracy - workspace_greedy_accuracy,
        "causal_token_f1": causal_f1,
        "workspace_token_f1": workspace_f1,
        "f1_ratio": workspace_f1 / causal_f1 if causal_f1 > 0 else None,
    }
    aggregate["readout_ablation"] = {
        "records": len(records),
        # Descriptive arm (the exact slice the Version 8.0 audit shipped):
        # memory tokens removed, gated synthesis tokens kept in both arms.
        "memory_tokens_graded_accuracy_delta": workspace_greedy_accuracy
        - mean(float(record["ablated_valid"]) for record in records),
        "mean_greedy_f1_delta": mean(
            record["workspace_greedy_token_f1"] - record["ablated_token_f1"]
            for record in records
        ),
        # Fixed instrument (Version 9.0): the WHOLE prefix removed.
        "full_prefix_graded_accuracy_delta": workspace_greedy_accuracy
        - mean(float(record["full_ablation_valid"]) for record in records),
    }

    # ------------------------------------------------------------------
    # Claim L (Version 9.0): information-asymmetric masked arms. The masked
    # floor (prefix removed on the masked prompt) must crater, and the
    # workspace arm must recover what only the latent can carry.
    masked_records = [record for record in records if record.get("masked")]
    unmasked_records = [record for record in records if not record.get("masked")]
    unmasked_causal = mean(
        float(record["arms"]["greedy"]) for record in unmasked_records
    )
    unmasked_workspace = mean(
        float(record["workspace_greedy_valid"]) for record in unmasked_records
    )
    gist_values = [
        record["masked_arms"].get("gist_valid")
        for record in masked_records
        if record.get("masked_arms")
    ]
    gist_ran = bool(gist_values) and all(value is not None for value in gist_values)
    premise_values = [
        record["masked_arms"].get("premise_probe_accuracy")
        for record in masked_records
        if record.get("masked_arms")
        and record["masked_arms"].get("premise_probe_accuracy") is not None
    ]
    aggregate["masked_channel"] = {
        "masked_rows": len(masked_records),
        "leak_rows": sum(
            1
            for record in masked_records
            if record.get("masked_arms", {}).get("leak")
        ),
        "workspace_accuracy": mean(
            float(record["masked_arms"]["workspace_valid"])
            for record in masked_records
        )
        if masked_records
        else None,
        "floor_accuracy": mean(
            float(record["masked_arms"]["floor_valid"]) for record in masked_records
        )
        if masked_records
        else None,
        "gist_accuracy": mean(float(bool(value)) for value in gist_values)
        if gist_ran
        else None,
        "premise_probe_accuracy": mean(premise_values) if premise_values else None,
        "unmasked_causal_accuracy": unmasked_causal,
        "unmasked_workspace_accuracy": unmasked_workspace,
        "unmasked_parity_gap": unmasked_causal - unmasked_workspace,
    }

    # ------------------------------------------------------------------
    # Claim S: same-pool selection arms with the difficulty-stratified
    # headline (weighted SC vs plain SC, McNemar mid-p, medium bin).
    def arm_accuracy(subset: List[Dict[str, Any]], arm: str) -> float:
        return mean(float(record["arms"][arm]) for record in subset)

    # Version 9.0: every selection statistic runs on behavior anchors only.
    # The shipped answer signature has no vote equivalence for near-unique
    # SQL strings, so SQL rows would distort any voting comparison; they stay
    # in the risk/abstention population and the domain metrics.
    selection_records = [record for record in records if record["is_behavior_anchor"]]
    selection: Dict[str, Any] = {
        "graded_rows": len(selection_records),
        "population": "behavior_anchors_only",
    }
    arm_names = ("greedy", "sc_vote", "weighted_sc", "bon_logprob", "heads_argmax", "oracle_any")
    by_bin: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for record in selection_records:
        by_bin[record["difficulty_bin"]].append(record)
    for scope_name, subset in (
        ("all", selection_records),
        ("easy", by_bin["easy"]),
        ("medium", by_bin["medium"]),
        ("hard", by_bin["hard"]),
    ):
        scope: Dict[str, Any] = {"rows": len(subset)}
        for arm in arm_names:
            scope[arm] = arm_accuracy(subset, arm) if subset else None
        if subset:
            wins = sum(
                1
                for record in subset
                if record["arms"]["weighted_sc"] and not record["arms"]["sc_vote"]
            )
            losses = sum(
                1
                for record in subset
                if record["arms"]["sc_vote"] and not record["arms"]["weighted_sc"]
            )
            scope["weighted_vs_sc"] = {
                "wins": wins,
                "losses": losses,
                "mcnemar_mid_p": mcnemar_mid_p(wins, losses),
            }
            scope["exactly_one_valid_fraction"] = mean(
                float(sum(record["pool"]["candidate_valid_flags"]) == 1)
                for record in subset
            )
        selection[scope_name] = scope
    selection["sc_tie_rate"] = mean(
        float(record["pool"]["sc_tied"]) for record in selection_records
    )
    # Version 9.0 report-only continuation of the retired Claim S: one
    # weighted-vs-plurality comparison over the banded anchors, VOID unless at
    # least 25 discordant pairs exist (the McNemar power floor: 25 pairs at a
    # 70/30 split gives mid-p < 0.05 about 80% of the time). No headline and
    # no gate rests on it.
    report_wins = sum(
        1
        for record in selection_records
        if record["arms"]["weighted_sc"] and not record["arms"]["sc_vote"]
    )
    report_losses = sum(
        1
        for record in selection_records
        if record["arms"]["sc_vote"] and not record["arms"]["weighted_sc"]
    )
    report_pairs = report_wins + report_losses
    selection["report_only_weighted_vs_sc"] = {
        "population": "banded_anchors",
        "wins": report_wins,
        "losses": report_losses,
        "discordant_pairs": report_pairs,
        "power_floor_pairs": 25,
        "status": "reported" if report_pairs >= 25 else "void_insufficient_pairs",
        "mcnemar_mid_p": mcnemar_mid_p(report_wins, report_losses)
        if report_pairs >= 25
        else None,
    }
    # Anchor-only sensitivity check (P12): the same headline without SQL rows.
    anchor_medium = [
        record
        for record in by_bin["medium"]
        if record["is_behavior_anchor"]
    ]
    if anchor_medium:
        wins = sum(
            1
            for record in anchor_medium
            if record["arms"]["weighted_sc"] and not record["arms"]["sc_vote"]
        )
        losses = sum(
            1
            for record in anchor_medium
            if record["arms"]["sc_vote"] and not record["arms"]["weighted_sc"]
        )
        selection["medium_anchors_only"] = {
            "rows": len(anchor_medium),
            "sc_vote": arm_accuracy(anchor_medium, "sc_vote"),
            "weighted_sc": arm_accuracy(anchor_medium, "weighted_sc"),
            "mcnemar_mid_p": mcnemar_mid_p(wins, losses),
        }
    aggregate["selection"] = selection

    # Score diagnostics: are the heads just re-implementing logprob or length?
    all_scores: List[float] = []
    all_logprobs: List[float] = []
    all_lengths: List[float] = []
    for record in records:
        all_scores.extend(record["pool"]["publish_scores"])
        all_logprobs.extend(record["pool"]["candidate_mean_logprobs"])
        all_lengths.extend(float(v) for v in record["pool"]["candidate_lengths"])
    aggregate["score_diagnostics"] = {
        "spearman_publish_vs_logprob": spearman_correlation(all_scores, all_logprobs),
        "spearman_publish_vs_length": spearman_correlation(all_scores, all_lengths),
        "candidates": len(all_scores),
    }

    # ------------------------------------------------------------------
    # Claim A: conformal abstention versus mean-logprob abstention.
    heads_confidences = [
        float(record["pool"]["selected_publish_score"]) for record in records
    ]
    logprob_confidences = [
        float(math.exp(record["pool"]["candidate_mean_logprobs"][record["pool"]["bon_index"]]))
        for record in records
    ]
    heads_correct = [bool(record["candidate_valid"]) for record in records]
    logprob_correct = [bool(record["arms"]["bon_logprob"]) for record in records]
    accepted = [record for record in records if record["committed"]]
    coverage = len(accepted) / max(1, len(records))
    heads_selective_accuracy = (
        mean(float(record["candidate_valid"]) for record in accepted)
        if accepted
        else 0.0
    )
    ranked = sorted(
        range(len(records)), key=lambda i: logprob_confidences[i], reverse=True
    )
    matched = ranked[: max(1, len(accepted))] if accepted else []
    logprob_selective_accuracy = (
        mean(float(logprob_correct[i]) for i in matched) if matched else 0.0
    )
    selective_errors = sum(1 for record in accepted if not record["candidate_valid"])
    # Paired-by-row AUGRC: each rule ranks the rows by its own confidence and
    # is scored on its own selected answer's correctness. Version 9.0's
    # headline restricts the area to the deployable coverage band [0.05,
    # 0.50]; the full-curve areas remain reported for cross-study continuity.
    heads_augrc = augrc(heads_confidences, heads_correct)
    logprob_augrc = augrc(logprob_confidences, logprob_correct)
    heads_partial = partial_augrc(heads_confidences, heads_correct)
    logprob_partial = partial_augrc(logprob_confidences, logprob_correct)
    aggregate["abstention"] = {
        "graded_rows": len(records),
        "coverage": coverage,
        "heads_selective_accuracy": heads_selective_accuracy,
        "logprob_selective_accuracy_at_matched_coverage": logprob_selective_accuracy,
        "heads_augrc": heads_augrc,
        "logprob_augrc": logprob_augrc,
        "heads_partial_augrc": heads_partial,
        "logprob_partial_augrc": logprob_partial,
        "partial_augrc_band": [0.05, 0.50],
        "published_rows": len(accepted),
        "selective_risk_ucb95": clopper_pearson_upper(
            selective_errors, len(accepted)
        ),
    }
    # Paired bootstrap over rows: each rule keeps its own confidence and its
    # own selected answer's correctness within every resample. The headline
    # statistic is the coverage-restricted area (Version 9.0); the full-curve
    # fraction is also reported.
    generator = torch.Generator().manual_seed(args.seed)
    wins = 0
    partial_wins = 0
    resamples = 1000
    count = len(records)
    for _ in range(resamples):
        indices = torch.randint(0, count, (count,), generator=generator).tolist()
        sampled_heads_conf = [heads_confidences[i] for i in indices]
        sampled_heads_ok = [heads_correct[i] for i in indices]
        sampled_lp_conf = [logprob_confidences[i] for i in indices]
        sampled_lp_ok = [logprob_correct[i] for i in indices]
        wins += int(
            augrc(sampled_heads_conf, sampled_heads_ok)
            < augrc(sampled_lp_conf, sampled_lp_ok)
        )
        partial_wins += int(
            partial_augrc(sampled_heads_conf, sampled_heads_ok)
            < partial_augrc(sampled_lp_conf, sampled_lp_ok)
        )
    aggregate["abstention"]["bootstrap_heads_win_fraction"] = wins / resamples
    aggregate["abstention"]["bootstrap_partial_win_fraction"] = (
        partial_wins / resamples
    )

    # Legacy field names consumed by cross-study tooling.
    aggregate["selective_comparison"] = {
        "graded_rows": len(records),
        "coverage": coverage,
        "heads_selective_accuracy": heads_selective_accuracy,
        "logprob_selective_accuracy_at_matched_coverage": logprob_selective_accuracy,
    }

    domain_names = sorted({str(record["domain"]) for record in records})
    domain_metrics: Dict[str, Any] = {}
    for domain_name in domain_names:
        domain_records = [
            record for record in records if str(record["domain"]) == domain_name
        ]
        domain_metrics[domain_name] = {
            "samples": len(domain_records),
            "workspace_greedy_accuracy": mean(
                float(record["workspace_greedy_valid"]) for record in domain_records
            ),
            "causal_greedy_accuracy": mean(
                float(record["arms"]["greedy"]) for record in domain_records
            ),
            "weighted_sc_accuracy": mean(
                float(record["arms"]["weighted_sc"]) for record in domain_records
            ),
            "commit_coverage": mean(
                float(record["committed"]) for record in domain_records
            ),
            "commit_accuracy": mean(
                float(record["commit_correct"]) for record in domain_records
            ),
            "quality_pass_rate": mean(
                float(record["quality"]["passed"]) for record in domain_records
            ),
        }
    aggregate["domain_metrics"] = domain_metrics
    grader_types = sorted(
        {
            str((record["answer_spec"] or {}).get("type", "none")).lower()
            for record in records
            if record["is_behavior_anchor"]
        }
    )
    aggregate["probe_accuracy_by_grader"] = {
        grader: mean(
            float(record["workspace_greedy_valid"])
            for record in records
            if record["is_behavior_anchor"]
            and str((record["answer_spec"] or {}).get("type", "none")).lower() == grader
        )
        for grader in grader_types
    }

    calibration_path = args.checkpoint.parent / "commitment-calibration.json"
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.exists()
        else None
    )
    policy_path = args.checkpoint.parent / "policy-head-training.json"
    policy_report = (
        json.loads(policy_path.read_text(encoding="utf-8"))
        if policy_path.exists()
        else None
    )
    # Canary cleanliness from the training metrics log: format-marker rate
    # must be zero at every evaluation boundary from step 1536 (joint phase).
    canary_clean = None
    metrics_path = args.checkpoint.parent / "metrics.jsonl"
    if metrics_path.exists():
        canary_clean = True
        canary_seen = 0
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if '"canary"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "canary" not in entry:
                    continue
                canary_seen += 1
                if int(entry.get("step", 0)) >= 1536 and (
                    float(entry["canary"].get("format_marker_rate", 0.0)) > 0.0
                ):
                    canary_clean = False
        if canary_seen == 0:
            canary_clean = None

    # ------------------------------------------------------------------
    # Version 9.0 capability gates (22): six training/infra, six Claim L,
    # two attribution, three Claim V, four Claim A2, one general. The
    # notebook's verdict cell supplies training_complete and
    # zero_skipped_updates from summary.json and re-derives `passed`.
    masked_channel = aggregate["masked_channel"]
    masked_rows_present = int(masked_channel["masked_rows"] or 0) > 0
    masked_workspace_accuracy = float(masked_channel["workspace_accuracy"] or 0.0)
    masked_floor_accuracy = float(
        masked_channel["floor_accuracy"] if masked_channel["floor_accuracy"] is not None else 1.0
    )
    gist_accuracy = masked_channel["gist_accuracy"]
    premise_probe = masked_channel["premise_probe_accuracy"]
    published_rows = int(aggregate["abstention"]["published_rows"])
    risk_ucb_feasible = published_rows >= 20
    distinct_scores = len(
        {round(float(value), 6) for value in all_scores}
    )
    conformal_report = (calibration or {}).get("conformal") or {}
    gates = {
        # --- training / infrastructure (six; first two are notebook-supplied)
        "training_complete": True,
        "zero_skipped_updates": True,
        "truncation_integrity": bool(
            masked_rows_present
            and masked_channel["leak_rows"] == 0
        ),
        "validity_floor_met": bool((policy_report or {}).get("validity_floor_met")),
        "calibration_fitted": bool(
            (calibration or {}).get("publish_rule", {}).get("fitted")
            and (calibration or {}).get("publish_threshold_non_degenerate")
        ),
        "canary_clean": bool(canary_clean),
        # --- Claim L: the latent channel under structural necessity (six)
        "masked_leak_zero": bool(
            masked_rows_present and masked_channel["leak_rows"] == 0
        ),
        "masked_causal_floor": bool(
            masked_rows_present and masked_floor_accuracy <= 0.10
        ),
        "latent_channel_live": bool(
            masked_rows_present
            and (masked_workspace_accuracy - masked_floor_accuracy) >= 0.30
        ),
        "full_ablation_consistent": bool(
            masked_rows_present
            and (
                masked_workspace_accuracy
                - mean(
                    float(record["full_ablation_valid"])
                    for record in masked_records
                )
            )
            >= 0.30
        ),
        "unmasked_parity": (
            abs(masked_channel["unmasked_parity_gap"]) <= 0.05
        ),
        "aux_premise_decodable": bool(
            premise_probe is not None and premise_probe >= 0.50
        ),
        # --- attribution control (two)
        "gist_control_run": bool(gist_accuracy is not None),
        "canvas_beats_gist": bool(
            gist_accuracy is not None
            and (masked_workspace_accuracy - float(gist_accuracy)) >= 0.10
        ),
        # --- Claim V: verifier and publish-score integrity (three)
        "verifier_margin_positive_on_generated": aggregate["mean_verifier_error_margin"] > 0,
        "clean_commit_ranked_above_corrupt": aggregate["mean_commit_margin"] > 0,
        "publish_score_continuous": distinct_scores >= 100,
        # --- Claim A2: calibrated abstention (four)
        "coverage_in_band": 0.20 <= coverage <= 0.55,
        "risk_ucb_015": bool(
            (not risk_ucb_feasible)
            or aggregate["abstention"]["selective_risk_ucb95"] <= 0.15
        ),
        "abstention_beats_logprob_partial_augrc": (
            aggregate["abstention"]["bootstrap_partial_win_fraction"] >= 0.80
        ),
        "safe_abstention_probes": (
            aggregate["safe_abstention_commit_rate"] >= 0.50
            and aggregate["safe_abstention_accuracy"] >= 0.70
        ),
        # --- general (one)
        "complete_answer_rate_at_least_50pct": aggregate["complete_answer_rate"] >= 0.50,
    }
    gates["passed"] = all(bool(value) for value in gates.values())
    aggregate["gate_notes"] = {
        "risk_ucb_void_low_n": not risk_ucb_feasible,
        "published_rows": published_rows,
        "distinct_publish_scores": distinct_scores,
        "canary_boundaries_seen": canary_clean is not None,
        "no_scaffold_leak_rate": aggregate["prompt_leak_rate"],
        "quality_pass_rate": aggregate["quality_pass_rate"],
        "lane_summary_cosine_report_only": aggregate["mean_lane_summary_cosine"],
        "prefix_gate_tanh_telemetry_only": aggregate["prefix_gate_tanh"],
    }
    gate_path = args.output_dir / "v9.0-capability-gate.json"
    gate_path.write_text(
        json.dumps(gates, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )

    report = {
        "status": "checkpoint_reload_and_generation_complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(payload["step"]),
        "base_model": payload["base_model"],
        "base_revision": payload["base_revision"],
        "policy_thresholds": {
            "commitment": model.config.commitment_threshold,
            "risk": model.config.risk_threshold,
            "verifier_error": model.config.verifier_error_threshold,
        },
        "publish_rule": {
            "weights": list(model.config.publish_weights or ()),
            "bias": model.config.publish_bias,
            "threshold": model.config.publish_threshold,
        },
        "conformal": conformal_report,
        "validation_calibration": calibration,
        "gates": gates,
        "aggregate": aggregate,
        "records": records,
        "boundary": (
            "Diagnostic outputs only. Thresholds use verified validation anchors and test "
            "records remain held out, but broad capability and production safety are not established."
        ),
    }
    report_path = args.output_dir / "checkpoint-evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "gates": gates, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
