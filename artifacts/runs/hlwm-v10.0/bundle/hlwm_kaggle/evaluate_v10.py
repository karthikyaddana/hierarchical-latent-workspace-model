"""Version 10.0 audit arms, channel gates, verdict assembly and ladder.

This module implements Section 4 of the Version 10 plan AS AMENDED by the
binding red-team amendments (Section 10, A3/A4/A5/A6 of
``reports/hlwm-v10-plan-2026-09-03.md``).  It consumes the Version 10.0
model methods (``produce_latent_thoughts``, ``student_channel_teacher_force``,
``decode_candidate_v10``, ``_decode_candidate``, ``route_family_logits``) and
reuses the Version 9.0 audit's grading and leak-scan surface from
``evaluate_checkpoint`` plus the preregistered statistics in
``selective_stats``.  It never modifies those modules.

Preregistered conventions implemented here (each documented at the point of
use as well):

* **Masked-core ordering rule.**  The fixed 160-row masked stratum is the
  first ``masked_core_rows`` masked audit rows under an ascending
  lexicographic sort of ``str(row["episode_id"])`` (plain Python string
  comparison of the anchor id).  The rule is a pure function of the anchor
  ids: it cannot drift with dataloader order, and duplicate ids raise so the
  ordering is always total.  The same rule orders the per-family expert-arm
  subsets, the abstention streaming order (the e-process consumes rows in
  this order), and the probe's held-out rows.

* **Paired one-sided 90% lower confidence bound (A3).**  Each channel margin
  gate compares two arms on IDENTICAL rows.  With per-row correctness deltas
  ``d_i = treatment_i - control_i`` (values in {-1, 0, +1}),

      LCB = mean(d) - z_{0.90} * sd(d) / sqrt(n),   z_{0.90} = 1.2815515655,

  where ``sd`` is the sample standard deviation (ddof=1).  The gate passes
  when LCB > 0.  Point margins from Section 4 are reported descriptively in
  ``gate_notes`` (the 0.05/0.10 tier distinction is deleted per A3).  n < 2
  is flagged degenerate and never passes.

* **Discrete semantic entropy clustering rule.**  Pool candidates are
  clustered by EXACT string equality of their graded-answer signature
  (``evaluate_checkpoint.answer_signature`` — the equivalence key already
  matched to the semantic graders), and the score is the Shannon entropy
  (natural log) of the cluster distribution over the N-candidate pool.  The
  abstention score is the NEGATIVE entropy (higher = more confident).

* **Abstention statistics roles (A5).**  ``delong_paired_one_sided`` +
  ``stouffer_pool`` are FUTILITY-ONLY; ``eprocess_rank_bets`` is the single
  confirmatory analysis from row 1; ``nested_cross_conformal`` supplies the
  separate operational coverage/risk readout.  None of them enter the
  ``passed`` conjunction — the abstention claim resolves independently of
  the channel rungs (Section 7).

* **Arm return convention.**  Every arm returns a dict with at least
  ``decode_ids`` (a list of ``[1, T]`` long tensors, one per prepared row)
  plus arm-specific disclosures (shuffle donors, routes used, the shared
  pause vectors).  The dict form is a deliberate widening of "returns
  per-row decode ids" so the audit record can disclose exactly what each
  control did.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # Package-style execution (repo tests) and flat Kaggle execution.
    from .evaluate_checkpoint import (
        TURN_SCAFFOLD,
        answer_signature,
        encode_preserving_ends,
        mean,
    )
    from .modeling_hlwm import HLWMForConditionalGeneration
    from .selective_stats import (
        delong_paired_one_sided,
        eprocess_rank_bets,
        failure_auroc,
        nested_cross_conformal,
    )
    from .semantic_grading import grade_semantic_answer
except ImportError:  # Direct execution inside the Kaggle dataset directory.
    from evaluate_checkpoint import (
        TURN_SCAFFOLD,
        answer_signature,
        encode_preserving_ends,
        mean,
    )
    from modeling_hlwm import HLWMForConditionalGeneration
    from selective_stats import (
        delong_paired_one_sided,
        eprocess_rank_bets,
        failure_auroc,
        nested_cross_conformal,
    )
    from semantic_grading import grade_semantic_answer


Z_90 = 1.2815515655446004
"""One-sided 90% standard-normal quantile used by every paired margin gate."""

FAMILY_ORDER = ("numeric", "unit", "ordering", "abstention")
"""Gold family order matching the V10 batch contract (family_index 0..3)."""

EXPERT_SWAP = {0: 1, 1: 0}
"""Cross-routing ablation map for the two routed experts (plan Section 3)."""

ORDERING_RULE = (
    "masked core = first masked_core_rows masked rows under ascending "
    "lexicographic sort of str(row['episode_id']); the same anchor-id sort "
    "orders expert-arm subsets, the abstention stream and probe rows"
)

CHANNEL_GATE_KEYS = (
    "masked_leak_zero",
    "masked_causal_floor",
    "latent_channel_live",
    "beats_pause_compute",
    "unmasked_parity",
    "probe_generation_gap",
)
"""The per-seed channel gate set the Section 7 ladder keys rungs 3/4 off."""

EXPERT_LIVENESS_FLOOR = 0.10
CROSS_ROUTING_MARGIN = 0.05
ROUTER_DELTA_MARGIN = 0.02
PROBE_LIVENESS_MARGIN = 0.10
PROBE_GENERATION_GAP = 0.10
MASKED_FLOOR_CEILING = 0.10
UNMASKED_PARITY_BAND = 0.05
STEERING_SIGMA = 0.5


# ----------------------------------------------------------------------
# Paired-delta statistics.


def paired_lower_confidence_bound(
    deltas: Sequence[float], confidence_z: float = Z_90
) -> Dict[str, Any]:
    """Normal-approximation one-sided lower bound on the mean paired delta.

    LCB = mean(d) - z * sd(d)/sqrt(n) with sd the ddof=1 sample standard
    deviation.  ``degenerate`` marks n < 2 (no variance estimate; such a
    contrast never passes a gate).  With sd exactly zero the bound equals
    the mean: every paired delta agreed, and the bound is the honest limit
    of the formula.
    """

    values = [float(value) for value in deltas]
    count = len(values)
    if count == 0:
        raise ValueError("paired LCB needs at least one row")
    delta_mean = sum(values) / count
    if count < 2:
        return {
            "n": count,
            "mean": delta_mean,
            "sd": 0.0,
            "lcb": delta_mean,
            "z": confidence_z,
            "degenerate": True,
        }
    variance = sum((value - delta_mean) ** 2 for value in values) / (count - 1)
    sd = math.sqrt(max(0.0, variance))
    return {
        "n": count,
        "mean": delta_mean,
        "sd": sd,
        "lcb": delta_mean - confidence_z * sd / math.sqrt(count),
        "z": confidence_z,
        "degenerate": False,
    }


def _paired_contrast(
    treatment: Sequence[bool], control: Sequence[bool]
) -> Dict[str, Any]:
    if len(treatment) != len(control):
        raise ValueError("paired contrast needs equal-length correctness vectors")
    return paired_lower_confidence_bound(
        [float(bool(a)) - float(bool(b)) for a, b in zip(treatment, control)]
    )


# ----------------------------------------------------------------------
# Row bookkeeping.


def anchor_id(row: Mapping[str, Any]) -> str:
    value = row.get("episode_id")
    if value is None:
        raise ValueError("audit rows must carry an episode_id anchor id")
    return str(value)


def sorted_by_anchor(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The ONE preregistered ordering: ascending lexicographic anchor id."""

    ids = [anchor_id(row) for row in rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(
            "anchor ids must be unique for the preregistered ordering: %s"
            % duplicates[:5]
        )
    return sorted(rows, key=anchor_id)


def masked_core_selection(
    rows: Sequence[Mapping[str, Any]], masked_core_rows: int
) -> List[Mapping[str, Any]]:
    """First ``masked_core_rows`` masked rows in preregistered anchor order."""

    masked = sorted_by_anchor([row for row in rows if bool(row.get("masked"))])
    return masked[: max(0, int(masked_core_rows))]


def expert_arm_selection(
    rows: Sequence[Mapping[str, Any]], per_family: int
) -> List[Mapping[str, Any]]:
    """First ``per_family`` rows of each family, preregistered anchor order."""

    selected: List[Mapping[str, Any]] = []
    ordered = sorted_by_anchor(rows)
    for family in FAMILY_ORDER:
        family_rows = [row for row in ordered if family_of_row(row) == family]
        selected.extend(family_rows[: max(0, int(per_family))])
    return selected


def family_of_row(row: Mapping[str, Any]) -> str:
    family = row.get("family")
    if family is None:
        family = str((row.get("answer_spec") or {}).get("type", "")).lower()
    return str(family)


def family_index_of_row(row: Mapping[str, Any]) -> int:
    if "family_index" in row:
        return int(row["family_index"])
    family = family_of_row(row)
    if family not in FAMILY_ORDER:
        raise ValueError("row %s has unknown family %r" % (anchor_id(row), family))
    return FAMILY_ORDER.index(family)


def v10_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], int]:
    """Four-family audit population plus the count of excluded legacy rows.

    The master splits still carry pre-Version-10 corpus rows (io_tests,
    python_tests, sql_exact, untyped); the Version 10.0 claims are
    preregistered over the four anchor families only, and the family-indexed
    routing cannot represent anything else (session I-4 crash on
    'io_tests').  The excluded count is disclosed in the audit record.
    """

    kept = [row for row in rows if family_of_row(row) in FAMILY_ORDER]
    return kept, len(rows) - len(kept)


def route_for_family_index(
    family_index: int, expert_count: int, router_families: int = 4
) -> Optional[int]:
    """Deterministic family-to-expert map ({numeric,unit}->E0, rest->E1).

    Contiguous blocks of ``router_families // expert_count`` families share
    one expert, matching the plan's {numeric, unit} vs {ordering, abstention}
    pairing at 4 families / 2 experts.
    """

    if expert_count <= 0:
        return None
    group = max(1, int(router_families) // int(expert_count))
    return min(int(expert_count) - 1, int(family_index) // group)


def route_of_row(row: Mapping[str, Any], config: Any) -> Optional[int]:
    if "route_index" in row:
        return int(row["route_index"])
    expert_count = int(getattr(config, "mlp_expert_count", 0))
    if expert_count <= 0:
        return None
    return route_for_family_index(
        family_index_of_row(row), expert_count, int(getattr(config, "router_families", 4))
    )


# ----------------------------------------------------------------------
# Prepared audit rows.


@dataclass
class PreparedRow:
    """One audit row with both encoded prompts and its produced thoughts.

    Thoughts always come from the FULL prompt (the information-asymmetric
    producer surface); the student decode reads the MASKED prompt on masked
    rows and the full prompt otherwise (``masked_ids`` equals ``full_ids``
    on unmasked rows).
    """

    row: Mapping[str, Any]
    anchor_id: str
    family: str
    family_index: int
    route_index: Optional[int]
    full_ids: Tensor
    full_mask: Tensor
    masked_ids: Tensor
    masked_mask: Tensor
    thought_embeds: Tensor
    thought_states: Tensor


def _route_tensor(prepared: PreparedRow, device: torch.device) -> Optional[Tensor]:
    if prepared.route_index is None:
        return None
    return torch.tensor([int(prepared.route_index)], dtype=torch.long, device=device)


def prepare_rows(
    model: HLWMForConditionalGeneration,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    context_tokens: int = 320,
    device: Optional[torch.device] = None,
) -> List[PreparedRow]:
    """Encode prompts (shared unpadded primitive) and produce latent thoughts."""

    if device is None:
        device = next(model.parameters()).device
    prepared: List[PreparedRow] = []
    with torch.no_grad():
        for row in rows:
            prompt = str(row.get("public_prompt") or row.get("prompt") or "")
            if not prompt:
                raise ValueError("row %s has no prompt text" % anchor_id(row))
            masked_prompt = (
                str(row.get("masked_prompt", prompt))
                if bool(row.get("masked"))
                else prompt
            )
            full = encode_preserving_ends(tokenizer, prompt, context_tokens, device)
            masked = encode_preserving_ends(
                tokenizer, masked_prompt, context_tokens, device
            )
            embeds, states = model.produce_latent_thoughts(
                full["input_ids"], full["attention_mask"]
            )
            prepared.append(
                PreparedRow(
                    row=row,
                    anchor_id=anchor_id(row),
                    family=family_of_row(row),
                    family_index=family_index_of_row(row),
                    route_index=route_of_row(row, model.config),
                    full_ids=full["input_ids"],
                    full_mask=full["attention_mask"],
                    masked_ids=masked["input_ids"],
                    masked_mask=masked["attention_mask"],
                    thought_embeds=embeds,
                    thought_states=states,
                )
            )
    return prepared


# ----------------------------------------------------------------------
# Audit arms (a)-(h).  All decode row-by-row (batch 1, the unpadded audit
# surface) and return {"decode_ids": [ [1,T] tensor per row ], ...extras}.


def _decode_v10_row(
    model: HLWMForConditionalGeneration,
    prepared: PreparedRow,
    embeds: Tensor,
    states: Optional[Tensor],
    *,
    use_prefix_slots: bool,
    route_index: Optional[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    device = prepared.masked_ids.device
    route = (
        None
        if route_index is None
        else torch.tensor([int(route_index)], dtype=torch.long, device=device)
    )
    return model.decode_candidate_v10(
        prepared.masked_ids,
        prepared.masked_mask,
        embeds,
        states,
        use_prefix_slots=use_prefix_slots,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        generator=generator,
        route_index=route,
    )


def arm_full(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """(a) Full pipeline: thoughts from the FULL prompt, student decode on
    the masked prompt through slots + gold-label route."""

    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for index, item in enumerate(prepared):
            generator = None
            if temperature > 0.0:
                generator = torch.Generator(device=str(item.masked_ids.device))
                generator.manual_seed(int(seed) + index)
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    item.thought_states,
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    generator=generator,
                )
            )
    return {"arm": "full", "decode_ids": decode_ids}


def arm_floor(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_new_tokens: int = 48,
    use_full_prompt: bool = False,
) -> Dict[str, Any]:
    """(b) No-prefix floor: plain causal ``_decode_candidate`` on the masked
    prompt with ``workspace_prefix=None`` — no thoughts, no slots, no route.

    ``use_full_prompt=True`` is the plain-causal arm of the unmasked-parity
    gate (identical code path on the full prompt)."""

    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for item in prepared:
            ids = item.full_ids if use_full_prompt else item.masked_ids
            mask = item.full_mask if use_full_prompt else item.masked_mask
            decode_ids.append(
                model._decode_candidate(
                    ids, mask, None, max_new_tokens=max_new_tokens, temperature=0.0
                )
            )
    return {"arm": "floor", "decode_ids": decode_ids}


def _within_family_derangement(
    families: Sequence[str], seed: int
) -> List[int]:
    """Seeded within-family derangement: donor[i] != i for every row.

    Rows of each family are put in a seeded random order and rotated by one
    (a rotation of any ordering of n >= 2 elements is a derangement); a
    2-row family therefore degenerates to exactly the pairwise swap.  A
    1-row family cannot receive another row's thoughts, so it raises — the
    preregistered core must be family-balanced, and silently passing a row
    its own thoughts would fake a null.
    """

    singletons = sorted(
        {family for family in families if families.count(family) == 1}
    )
    if singletons:
        raise ValueError(
            "within-family shuffle impossible: singleton families %s"
            % singletons
        )
    donor = [-1] * len(families)
    generator = torch.Generator().manual_seed(int(seed))
    for family in dict.fromkeys(families):
        members = [index for index, value in enumerate(families) if value == family]
        order = [
            members[position]
            for position in torch.randperm(len(members), generator=generator).tolist()
        ]
        for position, row_index in enumerate(order):
            donor[row_index] = order[(position + 1) % len(order)]
    assert all(value >= 0 for value in donor)
    assert all(donor[index] != index for index in range(len(donor))), (
        "shuffled-prefix derangement is incomplete: a row kept its own thoughts"
    )
    return donor


def arm_shuffled(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    seed: int = 0,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(c) Within-family shuffled prefix: every row decodes with ANOTHER
    same-family row's (thought_embeds, thought_states)."""

    donor = _within_family_derangement([item.family for item in prepared], seed)
    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for index, item in enumerate(prepared):
            source = prepared[donor[index]]
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    source.thought_embeds,
                    source.thought_states,
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {"arm": "shuffled", "decode_ids": decode_ids, "donor_indices": donor}


def arm_pause(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    seed: int = 0,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(d) Compute-matched pause: ONE seeded random draw of K content-free
    vectors, grounded via ``model.ground_latents`` and shared by EVERY row.

    The draw depends only on ``seed`` and the model configuration, never on
    any row, so the pause thoughts carry zero mutual information with the
    row by construction.  The grounded vectors get the same synthesis-mode
    embedding real thoughts receive, so the pause arm matches the real
    channel's construction in everything except content."""

    if not prepared:
        return {"arm": "pause", "decode_ids": [], "pause_embeds": None, "pause_states": None}
    device = prepared[0].masked_ids.device
    thoughts = int(model.config.latent_thoughts)
    width = int(model.config.hidden_size)
    generator = torch.Generator().manual_seed(int(seed))
    raw_embeds = torch.randn(1, thoughts, width, generator=generator)
    raw_states = torch.randn(1, thoughts, width, generator=generator)
    with torch.no_grad():
        mode = model.mode_embedding.weight[model.MODE_SYNTHESIS].detach().cpu().float()
        pause_embeds = (
            model.ground_latents(raw_embeds.to(device)).float().cpu() + mode.view(1, 1, -1)
        )
        pause_states = model.ground_latents(raw_states.to(device)).float().cpu()
    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for item in prepared:
            dtype = item.thought_embeds.dtype
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    pause_embeds.to(device=device, dtype=dtype),
                    pause_states.to(device=device, dtype=item.thought_states.dtype),
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {
        "arm": "pause",
        "decode_ids": decode_ids,
        "pause_embeds": pause_embeds,
        "pause_states": pause_states,
    }


def arm_slot_ablation(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(e) Slot ablation: real thoughts in-band, ``use_prefix_slots=False``.

    Thought states are not passed at all, so the arm is structurally
    invariant to any perturbation of the slot input."""

    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for item in prepared:
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    None,
                    use_prefix_slots=False,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {"arm": "slot_ablation", "decode_ids": decode_ids}


def arm_steering(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    seed: int = 0,
    sigma: float = STEERING_SIGMA,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(f) Steering: real thoughts perturbed with seeded Gaussian noise at
    ``sigma`` AFTER grounding (``produce_latent_thoughts`` already grounds
    the embeds; states are perturbed at the same sigma)."""

    decode_ids: List[Tensor] = []
    with torch.no_grad():
        for index, item in enumerate(prepared):
            generator = torch.Generator().manual_seed(int(seed) + 7919 * index)
            embed_noise = torch.randn(
                item.thought_embeds.shape, generator=generator
            ).to(device=item.thought_embeds.device, dtype=item.thought_embeds.dtype)
            state_noise = torch.randn(
                item.thought_states.shape, generator=generator
            ).to(device=item.thought_states.device, dtype=item.thought_states.dtype)
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds + sigma * embed_noise,
                    item.thought_states + sigma * state_noise,
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {"arm": "steering", "decode_ids": decode_ids, "sigma": float(sigma)}


def arm_cross_routing(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(g) Cross-routing ablation: gold route mapped through {0->1, 1->0}."""

    decode_ids: List[Tensor] = []
    routes_used: List[Optional[int]] = []
    with torch.no_grad():
        for item in prepared:
            if item.route_index is None:
                raise ValueError(
                    "cross-routing needs routed experts; row %s has no route"
                    % item.anchor_id
                )
            swapped = EXPERT_SWAP.get(int(item.route_index))
            if swapped is None:
                raise ValueError(
                    "cross-routing swap is defined for experts 0/1; row %s "
                    "has route %d" % (item.anchor_id, item.route_index)
                )
            routes_used.append(swapped)
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    item.thought_states,
                    use_prefix_slots=True,
                    route_index=swapped,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {"arm": "cross_routing", "decode_ids": decode_ids, "routes_used": routes_used}


def arm_router_routed(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """(h) Router-routed: route from ``route_family_logits`` argmax through
    the same family-to-expert map, instead of the gold label."""

    expert_count = int(getattr(model.config, "mlp_expert_count", 0))
    router_families = int(getattr(model.config, "router_families", 4))
    decode_ids: List[Tensor] = []
    predicted_families: List[int] = []
    router_routes: List[Optional[int]] = []
    with torch.no_grad():
        for item in prepared:
            logits = model.route_family_logits(item.thought_states)
            predicted = int(logits.argmax(dim=-1).item())
            predicted_families.append(predicted)
            route = route_for_family_index(predicted, expert_count, router_families)
            router_routes.append(route)
            decode_ids.append(
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    item.thought_states,
                    use_prefix_slots=True,
                    route_index=route,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {
        "arm": "router_routed",
        "decode_ids": decode_ids,
        "predicted_families": predicted_families,
        "router_routes": router_routes,
    }


# ----------------------------------------------------------------------
# Grading, leak scan, per-arm accuracy.


def grade_decode(
    tokenizer: Any, decode_ids: Tensor, spec: Mapping[str, Any]
) -> Tuple[str, bool]:
    text = tokenizer.decode(
        decode_ids[0].detach().cpu(), skip_special_tokens=True
    ).strip()
    return text, bool(grade_semantic_answer(text, spec or {})["correct"])


def grade_arm(
    tokenizer: Any,
    prepared: Sequence[PreparedRow],
    arm_result: Mapping[str, Any],
) -> Dict[str, Any]:
    texts: List[str] = []
    correct: List[bool] = []
    for item, ids in zip(prepared, arm_result["decode_ids"]):
        text, valid = grade_decode(tokenizer, ids, item.row.get("answer_spec") or {})
        texts.append(text)
        correct.append(valid)
    return {
        "correct": correct,
        "accuracy": mean(float(value) for value in correct) if correct else 0.0,
        "texts": [text[:400] for text in texts],
    }


def masked_leak_scan(row: Mapping[str, Any], decode_text: str) -> Dict[str, bool]:
    """Version 9.0 leak scan, reused verbatim in spirit: (i) a withheld
    literal surviving inside the masked prompt (input leak), (ii) the
    line-anchored TURN_SCAFFOLD regex anywhere in the decoded answer."""

    prompt = str(row.get("public_prompt") or row.get("prompt") or "")
    masked_prompt = str(row.get("masked_prompt", prompt))
    withheld = [str(item) for item in row.get("withheld_literals", [])]
    input_leak = any(literal and literal in masked_prompt for literal in withheld)
    scaffold = bool(TURN_SCAFFOLD.search(decode_text or ""))
    return {
        "input_leak": bool(input_leak),
        "scaffold": scaffold,
        "leak": bool(input_leak or scaffold),
    }


# ----------------------------------------------------------------------
# Expert liveness (A3 precondition for the cross-routing gate).


def expert_liveness(
    model: HLWMForConditionalGeneration,
    prepared: Sequence[PreparedRow],
    *,
    max_rows: int = 8,
    max_new_tokens: int = 1,
) -> Dict[str, Any]:
    """Mean relative routed-expert output norm via forward hooks.

    ratio = mean per-position L2 norm of the routed expert's LoRA output
    (scale included) / mean per-position L2 norm of the shared path at the
    same layers.  The shared path is the always-on shared down-projection
    LoRA when it exists, else the frozen base down projection (disclosed in
    ``denominator``).  Below ``EXPERT_LIVENESS_FLOOR`` the cross-routing
    null is "experts_never_engaged" — an infrastructure finding, not
    negative evidence.
    """

    expert_norms: List[float] = []
    shared_norms: List[float] = []
    handles: List[Any] = []
    denominator = None

    def _norm_recorder(store: List[float]) -> Callable[..., None]:
        def _hook(module: nn.Module, inputs: Any, output: Tensor) -> None:
            store.append(float(output.detach().float().norm(dim=-1).mean()))

        return _hook

    for layer in model.backbone.layers:
        mlp = layer.mlp
        experts = getattr(mlp, "mlp_experts", None)
        if experts is None:
            continue
        for expert in experts:
            handles.append(expert.register_forward_hook(_norm_recorder(expert_norms)))
        if mlp.down_lora.enabled:
            shared_module, denominator = mlp.down_lora, "shared_lora"
        else:
            shared_module, denominator = mlp.down_proj, "base_down_proj"
        handles.append(shared_module.register_forward_hook(_norm_recorder(shared_norms)))

    if not handles:
        return {
            "state": "no_expert_modules",
            "ratio": 0.0,
            "engaged": False,
            "expert_events": 0,
            "shared_events": 0,
            "denominator": None,
        }
    try:
        with torch.no_grad():
            for item in prepared[: max(1, int(max_rows))]:
                _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    item.thought_states,
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                )
    finally:
        for handle in handles:
            handle.remove()
    mean_expert = mean(expert_norms) if expert_norms else 0.0
    mean_shared = mean(shared_norms) if shared_norms else 0.0
    ratio = float(mean_expert / max(mean_shared, 1.0e-9))
    if not math.isfinite(ratio):
        ratio = 0.0
    return {
        "state": "measured",
        "ratio": ratio,
        "engaged": bool(expert_norms and ratio >= EXPERT_LIVENESS_FLOOR),
        "expert_events": len(expert_norms),
        "shared_events": len(shared_norms),
        "denominator": denominator,
    }


# ----------------------------------------------------------------------
# In-audit premise-digit probe (A3: preconditioned gate).


def probe_digit_label(row: Mapping[str, Any]) -> Optional[int]:
    """Digit class for the probe: explicit ``probe_digit`` if present, else
    the first digit character of the first withheld literal."""

    if "probe_digit" in row:
        return int(row["probe_digit"]) % 10
    for literal in row.get("withheld_literals", []) or []:
        for char in str(literal):
            if char.isdigit():
                return int(char)
    return None


def train_premise_probe(
    latents: Tensor,
    labels: Sequence[int],
    *,
    seed: int,
    epochs: int = 300,
    lr: float = 0.05,
    train_fraction: float = 0.75,
    num_classes: int = 10,
    min_rows: int = 8,
) -> Dict[str, Any]:
    """Small closed training loop: one seeded linear head on held-out latents.

    Latents are standardized per dimension; the head is trained with Adam on
    a seeded train split and scored on the disjoint eval split.  ``chance``
    is the majority-class frequency over ALL provided labels (the accuracy a
    latent-blind guesser achieves), so the liveness precondition
    ``probe >= chance + 0.10`` cannot be met by class imbalance alone.
    """

    labels_list = [int(value) for value in labels]
    count = int(latents.shape[0])
    if count != len(labels_list):
        raise ValueError("latents and labels must align")
    if count < min_rows or len(set(labels_list)) < 2:
        return {
            "state": "insufficient_rows",
            "probe_accuracy": None,
            "chance": None,
            "live": False,
            "n": count,
            "n_train": 0,
            "n_eval": 0,
        }
    features = latents.detach().float().cpu()
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    features = (features - feature_mean) / feature_std
    targets = torch.tensor(labels_list, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    split = max(1, min(count - 1, int(round(count * train_fraction))))
    train_index, eval_index = permutation[:split], permutation[split:]
    weight = (
        torch.randn(num_classes, features.shape[1], generator=generator) * 0.01
    ).requires_grad_(True)
    bias = torch.zeros(num_classes, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    train_x, train_y = features[train_index], targets[train_index]
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        loss = F.cross_entropy(train_x @ weight.t() + bias, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        eval_logits = features[eval_index] @ weight.t() + bias
        accuracy = float(
            (eval_logits.argmax(dim=-1) == targets[eval_index]).float().mean()
        )
    counts: Dict[int, int] = {}
    for value in labels_list:
        counts[value] = counts.get(value, 0) + 1
    chance = max(counts.values()) / count
    return {
        "state": "trained",
        "probe_accuracy": accuracy,
        "chance": float(chance),
        "live": bool(accuracy >= chance + PROBE_LIVENESS_MARGIN),
        "n": count,
        "n_train": int(split),
        "n_eval": int(count - split),
    }


# ----------------------------------------------------------------------
# Abstention block (A5): scores, futility DeLong, confirmatory e-process,
# mandatory semantic-entropy baseline, operational conformal readout.


def discrete_semantic_entropy(
    pool_texts: Sequence[str], spec: Mapping[str, Any]
) -> Dict[str, Any]:
    """Shannon entropy (natural log) of the graded-answer cluster distribution.

    Clustering rule: candidates belong to one cluster iff their
    ``answer_signature`` strings are EXACTLY equal — the grader-matched
    equivalence key, so "7" and "the answer is 7 kg" cluster together on a
    numeric row while "8" does not."""

    signatures = [answer_signature(text, spec or {}) for text in pool_texts]
    counts: Dict[str, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    total = len(signatures)
    entropy = 0.0
    for value in counts.values():
        p = value / total
        entropy -= p * math.log(p)
    return {
        "entropy": float(entropy),
        "n_clusters": len(counts),
        "cluster_sizes": sorted(counts.values(), reverse=True),
        "n": total,
    }


def student_channel_mean_logprob(
    model: HLWMForConditionalGeneration,
    prepared: PreparedRow,
    decode_ids: Tensor,
) -> float:
    """Default calibrated "heads" score: the V10 channel's own mean logprob
    of its emission (masked prompt + thoughts + slots + route)."""

    if decode_ids.shape[1] == 0:
        return -1.0e9
    device = prepared.masked_ids.device
    with torch.no_grad():
        out = model.student_channel_teacher_force(
            prepared.masked_ids,
            prepared.masked_mask,
            prepared.thought_embeds,
            prepared.thought_states,
            decode_ids,
            torch.ones_like(decode_ids),
            use_prefix_slots=True,
            route_index=_route_tensor(prepared, device),
        )
    log_probabilities = (
        F.log_softmax(out["logits"].float(), dim=-1)
        .gather(-1, decode_ids[:, :, None])
        .squeeze(-1)
    )
    return float(log_probabilities.mean())


def causal_mean_logprob(
    model: HLWMForConditionalGeneration,
    prepared: PreparedRow,
    decode_ids: Tensor,
) -> float:
    """Max-logprob baseline score: plain-channel mean answer logprob."""

    if decode_ids.shape[1] == 0:
        return -1.0e9
    with torch.no_grad():
        value = model.causal_answer_mean_logprob(
            prepared.masked_ids, prepared.masked_mask, decode_ids
        )
    return float(value[0])


def gold_answer_forced_ce(
    model: HLWMForConditionalGeneration,
    prepared: PreparedRow,
    tokenizer: Any,
    *,
    max_gold_tokens: int = 48,
) -> Optional[Dict[str, float]]:
    """Teacher-forced CE (nats/token) of the GOLD answer, four ways.

    Every previous study scored what the channel EMITS.  Nothing scored what
    it assigns to the RIGHT answer, so a null emission result could not
    distinguish two very different failures: an exposure gap (the premise is
    physically absent, so the answer is genuinely unpredictable) from a
    content gap (the answer stays unpredictable even with the premise in the
    decoder ids).  The four arms hold everything constant but the two things
    that differ -- whether the decoder context carries the premise, and
    whether generation is routed through the V10 channel:

      channel_masked  V10 channel, premise removed  (the shipped condition)
      channel_full    V10 channel, premise present  (exposure controlled)
      causal_masked   plain channel, premise removed
      causal_full     plain channel, premise present  (the substrate ceiling)

    Returns ``None`` on rows without a gold answer string.
    """

    target = str(prepared.row.get("public_target") or "").strip()
    if not target:
        return None
    token_ids = list(tokenizer.encode(target, add_special_tokens=False))[
        : max(1, int(max_gold_tokens))
    ]
    if not token_ids:
        return None
    device = prepared.masked_ids.device
    gold = torch.tensor([token_ids], dtype=torch.long, device=device)
    gold_mask = torch.ones_like(gold)
    route = _route_tensor(prepared, device)

    def channel_ce(context_ids: Tensor, context_mask: Tensor) -> float:
        with torch.no_grad():
            out = model.student_channel_teacher_force(
                context_ids,
                context_mask,
                prepared.thought_embeds,
                prepared.thought_states,
                gold,
                gold_mask,
                use_prefix_slots=True,
                route_index=route,
            )
        log_probabilities = (
            F.log_softmax(out["logits"].float(), dim=-1)
            .gather(-1, gold[:, :, None])
            .squeeze(-1)
        )
        return float(-log_probabilities.mean())

    with torch.no_grad():
        causal_masked = -float(
            model.causal_answer_mean_logprob(
                prepared.masked_ids, prepared.masked_mask, gold
            )[0]
        )
        causal_full = -float(
            model.causal_answer_mean_logprob(
                prepared.full_ids, prepared.full_mask, gold
            )[0]
        )
    channel_masked = channel_ce(prepared.masked_ids, prepared.masked_mask)
    channel_full = channel_ce(prepared.full_ids, prepared.full_mask)
    return {
        "channel_masked": channel_masked,
        "channel_full": channel_full,
        "causal_masked": causal_masked,
        "causal_full": causal_full,
        # premise removal cost, channel held fixed
        "exposure_gap": channel_masked - channel_full,
        # cost of routing through the channel, exposure held fixed
        "channel_cost": channel_full - causal_full,
        "gold_tokens": len(token_ids),
    }


def gold_forced_ce_block(
    model: HLWMForConditionalGeneration,
    prepared_items: Sequence[PreparedRow],
    tokenizer: Any,
    *,
    rows: int = 128,
    max_gold_tokens: int = 48,
) -> Dict[str, Any]:
    """Bounded, deterministic, stratified gold-CE sweep.

    Rows are taken round-robin over ``(family, masked)`` buckets in anchor
    order so the sample is balanced and reproducible without an RNG, and the
    cost stays bounded: four teacher-forced forwards per selected row against
    the pool's eight decodes per audit row.
    """

    buckets: Dict[Tuple[str, bool], List[PreparedRow]] = {}
    for item in prepared_items:
        key = (item.family, bool(item.row.get("masked")))
        buckets.setdefault(key, []).append(item)
    order = sorted(buckets)
    selected: List[PreparedRow] = []
    depth = 0
    while len(selected) < max(0, int(rows)):
        added = False
        for key in order:
            if depth < len(buckets[key]) and len(selected) < int(rows):
                selected.append(buckets[key][depth])
                added = True
        if not added:
            break
        depth += 1

    per_row: List[Dict[str, Any]] = []
    for item in selected:
        measured = gold_answer_forced_ce(
            model, item, tokenizer, max_gold_tokens=max_gold_tokens
        )
        if measured is None:
            continue
        record = dict(measured)
        record["anchor_id"] = item.anchor_id
        record["family"] = item.family
        record["masked"] = bool(item.row.get("masked"))
        per_row.append(record)

    keys = (
        "channel_masked",
        "channel_full",
        "causal_masked",
        "causal_full",
        "exposure_gap",
        "channel_cost",
    )

    def summarize(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0, **{key: None for key in keys}}
        return {
            "n": len(subset),
            **{
                key: float(sum(float(row[key]) for row in subset) / len(subset))
                for key in keys
            },
        }

    families = sorted({str(row["family"]) for row in per_row})
    return {
        "n_selected": len(selected),
        "n_scored": len(per_row),
        "max_gold_tokens": int(max_gold_tokens),
        "overall": summarize(per_row),
        "masked": summarize([row for row in per_row if row["masked"]]),
        "unmasked": summarize([row for row in per_row if not row["masked"]]),
        "by_family": {
            family: summarize([row for row in per_row if row["family"] == family])
            for family in families
        },
        "records": per_row,
    }


def abstention_block(
    *,
    head_scores: Sequence[float],
    logprob_scores: Sequence[float],
    entropy_scores: Sequence[float],
    correct: Sequence[bool],
    anchor_ids: Sequence[str],
    target_coverage: float = 0.40,
    seed: int = 0,
    eprocess_lam: float = 0.5,
    eprocess_resume: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """A5 statistics stack over one seed's full-arm rows (preregistered
    anchor-id streaming order supplied by the caller).

    DeLong is FUTILITY-ONLY; the e-process is the single confirmatory
    analysis (resumable across the extension session via
    ``eprocess_resume`` = a previous result's wealth/max_wealth/pending).
    The semantic-entropy baseline is mandatory; conformal coverage/risk is
    the separate operational readout, never the ranking claim.
    """

    n = len(correct)
    if not (len(head_scores) == len(logprob_scores) == len(entropy_scores) == n == len(anchor_ids)):
        raise ValueError("abstention block needs aligned per-row vectors")
    labels = [bool(value) for value in correct]
    result: Dict[str, Any] = {
        "n": n,
        "n_correct": sum(labels),
        "n_incorrect": n - sum(labels),
        "degenerate": not (0 < sum(labels) < n),
    }
    resume = dict(eprocess_resume or {})
    eproc = eprocess_rank_bets(
        list(head_scores),
        list(logprob_scores),
        labels,
        lam=eprocess_lam,
        resume_wealth=float(resume.get("wealth", 1.0)),
        resume_max_wealth=resume.get("max_wealth"),
        resume_pending=resume.get("pending_wrongs"),
    )
    result["eprocess"] = eproc
    result["eprocess_confirmatory"] = True
    scores_by_name = {
        "head": list(head_scores),
        "logprob": list(logprob_scores),
        "neg_semantic_entropy": list(entropy_scores),
    }
    if result["degenerate"]:
        result["failure_auroc"] = {name: None for name in scores_by_name}
        result["delong_head_vs_logprob"] = None
        result["delong_head_vs_entropy"] = None
    else:
        result["failure_auroc"] = {
            name: float(failure_auroc(values, labels))
            for name, values in scores_by_name.items()
        }
        futility = dict(
            delong_paired_one_sided(list(head_scores), list(logprob_scores), labels)
        )
        futility["futility_only"] = True
        result["delong_head_vs_logprob"] = futility
        entropy_check = dict(
            delong_paired_one_sided(list(head_scores), list(entropy_scores), labels)
        )
        entropy_check["futility_only"] = True
        result["delong_head_vs_entropy"] = entropy_check
    unique_anchors = len(dict.fromkeys(anchor_ids))
    n_folds = min(5, unique_anchors)
    try:
        if n_folds < 2:
            raise ValueError("need at least two unique anchors for conformal folds")
        conformal = nested_cross_conformal(
            list(anchor_ids),
            list(head_scores),
            labels,
            n_folds=n_folds,
            target_coverage=float(target_coverage),
            seed=int(seed),
        )
        result["conformal"] = {
            "n_folds": conformal["n_folds"],
            "target_coverage": conformal["target_coverage"],
            "out_of_fold_coverage_correct": conformal["out_of_fold_coverage_correct"],
            "publish_rate": conformal["publish_rate"],
            "fold_thresholds": conformal["fold_thresholds"],
            "selective_risk": _selective_risk(labels, conformal["row_published"]),
        }
    except (ValueError, AssertionError) as error:
        result["conformal"] = {"error": str(error)}
    return result


def _selective_risk(labels: Sequence[bool], published: Sequence[bool]) -> Optional[float]:
    accepted = [label for label, keep in zip(labels, published) if keep]
    if not accepted:
        return None
    return sum(1 for label in accepted if not label) / len(accepted)


# ----------------------------------------------------------------------
# w/o-L1 branch comparison (A3: descriptive diagnostic; A6 item 6 hygiene).


def state_dict_sha256(model_or_state: Any) -> str:
    """Deterministic sha256 over a state dict (sorted names, raw bytes)."""

    if isinstance(model_or_state, nn.Module):
        state = model_or_state.state_dict()
    else:
        state = dict(model_or_state)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        tensor = tensor.detach().cpu().contiguous().view(-1)
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(int(tensor.numel())).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def wo_l1_delta(
    model_full: HLWMForConditionalGeneration,
    model_ablated: HLWMForConditionalGeneration,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    expected_full_sha256: Optional[str],
    seed: int = 0,
    context_tokens: int = 320,
    max_new_tokens: int = 48,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Descriptive w/o-L1 diagnostic (A3 demoted it from Claim B conjunct).

    Both models run their own FULL arm on identical rows.  A6 item 6
    hygiene, enforced before any decode: the caller MUST pass the recorded
    full-arm weight hash (refusing to run otherwise), the live full arm
    must match it, and the two state dicts must differ — a matching pair
    means the "branch" trained the live object in place and the contrast is
    contaminated.
    """

    if not expected_full_sha256:
        raise ValueError(
            "A6 item 6: the recorded full-arm weight sha256 is required; "
            "refusing to compare unverified checkpoints"
        )
    full_hash = state_dict_sha256(model_full)
    if full_hash != str(expected_full_sha256):
        raise RuntimeError(
            "full-arm weights do not match the recorded branch-point hash "
            "(%s != %s)" % (full_hash[:12], str(expected_full_sha256)[:12])
        )
    ablated_hash = state_dict_sha256(model_ablated)
    if ablated_hash == full_hash:
        raise RuntimeError(
            "w/o-L1 branch weights are byte-identical to the full arm: "
            "in-place contamination — the branch must train a reloaded copy"
        )
    ordered = sorted_by_anchor(v10_audit_rows(rows)[0])
    results: Dict[str, List[bool]] = {}
    for label, model in (("full", model_full), ("ablated", model_ablated)):
        prepared = prepare_rows(
            model, tokenizer, ordered, context_tokens=context_tokens, device=device
        )
        graded = grade_arm(
            tokenizer,
            prepared,
            arm_full(model, prepared, max_new_tokens=max_new_tokens, seed=seed),
        )
        results[label] = graded["correct"]
    paired = _paired_contrast(results["full"], results["ablated"])
    full_accuracy = mean(float(v) for v in results["full"]) if ordered else 0.0
    ablated_accuracy = mean(float(v) for v in results["ablated"]) if ordered else 0.0
    return {
        "descriptive_only": True,
        "n": len(ordered),
        "full_sha256": full_hash,
        "ablated_sha256": ablated_hash,
        "full_accuracy": full_accuracy,
        "ablated_accuracy": ablated_accuracy,
        "ablated_minus_full_mean": ablated_accuracy - full_accuracy,
        "full_minus_ablated": paired,
    }


# ----------------------------------------------------------------------
# The audit orchestrator.


def v10_audit(
    model: HLWMForConditionalGeneration,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Any,
    *,
    masked_core_rows: int = 160,
    seed: int,
    expert_rows_per_family: int = 128,
    context_tokens: int = 320,
    max_new_tokens: int = 48,
    pool_temperatures: Sequence[float] = (0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    probe_latent_rows: int = 512,
    gold_ce_rows: int = 128,
    target_coverage: float = 0.40,
    device: Optional[torch.device] = None,
    head_score_fn: Optional[Callable[..., float]] = None,
    eprocess_resume: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full Version 10.0 audit battery for one seed.

    Arms (a)-(f) run on the FIXED masked core (ordering rule:
    ``ORDERING_RULE``); arms (g)-(h) on ``expert_rows_per_family`` rows per
    family; the full arm additionally decodes an N-candidate pool on EVERY
    audit row for the abstention analysis (pool convention: greedy first,
    then flat-temperature samples — the Version 9.0 N=8 recipe).  Every
    decode is graded through ``grade_semantic_answer``.
    """

    if not list(pool_temperatures) or float(pool_temperatures[0]) != 0.0:
        raise ValueError("pool temperatures must start with 0.0 (greedy first)")
    if device is None:
        device = next(model.parameters()).device
    audit_rows, legacy_rows_excluded = v10_audit_rows(rows)
    ordered_rows = sorted_by_anchor(audit_rows)

    prepared_all = prepare_rows(
        model, tokenizer, ordered_rows, context_tokens=context_tokens, device=device
    )
    prepared_by_id = {item.anchor_id: item for item in prepared_all}

    # ---- masked core, arms (a)-(f).
    core_rows = masked_core_selection(ordered_rows, masked_core_rows)
    core = [prepared_by_id[anchor_id(row)] for row in core_rows]
    core_arm_results = {
        "full": arm_full(model, core, max_new_tokens=max_new_tokens, seed=seed),
        "floor": arm_floor(model, core, max_new_tokens=max_new_tokens),
        "shuffled": arm_shuffled(model, core, seed=seed + 1, max_new_tokens=max_new_tokens),
        "pause": arm_pause(model, core, seed=seed + 2, max_new_tokens=max_new_tokens),
        "slot_ablation": arm_slot_ablation(model, core, max_new_tokens=max_new_tokens),
        "steering": arm_steering(model, core, seed=seed + 3, max_new_tokens=max_new_tokens),
    }
    core_arms = {
        name: grade_arm(tokenizer, core, result)
        for name, result in core_arm_results.items()
    }
    leak_flags = [
        masked_leak_scan(item.row, text)
        for item, text in zip(core, core_arms["full"]["texts"])
    ]
    masked_core = {
        "n": len(core),
        "anchor_ids": [item.anchor_id for item in core],
        "arms": core_arms,
        "leak": {
            "input_leak_rows": sum(1 for flag in leak_flags if flag["input_leak"]),
            "scaffold_rows": sum(1 for flag in leak_flags if flag["scaffold"]),
            "leak_rows": sum(1 for flag in leak_flags if flag["leak"]),
        },
        "shuffled_donors": core_arm_results["shuffled"]["donor_indices"],
    }
    paired = {
        "full_minus_shuffled": _paired_contrast(
            core_arms["full"]["correct"], core_arms["shuffled"]["correct"]
        ),
        "full_minus_pause": _paired_contrast(
            core_arms["full"]["correct"], core_arms["pause"]["correct"]
        ),
        "full_minus_slot_ablation": _paired_contrast(
            core_arms["full"]["correct"], core_arms["slot_ablation"]["correct"]
        ),
        "full_minus_steering": _paired_contrast(
            core_arms["full"]["correct"], core_arms["steering"]["correct"]
        ),
        "full_minus_floor": _paired_contrast(
            core_arms["full"]["correct"], core_arms["floor"]["correct"]
        ),
    } if core else {}

    # ---- unmasked parity (full V10 channel vs plain causal, full prompt).
    unmasked_rows = [row for row in ordered_rows if not bool(row.get("masked"))]
    unmasked = {"n": len(unmasked_rows), "full_accuracy": None, "causal_accuracy": None, "parity_gap": None}
    if unmasked_rows:
        prepared_unmasked = [prepared_by_id[anchor_id(row)] for row in unmasked_rows]
        full_graded = grade_arm(
            tokenizer,
            prepared_unmasked,
            arm_full(model, prepared_unmasked, max_new_tokens=max_new_tokens, seed=seed),
        )
        causal_graded = grade_arm(
            tokenizer,
            prepared_unmasked,
            arm_floor(
                model, prepared_unmasked, max_new_tokens=max_new_tokens, use_full_prompt=True
            ),
        )
        unmasked.update(
            {
                "full_accuracy": full_graded["accuracy"],
                "causal_accuracy": causal_graded["accuracy"],
                "parity_gap": full_graded["accuracy"] - causal_graded["accuracy"],
            }
        )

    # ---- expert arms (g)-(h) on the per-family subsets.
    experts: Dict[str, Any] = {"state": "experts_disabled"}
    if int(getattr(config, "mlp_expert_count", 0)) > 0:
        expert_rows = expert_arm_selection(ordered_rows, expert_rows_per_family)
        prepared_experts = [prepared_by_id[anchor_id(row)] for row in expert_rows]
        label_graded = grade_arm(
            tokenizer,
            prepared_experts,
            arm_full(model, prepared_experts, max_new_tokens=max_new_tokens, seed=seed),
        )
        cross_result = arm_cross_routing(
            model, prepared_experts, max_new_tokens=max_new_tokens
        )
        cross_graded = grade_arm(tokenizer, prepared_experts, cross_result)
        router_result = arm_router_routed(
            model, prepared_experts, max_new_tokens=max_new_tokens
        )
        router_graded = grade_arm(tokenizer, prepared_experts, router_result)
        agreement = mean(
            float(predicted == item.family_index)
            for predicted, item in zip(
                router_result["predicted_families"], prepared_experts
            )
        ) if prepared_experts else 0.0
        per_family_n: Dict[str, int] = {}
        for item in prepared_experts:
            per_family_n[item.family] = per_family_n.get(item.family, 0) + 1
        experts = {
            "state": "measured",
            "n_rows": len(prepared_experts),
            "per_family_n": per_family_n,
            "label_routed_accuracy": label_graded["accuracy"],
            "cross_routed_accuracy": cross_graded["accuracy"],
            "router_routed_accuracy": router_graded["accuracy"],
            "cross_routing_cost": label_graded["accuracy"] - cross_graded["accuracy"],
            "cross_routing_paired": _paired_contrast(
                label_graded["correct"], cross_graded["correct"]
            ) if prepared_experts else None,
            "router_delta": abs(
                label_graded["accuracy"] - router_graded["accuracy"]
            ),
            "router_agreement": agreement,
            "liveness": expert_liveness(model, prepared_experts),
        }

    # ---- abstention: full-arm pool on EVERY audit row, preregistered order.
    score_head = head_score_fn or student_channel_mean_logprob
    records: List[Dict[str, Any]] = []
    head_scores: List[float] = []
    logprob_scores: List[float] = []
    entropy_scores: List[float] = []
    correct_flags: List[bool] = []
    anchor_ids: List[str] = []
    for index, item in enumerate(prepared_all):
        spec = item.row.get("answer_spec") or {}
        pool_texts: List[str] = []
        greedy_ids: Optional[Tensor] = None
        with torch.no_grad():
            for candidate, temperature in enumerate(pool_temperatures):
                generator = None
                if float(temperature) > 0.0:
                    generator = torch.Generator(device=str(device))
                    generator.manual_seed(
                        int(seed) * 1_000_003 + index * 101 + candidate
                    )
                ids = _decode_v10_row(
                    model,
                    item,
                    item.thought_embeds,
                    item.thought_states,
                    use_prefix_slots=True,
                    route_index=item.route_index,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature),
                    generator=generator,
                )
                if candidate == 0:
                    greedy_ids = ids
                pool_texts.append(
                    tokenizer.decode(
                        ids[0].detach().cpu(), skip_special_tokens=True
                    ).strip()
                )
        assert greedy_ids is not None
        greedy_text = pool_texts[0]
        row_correct = bool(grade_semantic_answer(greedy_text, spec)["correct"])
        entropy = discrete_semantic_entropy(pool_texts, spec)
        head = float(score_head(model, item, greedy_ids))
        logprob = causal_mean_logprob(model, item, greedy_ids)
        head_scores.append(head)
        logprob_scores.append(logprob)
        entropy_scores.append(-entropy["entropy"])
        correct_flags.append(row_correct)
        anchor_ids.append(item.anchor_id)
        records.append(
            {
                "anchor_id": item.anchor_id,
                "family": item.family,
                "masked": bool(item.row.get("masked")),
                "greedy_text": greedy_text[:400],
                "correct": row_correct,
                "head_score": head,
                "logprob_score": logprob,
                "semantic_entropy": entropy["entropy"],
                "pool_clusters": entropy["n_clusters"],
                "pool_size": entropy["n"],
            }
        )
    abstention = abstention_block(
        head_scores=head_scores,
        logprob_scores=logprob_scores,
        entropy_scores=entropy_scores,
        correct=correct_flags,
        anchor_ids=anchor_ids,
        target_coverage=target_coverage,
        seed=seed,
        eprocess_resume=eprocess_resume,
    )

    # ---- gold-answer teacher-forced CE: exposure gap vs content gap.
    gold_ce = gold_forced_ce_block(
        model,
        prepared_all,
        tokenizer,
        rows=gold_ce_rows,
        max_gold_tokens=max_new_tokens,
    )

    # ---- probe: held-out masked rows (never the core), anchor order.
    core_ids = set(masked_core["anchor_ids"])
    heldout = [
        item
        for item in prepared_all
        if bool(item.row.get("masked")) and item.anchor_id not in core_ids
    ][: max(0, int(probe_latent_rows))]
    probe: Dict[str, Any]
    labelled = [
        (item, probe_digit_label(item.row))
        for item in heldout
        if probe_digit_label(item.row) is not None
    ]
    if labelled:
        latents = torch.cat(
            [item.thought_states.float().mean(dim=1).cpu() for item, _ in labelled],
            dim=0,
        )
        probe = train_premise_probe(
            latents, [label for _, label in labelled], seed=seed + 4
        )
        probe["n_heldout_rows"] = len(labelled)
    else:
        probe = {
            "state": "no_heldout_rows",
            "probe_accuracy": None,
            "chance": None,
            "live": False,
            "n": 0,
            "n_heldout_rows": 0,
        }

    return {
        "seed": int(seed),
        "ordering_rule": ORDERING_RULE,
        "n_rows": len(ordered_rows),
        "legacy_rows_excluded": legacy_rows_excluded,
        "masked_core": masked_core,
        "paired": paired,
        "unmasked": unmasked,
        "experts": experts,
        "gold_forced_ce": gold_ce,
        "probe": probe,
        "abstention": abstention,
        "records": records,
    }


# ----------------------------------------------------------------------
# Verdict assembly.


def v10_gate_dict(
    audit: Mapping[str, Any],
    *,
    training_complete: bool = True,
    zero_skipped_updates: bool = True,
    warm_gate_passed: bool = True,
    infrastructure_ok: bool = True,
    wo_l1: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Flat JSON-serializable gate battery (Version 9.0 conventions).

    Bool gates + a ``passed`` conjunction + ``gate_notes`` carrying every
    disclosed diagnostic.  ``training_complete``/``zero_skipped_updates``/
    ``warm_gate_passed``/``infrastructure_ok`` are notebook-supplied, like
    the Version 9.0 training gates.  The abstention statistics NEVER enter
    ``passed`` — the claim resolves independently via the e-process
    (Section 7); they live in ``gate_notes``.  The w/o-L1 result is a
    descriptive diagnostic (A3) and likewise never a conjunct.
    """

    masked_core = audit.get("masked_core") or {}
    paired = audit.get("paired") or {}
    unmasked = audit.get("unmasked") or {}
    experts = audit.get("experts") or {}
    probe = audit.get("probe") or {}
    abstention = audit.get("abstention") or {}

    core_n = int(masked_core.get("n") or 0)
    leak_rows = int((masked_core.get("leak") or {}).get("leak_rows", 0))
    arms = masked_core.get("arms") or {}
    full_accuracy = float((arms.get("full") or {}).get("accuracy") or 0.0)
    floor_accuracy = float(
        (arms.get("floor") or {}).get("accuracy")
        if (arms.get("floor") or {}).get("accuracy") is not None
        else 1.0
    )

    def _lcb(name: str) -> Optional[Mapping[str, Any]]:
        return paired.get(name)

    live_contrast = _lcb("full_minus_shuffled")
    pause_contrast = _lcb("full_minus_pause")
    latent_channel_live = bool(
        core_n > 0
        and live_contrast is not None
        and not live_contrast["degenerate"]
        and live_contrast["lcb"] > 0.0
    )
    beats_pause = bool(
        core_n > 0
        and pause_contrast is not None
        and not pause_contrast["degenerate"]
        and pause_contrast["lcb"] > 0.0
    )

    parity_gap = unmasked.get("parity_gap")
    if int(unmasked.get("n") or 0) > 0 and parity_gap is not None:
        unmasked_parity = bool(abs(float(parity_gap)) <= UNMASKED_PARITY_BAND)
        parity_state = "measured"
    else:
        unmasked_parity = True
        parity_state = "not_evaluated_no_unmasked_rows"

    # Probe gate: PRECONDITIONED (A3).  Evaluated only when the channel is
    # live AND the probe clears its own liveness floor; otherwise vacuous
    # True with the precondition state disclosed.
    probe_accuracy = probe.get("probe_accuracy")
    probe_live = bool(probe.get("live"))
    if not latent_channel_live:
        probe_gate, probe_state = True, "not_evaluated_channel_dead"
    elif not probe_live or probe_accuracy is None:
        probe_gate, probe_state = True, "not_evaluated_probe_below_liveness"
    else:
        probe_gate = bool(
            float(probe_accuracy) - full_accuracy <= PROBE_GENERATION_GAP
        )
        probe_state = "evaluated"

    # Experts: cross-routing gate carries the liveness precondition; a
    # never-engaged expert bank is an infrastructure finding.
    liveness = (experts.get("liveness") or {}) if isinstance(experts, Mapping) else {}
    experts_measured = experts.get("state") == "measured"

    def _expert_metric(name: str, default: float) -> float:
        value = experts.get(name)
        return default if value is None else float(value)

    if not experts_measured:
        cross_gate, cross_state = True, "not_evaluated_experts_disabled"
        router_gate, router_state = True, "not_evaluated_experts_disabled"
    elif not bool(liveness.get("engaged")):
        cross_gate, cross_state = False, "experts_never_engaged"
        router_gate = bool(
            _expert_metric("router_delta", 1.0) <= ROUTER_DELTA_MARGIN
        )
        router_state = "measured"
    else:
        cross_gate = bool(
            _expert_metric("cross_routing_cost", 0.0) >= CROSS_ROUTING_MARGIN
        )
        cross_state = "measured"
        router_gate = bool(
            _expert_metric("router_delta", 1.0) <= ROUTER_DELTA_MARGIN
        )
        router_state = "measured"

    gates: Dict[str, Any] = {
        # --- notebook-supplied training/infrastructure gates.
        "training_complete": bool(training_complete),
        "zero_skipped_updates": bool(zero_skipped_updates),
        "warm_gate_passed": bool(warm_gate_passed),
        "infrastructure_ok": bool(infrastructure_ok),
        # --- channel gates (A3: paired 90% LCB > 0).
        "masked_leak_zero": bool(core_n > 0 and leak_rows == 0),
        "masked_causal_floor": bool(core_n > 0 and floor_accuracy <= MASKED_FLOOR_CEILING),
        "latent_channel_live": latent_channel_live,
        "beats_pause_compute": beats_pause,
        "unmasked_parity": unmasked_parity,
        "probe_generation_gap": probe_gate,
        # --- experts.
        "cross_routing_cost": cross_gate,
        "router_delta": router_gate,
    }
    gates["passed"] = all(bool(value) for value in gates.values())

    def _round(value: Any) -> Any:
        return None if value is None else float(value)

    eprocess = abstention.get("eprocess") or {}
    delong = abstention.get("delong_head_vs_logprob") or {}
    conformal = abstention.get("conformal") or {}
    aurocs = abstention.get("failure_auroc") or {}
    slot_contrast = _lcb("full_minus_slot_ablation")
    steer_contrast = _lcb("full_minus_steering")
    floor_contrast = _lcb("full_minus_floor")
    gates["gate_notes"] = {
        "ordering_rule": str(audit.get("ordering_rule", ORDERING_RULE)),
        # inter-arm ns.
        "masked_core_n": core_n,
        "unmasked_n": int(unmasked.get("n") or 0),
        "expert_rows_n": int(experts.get("n_rows") or 0) if experts_measured else 0,
        "abstention_n": int(abstention.get("n") or 0),
        "probe_heldout_n": int(probe.get("n_heldout_rows") or 0),
        # LCB values (gates + descriptive margins; the Section 4 point
        # margins are the means reported here, per A3).
        "latent_channel_live_lcb": _round((live_contrast or {}).get("lcb")),
        "latent_channel_live_mean_delta": _round((live_contrast or {}).get("mean")),
        "beats_pause_lcb": _round((pause_contrast or {}).get("lcb")),
        "beats_pause_mean_delta": _round((pause_contrast or {}).get("mean")),
        "slot_margin_descriptive_mean": _round((slot_contrast or {}).get("mean")),
        "slot_margin_descriptive_lcb": _round((slot_contrast or {}).get("lcb")),
        "steering_drop_descriptive_mean": _round((steer_contrast or {}).get("mean")),
        "floor_margin_descriptive_mean": _round((floor_contrast or {}).get("mean")),
        "masked_floor_accuracy": _round(floor_accuracy if core_n else None),
        "masked_full_accuracy": _round(full_accuracy if core_n else None),
        "unmasked_parity_state": parity_state,
        "unmasked_parity_gap": _round(parity_gap),
        # leak decomposition.
        "input_leak_rows": int((masked_core.get("leak") or {}).get("input_leak_rows", 0)),
        "scaffold_rows": int((masked_core.get("leak") or {}).get("scaffold_rows", 0)),
        # expert liveness.
        "expert_liveness_ratio": _round(liveness.get("ratio")),
        "expert_liveness_state": str(liveness.get("state", "not_measured")),
        "expert_liveness_denominator": liveness.get("denominator"),
        "cross_routing_state": cross_state,
        "cross_routing_cost": _round(experts.get("cross_routing_cost")) if experts_measured else None,
        "router_state": router_state,
        "router_delta_value": _round(experts.get("router_delta")) if experts_measured else None,
        "router_agreement": _round(experts.get("router_agreement")) if experts_measured else None,
        # probe precondition state.
        "probe_state": probe_state,
        "probe_accuracy": _round(probe_accuracy),
        "probe_chance": _round(probe.get("chance")),
        # abstention block (never in `passed`).
        "eprocess_wealth": _round(eprocess.get("wealth")),
        "eprocess_max_wealth": _round(eprocess.get("max_wealth")),
        "eprocess_rejects_05": bool(
            (eprocess.get("rejects_at_alpha") or {}).get("0.05", False)
        ),
        "eprocess_n_discordant": int(eprocess.get("n_discordant") or 0),
        "delong_futility_only": True,
        "delong_z": _round(delong.get("z")),
        "delong_p_one_sided": _round(delong.get("p_one_sided")),
        "failure_auroc_head": _round(aurocs.get("head")),
        "failure_auroc_logprob": _round(aurocs.get("logprob")),
        "failure_auroc_neg_semantic_entropy": _round(
            aurocs.get("neg_semantic_entropy")
        ),
        "abstention_degenerate": bool(abstention.get("degenerate", False)),
        "conformal_coverage": _round(conformal.get("out_of_fold_coverage_correct")),
        "conformal_publish_rate": _round(conformal.get("publish_rate")),
        "conformal_error": conformal.get("error"),
        # w/o-L1 descriptive diagnostic (A3/A6 item 6).
        "wo_l1_descriptive": (
            {
                "n": int(wo_l1.get("n") or 0),
                "ablated_minus_full_mean": _round(wo_l1.get("ablated_minus_full_mean")),
                "full_sha256": str(wo_l1.get("full_sha256", ""))[:16],
                "ablated_sha256": str(wo_l1.get("ablated_sha256", ""))[:16],
            }
            if wo_l1
            else None
        ),
    }
    return gates


def binding_rung(gates_by_seed: Mapping[Any, Mapping[str, Any]]) -> str:
    """Section 7 ladder over per-seed gate dicts, plus the independent
    abstention resolution.

    Returns ``"rung_<k>+abstention_<state>"``.  Rungs: 0 infrastructure
    failure (fix and rerun authorized); 1 warm gate failed on every seed
    (channel program terminal for this design class); 2 channel dead / no
    seed passes the channel gate set; 3 channel gate set passes on some but
    not all seeds (single-seed report, one replication session); 4 all
    seeds pass (rung 4 with fewer than two seeds is impossible — a lone
    passing seed is rung 3 because Section 7 requires both-seed
    replication).  Abstention resolves independently: the per-seed
    e-processes are independent, so the product of their FINAL wealths is
    itself an e-value; ``abstention_confirmed`` at product >= 20 (alpha
    0.05 by Ville), else ``abstention_open`` (anytime-valid — the martingale
    simply continues in the extension session; there is no "failed").
    """

    if not gates_by_seed:
        raise ValueError("binding_rung needs at least one seed's gates")
    seeds = [gates_by_seed[key] for key in sorted(gates_by_seed, key=str)]

    pooled_wealth = 1.0
    for gates in seeds:
        notes = gates.get("gate_notes") or {}
        wealth = notes.get("eprocess_wealth")
        pooled_wealth *= float(wealth) if wealth is not None else 1.0
    abstention = (
        "abstention_confirmed" if pooled_wealth >= 20.0 else "abstention_open"
    )

    if any(not bool(gates.get("infrastructure_ok", True)) for gates in seeds):
        rung = "rung_0"
    elif all(not bool(gates.get("warm_gate_passed", True)) for gates in seeds):
        rung = "rung_1"
    elif all(not bool(gates.get("latent_channel_live", False)) for gates in seeds):
        rung = "rung_2"
    else:
        # Rungs 3/4 are replication claims about the PREREGISTERED run; a
        # seed whose training aborted cannot support them no matter what
        # its abort-checkpoint audit shows (one-sided 90% LCBs give ~10%
        # per-contrast null FPR). Aborted-but-channel-live caps at rung 2
        # with the evidence preserved in the per-seed gate dicts.
        passes = [
            all(bool(gates.get(key, False)) for key in CHANNEL_GATE_KEYS)
            and bool(gates.get("training_complete", False))
            for gates in seeds
        ]
        if all(passes) and len(passes) >= 2:
            rung = "rung_4"
        elif any(passes):
            rung = "rung_3"
        else:
            rung = "rung_2"
    return "%s+%s" % (rung, abstention)


def training_verdict_from_metrics(metrics_path: Any) -> Dict[str, Any]:
    """Last ``v10_verdict`` record written by the trainer, or ``{}``."""

    import json
    from pathlib import Path

    path = Path(metrics_path)
    verdict: Dict[str, Any] = {}
    if not path.exists():
        return verdict
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "v10_verdict" in record:
            verdict = record["v10_verdict"] or {}
    return verdict


def run_seed_audit(
    output_dir: Any,
    *,
    data_dir: Any,
    seed: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Audit one seed's checkpoints and write ``v10-audit``/``v10-gates``.

    Factored out of the notebook so the two seeds can be audited
    CONCURRENTLY, one per T4, and so a crash in one seed's audit can no
    longer destroy the other's (sessions I-4 and I-5 both lost a finished
    seed to an exception raised while auditing the seed before it).
    """

    import gc
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from data import Reasoning9000Dataset
    from evaluate_checkpoint import load_hlwm

    output = Path(output_dir)
    data = Path(data_dir)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    verdict = training_verdict_from_metrics(output / "metrics.jsonl")
    full_path = output / "checkpoint-v10-full.pt"
    if not full_path.exists():
        candidates = sorted(output.glob("checkpoint-step-*.pt"))
        if not candidates:
            raise FileNotFoundError(f"no checkpoint for seed {seed} under {output}")
        full_path = candidates[-1]

    payload = torch.load(full_path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(
        payload["base_model"], revision=payload["base_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_hlwm(payload, device)
    model.eval()
    rows = Reasoning9000Dataset(data / "master" / "test.jsonl", num_lanes=1).rows
    audit = v10_audit(
        model, rows, tokenizer, model.config, seed=int(seed), device=device
    )
    audit["checkpoint"] = str(full_path)

    wo_l1: Optional[Dict[str, Any]] = None
    branch_path = output / "checkpoint-v10-wo-l1-branch.pt"
    if branch_path.exists() and verdict.get("full_checkpoint_sha256"):
        branch_payload = torch.load(branch_path, map_location="cpu", weights_only=False)
        model_ablated = load_hlwm(branch_payload, device)
        model_ablated.eval()
        try:
            wo_l1 = wo_l1_delta(
                model,
                model_ablated,
                rows,
                tokenizer,
                expected_full_sha256=verdict.get("full_state_sha256"),
                seed=int(seed),
                device=device,
            )
        except Exception as error:  # descriptive diagnostic: never fatal
            wo_l1 = {"error": str(error)}
        del model_ablated, branch_payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gates = v10_gate_dict(
        audit,
        training_complete=not verdict.get("aborted"),
        zero_skipped_updates=True,
        warm_gate_passed=bool((verdict.get("warm_gate") or {}).get("passed", False)),
        infrastructure_ok=True,
        wo_l1=wo_l1,
    )
    (output / "v10-audit.json").write_text(
        json.dumps(audit, indent=1, sort_keys=True, default=str)
    )
    (output / "v10-gates.json").write_text(
        json.dumps(gates, indent=1, sort_keys=True, default=str)
    )
    del model, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gates


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit one Version 10.0 seed.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    gates = run_seed_audit(
        args.output_dir, data_dir=args.data_dir, seed=args.seed
    )
    print(
        "seed",
        args.seed,
        "gates:",
        {name: value for name, value in gates.items() if isinstance(value, bool)},
    )


__all__ = [
    "CHANNEL_GATE_KEYS",
    "EXPERT_SWAP",
    "FAMILY_ORDER",
    "ORDERING_RULE",
    "PreparedRow",
    "Z_90",
    "abstention_block",
    "arm_cross_routing",
    "arm_floor",
    "arm_full",
    "arm_pause",
    "arm_router_routed",
    "arm_shuffled",
    "arm_slot_ablation",
    "arm_steering",
    "binding_rung",
    "causal_mean_logprob",
    "discrete_semantic_entropy",
    "expert_arm_selection",
    "expert_liveness",
    "gold_answer_forced_ce",
    "gold_forced_ce_block",
    "grade_arm",
    "grade_decode",
    "masked_core_selection",
    "masked_leak_scan",
    "paired_lower_confidence_bound",
    "prepare_rows",
    "probe_digit_label",
    "route_for_family_index",
    "run_seed_audit",
    "sorted_by_anchor",
    "state_dict_sha256",
    "student_channel_mean_logprob",
    "train_premise_probe",
    "training_verdict_from_metrics",
    "v10_audit",
    "v10_audit_rows",
    "v10_gate_dict",
    "wo_l1_delta",
]


if __name__ == "__main__":
    main()
