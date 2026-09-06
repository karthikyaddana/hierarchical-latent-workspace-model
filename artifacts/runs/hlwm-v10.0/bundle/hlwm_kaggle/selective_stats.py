"""Preregistered selective-prediction statistics for HLWM Version 10.0.

Version 9.0's confirmatory abstention comparison (delta-AUGRC under a paired
bootstrap against a 0.80 win-fraction bar) was underpowered at n=288 rows per
seed: win fractions landed at 0.746/0.792 and the fixed-n bootstrap offered
no valid way to add data.  Version 10.0's preregistered replacement, all of
it implemented here so the notebook and the audit share one code path:

* the contrast is reduced from delta-AUGRC to a paired failure-AUROC
  difference — for a fixed row set with fixed correctness labels the
  empirical AUGRC is an exact decreasing affine function of failure AUROC
  (Traub et al., arXiv:2407.01032, Eq. 8: AUGRC = acc * (1 - acc) *
  (1 - AUROC_f)), so the reduction changes the test statistic, not the
  estimand;
* one-sided paired DeLong per seed, pooled across seeds with fixed-effect
  Stouffer, serves as the FUTILITY look only;
* an anytime-valid e-process over rank bets is THE single confirmatory
  analysis: it remains valid under optional stopping and under resuming the
  same martingale in a later session, which is exactly what the fixed-n
  bootstrap could not survive.

``augrc`` and ``partial_augrc`` are ported verbatim from
``evaluate_checkpoint.py`` so Version 10.0 numbers stay comparable to the
Version 8.0/9.0 audit records.  Everything here is pure Python plus torch
CPU tensors and is deterministic given the explicit seed arguments.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _normal_sf(z: float) -> float:
    """Standard normal survival function via erfc (stable in the far tail)."""

    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _midranks(values: Sequence[float]) -> List[float]:
    """One-based midranks (ties averaged) aligned to the input order.

    Midranks are the tie convention behind both the 0.5-credit AUROC and the
    fast DeLong structural components (Sun & Xu 2014), so a single helper
    keeps the point estimate and its variance on the same tie rule.
    """

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        tail = position
        while (
            tail + 1 < len(order)
            and values[order[tail + 1]] == values[order[position]]
        ):
            tail += 1
        midrank = (position + tail) / 2.0 + 1.0
        for index in order[position : tail + 1]:
            ranks[index] = midrank
        position = tail + 1
    return ranks


def failure_auroc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """AUROC of the confidence score as a failure detector.

    The probability (ties earn 0.5 credit) that a uniformly drawn correct row
    outscores a uniformly drawn incorrect row — the AUROC_f of Traub et al.
    For a fixed row set with fixed labels the empirical AUGRC below is an
    exact decreasing affine function of this quantity, which is why Version
    10.0 tests it in place of delta-AUGRC.  A one-class slice raises
    ``ValueError``: returning 0.5 silently would let a degenerate audit
    stratum pass a gate.
    """

    if len(scores) != len(correct):
        raise ValueError("scores and correct must have equal length")
    positives = [float(score) for score, label in zip(scores, correct) if label]
    negatives = [float(score) for score, label in zip(scores, correct) if not label]
    if not positives or not negatives:
        raise ValueError(
            "failure AUROC needs at least one correct and one incorrect row"
        )
    combined_ranks = _midranks(positives + negatives)
    positive_count = len(positives)
    rank_sum = sum(combined_ranks[:positive_count])
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * len(negatives)
    )


def _delong_components(
    positives: Sequence[float], negatives: Sequence[float]
) -> Tuple[float, List[float], List[float]]:
    """AUROC plus the DeLong structural components V10 / V01 via midranks."""

    positive_count = len(positives)
    negative_count = len(negatives)
    combined = _midranks(list(positives) + list(negatives))
    within_positive = _midranks(positives)
    within_negative = _midranks(negatives)
    v10 = [
        (combined[index] - within_positive[index]) / negative_count
        for index in range(positive_count)
    ]
    v01 = [
        1.0 - (combined[positive_count + index] - within_negative[index]) / positive_count
        for index in range(negative_count)
    ]
    auroc = (
        sum(combined[:positive_count])
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return auroc, v10, v01


def _covariance(u: Sequence[float], v: Sequence[float]) -> float:
    """Unbiased sample covariance; 0.0 when fewer than two observations."""

    count = len(u)
    if count < 2:
        return 0.0
    mean_u = sum(u) / count
    mean_v = sum(v) / count
    return sum(
        (u_value - mean_u) * (v_value - mean_v) for u_value, v_value in zip(u, v)
    ) / (count - 1)


def delong_paired_one_sided(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    correct: Sequence[bool],
) -> Dict[str, Any]:
    """One-sided paired DeLong test of H1: failure-AUROC(A) > failure-AUROC(B).

    Fast DeLong via midranks (Sun & Xu 2014): one structural component per
    correct row (V10) and per incorrect row (V01) for each score, empirical
    covariances between the paired component vectors, and
    Var(delta) = (s10_aa + s10_bb - 2 s10_ab) / n_pos
               + (s01_aa + s01_bb - 2 s01_ab) / n_neg.
    Both scores MUST rank the same rows under the same labels — the shared
    components are what buy the paired variance reduction over two
    independent AUROC estimates.  Preregistered role: futility look only;
    the confirmatory analysis is ``eprocess_rank_bets``.  A degenerate
    variance (identical scores, or a class too small for a covariance)
    returns z=0, p=0.5 and ``degenerate=True`` instead of dividing by zero
    mid-audit.
    """

    if not len(scores_a) == len(scores_b) == len(correct):
        raise ValueError("scores_a, scores_b and correct must have equal length")
    positives_a = [float(s) for s, label in zip(scores_a, correct) if label]
    negatives_a = [float(s) for s, label in zip(scores_a, correct) if not label]
    positives_b = [float(s) for s, label in zip(scores_b, correct) if label]
    negatives_b = [float(s) for s, label in zip(scores_b, correct) if not label]
    if not positives_a or not negatives_a:
        raise ValueError(
            "paired DeLong needs at least one correct and one incorrect row"
        )
    auroc_a, v10_a, v01_a = _delong_components(positives_a, negatives_a)
    auroc_b, v10_b, v01_b = _delong_components(positives_b, negatives_b)
    positive_count = len(positives_a)
    negative_count = len(negatives_a)
    variance = (
        _covariance(v10_a, v10_a)
        + _covariance(v10_b, v10_b)
        - 2.0 * _covariance(v10_a, v10_b)
    ) / positive_count + (
        _covariance(v01_a, v01_a)
        + _covariance(v01_b, v01_b)
        - 2.0 * _covariance(v01_a, v01_b)
    ) / negative_count
    delta = auroc_a - auroc_b
    degenerate = not (math.isfinite(variance) and variance > 1.0e-16)
    if degenerate:
        z = 0.0
        p_one_sided = 0.5
    else:
        z = delta / math.sqrt(variance)
        p_one_sided = _normal_sf(z)
    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "delta": float(delta),
        "variance": float(variance),
        "z": float(z),
        "p_one_sided": float(p_one_sided),
        "degenerate": degenerate,
    }


def stouffer_pool(z_values: Sequence[float]) -> Dict[str, float]:
    """Fixed-effect, equal-weight Stouffer pooling of one-sided z scores.

    The seeds are independent replications of one preregistered contrast, so
    the pooled statistic is sum(z_i) / sqrt(k) ~ N(0, 1) under the shared
    null.  Equal weights are deliberate: per-seed effective sample sizes are
    close enough that fitting weights would be a researcher degree of
    freedom, not a power gain.
    """

    if len(z_values) == 0:
        raise ValueError("cannot pool an empty list of z values")
    z_pool = sum(float(z) for z in z_values) / math.sqrt(len(z_values))
    return {"z_pool": float(z_pool), "p_one_sided": _normal_sf(z_pool)}


def eprocess_rank_bets(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    correct: Sequence[bool],
    *,
    lam: float = 0.5,
    resume_wealth: float = 1.0,
    resume_max_wealth: Optional[float] = None,
    resume_pending: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Anytime-valid e-process for "A ranks failures below successes better".

    Rows stream in the given (preregistered) order.  Every incorrect row
    joins a FIFO queue; the next correct row completes a (wrong, right) pair
    with the oldest queued incorrect row, and both rows are consumed —
    row-disjoint pairs keep the bets conditionally independent, which the
    supermartingale argument requires.  On each pair,
    a_correct = scores_a[right] > scores_a[wrong] and likewise for B;
    concordant pairs carry no information about the CONTRAST and are
    discarded.  On a discordant pair the null (A and B exchangeable given
    correctness) makes "A is the correct ranker" a fair coin, so the wealth
    update W *= 1 + lam * (2 x - 1) has conditional expectation exactly W:
    wealth is a nonnegative martingale with W_0 = 1 and Ville's inequality
    gives P(sup_t W_t >= 1/alpha) <= alpha.  Rejection therefore keys off
    the running maximum — an anytime-valid test may stop at first crossing.

    ``lam`` must stay inside (0, 1) so wealth remains positive.  Resuming:
    pass the previous session's ``wealth``, ``max_wealth`` and
    ``pending_wrongs`` back in as ``resume_wealth`` / ``resume_max_wealth``
    / ``resume_pending`` and a split run reproduces the unsplit run exactly;
    continuation is a predictable decision, so anytime validity is preserved
    across sessions.
    """

    if not len(scores_a) == len(scores_b) == len(correct):
        raise ValueError("scores_a, scores_b and correct must have equal length")
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie in (0, 1) so wealth stays positive")
    if resume_wealth <= 0.0:
        raise ValueError("resume_wealth must be positive")
    wealth = float(resume_wealth)
    max_wealth = (
        float(resume_max_wealth) if resume_max_wealth is not None else wealth
    )
    max_wealth = max(max_wealth, wealth)
    pending: deque[Tuple[float, float]] = deque(
        (float(a), float(b)) for a, b in (resume_pending or [])
    )
    n_pairs = 0
    n_discordant = 0
    for a_value, b_value, is_correct in zip(scores_a, scores_b, correct):
        if not is_correct:
            pending.append((float(a_value), float(b_value)))
            continue
        if not pending:
            continue
        wrong_a, wrong_b = pending.popleft()
        n_pairs += 1
        a_correct = float(a_value) > wrong_a
        b_correct = float(b_value) > wrong_b
        if a_correct == b_correct:
            continue
        n_discordant += 1
        wealth *= 1.0 + lam * (1.0 if a_correct else -1.0)
        if wealth > max_wealth:
            max_wealth = wealth
    return {
        "wealth": wealth,
        "log_wealth": math.log(wealth),
        "max_wealth": max_wealth,
        "n_pairs": n_pairs,
        "n_discordant": n_discordant,
        "pending_wrongs": list(pending),
        "rejects_at_alpha": {
            "0.05": max_wealth >= 20.0,
            "0.01": max_wealth >= 100.0,
        },
    }


def augrc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the generalized risk-coverage curve (Traub et al. 2024).

    Ported verbatim from ``evaluate_checkpoint.augrc`` (the Version 8.0/9.0
    audit surface) so Version 10.0 numbers stay comparable across studies.
    Generalized risk at coverage c is P(error AND accepted); averaging it
    over the coverage sweep is robust to the few-high-confidence-failures
    distortion that affects plain AURC.  Lower is better.
    """

    count = len(scores)
    if count == 0:
        return 0.0
    order = sorted(range(count), key=lambda i: scores[i], reverse=True)
    area = 0.0
    accepted_errors = 0
    for accepted, index in enumerate(order, start=1):
        accepted_errors += int(not correct[index])
        area += accepted_errors / count
    return area / count


def partial_augrc(
    scores: Sequence[float],
    correct: Sequence[bool],
    low: float = 0.05,
    high: float = 0.50,
) -> float:
    """AUGRC restricted to a coverage band (Version 9.0 headline statistic).

    Ported verbatim from ``evaluate_checkpoint.partial_augrc``.  Version
    8.0's full-curve AUGRC on a 59%-easy pool measured the baseline's
    saturation region, not abstention quality; the deployable region is the
    low-coverage band where abstention actually operates.  Lower is better.
    """

    count = len(scores)
    if count == 0:
        return 0.0
    order = sorted(range(count), key=lambda i: scores[i], reverse=True)
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


def _split_conformal_threshold(
    calibration_scores: Sequence[float],
    *,
    target_coverage: float,
    calibration_floor: int,
    seed: int,
) -> Dict[str, Any]:
    """Split-conformal order-statistic threshold, refusing to clamp.

    Same convention as ``train_kaggle.conformal_publish_threshold`` — tau is
    the k-th smallest calibration score with
    k = floor((n + 1) * (1 - target_coverage)), guaranteeing expected
    coverage (n + 1 - k) / (n + 1) >= target by exchangeability, with the
    preregistered seeded 1e-7 jitter when more than 20% of scores tie at the
    chosen order statistic.  Unlike the training-time helper this raises
    ``ValueError`` instead of clamping k into range: a clamp would silently
    void the coverage guarantee, and a previous session's silent-guard
    pattern is exactly what crashed rung 0.
    """

    scores = [float(value) for value in calibration_scores]
    count = len(scores)
    if count < calibration_floor:
        raise ValueError(
            "conformal calibration needs at least %d correct rows, found %d"
            % (calibration_floor, count)
        )
    k = int(math.floor((count + 1) * (1.0 - target_coverage)))
    if k < 1 or k > count:
        raise ValueError(
            "order statistic k=%d is outside [1, %d] at target coverage %.4f"
            % (k, count, target_coverage)
        )
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
    return {
        "threshold": float(threshold),
        "calibration_count": count,
        "order_statistic_k": k,
        "guaranteed_expected_coverage": (count + 1 - k) / (count + 1),
        "tie_fraction": tie_fraction,
        "tie_jitter_applied": jittered,
    }


def nested_cross_conformal(
    anchor_ids: Sequence[Any],
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_folds: int = 5,
    target_coverage: float,
    seed: int,
) -> Dict[str, Any]:
    """Anchor-grouped nested cross-conformal publish thresholds.

    Rows of one anchor are near-duplicate candidates of the same prompt, so
    the exchangeability unit is the anchor, not the row: outer folds
    partition ANCHORS (deterministic seeded permutation, round-robin
    assignment), and the code asserts that no anchor's rows straddle a fold
    boundary.  For each outer fold the threshold is fitted on the OTHER
    folds' CORRECT rows via the split-conformal convention of
    ``train_kaggle.conformal_publish_threshold`` (k-th smallest score,
    k = floor((n + 1) * (1 - target_coverage)), seeded tie jitter), and
    every row receives an out-of-fold publish decision
    ``score >= tau_fold``.  Calibrating on correct rows makes the
    (n + 1 - k) / (n + 1) >= target guarantee apply to the published
    fraction of held-out CORRECT rows — the recall-style coverage the
    Version 10.0 publish gate audits.

    Coverage floor: every fold's calibration count n must satisfy
    n >= ceil(1 / (1 - target_coverage)), the smallest n for which
    k = floor((n + 1) * (1 - target_coverage)) >= 1 — i.e. for which the
    order statistic exists without the min/max clamp that would silently
    void the guarantee.  Violations raise ``ValueError`` (as does the
    defensive k > n check) rather than degrading quietly.
    """

    if not len(anchor_ids) == len(scores) == len(labels):
        raise ValueError("anchor_ids, scores and labels must have equal length")
    if n_folds < 2:
        raise ValueError("nested cross-conformal needs at least two outer folds")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be inside (0, 1)")
    unique_anchors = list(dict.fromkeys(anchor_ids))
    if len(unique_anchors) < n_folds:
        raise ValueError(
            "need at least %d unique anchors for %d folds, found %d"
            % (n_folds, n_folds, len(unique_anchors))
        )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(unique_anchors), generator=generator).tolist()
    anchor_folds = {
        unique_anchors[anchor_index]: position % n_folds
        for position, anchor_index in enumerate(permutation)
    }
    row_folds = [anchor_folds[anchor] for anchor in anchor_ids]
    observed_folds: Dict[Any, int] = {}
    for anchor, fold in zip(anchor_ids, row_folds):
        previous = observed_folds.setdefault(anchor, fold)
        assert previous == fold, "anchor %r straddles an outer fold boundary" % (
            anchor,
        )

    calibration_floor = int(math.ceil(1.0 / (1.0 - target_coverage)))
    fold_thresholds: List[float] = []
    fold_calibration_counts: List[int] = []
    fold_order_statistics: List[int] = []
    fold_guaranteed_coverage: List[float] = []
    fold_tie_jittered: List[bool] = []
    for fold in range(n_folds):
        calibration_scores = [
            float(score)
            for score, label, row_fold in zip(scores, labels, row_folds)
            if label and row_fold != fold
        ]
        fitted = _split_conformal_threshold(
            calibration_scores,
            target_coverage=target_coverage,
            calibration_floor=calibration_floor,
            seed=seed + 1 + fold,
        )
        fold_thresholds.append(fitted["threshold"])
        fold_calibration_counts.append(fitted["calibration_count"])
        fold_order_statistics.append(fitted["order_statistic_k"])
        fold_guaranteed_coverage.append(fitted["guaranteed_expected_coverage"])
        fold_tie_jittered.append(fitted["tie_jitter_applied"])

    row_published = [
        float(score) >= fold_thresholds[fold]
        for score, fold in zip(scores, row_folds)
    ]
    correct_total = sum(1 for label in labels if label)
    correct_published = sum(
        1 for label, published in zip(labels, row_published) if label and published
    )
    return {
        "n_folds": n_folds,
        "target_coverage": float(target_coverage),
        "calibration_floor": calibration_floor,
        "anchor_folds": anchor_folds,
        "row_folds": row_folds,
        "row_published": row_published,
        "fold_thresholds": fold_thresholds,
        "fold_calibration_counts": fold_calibration_counts,
        "fold_order_statistics": fold_order_statistics,
        "fold_guaranteed_coverage": fold_guaranteed_coverage,
        "fold_tie_jittered": fold_tie_jittered,
        "out_of_fold_coverage_correct": (
            correct_published / correct_total if correct_total else 0.0
        ),
        "publish_rate": (
            sum(row_published) / len(row_published) if row_published else 0.0
        ),
    }
