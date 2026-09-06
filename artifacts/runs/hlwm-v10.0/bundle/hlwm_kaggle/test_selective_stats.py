"""Tests for the Version 10.0 selective-prediction statistics module.

Every stochastic check runs on an explicit torch.Generator seed, so the suite
is deterministic: calibration bounds below are Monte Carlo tolerances around
the preregistered guarantees (DeLong null uniformity, Ville's inequality for
the e-process, the split-conformal coverage floor), not flaky thresholds.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

try:
    from hlwm_kaggle.selective_stats import (
        augrc,
        delong_paired_one_sided,
        eprocess_rank_bets,
        failure_auroc,
        nested_cross_conformal,
        partial_augrc,
        stouffer_pool,
    )
except ImportError:  # Direct execution inside the Kaggle dataset directory.
    from selective_stats import (
        augrc,
        delong_paired_one_sided,
        eprocess_rank_bets,
        failure_auroc,
        nested_cross_conformal,
        partial_augrc,
        stouffer_pool,
    )


def test_failure_auroc_hand_computed_with_tie():
    scores = [0.9, 0.8, 0.8, 0.4, 0.3, 0.2]
    correct = [True, True, False, True, False, False]
    # Positives {0.9, 0.8, 0.4} vs negatives {0.8, 0.3, 0.2}: 7 wins, one
    # 0.8-vs-0.8 tie at half credit, one loss -> 7.5 / 9.
    assert abs(failure_auroc(scores, correct) - 7.5 / 9.0) < 1.0e-12


def test_failure_auroc_rejects_one_class_slices():
    with pytest.raises(ValueError):
        failure_auroc([0.1, 0.2], [True, True])
    with pytest.raises(ValueError):
        failure_auroc([0.1, 0.2], [False, False])
    with pytest.raises(ValueError):
        failure_auroc([0.1, 0.2], [True])


def test_delong_paired_detects_genuine_improvement():
    generator = torch.Generator().manual_seed(101)
    labels = torch.rand(400, generator=generator) > 0.5
    signal = labels.float()
    scores_a = signal + 0.3 * torch.randn(400, generator=generator)
    scores_b = signal + 1.0 * torch.randn(400, generator=generator)
    result = delong_paired_one_sided(
        scores_a.tolist(), scores_b.tolist(), labels.tolist()
    )
    assert result["auroc_a"] > result["auroc_b"]
    assert result["delta"] > 0.1
    assert result["z"] > 2.0
    assert result["p_one_sided"] < 0.01
    assert not result["degenerate"]


def test_delong_paired_null_is_calibrated():
    labels = [True] * 100 + [False] * 100
    p_values = []
    z_values = []
    for resample in range(200):
        generator = torch.Generator().manual_seed(1000 + resample)
        scores_a = torch.randn(200, generator=generator)
        scores_b = torch.randn(200, generator=generator)
        result = delong_paired_one_sided(
            scores_a.tolist(), scores_b.tolist(), labels
        )
        p_values.append(result["p_one_sided"])
        z_values.append(result["z"])
    mean_z = sum(z_values) / len(z_values)
    reject_rate = sum(1 for p in p_values if p <= 0.05) / len(p_values)
    assert abs(mean_z) < 0.2
    assert 0.02 <= reject_rate <= 0.09


def test_delong_paired_degenerate_variance_flagged():
    scores = [0.9, 0.7, 0.4, 0.2]
    correct = [True, True, False, False]
    result = delong_paired_one_sided(scores, scores, correct)
    assert result["degenerate"]
    assert result["z"] == 0.0
    assert result["p_one_sided"] == 0.5


def test_stouffer_pool_hand_check():
    result = stouffer_pool([1.0, 2.0])
    expected_z = 3.0 / math.sqrt(2.0)
    assert abs(result["z_pool"] - expected_z) < 1.0e-12
    expected_p = 0.5 * math.erfc(expected_z / math.sqrt(2.0))
    assert abs(result["p_one_sided"] - expected_p) < 1.0e-12
    with pytest.raises(ValueError):
        stouffer_pool([])


def _null_stream(seed: int, count: int):
    generator = torch.Generator().manual_seed(seed)
    correct = (torch.rand(count, generator=generator) > 0.5).tolist()
    scores_a = torch.randn(count, generator=generator).tolist()
    scores_b = torch.randn(count, generator=generator).tolist()
    return scores_a, scores_b, correct


def _informative_stream(seed: int, count: int):
    generator = torch.Generator().manual_seed(seed)
    correct = (torch.rand(count, generator=generator) > 0.5).tolist()
    signal = torch.tensor([1.0 if flag else 0.0 for flag in correct])
    scores_a = (signal + 0.5 * torch.randn(count, generator=generator)).tolist()
    scores_b = torch.randn(count, generator=generator).tolist()
    return scores_a, scores_b, correct


def test_eprocess_validity_ville_bound():
    crossings = 0
    for stream in range(500):
        scores_a, scores_b, correct = _null_stream(2000 + stream, 300)
        result = eprocess_rank_bets(scores_a, scores_b, correct)
        crossings += int(result["max_wealth"] >= 20.0)
    assert crossings / 500 <= 0.05 + 0.02


def test_eprocess_power_grows_with_n():
    medians = []
    for count in (100, 200, 400):
        log_wealths = []
        for stream in range(50):
            scores_a, scores_b, correct = _informative_stream(
                5000 + stream, count
            )
            log_wealths.append(
                eprocess_rank_bets(scores_a, scores_b, correct)["log_wealth"]
            )
        medians.append(statistics.median(log_wealths))
    assert medians[0] < medians[1] < medians[2]
    assert medians[2] > 0.0


def test_eprocess_resume_is_exact():
    scores_a, scores_b, correct = _informative_stream(9001, 300)
    full = eprocess_rank_bets(scores_a, scores_b, correct)
    first = eprocess_rank_bets(scores_a[:150], scores_b[:150], correct[:150])
    second = eprocess_rank_bets(
        scores_a[150:],
        scores_b[150:],
        correct[150:],
        resume_wealth=first["wealth"],
        resume_max_wealth=first["max_wealth"],
        resume_pending=first["pending_wrongs"],
    )
    assert second["wealth"] == full["wealth"]
    assert second["log_wealth"] == full["log_wealth"]
    assert second["max_wealth"] == full["max_wealth"]
    assert first["n_pairs"] + second["n_pairs"] == full["n_pairs"]
    assert first["n_discordant"] + second["n_discordant"] == full["n_discordant"]
    assert second["pending_wrongs"] == full["pending_wrongs"]
    assert second["rejects_at_alpha"] == full["rejects_at_alpha"]


def test_eprocess_rejects_bad_arguments():
    with pytest.raises(ValueError):
        eprocess_rank_bets([0.1], [0.2], [True], lam=1.0)
    with pytest.raises(ValueError):
        eprocess_rank_bets([0.1], [0.2], [True], resume_wealth=0.0)
    with pytest.raises(ValueError):
        eprocess_rank_bets([0.1, 0.2], [0.2], [True, False])


def _conformal_rows(seed: int, anchor_count: int, rows_per_anchor: int):
    generator = torch.Generator().manual_seed(seed)
    anchor_ids = []
    scores = []
    labels = []
    for anchor in range(anchor_count):
        for _ in range(rows_per_anchor):
            anchor_ids.append("anchor-%04d" % anchor)
            label = bool(torch.rand(1, generator=generator) > 0.5)
            uniform = float(torch.rand(1, generator=generator))
            scores.append(0.5 + 0.5 * uniform if label else 0.5 * uniform)
            labels.append(label)
    return anchor_ids, scores, labels


def test_nested_cross_conformal_hits_target_coverage():
    anchor_ids, scores, labels = _conformal_rows(31, 500, 2)
    result = nested_cross_conformal(
        anchor_ids, scores, labels, n_folds=5, target_coverage=0.8, seed=7
    )
    assert abs(result["out_of_fold_coverage_correct"] - 0.8) <= 0.08
    assert len(result["fold_thresholds"]) == 5
    assert all(count >= 5 for count in result["fold_calibration_counts"])
    assert all(
        guaranteed >= 0.8 for guaranteed in result["fold_guaranteed_coverage"]
    )


def test_nested_cross_conformal_anchors_never_straddle_folds():
    anchor_ids, scores, labels = _conformal_rows(53, 60, 3)
    result = nested_cross_conformal(
        anchor_ids, scores, labels, n_folds=4, target_coverage=0.7, seed=11
    )
    fold_by_anchor = {}
    for anchor, fold in zip(anchor_ids, result["row_folds"]):
        fold_by_anchor.setdefault(anchor, set()).add(fold)
    assert all(len(folds) == 1 for folds in fold_by_anchor.values())
    assert all(
        result["anchor_folds"][anchor] == next(iter(folds))
        for anchor, folds in fold_by_anchor.items()
    )


def test_nested_cross_conformal_raises_below_calibration_floor():
    # target 0.9 -> floor ceil(1 / 0.1) = 10 correct calibration rows per
    # fold; ten single-row all-correct anchors leave only eight per fold.
    anchor_ids = ["a-%d" % index for index in range(10)]
    scores = [0.5 + 0.04 * index for index in range(10)]
    labels = [True] * 10
    with pytest.raises(ValueError):
        nested_cross_conformal(
            anchor_ids, scores, labels, n_folds=5, target_coverage=0.9, seed=3
        )


def test_nested_cross_conformal_rejects_bad_partitions():
    anchor_ids, scores, labels = _conformal_rows(71, 3, 2)
    with pytest.raises(ValueError):
        nested_cross_conformal(
            anchor_ids, scores, labels, n_folds=5, target_coverage=0.5, seed=1
        )
    with pytest.raises(ValueError):
        nested_cross_conformal(
            anchor_ids, scores, labels, n_folds=2, target_coverage=1.0, seed=1
        )


def test_traub_identity_sign_agreement():
    # AUGRC = acc * (1 - acc) * (1 - AUROC_f) holds exactly only at fixed
    # accuracy; on a shared row set with shared labels the accuracy factor
    # cancels, so sign agreement of the deltas is the testable invariant.
    for draw in range(20):
        generator = torch.Generator().manual_seed(400 + draw)
        correct = (torch.rand(300, generator=generator) > 0.4).tolist()
        signal = torch.tensor([1.0 if flag else 0.0 for flag in correct])
        scores_a = (signal + 0.5 * torch.randn(300, generator=generator)).tolist()
        scores_b = (signal + 1.5 * torch.randn(300, generator=generator)).tolist()
        delta_augrc = augrc(scores_a, correct) - augrc(scores_b, correct)
        delta_auroc = failure_auroc(scores_b, correct) - failure_auroc(
            scores_a, correct
        )
        assert delta_augrc != 0.0 and delta_auroc != 0.0
        assert math.copysign(1.0, delta_augrc) == math.copysign(1.0, delta_auroc)


def test_partial_augrc_matches_full_curve_on_full_band():
    generator = torch.Generator().manual_seed(77)
    correct = (torch.rand(64, generator=generator) > 0.5).tolist()
    scores = torch.randn(64, generator=generator).tolist()
    # Averaging the banded sum over its own weight recovers the full AUGRC
    # when the band spans every coverage level.
    assert abs(partial_augrc(scores, correct, 0.0, 1.0) - augrc(scores, correct)) < 1.0e-12
    assert partial_augrc([], [], 0.05, 0.5) == 0.0
