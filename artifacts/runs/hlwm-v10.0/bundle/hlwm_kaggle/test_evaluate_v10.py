"""Tests for the Version 10.0 audit arms, gates, verdict and ladder."""

from __future__ import annotations

import copy
import json
import math
import zlib

import torch

# Pin this file's own directory ahead of anything else on sys.path. The old
# preamble preferred ``experiments.kaggle_hlwm`` -- the ARCHIVED pre-hotfix
# Version 6.0 tree -- whenever the repo root was importable, which silently
# tested the wrong copy of the code.
import sys
from pathlib import Path as _Path

_BUNDLE = str(_Path(__file__).resolve().parent)
if sys.path[:1] != [_BUNDLE]:
    sys.path.insert(0, _BUNDLE)

from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration
import evaluate_v10 as ev


# ----------------------------------------------------------------------
# Fixtures: deterministic offline tokenizer, tiny V10 models, synthetic rows.


class TinyTokenizer:
    """Deterministic word tokenizer into the 41-token tiny vocabulary."""

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [
            3 + (zlib.crc32(word.encode("utf-8")) % 38)
            for word in str(text).split()
        ]
        return ids or [3]

    def decode(self, ids, skip_special_tokens=True):
        values = [int(value) for value in torch.as_tensor(ids).flatten().tolist()]
        if skip_special_tokens:
            values = [value for value in values if value > 2]
        return " ".join(str(value) for value in values)


def _config(**overrides):
    values = dict(
        latent_thoughts=3,
        kv_prefix_slots=4,
        kv_prefix_rank=8,
        response_cue_ids=(5, 7),
        lora_rank=4,
        lora_tail_layers=2,
        mlp_expert_count=2,
        mlp_expert_rank=4,
        lora_dropout=0.0,
    )
    values.update(overrides)
    return HLWMConfig.tiny(**values)


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    return HLWMForConditionalGeneration(_config(**overrides)).eval()


_SPECS = {
    "numeric": {"type": "numeric", "expected": 7},
    "unit": {"type": "unit", "expected": 7, "unit": "kg"},
    "ordering": {"type": "ordering", "expected": [1, 2]},
    "abstention": {"type": "abstention"},
}


def _row(index, family, masked=True):
    literal = "%d8%02d" % (1 + index % 5, index)
    prompt = "compute value %s plus tax now please" % literal
    masked_prompt = prompt.replace(literal, "[withheld]")
    return {
        "episode_id": "row-%03d" % index,
        "family": family,
        "masked": bool(masked),
        "public_prompt": prompt,
        "masked_prompt": masked_prompt if masked else prompt,
        "public_target": "the answer is %d" % (7 + index),
        "answer_spec": dict(_SPECS[family]),
        "withheld_literals": [literal] if masked else [],
    }


def _core_rows():
    # Two families, two rows each: shuffling is a swap inside each family.
    return [
        _row(0, "numeric"),
        _row(1, "numeric"),
        _row(2, "unit"),
        _row(3, "unit"),
    ]


# ----------------------------------------------------------------------
# Paired-LCB math.


def test_paired_lcb_matches_hand_computation():
    deltas = [1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0]
    n = len(deltas)
    delta_mean = sum(deltas) / n
    sd = math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / (n - 1))
    expected = delta_mean - ev.Z_90 * sd / math.sqrt(n)
    result = ev.paired_lower_confidence_bound(deltas)
    assert not result["degenerate"]
    assert abs(result["mean"] - delta_mean) < 1e-12
    assert abs(result["sd"] - sd) < 1e-12
    assert abs(result["lcb"] - expected) < 1e-12

    all_agree = ev.paired_lower_confidence_bound([1.0] * 5)
    assert all_agree["sd"] == 0.0 and all_agree["lcb"] == 1.0
    single = ev.paired_lower_confidence_bound([1.0])
    assert single["degenerate"] and single["lcb"] == 1.0
    try:
        ev.paired_lower_confidence_bound([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty delta vector must raise")


# ----------------------------------------------------------------------
# Preregistered ordering rule.


def test_masked_core_ordering_rule_is_lexicographic_anchor_order():
    rows = [
        _row(3, "unit"),
        _row(0, "numeric"),
        _row(5, "numeric", masked=False),
        _row(2, "unit"),
        _row(1, "numeric"),
    ]
    core = ev.masked_core_selection(rows, 3)
    assert [r["episode_id"] for r in core] == ["row-000", "row-001", "row-002"]
    # Unmasked rows never enter; the rule ignores input order entirely.
    core_all = ev.masked_core_selection(list(reversed(rows)), 99)
    assert [r["episode_id"] for r in core_all] == [
        "row-000",
        "row-001",
        "row-002",
        "row-003",
    ]
    duplicated = rows + [dict(rows[0])]
    try:
        ev.masked_core_selection(duplicated, 3)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate anchor ids must raise")


# ----------------------------------------------------------------------
# Arms: shapes, invariances, controls.


def test_arms_return_per_row_decodes_and_are_deterministic():
    model = _model(seed=11)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    arms = {
        "full": ev.arm_full(model, prepared, max_new_tokens=5),
        "floor": ev.arm_floor(model, prepared, max_new_tokens=5),
        "shuffled": ev.arm_shuffled(model, prepared, seed=3, max_new_tokens=5),
        "pause": ev.arm_pause(model, prepared, seed=4, max_new_tokens=5),
        "slot_ablation": ev.arm_slot_ablation(model, prepared, max_new_tokens=5),
        "steering": ev.arm_steering(model, prepared, seed=5, max_new_tokens=5),
        "cross_routing": ev.arm_cross_routing(model, prepared, max_new_tokens=5),
        "router_routed": ev.arm_router_routed(model, prepared, max_new_tokens=5),
    }
    for name, result in arms.items():
        decode_ids = result["decode_ids"]
        assert len(decode_ids) == len(prepared), name
        for ids in decode_ids:
            assert ids.ndim == 2 and ids.shape[0] == 1 and 1 <= ids.shape[1] <= 5
            assert ids.dtype == torch.long
    # Greedy decodes are deterministic across repeated arm invocations.
    again = ev.arm_full(model, prepared, max_new_tokens=5)
    for first, second in zip(arms["full"]["decode_ids"], again["decode_ids"]):
        assert torch.equal(first, second)


def test_shuffled_is_complete_derangement_and_singleton_raises():
    model = _model(seed=13)
    tokenizer = TinyTokenizer()
    rows = [
        _row(0, "numeric"),
        _row(1, "numeric"),
        _row(2, "numeric"),
        _row(3, "unit"),
        _row(4, "unit"),
    ]
    prepared = ev.prepare_rows(model, tokenizer, rows, context_tokens=24)
    result = ev.arm_shuffled(model, prepared, seed=9, max_new_tokens=4)
    donors = result["donor_indices"]
    assert sorted(donors) == list(range(len(prepared)))  # a permutation
    for index, donor in enumerate(donors):
        assert donor != index, "row received its own thoughts"
        assert prepared[donor].family == prepared[index].family
    # Two-row family degenerates to the pairwise swap.
    assert {donors[3], donors[4]} == {4, 3}
    singleton_rows = rows + [_row(5, "ordering")]
    prepared_singleton = ev.prepare_rows(
        model, tokenizer, singleton_rows, context_tokens=24
    )
    try:
        ev.arm_shuffled(model, prepared_singleton, seed=9, max_new_tokens=4)
    except ValueError as error:
        assert "ordering" in str(error)
    else:
        raise AssertionError("singleton family must raise")


def test_pause_thoughts_are_row_invariant_and_seed_deterministic():
    model = _model(seed=17)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    first = ev.arm_pause(model, prepared[:2], seed=21, max_new_tokens=4)
    second = ev.arm_pause(model, prepared[2:], seed=21, max_new_tokens=4)
    # Zero mutual information with the row by construction: the pause
    # vectors depend only on the seed, never on which rows are present.
    assert torch.equal(first["pause_embeds"], second["pause_embeds"])
    assert torch.equal(first["pause_states"], second["pause_states"])
    other_seed = ev.arm_pause(model, prepared[:2], seed=22, max_new_tokens=4)
    assert not torch.equal(first["pause_embeds"], other_seed["pause_embeds"])


def test_slot_ablation_is_invariant_to_thought_state_perturbation():
    model = _model(seed=19)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    perturbed = copy.copy(prepared)
    perturbed = [copy.copy(item) for item in prepared]
    generator = torch.Generator().manual_seed(99)
    for item in perturbed:
        item.thought_states = item.thought_states + 2.0 * torch.randn(
            item.thought_states.shape, generator=generator
        )
    base = ev.arm_slot_ablation(model, prepared, max_new_tokens=5)
    moved = ev.arm_slot_ablation(model, perturbed, max_new_tokens=5)
    for first, second in zip(base["decode_ids"], moved["decode_ids"]):
        assert torch.equal(first, second), (
            "slot ablation saw the thought states it claims to remove"
        )


def test_steering_is_seed_deterministic_and_reports_sigma():
    model = _model(seed=23)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    first = ev.arm_steering(model, prepared, seed=31, max_new_tokens=4)
    second = ev.arm_steering(model, prepared, seed=31, max_new_tokens=4)
    assert first["sigma"] == 0.5
    for a, b in zip(first["decode_ids"], second["decode_ids"]):
        assert torch.equal(a, b)


def test_cross_routing_applies_the_expert_swap():
    model = _model(seed=29)
    tokenizer = TinyTokenizer()
    rows = [_row(0, "numeric"), _row(1, "ordering")]
    prepared = ev.prepare_rows(model, tokenizer, rows, context_tokens=24)
    assert [item.route_index for item in prepared] == [0, 1]
    result = ev.arm_cross_routing(model, prepared, max_new_tokens=4)
    assert result["routes_used"] == [1, 0]


def test_router_routed_matches_probe_argmax_through_the_family_map():
    model = _model(seed=31)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    result = ev.arm_router_routed(model, prepared, max_new_tokens=4)
    with torch.no_grad():
        for item, predicted, route in zip(
            prepared, result["predicted_families"], result["router_routes"]
        ):
            logits = model.route_family_logits(item.thought_states)
            assert predicted == int(logits.argmax(dim=-1).item())
            assert route == ev.route_for_family_index(
                predicted,
                model.config.mlp_expert_count,
                model.config.router_families,
            )
            assert route == predicted // 2  # 4 families over 2 experts


# ----------------------------------------------------------------------
# Expert liveness hook.


def test_expert_liveness_hook_returns_finite_ratio_and_cleans_up():
    model = _model(seed=37)
    tokenizer = TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _core_rows(), context_tokens=24)
    first = ev.expert_liveness(model, prepared, max_rows=2)
    assert first["state"] == "measured"
    assert math.isfinite(first["ratio"]) and first["ratio"] >= 0.0
    assert first["expert_events"] > 0 and first["shared_events"] > 0
    assert first["denominator"] in ("shared_lora", "base_down_proj")
    # Hooks must not leak: a second run records the same number of events.
    second = ev.expert_liveness(model, prepared, max_rows=2)
    assert second["expert_events"] == first["expert_events"]
    assert second["shared_events"] == first["shared_events"]
    no_experts = _model(seed=37, mlp_expert_count=0)
    bare = ev.expert_liveness(
        no_experts,
        ev.prepare_rows(no_experts, tokenizer, _core_rows(), context_tokens=24),
    )
    assert bare["state"] == "no_expert_modules" and not bare["engaged"]


# ----------------------------------------------------------------------
# w/o-L1 hygiene (A6 item 6).


def test_wo_l1_delta_enforces_hash_discipline():
    model = _model(seed=41)
    tokenizer = TinyTokenizer()
    rows = _core_rows()
    recorded = ev.state_dict_sha256(model)

    try:
        ev.wo_l1_delta(model, model, rows, tokenizer, expected_full_sha256=None)
    except ValueError as error:
        assert "A6" in str(error)
    else:
        raise AssertionError("missing recorded hash must raise")

    try:
        ev.wo_l1_delta(
            model, model, rows, tokenizer, expected_full_sha256="deadbeef"
        )
    except RuntimeError as error:
        assert "branch-point hash" in str(error)
    else:
        raise AssertionError("wrong recorded hash must raise")

    try:
        ev.wo_l1_delta(
            model, model, rows, tokenizer, expected_full_sha256=recorded
        )
    except RuntimeError as error:
        assert "contamination" in str(error)
    else:
        raise AssertionError("identical state dicts must raise")

    branched = copy.deepcopy(model)
    with torch.no_grad():
        branched.latent_projection[1].weight += 0.01
    report = ev.wo_l1_delta(
        model,
        branched,
        rows,
        tokenizer,
        expected_full_sha256=recorded,
        context_tokens=24,
        max_new_tokens=4,
    )
    assert report["descriptive_only"] is True
    assert report["n"] == len(rows)
    assert report["full_sha256"] == recorded != report["ablated_sha256"]
    assert -1.0 <= report["ablated_minus_full_mean"] <= 1.0
    assert report["full_minus_ablated"]["n"] == len(rows)


# ----------------------------------------------------------------------
# Semantic entropy, probe, abstention statistics.


def test_semantic_entropy_clusters_by_graded_answer_signature():
    spec = {"type": "numeric", "expected": 7}
    result = ev.discrete_semantic_entropy(["7", "value is 7 kg", "8"], spec)
    expected = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert result["n_clusters"] == 2
    assert result["cluster_sizes"] == [2, 1]
    assert abs(result["entropy"] - expected) < 1e-12
    unanimous = ev.discrete_semantic_entropy(["7", "7"], spec)
    assert unanimous["entropy"] == 0.0 and unanimous["n_clusters"] == 1


def test_premise_probe_closed_loop_learns_separable_digits():
    generator = torch.Generator().manual_seed(7)
    count, width = 80, 16
    labels = [index % 4 for index in range(count)]
    latents = torch.randn(count, width, generator=generator) * 0.1
    for index, label in enumerate(labels):
        latents[index, label] += 3.0
    result = ev.train_premise_probe(latents, labels, seed=5)
    assert result["state"] == "trained"
    assert result["live"] and result["probe_accuracy"] >= result["chance"] + 0.10
    again = ev.train_premise_probe(latents, labels, seed=5)
    assert again["probe_accuracy"] == result["probe_accuracy"]
    tiny = ev.train_premise_probe(latents[:3], labels[:3], seed=5)
    assert tiny["state"] == "insufficient_rows" and not tiny["live"]


def test_probe_digit_label_reads_first_withheld_digit():
    assert ev.probe_digit_label({"withheld_literals": ["4821"]}) == 4
    assert ev.probe_digit_label({"withheld_literals": ["x9"]}) == 9
    assert ev.probe_digit_label({"probe_digit": 13}) == 3
    assert ev.probe_digit_label({"withheld_literals": []}) is None


def test_abstention_block_futility_confirmatory_and_baseline_roles():
    count = 60
    correct = [index % 3 != 0 for index in range(count)]
    head = [1.0 + 0.001 * index if flag else 0.001 * index for index, flag in enumerate(correct)]
    logprob = [0.001 * ((index * 7) % count) for index in range(count)]
    entropy = [0.5 if flag else -0.5 for flag in correct]
    anchors = ["a%03d" % index for index in range(count)]
    result = ev.abstention_block(
        head_scores=head,
        logprob_scores=logprob,
        entropy_scores=entropy,
        correct=correct,
        anchor_ids=anchors,
        target_coverage=0.40,
        seed=3,
    )
    assert not result["degenerate"]
    assert result["failure_auroc"]["head"] == 1.0
    assert result["delong_head_vs_logprob"]["futility_only"] is True
    assert result["delong_head_vs_logprob"]["z"] > 0.0
    assert result["eprocess_confirmatory"] is True
    assert result["eprocess"]["wealth"] >= 1.0
    assert result["eprocess"]["n_discordant"] > 0
    assert "publish_rate" in result["conformal"]
    assert 0.0 <= result["conformal"]["out_of_fold_coverage_correct"] <= 1.0

    degenerate = ev.abstention_block(
        head_scores=head,
        logprob_scores=logprob,
        entropy_scores=entropy,
        correct=[False] * count,
        anchor_ids=anchors,
        target_coverage=0.40,
        seed=3,
    )
    assert degenerate["degenerate"]
    assert degenerate["delong_head_vs_logprob"] is None
    assert "error" in degenerate["conformal"]

    resumed = ev.abstention_block(
        head_scores=head,
        logprob_scores=logprob,
        entropy_scores=entropy,
        correct=correct,
        anchor_ids=anchors,
        target_coverage=0.40,
        seed=3,
        eprocess_resume={"wealth": 2.0, "max_wealth": 2.0, "pending_wrongs": []},
    )
    assert resumed["eprocess"]["wealth"] == 2.0 * result["eprocess"]["wealth"]


# ----------------------------------------------------------------------
# Leak scan.


def test_masked_leak_scan_detects_scaffold_and_input_leak():
    clean_row = _row(0, "numeric")
    clean = ev.masked_leak_scan(clean_row, "the value is 7")
    assert not clean["leak"]
    scaffold = ev.masked_leak_scan(
        clean_row, "the value is 7\nAssistant: and another turn"
    )
    assert scaffold["scaffold"] and scaffold["leak"] and not scaffold["input_leak"]
    leaky_row = dict(clean_row)
    leaky_row["masked_prompt"] = clean_row["public_prompt"]  # literal survives
    leaked = ev.masked_leak_scan(leaky_row, "the value is 7")
    assert leaked["input_leak"] and leaked["leak"]


# ----------------------------------------------------------------------
# End-to-end audit + gate dict.


def test_v10_audit_end_to_end_structure_and_gate_dict_serializes():
    model = _model(seed=47)
    tokenizer = TinyTokenizer()
    rows = _core_rows() + [
        _row(4, "numeric"),  # held out of the 4-row core -> probe pool
        _row(5, "unit"),
        _row(6, "ordering", masked=False),
        _row(7, "abstention", masked=False),
        _row(8, "numeric", masked=False),
    ]
    audit = ev.v10_audit(
        model,
        rows,
        tokenizer,
        model.config,
        masked_core_rows=4,
        seed=17,
        expert_rows_per_family=2,
        context_tokens=24,
        max_new_tokens=5,
        pool_temperatures=(0.0, 0.8),
        probe_latent_rows=8,
        target_coverage=0.40,
    )
    assert audit["ordering_rule"] == ev.ORDERING_RULE
    core = audit["masked_core"]
    assert core["n"] == 4
    assert core["anchor_ids"] == ["row-000", "row-001", "row-002", "row-003"]
    for arm in ("full", "floor", "shuffled", "pause", "slot_ablation", "steering"):
        assert len(core["arms"][arm]["correct"]) == 4
        assert 0.0 <= core["arms"][arm]["accuracy"] <= 1.0
    assert core["leak"]["leak_rows"] == 0
    for contrast in (
        "full_minus_shuffled",
        "full_minus_pause",
        "full_minus_slot_ablation",
        "full_minus_steering",
        "full_minus_floor",
    ):
        assert audit["paired"][contrast]["n"] == 4
    assert audit["unmasked"]["n"] == 3
    assert isinstance(audit["unmasked"]["parity_gap"], float)
    assert audit["experts"]["state"] == "measured"
    # 2 per family where available: numeric 2, unit 2, ordering 1, abstention 1.
    assert audit["experts"]["n_rows"] == 6
    assert audit["experts"]["per_family_n"] == {
        "numeric": 2,
        "unit": 2,
        "ordering": 1,
        "abstention": 1,
    }
    assert math.isfinite(audit["experts"]["liveness"]["ratio"])
    assert len(audit["records"]) == len(rows)
    assert all(record["pool_size"] == 2 for record in audit["records"])
    assert audit["abstention"]["n"] == len(rows)
    assert audit["probe"]["n_heldout_rows"] == 2  # rows 4 and 5
    assert audit["probe"]["state"] == "insufficient_rows"
    gold = audit["gold_forced_ce"]
    assert gold["n_scored"] == len(rows)
    assert math.isfinite(gold["overall"]["channel_masked"])
    assert math.isfinite(gold["overall"]["causal_full"])
    assert gold["unmasked"]["exposure_gap"] == 0.0

    gates = ev.v10_gate_dict(audit)
    payload = json.dumps(gates)  # must be JSON-serializable end to end
    assert '"passed"' in payload
    assert isinstance(gates["passed"], bool)
    notes = gates["gate_notes"]
    assert notes["masked_core_n"] == 4
    assert notes["ordering_rule"] == ev.ORDERING_RULE
    assert notes["eprocess_wealth"] is not None
    assert notes["delong_futility_only"] is True
    assert notes["probe_state"] in (
        "evaluated",
        "not_evaluated_channel_dead",
        "not_evaluated_probe_below_liveness",
    )
    with_wol1 = ev.v10_gate_dict(
        audit,
        wo_l1={
            "n": 4,
            "ablated_minus_full_mean": -0.25,
            "full_sha256": "a" * 64,
            "ablated_sha256": "b" * 64,
        },
    )
    assert with_wol1["gate_notes"]["wo_l1_descriptive"]["n"] == 4
    json.dumps(with_wol1)


def test_v10_audit_excludes_legacy_family_rows():
    """Session I-4 regression: the master splits still carry pre-Version-10
    corpus rows (io_tests, python_tests, sql_exact, untyped); the audit must
    run on the four preregistered families and disclose the excluded count
    instead of crashing in family_index_of_row on the first apps-train row."""

    legacy_io = {
        "episode_id": "apps-train-000000legacy",
        "public_prompt": "write a program that reads stdin",
        "masked": False,
        "answer_spec": {"type": "io_tests"},
    }
    legacy_untyped = {
        "episode_id": "old-workspace-row",
        "public_prompt": "an untyped legacy row",
        "masked": False,
    }
    rows = _core_rows()
    kept, excluded = ev.v10_audit_rows(rows + [legacy_io, legacy_untyped])
    assert [row["episode_id"] for row in kept] == [row["episode_id"] for row in rows]
    assert excluded == 2

    model = _model(seed=59)
    tokenizer = TinyTokenizer()
    audit = ev.v10_audit(
        model,
        rows + [legacy_io, legacy_untyped],
        tokenizer,
        model.config,
        masked_core_rows=4,
        seed=17,
        expert_rows_per_family=1,
        context_tokens=24,
        max_new_tokens=3,
        pool_temperatures=(0.0, 0.8),
    )
    assert audit["legacy_rows_excluded"] == 2
    assert audit["n_rows"] == len(rows)
    assert len(audit["records"]) == len(rows)
    json.dumps(ev.v10_gate_dict(audit))


def test_pool_temperatures_must_start_greedy():
    model = _model(seed=53)
    tokenizer = TinyTokenizer()
    try:
        ev.v10_audit(
            model,
            _core_rows(),
            tokenizer,
            model.config,
            masked_core_rows=2,
            seed=1,
            pool_temperatures=(0.8, 0.0),
            context_tokens=24,
            max_new_tokens=3,
        )
    except ValueError as error:
        assert "greedy" in str(error)
    else:
        raise AssertionError("non-greedy-first pool must raise")


# ----------------------------------------------------------------------
# Gate preconditions on synthetic audit dicts.


def _synthetic_audit(
    *,
    full,
    shuffled,
    pause,
    floor,
    slot=None,
    steering=None,
    leak_rows=0,
    parity_gap=0.0,
    probe=None,
    experts=None,
    eprocess_wealth=1.0,
):
    n = len(full)
    slot = slot if slot is not None else list(full)
    steering = steering if steering is not None else list(shuffled)

    def _arm(correct):
        return {
            "correct": list(correct),
            "accuracy": sum(map(float, correct)) / max(1, len(correct)),
            "texts": [""] * len(correct),
        }

    def _contrast(a, b):
        return ev.paired_lower_confidence_bound(
            [float(bool(x)) - float(bool(y)) for x, y in zip(a, b)]
        )

    return {
        "seed": 17,
        "ordering_rule": ev.ORDERING_RULE,
        "n_rows": n,
        "masked_core": {
            "n": n,
            "anchor_ids": ["row-%03d" % index for index in range(n)],
            "arms": {
                "full": _arm(full),
                "floor": _arm(floor),
                "shuffled": _arm(shuffled),
                "pause": _arm(pause),
                "slot_ablation": _arm(slot),
                "steering": _arm(steering),
            },
            "leak": {
                "input_leak_rows": 0,
                "scaffold_rows": leak_rows,
                "leak_rows": leak_rows,
            },
            "shuffled_donors": list(range(1, n)) + [0],
        },
        "paired": {
            "full_minus_shuffled": _contrast(full, shuffled),
            "full_minus_pause": _contrast(full, pause),
            "full_minus_slot_ablation": _contrast(full, slot),
            "full_minus_steering": _contrast(full, steering),
            "full_minus_floor": _contrast(full, floor),
        },
        "unmasked": {
            "n": 20,
            "full_accuracy": 0.5,
            "causal_accuracy": 0.5 - parity_gap,
            "parity_gap": parity_gap,
        },
        "experts": experts
        or {
            "state": "measured",
            "n_rows": 32,
            "per_family_n": {},
            "label_routed_accuracy": 0.5,
            "cross_routed_accuracy": 0.4,
            "router_routed_accuracy": 0.5,
            "cross_routing_cost": 0.10,
            "cross_routing_paired": None,
            "router_delta": 0.0,
            "router_agreement": 1.0,
            "liveness": {
                "state": "measured",
                "ratio": 0.5,
                "engaged": True,
                "expert_events": 8,
                "shared_events": 8,
                "denominator": "shared_lora",
            },
        },
        "probe": probe
        or {
            "state": "trained",
            "probe_accuracy": 0.9,
            "chance": 0.25,
            "live": True,
            "n": 512,
            "n_heldout_rows": 512,
        },
        "abstention": {
            "n": 40,
            "n_correct": 20,
            "n_incorrect": 20,
            "degenerate": False,
            "eprocess": {
                "wealth": eprocess_wealth,
                "max_wealth": eprocess_wealth,
                "n_pairs": 10,
                "n_discordant": 4,
                "rejects_at_alpha": {"0.05": eprocess_wealth >= 20.0},
            },
            "eprocess_confirmatory": True,
            "failure_auroc": {
                "head": 0.8,
                "logprob": 0.6,
                "neg_semantic_entropy": 0.7,
            },
            "delong_head_vs_logprob": {
                "z": 1.0,
                "p_one_sided": 0.16,
                "futility_only": True,
            },
            "conformal": {
                "out_of_fold_coverage_correct": 0.4,
                "publish_rate": 0.3,
            },
        },
        "records": [],
    }


def test_gate_dict_probe_precondition_states():
    live = _synthetic_audit(
        full=[True] * 30,
        shuffled=[False] * 30,
        pause=[False] * 30,
        floor=[False] * 30,
    )
    gates = ev.v10_gate_dict(live)
    assert gates["latent_channel_live"] and gates["beats_pause_compute"]
    assert gates["gate_notes"]["probe_state"] == "evaluated"
    # probe 0.9 - full 1.0 <= 0.10 -> gate holds.
    assert gates["probe_generation_gap"]

    gap = _synthetic_audit(
        full=[True] * 15 + [False] * 15,
        shuffled=[False] * 30,
        pause=[False] * 30,
        floor=[False] * 30,
        probe={
            "state": "trained",
            "probe_accuracy": 0.95,
            "chance": 0.25,
            "live": True,
            "n": 512,
            "n_heldout_rows": 512,
        },
    )
    gap_gates = ev.v10_gate_dict(gap)
    assert gap_gates["latent_channel_live"]
    assert not gap_gates["probe_generation_gap"]  # 0.95 - 0.5 > 0.10

    dead = _synthetic_audit(
        full=[False] * 30,
        shuffled=[False] * 30,
        pause=[False] * 30,
        floor=[False] * 30,
    )
    dead_gates = ev.v10_gate_dict(dead)
    assert not dead_gates["latent_channel_live"]
    assert dead_gates["probe_generation_gap"]  # vacuous
    assert dead_gates["gate_notes"]["probe_state"] == "not_evaluated_channel_dead"

    dull_probe = _synthetic_audit(
        full=[True] * 30,
        shuffled=[False] * 30,
        pause=[False] * 30,
        floor=[False] * 30,
        probe={
            "state": "trained",
            "probe_accuracy": 0.30,
            "chance": 0.25,
            "live": False,
            "n": 512,
            "n_heldout_rows": 512,
        },
    )
    dull_gates = ev.v10_gate_dict(dull_probe)
    assert dull_gates["probe_generation_gap"]  # vacuous
    assert (
        dull_gates["gate_notes"]["probe_state"]
        == "not_evaluated_probe_below_liveness"
    )


def test_gate_dict_expert_liveness_precondition_and_router_delta():
    engaged = _synthetic_audit(
        full=[True] * 20,
        shuffled=[False] * 20,
        pause=[False] * 20,
        floor=[False] * 20,
    )
    gates = ev.v10_gate_dict(engaged)
    assert gates["cross_routing_cost"] and gates["router_delta"]
    assert gates["gate_notes"]["cross_routing_state"] == "measured"

    dormant = _synthetic_audit(
        full=[True] * 20,
        shuffled=[False] * 20,
        pause=[False] * 20,
        floor=[False] * 20,
        experts={
            "state": "measured",
            "n_rows": 32,
            "per_family_n": {},
            "label_routed_accuracy": 0.5,
            "cross_routed_accuracy": 0.5,
            "router_routed_accuracy": 0.45,
            "cross_routing_cost": 0.0,
            "cross_routing_paired": None,
            "router_delta": 0.05,
            "router_agreement": 0.9,
            "liveness": {
                "state": "measured",
                "ratio": 0.02,
                "engaged": False,
                "expert_events": 8,
                "shared_events": 8,
                "denominator": "shared_lora",
            },
        },
    )
    dormant_gates = ev.v10_gate_dict(dormant)
    assert not dormant_gates["cross_routing_cost"]
    assert (
        dormant_gates["gate_notes"]["cross_routing_state"]
        == "experts_never_engaged"
    )
    assert not dormant_gates["router_delta"]  # 0.05 > 0.02
    assert not dormant_gates["passed"]

    leaky = _synthetic_audit(
        full=[True] * 20,
        shuffled=[False] * 20,
        pause=[False] * 20,
        floor=[False] * 20,
        leak_rows=1,
    )
    assert not ev.v10_gate_dict(leaky)["masked_leak_zero"]

    drifted = _synthetic_audit(
        full=[True] * 20,
        shuffled=[False] * 20,
        pause=[False] * 20,
        floor=[False] * 20,
        parity_gap=0.2,
    )
    assert not ev.v10_gate_dict(drifted)["unmasked_parity"]


# ----------------------------------------------------------------------
# Ladder.


def _seed_gates(
    *,
    infrastructure_ok=True,
    warm_gate_passed=True,
    channel=True,
    live=None,
    wealth=1.0,
    training_complete=True,
):
    live = channel if live is None else live
    gates = {
        "infrastructure_ok": infrastructure_ok,
        "training_complete": training_complete,
        "warm_gate_passed": warm_gate_passed,
        "masked_leak_zero": channel,
        "masked_causal_floor": channel,
        "latent_channel_live": live,
        "beats_pause_compute": channel,
        "unmasked_parity": channel,
        "probe_generation_gap": channel,
        "cross_routing_cost": True,
        "router_delta": True,
        "gate_notes": {"eprocess_wealth": wealth},
    }
    gates["passed"] = all(
        bool(value) for key, value in gates.items() if key != "gate_notes"
    )
    return gates


def test_binding_rung_covers_every_rung_and_abstention_resolution():
    # Rung 0: infrastructure failure on any seed.
    assert ev.binding_rung(
        {17: _seed_gates(infrastructure_ok=False), 29: _seed_gates()}
    ).startswith("rung_0")
    # Rung 1: warm gate failed on BOTH seeds.
    assert ev.binding_rung(
        {
            17: _seed_gates(warm_gate_passed=False, channel=False),
            29: _seed_gates(warm_gate_passed=False, channel=False),
        }
    ).startswith("rung_1")
    # One warm failure only does NOT trigger rung 1.
    mixed_warm = ev.binding_rung(
        {
            17: _seed_gates(warm_gate_passed=False, channel=False),
            29: _seed_gates(channel=True),
        }
    )
    assert not mixed_warm.startswith("rung_1")
    # Rung 2: masked transfer null on both seeds.
    assert ev.binding_rung(
        {17: _seed_gates(channel=False), 29: _seed_gates(channel=False)}
    ).startswith("rung_2")
    # Rung 2 also when live on one seed but no seed passes the full set.
    assert ev.binding_rung(
        {
            17: _seed_gates(channel=False, live=True),
            29: _seed_gates(channel=False),
        }
    ).startswith("rung_2")
    # Rung 3: exactly one seed passes the channel gate set.
    assert ev.binding_rung(
        {17: _seed_gates(channel=True), 29: _seed_gates(channel=False, live=True)}
    ).startswith("rung_3")
    # A lone seed passing everything is rung 3 (no replication claim).
    assert ev.binding_rung({17: _seed_gates(channel=True)}).startswith("rung_3")
    # Rung 4: both seeds pass.
    assert ev.binding_rung(
        {17: _seed_gates(channel=True), 29: _seed_gates(channel=True)}
    ).startswith("rung_4")
    # Abstention resolves independently via the product e-value.
    confirmed = ev.binding_rung(
        {
            17: _seed_gates(channel=False, wealth=10.0),
            29: _seed_gates(channel=False, wealth=4.0),
        }
    )
    assert confirmed == "rung_2+abstention_confirmed"  # 10 * 4 >= 20
    open_case = ev.binding_rung(
        {
            17: _seed_gates(channel=True, wealth=2.0),
            29: _seed_gates(channel=True, wealth=3.0),
        }
    )
    assert open_case == "rung_4+abstention_open"
    try:
        ev.binding_rung({})
    except ValueError:
        pass
    else:
        raise AssertionError("empty gates_by_seed must raise")


def test_binding_rung_never_claims_a_pass_on_aborted_training():
    """A seed whose training aborted at a tripwire cannot reach rung 3 or 4.

    The channel gates are computed on the abort checkpoint for the failure
    branch, so they can look clean while the preregistered run never
    happened; without this conjunct a truncated run could be reported as an
    established claim.
    """

    both_aborted = ev.binding_rung(
        {
            17: _seed_gates(channel=True, training_complete=False),
            29: _seed_gates(channel=True, training_complete=False),
        }
    )
    assert not both_aborted.startswith("rung_4"), both_aborted
    assert not both_aborted.startswith("rung_3"), both_aborted
    one_aborted = ev.binding_rung(
        {
            17: _seed_gates(channel=True, training_complete=False),
            29: _seed_gates(channel=True, training_complete=True),
        }
    )
    assert one_aborted.startswith("rung_3"), one_aborted
    # Sanity: identical gates with training complete DO reach rung 4.
    assert ev.binding_rung(
        {17: _seed_gates(channel=True), 29: _seed_gates(channel=True)}
    ).startswith("rung_4")


# ----------------------------------------------------------------------
# Gold-answer teacher-forced CE: exposure gap vs content gap.


def test_gold_forced_ce_separates_exposure_from_channel_cost():
    model = _model(seed=5)
    tokenizer = TinyTokenizer()
    rows = []
    for index in range(8):
        row = _row(index, "numeric" if index % 2 == 0 else "unit",
                   masked=index % 4 < 2)
        row["public_target"] = "the result is %d" % (100 + index)
        rows.append(row)
    prepared = ev.prepare_rows(model, tokenizer, ev.sorted_by_anchor(rows),
                               context_tokens=32, device=None)

    block = ev.gold_forced_ce_block(model, prepared, tokenizer, rows=6,
                                    max_gold_tokens=8)
    assert block["n_scored"] == 6, block["n_scored"]
    # Stratified round-robin: every (family, masked) bucket is represented
    # before any bucket is sampled twice.
    buckets = {(record["family"], record["masked"]) for record in block["records"]}
    assert len(buckets) == 4, buckets
    for record in block["records"]:
        for key in ("channel_masked", "channel_full", "causal_masked", "causal_full"):
            assert math.isfinite(record[key]), (key, record)
            assert record[key] >= 0.0, (key, record)  # CE, not logprob
        assert math.isclose(
            record["exposure_gap"],
            record["channel_masked"] - record["channel_full"],
            rel_tol=1e-9, abs_tol=1e-9,
        )
        assert math.isclose(
            record["channel_cost"],
            record["channel_full"] - record["causal_full"],
            rel_tol=1e-9, abs_tol=1e-9,
        )
        if not record["masked"]:
            # Unmasked rows read the same ids in both arms, so the exposure
            # contrast must be exactly zero -- this is the arm that would
            # expose a prompt-plumbing bug in the masked arm.
            assert record["exposure_gap"] == 0.0, record
    assert block["unmasked"]["exposure_gap"] == 0.0
    assert block["masked"]["n"] + block["unmasked"]["n"] == block["n_scored"]

    # Deterministic: same model, same rows, same numbers.
    again = ev.gold_forced_ce_block(model, prepared, tokenizer, rows=6,
                                    max_gold_tokens=8)
    assert again["records"] == block["records"]

    # A row without a gold answer is skipped, not crashed on.
    stripped = copy.deepcopy(rows)
    for row in stripped:
        row.pop("public_target", None)
    prepared_stripped = ev.prepare_rows(
        model, tokenizer, ev.sorted_by_anchor(stripped), context_tokens=32,
        device=None,
    )
    empty = ev.gold_forced_ce_block(model, prepared_stripped, tokenizer, rows=4)
    assert empty["n_scored"] == 0
    assert empty["overall"]["channel_masked"] is None


def test_training_verdict_from_metrics_takes_the_last_record(tmp_path=None):
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    directory = _Path(tempfile.mkdtemp())
    path = directory / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                _json.dumps({"step": 1, "loss": 1.0}),
                "not json at all",
                _json.dumps({"v10_verdict": {"aborted": "warm_gate"}}),
                _json.dumps({"v10_verdict": {"aborted": None, "warm_gate": {"passed": True}}}),
            ]
        )
        + "\n"
    )
    verdict = ev.training_verdict_from_metrics(path)
    assert verdict["aborted"] is None
    assert verdict["warm_gate"]["passed"] is True
    assert ev.training_verdict_from_metrics(directory / "missing.jsonl") == {}


def test_run_seed_audit_plumbing_on_the_session_i5_output_shape():
    """The per-GPU audit entry point, on an aborted seed's real layout.

    ``run_seed_audit`` runs only as a subprocess (notebook cell 9), so no
    other test reaches its checkpoint discovery, verdict wiring or file
    writes. The three heavy loads are stubbed; everything between them is
    shipped code. The on-disk shape mirrors the session I-5 kernel output
    exactly: no ``checkpoint-v10-full.pt``, raw step checkpoints plus their
    ``.sha256`` sidecars, and a verdict that aborted at the go/no-go.
    """

    import json as _json
    import sys as _sys
    import tempfile
    import types
    from pathlib import Path as _Path

    directory = _Path(tempfile.mkdtemp())
    output = directory / "hlwm-v10.0-seed-17"
    output.mkdir()
    (output / "metrics.jsonl").write_text(
        _json.dumps({"step": 600, "warm_recon_digit_em": 0.6})
        + "\n"
        + _json.dumps(
            {
                "v10_verdict": {
                    "aborted": "gonogo",
                    "gonogo_masked_numeric_em": 0.0,
                    "warm_gate": {"passed": True},
                }
            }
        )
        + "\n"
    )
    for step in (500, 1200):
        torch.save(
            {"base_model": "tiny", "base_revision": "rev"},
            output / ("checkpoint-step-%06d.pt" % step),
        )
        (output / ("checkpoint-step-%06d.pt.sha256" % step)).write_text("sha  name\n")

    model = _model(3)
    rows = _core_rows() + [
        _row(4, "numeric", masked=False),
        _row(5, "unit", masked=False),
    ]
    tokenizer = TinyTokenizer()
    real_audit = ev.v10_audit
    observed = {}

    def _small_audit(audited_model, audited_rows, audited_tokenizer, config, **kwargs):
        observed.update(seed=kwargs.get("seed"), n_rows=len(audited_rows))
        kwargs.update(
            masked_core_rows=4,
            expert_rows_per_family=2,
            probe_latent_rows=8,
            max_new_tokens=4,
            gold_ce_rows=4,
        )
        return real_audit(audited_model, audited_rows, audited_tokenizer, config, **kwargs)

    stubs = {
        "transformers": types.ModuleType("transformers"),
        "evaluate_checkpoint": types.ModuleType("evaluate_checkpoint"),
        "data": types.ModuleType("data"),
    }
    loaded_on = {}

    def _load_hlwm(payload, device):
        loaded_on["device"] = device
        return model

    stubs["transformers"].AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: tokenizer
    )
    stubs["evaluate_checkpoint"].load_hlwm = _load_hlwm
    stubs["data"].Reasoning9000Dataset = lambda *args, **kwargs: types.SimpleNamespace(
        rows=rows
    )
    saved = {name: _sys.modules.get(name) for name in stubs}
    _sys.modules.update(stubs)
    ev.v10_audit = _small_audit
    try:
        # Pin the device. The default is cuda:0 whenever a GPU exists, which
        # strands this CPU-resident tiny model and made a pure plumbing test
        # host-dependent -- it passed locally and failed on the Kaggle runner.
        gates = ev.run_seed_audit(
            output, data_dir=directory, seed=17, device=torch.device("cpu")
        )
    finally:
        ev.v10_audit = real_audit
        for name, module in saved.items():
            if module is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = module

    assert observed == {"seed": 17, "n_rows": len(rows)}
    # Pinning above only stays honest if the device really reaches the loader.
    assert loaded_on["device"] == torch.device("cpu")
    audit = _json.loads((output / "v10-audit.json").read_text())
    written = _json.loads((output / "v10-gates.json").read_text())
    # Newest raw step wins, and the .sha256 sidecar is never mistaken for one.
    assert audit["checkpoint"].endswith("checkpoint-step-001200.pt")
    # An aborted run cannot report itself complete, and so cannot bind.
    assert gates["training_complete"] is False
    assert written["training_complete"] is False
    assert gates["warm_gate_passed"] is True
    rung = ev.binding_rung({17: gates, 29: gates})
    assert not rung.startswith(("rung_3", "rung_4"))


if __name__ == "__main__":
    import sys

    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS", name)
            except AssertionError as error:
                failures += 1
                print("FAIL", name, error)
    sys.exit(1 if failures else 0)
