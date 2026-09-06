#!/usr/bin/env python3
"""Tests for the HLWM v10 gold-trace library (scripts/hlwm_v10_traces.py).

Run from the repo root: ``.venv/bin/python -m pytest scripts/test_hlwm_v10_traces.py -q``

Consistency tests re-derive the v9 key arithmetic (copied verbatim from
``behavior_anchor_rows_v9`` kinds 0/1/2, with ``stable_key``/``_operand``
imported from the builder so there is no drift) and assert the trace finals
match the integers rendered in the rows' answer strings.

Token-budget note (measured under the pinned Qwen3 tokenizer, which splits
numbers into single digits): unit/abstention/ordering traces and chain-1
numeric traces are asserted 100% ok at max_tokens=30. Multi-step numeric
traces cannot fit 30 whole-trace tokens by digit count alone (worst case
chain-3 is ~74 tokens), so for numeric the 30-token assertion is applied to
chain-1 rows, every individual step is asserted <= 30 tokens, and the
whole-trace ok-rate at 30 is reported alongside ordering's.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import hlwm_v10_traces as tr
from scripts.build_hlwm_v90_bundle import (
    _operand,
    behavior_anchor_rows_v9,
    stable_key,
)

SEED_CONSISTENCY = 17
SEED_BUDGET = 29
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
QWEN_EOS_PAD = 151643

EQUATION_RE = re.compile(r"<<(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)>>")


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


# ---------------------------------------------------------------------------
# v9 key arithmetic, copied verbatim from behavior_anchor_rows_v9 (kinds 0-2).
# ---------------------------------------------------------------------------

def derive_chain_digits(key):
    return 1 + (key >> 3) % 3, 2 + (key >> 21) % 5


def derive_numeric(key, chain, digits):
    operands = [_operand(key, digits, 31)]
    operators = []
    value = operands[0]
    for step in range(chain):
        op_pick = (key >> (7 + 5 * step)) % 3
        if op_pick == 2:
            factor = 2 + ((key >> (11 + 5 * step)) % 11)
            operators.append("*")
            operands.append(factor)
            value = value * factor
        elif op_pick == 1 and value > 2:
            term = 1 + ((key >> (13 + 5 * step)) % max(2, min(value - 1, 10 ** digits)))
            operators.append("-")
            operands.append(term)
            value = value - term
        else:
            term = _operand(key, max(2, digits - 1), 17 + 5 * step)
            operators.append("+")
            operands.append(term)
            value = value + term
    return operands, operators, value


def derive_unit(key, chain, digits):
    hops = {
        1: (("minutes", "seconds", 60), ("hours", "minutes", 60), ("days", "hours", 24)),
        2: (("hours", "seconds", 3600), ("days", "minutes", 1440)),
        3: (("days", "seconds", 86400),),
    }[chain]
    source_unit, target_unit, factor = hops[(key >> 8) % len(hops)]
    quantity = 2 + key % (10 ** min(digits, 4))
    return quantity, factor, quantity * factor


def derive_ordering(key, chain, digits):
    length = {1: 4, 2: 6, 3: 8}[chain]
    base = _operand(key, digits, 29)
    values = [base]
    for step in range(length - 1):
        values.append(values[-1] + 1 + ((key >> (9 + 3 * step)) % 13))
    ordered = list(values)
    shuffled = list(ordered)
    for position in range(length - 1, 0, -1):
        swap = (key >> (position * 2 + 1)) % (position + 1)
        shuffled[position], shuffled[swap] = shuffled[swap], shuffled[position]
    if shuffled == ordered:
        shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
    return shuffled, ordered


def build_trace_for_row(row, key):
    kind = row["difficulty_params"]["kind"]
    chain = row["difficulty_params"]["chain"]
    digits = row["difficulty_params"]["digits"]
    assert (chain, digits) == derive_chain_digits(key)
    if kind == "numeric":
        operands, operators, _ = derive_numeric(key, chain, digits)
        return tr.numeric_trace(operands, operators)
    if kind == "unit":
        quantity, factor, converted = derive_unit(key, chain, digits)
        return tr.unit_trace(quantity, factor, converted)
    if kind == "ordering":
        shuffled, ordered = derive_ordering(key, chain, digits)
        return tr.ordering_trace(shuffled, ordered)
    return tr.abstention_trace()


def _apply(left, operator, right):
    return {"+": left + right, "-": left - right, "*": left * right}[operator]


# ---------------------------------------------------------------------------
# 1. Worked example: exact strings.
# ---------------------------------------------------------------------------

def test_numeric_trace_worked_example():
    out = tr.numeric_trace([417, 88, 3], ["+", "-"])
    assert out["steps"] == ["<<417 + 88 = 505>>", "<<505 - 3 = 502>>"]
    assert out["values"] == [505, 502]
    assert out["trace"] == "<<417 + 88 = 505>> <<505 - 3 = 502>>"
    assert out["final"] == 502


def test_numeric_trace_multiplication_renders_star():
    out = tr.numeric_trace([7, 12], ["*"])
    assert out["steps"] == ["<<7 * 12 = 84>>"]
    assert out["final"] == 84


# ---------------------------------------------------------------------------
# 2. Consistency with the v9 generator: 200/200 train rows.
# ---------------------------------------------------------------------------

def test_v9_generator_consistency_200_rows():
    rows = behavior_anchor_rows_v9("train", 200, SEED_CONSISTENCY)
    assert len(rows) == 200
    checked = 0
    for index, row in enumerate(rows):
        key = stable_key(SEED_CONSISTENCY, "train", index)
        kind = row["difficulty_params"]["kind"]
        answer = row["integration"]["published_answer"]
        request = row["input"]["user_request"]
        trace = build_trace_for_row(row, key)
        if kind == "numeric":
            match = re.search(r"The result is (-?\d+)\.", answer)
            assert match, answer
            assert trace["final"] == int(match.group(1))
            assert trace["final"] == row["evaluation"]["answer_spec"]["expected"]
            rendered = str(trace["operands"][0])
            for operator, operand in zip(trace["operators"], trace["operands"][1:]):
                symbol = {"+": "+", "-": "-", "*": "x"}[operator]
                rendered += " %s %d" % (symbol, operand)
            assert rendered in request  # operands match the rendered request
        elif kind == "unit":
            match = re.search(r"equals (\d+) ", answer)
            assert match, answer
            assert trace["final"] == int(match.group(1))
            assert trace["final"] == row["evaluation"]["answer_spec"]["expected"]
            assert str(trace["quantity"]) in request
        elif kind == "ordering":
            match = re.search(r"Ascending order: ([0-9, ]+)\.", answer)
            assert match, answer
            expected = [int(token) for token in match.group(1).split(", ")]
            assert trace["final"] == expected
            assert trace["final"] == row["evaluation"]["answer_spec"]["expected"]
        else:
            assert kind == "abstention"
            assert trace["steps"] == ["<<no value provided = abstain>>"]
        checked += 1
    assert checked == 200


# ---------------------------------------------------------------------------
# 3. corrupt_one_step: exactly one wrong derivation, cascaded, per 100 keys.
# ---------------------------------------------------------------------------

def test_corrupt_one_step_100_keyed_three_step_traces():
    corruption_indices = set()
    for run in range(100):
        key = stable_key(99, "corrupt", run)
        _, digits = derive_chain_digits(key)
        operands, operators, _ = derive_numeric(key, 3, digits)
        clean = tr.numeric_trace(operands, operators)
        assert len(clean["steps"]) == 3
        corrupt = tr.corrupt_one_step(clean, key)
        corruption_indices.add(corrupt["corrupted_step_index"])

        parsed = []
        for step in corrupt["steps"]:
            match = EQUATION_RE.fullmatch(step)
            assert match, step
            left, operator, right, result = match.groups()
            parsed.append((int(left), operator, int(right), int(result)))

        wrong = [
            index
            for index, (left, operator, right, result) in enumerate(parsed)
            if _apply(left, operator, right) != result
        ]
        assert wrong == [corrupt["corrupted_step_index"]]

        # Internal chain consistency: each step's left input is the previous
        # step's (possibly corrupted) result, and operators/operands are the
        # originals.
        for index in range(1, 3):
            assert parsed[index][0] == parsed[index - 1][3]
        for index, (left, operator, right, _) in enumerate(parsed):
            assert operator == operators[index]
            assert right == operands[index + 1]
        for index in range(corrupt["corrupted_step_index"]):
            assert corrupt["steps"][index] == clean["steps"][index]

        assert corrupt["final"] != clean["final"]
        bad_result = parsed[corrupt["corrupted_step_index"]][3]
        true_result = clean["values"][corrupt["corrupted_step_index"]]
        assert bad_result != true_result
        assert len(str(bad_result)) == len(str(true_result))  # digit-plausible
    assert corruption_indices == {0, 1, 2}  # key-derived pick covers all steps


# ---------------------------------------------------------------------------
# 4. teacher_trace_tokens: exclude the final answer-producing step.
# ---------------------------------------------------------------------------

def test_teacher_trace_tokens():
    multi = tr.numeric_trace([417, 88, 3], ["+", "-"])
    teacher = tr.teacher_trace_tokens(multi)
    assert teacher["teacher_steps"] == ["<<417 + 88 = 505>>"]
    assert teacher["excluded_step"] == "<<505 - 3 = 502>>"
    assert teacher["single_step"] is False

    single = tr.unit_trace(5, 60, 300)
    teacher_single = tr.teacher_trace_tokens(single)
    assert teacher_single["teacher_steps"] == []
    assert teacher_single["excluded_step"] == "<<5 * 60 = 300>>"
    assert teacher_single["single_step"] is True

    abstain = tr.teacher_trace_tokens(tr.abstention_trace())
    assert abstain["teacher_steps"] == []
    assert abstain["single_step"] is True


# ---------------------------------------------------------------------------
# 5. compression_windows: balanced, covering, guarded.
# ---------------------------------------------------------------------------

def test_compression_windows():
    ids = list(range(1, 11))
    windows = tr.compression_windows(ids, 3, QWEN_EOS_PAD)
    assert [len(window) for window in windows] == [4, 3, 3]
    assert [token for window in windows for token in window] == ids

    for n in range(1, 41):
        for k in range(1, n + 1):
            token_ids = list(range(1000, 1000 + n))
            windows = tr.compression_windows(token_ids, k, QWEN_EOS_PAD)
            sizes = [len(window) for window in windows]
            assert len(windows) == k
            assert all(sizes)  # never empty
            assert max(sizes) - min(sizes) <= 1
            assert [token for window in windows for token in window] == token_ids

    with pytest.raises(ValueError):
        tr.compression_windows([1, 2, 3], 4, QWEN_EOS_PAD)
    with pytest.raises(ValueError):
        tr.compression_windows([1, QWEN_EOS_PAD, 3], 2, QWEN_EOS_PAD)
    with pytest.raises(ValueError):
        tr.compression_windows([1, 7, 3], 2, [7, QWEN_EOS_PAD])


# ---------------------------------------------------------------------------
# 6. Token budget over 500 generated anchors, real tokenizer.
# ---------------------------------------------------------------------------

def test_token_budget_500_anchors(tokenizer):
    rows = behavior_anchor_rows_v9("train", 500, SEED_BUDGET)
    stats = {
        kind: {"n": 0, "ok30": 0, "max_tokens": 0}
        for kind in ("numeric", "unit", "ordering", "abstention")
    }
    numeric_chain1_failures = []
    numeric_step_over_30 = []
    for index, row in enumerate(rows):
        key = stable_key(SEED_BUDGET, "train", index)
        kind = row["difficulty_params"]["kind"]
        chain = row["difficulty_params"]["chain"]
        trace = build_trace_for_row(row, key)
        verdict = tr.validate_trace(trace, tokenizer, max_tokens=30, max_steps=3)
        # Parse and step-count budgets hold for every family at every band.
        budget_only = [
            reason for reason in verdict["reasons"]
            if not reason.startswith("token_budget_exceeded")
        ]
        assert budget_only == [], (kind, verdict)

        record = stats[kind]
        record["n"] += 1
        record["ok30"] += int(verdict["ok"])
        record["max_tokens"] = max(record["max_tokens"], verdict["n_tokens"])

        if kind in ("unit", "abstention", "ordering"):
            assert verdict["ok"], (kind, trace["trace"], verdict)
        else:
            # Numeric: every individual step fits the 30-token budget; the
            # whole trace fits it whenever chain == 1. Multi-step traces at
            # high digit bands cannot fit 30 whole-trace tokens under the
            # digit-per-token Qwen3 tokenizer (see module docstring).
            for step in trace["steps"]:
                step_tokens = len(tokenizer.encode(step, add_special_tokens=False))
                if step_tokens > 30:
                    numeric_step_over_30.append((index, step, step_tokens))
            if chain == 1 and not verdict["ok"]:
                numeric_chain1_failures.append((index, trace["trace"], verdict))

    assert numeric_step_over_30 == []
    assert numeric_chain1_failures == []
    for kind in ("unit", "abstention", "ordering"):
        assert stats[kind]["ok30"] == stats[kind]["n"]
    assert sum(record["n"] for record in stats.values()) == 500

    report = " | ".join(
        "%s: %d/%d ok@30, max %d tok"
        % (kind, record["ok30"], record["n"], record["max_tokens"])
        for kind, record in stats.items()
    )
    print("\n[token-budget] " + report)
    # Ordering ok-rate reported (and asserted) separately per spec: the final
    # shrunk format <<sorted: p1 p2 ... pn>> passes 100% at 30 tokens.
    assert stats["ordering"]["ok30"] == stats["ordering"]["n"]
    assert stats["ordering"]["max_tokens"] <= 30


# ---------------------------------------------------------------------------
# Corruption of the other families (single-step unit, permutation swap).
# ---------------------------------------------------------------------------

def test_corrupt_unit_and_ordering_and_abstention():
    unit = tr.unit_trace(90, 60, 5400)
    corrupt = tr.corrupt_one_step(unit, 12345)
    assert corrupt["corrupted_step_index"] == 0
    match = EQUATION_RE.fullmatch(corrupt["steps"][0])
    left, operator, right, result = match.groups()
    assert (int(left), operator, int(right)) == (90, "*", 60)
    assert int(result) != 5400
    assert len(result) == len("5400")

    ordering = tr.ordering_trace([120, 104, 131, 111], [104, 111, 120, 131])
    corrupt_order = tr.corrupt_one_step(ordering, 777)
    assert corrupt_order["final"] != ordering["final"]
    assert sorted(corrupt_order["final"]) == ordering["final"]
    assert tr.SORTED_STEP_RE.fullmatch(corrupt_order["steps"][0])

    with pytest.raises(ValueError):
        tr.corrupt_one_step(tr.abstention_trace(), 5)


def test_validate_trace_flags_bad_steps(tokenizer):
    bad = {"steps": ["<<1 + 1 = 2>", "<<a < b>>"], "trace": "<<1 + 1 = 2> <<a < b>>"}
    verdict = tr.validate_trace(bad, tokenizer)
    assert not verdict["ok"]
    assert any(reason.startswith("unparseable_step") for reason in verdict["reasons"])
