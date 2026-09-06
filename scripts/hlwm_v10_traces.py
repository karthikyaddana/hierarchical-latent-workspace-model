#!/usr/bin/env python3
"""Gold compact reasoning traces for HLWM v10 behavior anchors.

Pure functions the v10 bundle builder calls to attach a gold trace plus
supervision artifacts (teacher steps, process-supervision negatives, CoLaR
compression windows) to every behavior-anchor row. The arithmetic mirrors the
v9 generator (``behavior_anchor_rows_v9`` in ``build_hlwm_v90_bundle.py``)
exactly: callers pass the key-derived operands/operators/values, so traces are
consistent with the already-rendered request/answer text.

Trace formats (all steps match the parse regex ``<<[^<>]+>>``):

* numeric  -- GSM8k-Aug equation steps with the RUNNING value, e.g. operands
  [417, 88, 3] with operators ['+', '-'] gives
  ``<<417 + 88 = 505>> <<505 - 3 = 502>>``. Multiplication renders as ``*``.
* unit     -- one step ``<<quantity * factor = converted>>``.
* ordering -- ONE compact step listing, for each element of the ascending
  output, its 1-based position in the shuffled input:
  ``<<sorted: 4 1 3 2 6 5 8 7>>``. Rationale: the pinned Qwen3 tokenizer
  splits numbers into single digits, so the spec's value-listing format
  ``<<sorted: a < b < c>>`` costs 67 tokens for eight 6-digit values (over
  the 30-token budget even for four values) AND its inner ``<`` breaks the
  ``<<[^<>]+>>`` parse regex. The position format is a full permutation
  (combined with the request it determines the answer), is 20 tokens worst
  case (8 items), and contains no inner angle brackets.
* abstention -- one step ``<<no value provided = abstain>>``.

Token-budget reality under the real tokenizer (digit-per-token): every single
step of every family fits in 30 tokens (worst measured: numeric chain-3 third
step 27, unit 25, ordering 20, abstention 8), and whole traces of unit /
ordering / abstention and chain-1 numeric fit in 30 tokens. Multi-step numeric
traces (chain 2-3 at high digit bands) cannot: three full running-value
equations over 6-10 digit numbers are ~50-76 tokens by digit count alone.
``validate_trace`` reports this honestly so the builder can gate trace
supervision by difficulty band or raise the budget for numeric rows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Union

STEP_RE = re.compile(r"<<[^<>]+>>")
EQUATION_STEP_RE = re.compile(r"<<(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)>>")
SORTED_STEP_RE = re.compile(r"<<sorted: (\d+(?: \d+)*)>>")
ABSTAIN_STEP = "<<no value provided = abstain>>"

_OPERATORS = ("+", "-", "*")


def _apply(left: int, operator: str, right: int) -> int:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    raise ValueError("unsupported operator %r" % (operator,))


def _render_equation(left: int, operator: str, right: int, result: int) -> str:
    return "<<%d %s %d = %d>>" % (left, operator, right, result)


def _render_sorted(positions: Sequence[int]) -> str:
    return "<<sorted: %s>>" % " ".join(str(position) for position in positions)


def numeric_trace(operands: Sequence[int], operators: Sequence[str]) -> Dict[str, Any]:
    """GSM8k-Aug equation steps with the running value.

    ``operands`` is the v9 operand chain (first entry is the start value) and
    ``operators`` the v9 operator list, so ``len(operands) == len(operators) + 1``.
    """

    operand_list = [int(value) for value in operands]
    operator_list = [str(operator) for operator in operators]
    if len(operand_list) < 2 or len(operator_list) != len(operand_list) - 1:
        raise ValueError(
            "need n>=2 operands and n-1 operators, got %d/%d"
            % (len(operand_list), len(operator_list))
        )
    unknown = [operator for operator in operator_list if operator not in _OPERATORS]
    if unknown:
        raise ValueError("unsupported operators %r" % (unknown,))

    running = operand_list[0]
    steps: List[str] = []
    values: List[int] = []
    for operator, operand in zip(operator_list, operand_list[1:]):
        result = _apply(running, operator, operand)
        steps.append(_render_equation(running, operator, operand, result))
        values.append(result)
        running = result
    return {
        "kind": "numeric",
        "steps": steps,
        "values": values,
        "trace": " ".join(steps),
        "final": values[-1],
        "operands": operand_list,
        "operators": operator_list,
    }


def unit_trace(quantity: int, factor: int, converted: int) -> Dict[str, Any]:
    """Single-step conversion trace ``<<quantity * factor = converted>>``."""

    quantity, factor, converted = int(quantity), int(factor), int(converted)
    if quantity * factor != converted:
        raise ValueError(
            "inconsistent conversion: %d * %d != %d" % (quantity, factor, converted)
        )
    step = _render_equation(quantity, "*", factor, converted)
    return {
        "kind": "unit",
        "steps": [step],
        "values": [converted],
        "trace": step,
        "final": converted,
        "quantity": quantity,
        "factor": factor,
    }


def ordering_trace(shuffled: Sequence[int], ordered: Sequence[int]) -> Dict[str, Any]:
    """One compact permutation step for a sorting anchor.

    Final format (see module docstring for the shrink rationale): the step
    lists, for each element of the ascending output, its 1-based position in
    the shuffled input, e.g. ``<<sorted: 4 1 3 2 6 5 8 7>>``. Worst case under
    the pinned Qwen3 tokenizer is 20 tokens (8 items). ``final`` is the
    ordered list.
    """

    shuffled_list = [int(value) for value in shuffled]
    ordered_list = [int(value) for value in ordered]
    if sorted(shuffled_list) != ordered_list:
        raise ValueError("ordered is not the ascending sort of shuffled")

    used = [False] * len(shuffled_list)
    positions: List[int] = []
    for value in ordered_list:
        for index, candidate in enumerate(shuffled_list):
            if not used[index] and candidate == value:
                used[index] = True
                positions.append(index + 1)
                break
    step = _render_sorted(positions)
    return {
        "kind": "ordering",
        "steps": [step],
        "values": [],
        "trace": step,
        "final": ordered_list,
        "shuffled": shuffled_list,
        "positions": positions,
    }


def abstention_trace() -> Dict[str, Any]:
    """Single abstention step; there is no derivable value."""

    return {
        "kind": "abstention",
        "steps": [ABSTAIN_STEP],
        "values": [],
        "trace": ABSTAIN_STEP,
        "final": "abstain",
    }


def _corrupt_value(value: int, key: int) -> int:
    """Key-derived off-by-delta corruption that keeps the digit count.

    Prefers a corrupted value with the same number of decimal digits as the
    true value (plausible process negative), never equal to the true value,
    and never below 1. Defined for the positive values the v9 generators emit.
    """

    preferred_delta = 1 + ((key >> 7) % 9)
    preferred_sign = 1 if ((key >> 4) & 1) else -1
    digits = len(str(value))
    deltas = [preferred_delta] + [d for d in range(1, 10) if d != preferred_delta]
    for delta in deltas:
        for sign in (preferred_sign, -preferred_sign):
            candidate = value + sign * delta
            if candidate != value and candidate >= 1 and len(str(candidate)) == digits:
                return candidate
    return value + 1  # unreachable for value >= 1, kept as a safe fallback


def corrupt_one_step(trace_dict: Dict[str, Any], key: int) -> Dict[str, Any]:
    """Deterministic process-supervision negative: exactly one wrong derivation.

    For equation traces (numeric/unit) the key picks one step, its RESULT is
    corrupted by a key-derived delta (never equal to the true value, digit
    count preserved), and the corruption CASCADES: every later step re-derives
    truthfully from the corrupted running value, so the corrupted trace is
    internally consistent arithmetic except at exactly the corrupted step.
    For the single-step ordering trace, two key-derived adjacent positions of
    the permutation are swapped (the "result" is the permutation). Abstention
    traces carry no value and raise ``ValueError``.
    """

    key = int(key)
    steps = list(trace_dict["steps"])
    if not steps:
        raise ValueError("trace has no steps to corrupt")
    kind = trace_dict.get("kind")

    if kind == "abstention" or steps[0] == ABSTAIN_STEP:
        raise ValueError("abstention traces have no corruptible value")

    sorted_match = SORTED_STEP_RE.fullmatch(steps[0])
    if kind == "ordering" or (sorted_match and len(steps) == 1):
        if sorted_match is None or len(steps) != 1:
            raise ValueError("ordering trace must be a single <<sorted: ...>> step")
        positions = [int(token) for token in sorted_match.group(1).split(" ")]
        if len(positions) < 2:
            raise ValueError("ordering trace too short to corrupt")
        shuffled = trace_dict.get("shuffled")
        if shuffled is None:
            raise ValueError("ordering corruption requires the 'shuffled' field")
        swap_at = key % (len(positions) - 1)
        corrupted = list(positions)
        corrupted[swap_at], corrupted[swap_at + 1] = (
            corrupted[swap_at + 1],
            corrupted[swap_at],
        )
        step = _render_sorted(corrupted)
        final = [shuffled[position - 1] for position in corrupted]
        return {
            "kind": "ordering",
            "steps": [step],
            "values": [],
            "trace": step,
            "corrupted_step_index": 0,
            "final": final,
        }

    parsed: List[Any] = []
    for index, step in enumerate(steps):
        match = EQUATION_STEP_RE.fullmatch(step)
        if match is None:
            raise ValueError("step %d is not an equation step: %r" % (index, step))
        parsed.append(tuple(
            int(group) if position != 1 else group
            for position, group in enumerate(match.groups())
        ))

    corrupt_index = key % len(steps)
    new_steps: List[str] = []
    new_values: List[int] = []
    for index, (left, operator, right, result) in enumerate(parsed):
        if index < corrupt_index:
            new_steps.append(steps[index])
            new_values.append(result)
        elif index == corrupt_index:
            bad = _corrupt_value(result, key)
            new_steps.append(_render_equation(left, operator, right, bad))
            new_values.append(bad)
        else:
            cascaded_left = new_values[-1]
            cascaded_result = _apply(cascaded_left, operator, right)
            new_steps.append(
                _render_equation(cascaded_left, operator, right, cascaded_result)
            )
            new_values.append(cascaded_result)
    return {
        "kind": kind or "numeric",
        "steps": new_steps,
        "values": new_values,
        "trace": " ".join(new_steps),
        "corrupted_step_index": corrupt_index,
        "final": new_values[-1],
    }


def teacher_trace_tokens(trace_dict: Dict[str, Any]) -> Dict[str, Any]:
    """CODI-style teacher supervision: exclude the answer-producing final step.

    Including the final step creates a copy shortcut, so ``teacher_steps`` is
    every step but the last; single-step traces get an empty teacher list and
    ``single_step: True``.
    """

    steps = list(trace_dict["steps"])
    if not steps:
        raise ValueError("trace has no steps")
    return {
        "teacher_steps": steps[:-1],
        "excluded_step": steps[-1],
        "single_step": len(steps) == 1,
    }


def compression_windows(
    token_ids: Sequence[int],
    k: int,
    forbidden_ids: Union[int, Iterable[int]],
) -> List[List[int]]:
    """Partition ``token_ids`` into ``k`` contiguous CoLaR compression windows.

    Window sizes are as equal as possible (they differ by at most 1) and never
    empty; if ``len(token_ids) < k`` a ``ValueError`` is raised -- the caller
    must lower ``k`` or pad the TRACE, never the windows. ``forbidden_ids``
    (an id or an iterable of ids, e.g. the Qwen3 eos/pad alias 151643) must
    not appear anywhere in ``token_ids``; a planted forbidden id raises
    ``ValueError``.
    """

    ids = [int(token) for token in token_ids]
    if k < 1:
        raise ValueError("k must be >= 1, got %d" % k)
    if len(ids) < k:
        raise ValueError(
            "len(token_ids)=%d < k=%d: lower k or pad the trace, never the windows"
            % (len(ids), k)
        )
    if isinstance(forbidden_ids, int):
        forbidden = {int(forbidden_ids)}
    else:
        forbidden = {int(token) for token in forbidden_ids}
    planted = sorted(set(ids) & forbidden)
    if planted:
        raise ValueError("forbidden ids present in token_ids: %r" % (planted,))

    base, remainder = divmod(len(ids), k)
    windows: List[List[int]] = []
    cursor = 0
    for index in range(k):
        size = base + (1 if index < remainder else 0)
        windows.append(ids[cursor:cursor + size])
        cursor += size
    return windows


def validate_trace(
    trace_dict: Dict[str, Any],
    tokenizer: Any,
    max_tokens: int = 30,
    max_steps: int = 3,
) -> Dict[str, Any]:
    """Budget/parse validation under the REAL tokenizer.

    ``n_tokens`` is the token count of the full ``trace`` string with
    ``add_special_tokens=False``; ``ok`` requires the token budget, the step
    budget, every step fully matching ``<<[^<>]+>>``, and the trace being
    exactly the concatenation of the steps.
    """

    steps = list(trace_dict.get("steps", []))
    trace = str(trace_dict.get("trace", ""))
    reasons: List[str] = []
    if not steps:
        reasons.append("no_steps")
    for index, step in enumerate(steps):
        if STEP_RE.fullmatch(step) is None:
            reasons.append("unparseable_step_%d" % index)
    if STEP_RE.findall(trace) != steps:
        reasons.append("trace_steps_mismatch")

    n_tokens = len(tokenizer.encode(trace, add_special_tokens=False))
    n_steps = len(steps)
    if n_tokens > max_tokens:
        reasons.append("token_budget_exceeded:%d>%d" % (n_tokens, max_tokens))
    if n_steps > max_steps:
        reasons.append("step_budget_exceeded:%d>%d" % (n_steps, max_steps))
    return {"ok": not reasons, "n_tokens": n_tokens, "n_steps": n_steps, "reasons": reasons}
