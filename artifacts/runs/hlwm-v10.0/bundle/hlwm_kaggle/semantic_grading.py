"""Deterministic semantic graders for the narrow verified HLWM probes.

These graders intentionally cover only answer classes that can be checked
without a language model or external facts. They accept harmless wording
variation while still rejecting wrong numbers, units, ordering, and guessed
answers when evidence is absent.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CODE_FENCE = re.compile(r"```[ \t]*(?:python|py|sql)?[ \t]*\r?\n(.*?)```", re.DOTALL)
_ABSTENTION_PATTERNS = (
    re.compile(r"\bcannot (?:be )?determin(?:e|ed)\b"),
    re.compile(r"\bcan(?:not|'t) determine\b"),
    re.compile(r"\bunable to determine\b"),
    re.compile(r"\binsufficient (?:information|evidence|context|data)\b"),
    re.compile(r"\bnot enough (?:information|evidence|context|data)\b"),
    re.compile(r"\b(?:information|evidence|value|percentage) (?:is|was) not (?:provided|available|given)\b"),
    re.compile(r"\bunknown from (?:the )?(?:provided|available) (?:information|evidence|context|data)\b"),
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("−", "-").split())


def extract_numbers(text: str) -> List[float]:
    values: List[float] = []
    for match in _NUMBER.findall(str(text).replace("−", "-")):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def contains_safe_abstention(text: str) -> bool:
    normalized = normalize_text(text)
    return any(pattern.search(normalized) for pattern in _ABSTENTION_PATTERNS)


def _close(actual: float, expected: float, absolute: float, relative: float) -> bool:
    return math.isclose(actual, expected, abs_tol=absolute, rel_tol=relative)


def extract_code_block(text: str) -> str:
    """Return the first fenced code block, or the raw text when unfenced."""

    match = _CODE_FENCE.search(str(text))
    if match:
        return match.group(1).strip()
    return str(text).strip()


def normalize_program_output(text: str) -> str:
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def normalize_sql(text: str) -> str:
    candidate = extract_code_block(text)
    candidate = candidate.strip().rstrip(";").strip()
    return " ".join(candidate.lower().split())


def run_python_program(
    program: str,
    stdin_text: str = "",
    timeout_seconds: float = 6.0,
) -> Tuple[bool, str, str]:
    """Execute one isolated python subprocess; returns (ok, stdout, detail).

    ``-I`` runs isolated mode (no site customization, no user path). The
    subprocess boundary plus a hard timeout is the sandbox this prototype
    uses for grading generated code; graded programs are short, offline
    exercises and the audit host is a disposable notebook session.
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except (OSError, ValueError) as error:
        return False, "", "spawn_error: %s" % error
    detail = "" if completed.returncode == 0 else (
        (completed.stderr or "").strip().splitlines() or ["exit %d" % completed.returncode]
    )[-1][:300]
    return completed.returncode == 0, completed.stdout or "", detail


def grade_python_tests(
    text: str,
    tests: Sequence[str],
    setup: str = "",
    timeout_seconds: float = 6.0,
) -> Dict[str, Any]:
    code = extract_code_block(text)
    program_parts = [part for part in (setup, code) if part]
    program_parts.extend(str(test) for test in tests)
    ok, _stdout, detail = run_python_program(
        "\n\n".join(program_parts), timeout_seconds=timeout_seconds
    )
    return {"correct": bool(ok), "tests_run": len(tests), "detail": detail}


def grade_io_tests(
    text: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    timeout_seconds: float = 6.0,
    max_cases: int = 3,
) -> Dict[str, Any]:
    code = extract_code_block(text)
    cases = list(zip(inputs, outputs))[: max(1, int(max_cases))]
    if not cases:
        return {"correct": False, "cases_run": 0, "detail": "no io cases"}
    for index, (case_input, case_output) in enumerate(cases):
        ok, stdout, detail = run_python_program(
            code, stdin_text=str(case_input), timeout_seconds=timeout_seconds
        )
        if not ok:
            return {"correct": False, "cases_run": index + 1, "detail": detail or "runtime error"}
        if normalize_program_output(stdout) != normalize_program_output(case_output):
            return {"correct": False, "cases_run": index + 1, "detail": "wrong output on case %d" % index}
    return {"correct": True, "cases_run": len(cases), "detail": ""}


def grade_semantic_answer(text: str, spec: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Grade one answer against a structured, programmatically verified spec."""

    spec = dict(spec or {})
    grader = str(spec.get("type", "none")).lower()
    normalized = normalize_text(text)
    numbers = extract_numbers(text)
    result: Dict[str, Any] = {
        "grader": grader,
        "correct": True,
        "normalized": normalized,
        "numbers": numbers,
    }

    if grader == "none":
        result["reason"] = "no structured grader for this record"
        return result

    if grader in ("numeric", "unit"):
        expected = float(spec["expected"])
        absolute = float(spec.get("absolute_tolerance", 1.0e-9))
        relative = float(spec.get("relative_tolerance", 1.0e-9))
        numeric_ok = bool(numbers) and _close(numbers[-1], expected, absolute, relative)
        required_unit = normalize_text(str(spec.get("unit", "")))
        unit_ok = not required_unit or bool(
            re.search(r"\b%s\b" % re.escape(required_unit), normalized)
        )
        result.update(
            {
                "correct": bool(numeric_ok and unit_ok),
                "expected": expected,
                "numeric_ok": numeric_ok,
                "unit_ok": unit_ok,
                "reason": "last stated value and required unit must match",
            }
        )
        return result

    if grader == "ordering":
        expected_values = [float(value) for value in spec.get("expected", [])]
        actual_suffix = numbers[-len(expected_values) :] if expected_values else []
        correct = len(actual_suffix) == len(expected_values) and all(
            _close(actual, expected, 1.0e-9, 1.0e-9)
            for actual, expected in zip(actual_suffix, expected_values)
        )
        result.update(
            {
                "correct": bool(correct),
                "expected": expected_values,
                "actual_suffix": actual_suffix,
                "reason": "final numeric sequence must match the requested order",
            }
        )
        return result

    if grader == "python_tests":
        outcome = grade_python_tests(
            text,
            [str(item) for item in spec.get("tests", [])],
            setup=str(spec.get("setup", "")),
            timeout_seconds=float(spec.get("time_limit_seconds", 6.0)),
        )
        result.update(outcome)
        result["reason"] = "extracted program must pass the executed test list"
        return result

    if grader == "io_tests":
        outcome = grade_io_tests(
            text,
            [str(item) for item in spec.get("inputs", [])],
            [str(item) for item in spec.get("outputs", [])],
            timeout_seconds=float(spec.get("time_limit_seconds", 6.0)),
            max_cases=int(spec.get("max_cases", 3)),
        )
        result.update(outcome)
        result["reason"] = "extracted program must reproduce the official io pairs"
        return result

    if grader == "sql_exact":
        expected_sql = normalize_sql(str(spec.get("expected", "")))
        actual_sql = normalize_sql(text)
        result.update(
            {
                "correct": bool(expected_sql) and actual_sql == expected_sql,
                "expected": expected_sql,
                "actual": actual_sql,
                "reason": "normalized sql must match the official gold query",
            }
        )
        return result

    if grader == "abstention":
        abstained = contains_safe_abstention(text)
        forbid_numbers = bool(spec.get("forbid_numbers", True))
        no_guess = not numbers if forbid_numbers else True
        result.update(
            {
                "correct": bool(abstained and no_guess),
                "abstention_detected": abstained,
                "unsupported_number_absent": no_guess,
                "reason": "must explicitly state missing evidence without guessing",
            }
        )
        return result

    raise ValueError("unsupported semantic grader type: %s" % grader)


__all__ = [
    "contains_safe_abstention",
    "extract_code_block",
    "extract_numbers",
    "grade_io_tests",
    "grade_python_tests",
    "grade_semantic_answer",
    "normalize_program_output",
    "normalize_sql",
    "normalize_text",
    "run_python_program",
]
