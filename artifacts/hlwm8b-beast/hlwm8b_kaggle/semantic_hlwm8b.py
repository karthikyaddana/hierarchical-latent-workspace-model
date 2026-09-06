from __future__ import annotations

import re
from typing import Any, Mapping


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return NUMBER.findall(str(text).replace(",", ""))


def grade_programmatic(task_type: str, candidate: str, reference: str) -> bool:
    candidate_lower = str(candidate).strip().lower()
    reference_lower = str(reference).strip().lower()
    if not candidate_lower:
        return False
    if task_type in {"numeric", "unit"}:
        expected = _numbers(reference_lower)
        observed = _numbers(candidate_lower)
        return bool(expected and observed and observed[-1] == expected[-1])
    if task_type == "ordering":
        return _numbers(candidate_lower) == _numbers(reference_lower)
    if task_type == "abstention":
        uncertainty = (
            "cannot be determined",
            "cannot determine",
            "not enough information",
            "insufficient information",
            "not provided",
        )
        return any(phrase in candidate_lower for phrase in uncertainty)
    raise ValueError("unsupported programmatic task type %s" % task_type)


def grade_row(row: Mapping[str, Any], candidate: str) -> bool:
    task_type = str(row.get("task_type") or "")
    if not task_type:
        raise ValueError("on-policy grading requires a programmatic task_type")
    return grade_programmatic(task_type, candidate, str(row["chosen"]))

