"""Deterministic evaluation for end-to-end expert workflow answers.

This module deliberately does not use a language-model judge.  The evaluation
suite describes observable requirements (roles, product surfaces, UI states,
edge cases, constraints, implementation evidence and deliverables), and the
grader reports exactly which requirements were or were not present.

The grader measures coverage and basic evidence hygiene.  It does not pretend
that keyword coverage is a substitute for human judgement of taste or factual
quality, so subjective production claims still require a separate human audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SUITE_NAME = "hlwm-expert-workflows-v1"
DIMENSIONS = (
    "role_surface_coverage",
    "ui_states_accessibility",
    "edge_cases",
    "constraints",
    "reuse_decision",
    "implementation_tests_deliverables",
    "tool_schema",
    "unsupported_claims",
    "answer_completion",
)
DEFAULT_PASS_SCORE = 0.80
DEFAULT_DIMENSION_FLOOR = 0.60

WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:todo|tbd|lorem ipsum|coming soon|fill (?:this|it) in|placeholder)\b|<insert[^>]*>",
    re.IGNORECASE,
)

# Past-tense claims of actions that require an external tool result are unsafe
# when the response contains no attached evidence marker.
EXTERNAL_ACTION_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:opened|browsed|visited|inspected|downloaded|ran|executed|"
    r"tested|verified|deployed|measured|benchmarked)\b",
    re.IGNORECASE,
)
TEST_SUCCESS_PATTERN = re.compile(
    r"\b(?:all\s+)?tests?\s+(?:pass|passed)|\bverified\s+(?:as\s+)?working\b|"
    r"\bdeployed\s+successfully\b",
    re.IGNORECASE,
)
GUARANTEE_PATTERN = re.compile(
    r"\b(?:guaranteed?|will definitely|cannot fail|100%\s+(?:increase|improvement|return|conversion))\b",
    re.IGNORECASE,
)
UNSUPPORTED_UPLIFT_PATTERN = re.compile(
    r"\b(?:increase|improve|boost|reduce|grow)(?:s|d)?\b[^.!?\n]{0,70}\b\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)
COMPARATIVE_PATTERN = re.compile(
    r"\b(?:industry[- ]leading|best[- ]in[- ]class|outperforms?|beats all|market[- ]leading)\b",
    re.IGNORECASE,
)
EVIDENCE_MARKERS = (
    "tool result",
    "tool_output",
    "attached log",
    "provided log",
    "test output",
    "benchmark output",
    "source:",
    "evidence:",
    "according to the supplied",
)
HYPOTHESIS_MARKERS = (
    "target",
    "hypothesis",
    "estimate",
    "scenario",
    "baseline",
    "example",
    "assumption",
    "illustrative",
)


def normalize_text(value: str) -> str:
    return " ".join(WORD_PATTERN.findall(str(value).lower()))


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()


def _shingles(value: str, size: int = 8) -> Set[str]:
    words = normalize_text(value).split()
    if len(words) < size:
        return set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _phrase_present(normalized_answer: str, phrase: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    return bool(normalized_phrase and " %s " % normalized_phrase in " %s " % normalized_answer)


def _requirement_groups(value: Any) -> List[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("rubric requirement collection must be a list")
    groups: List[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each rubric requirement must be an object")
        label = str(item.get("label") or "").strip()
        alternatives = item.get("any")
        if not label or not isinstance(alternatives, list) or not alternatives:
            raise ValueError("each rubric requirement needs label and nonempty any")
        if not all(isinstance(option, str) and option.strip() for option in alternatives):
            raise ValueError("rubric alternatives must be nonempty strings")
        groups.append(item)
    return groups


def _grade_groups(answer: str, groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not groups:
        return {"enabled": False, "score": 1.0, "matched": [], "missing": []}
    normalized = normalize_text(answer)
    matched: List[str] = []
    missing: List[str] = []
    for group in groups:
        label = str(group["label"])
        if any(_phrase_present(normalized, str(option)) for option in group["any"]):
            matched.append(label)
        else:
            missing.append(label)
    return {
        "enabled": True,
        "score": len(matched) / len(groups),
        "matched": matched,
        "missing": missing,
        "required": len(groups),
    }


def _mean_enabled(parts: Sequence[Mapping[str, Any]]) -> float:
    enabled = [float(part["score"]) for part in parts if part.get("enabled")]
    return sum(enabled) / len(enabled) if enabled else 1.0


def _dimension_result(parts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    enabled = any(part.get("enabled") for part in parts.values())
    return {
        "enabled": enabled,
        "score": _mean_enabled(list(parts.values())),
        "parts": dict(parts),
    }


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _grade_tool_schema(answer: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    if not spec.get("required", False):
        return {"enabled": False, "score": 1.0, "valid_objects": 0, "errors": []}
    required_fields = [
        str(field)
        for field in spec.get(
            "required_fields",
            ["tool", "purpose", "inputs", "success_criteria", "fallback"],
        )
    ]
    parsed: List[Any] = []
    errors: List[str] = []
    for index, block in enumerate(JSON_FENCE_PATTERN.findall(answer)):
        try:
            parsed.append(json.loads(block))
        except json.JSONDecodeError as error:
            errors.append("JSON block %d is invalid: %s" % (index + 1, error.msg))
    valid_objects: List[Mapping[str, Any]] = []
    for value in parsed:
        for candidate in _walk_objects(value):
            if not all(field in candidate for field in required_fields):
                continue
            if all(_nonempty(candidate[field]) for field in required_fields):
                valid_objects.append(candidate)
    minimum_tools = max(1, int(spec.get("minimum_tools", 1)))
    score = min(1.0, len(valid_objects) / minimum_tools)
    if not parsed:
        errors.append("no fenced JSON tool plan found")
    elif not valid_objects:
        errors.append("no tool object contains all required nonempty fields")
    return {
        "enabled": True,
        "score": score,
        "valid_objects": len(valid_objects),
        "minimum_tools": minimum_tools,
        "required_fields": required_fields,
        "errors": errors,
    }


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _sentences(value: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", value) if part.strip()]


def _has_evidence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in EVIDENCE_MARKERS)


def _has_hypothesis_boundary(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in HYPOTHESIS_MARKERS)


def _grade_unsupported_claims(answer: str, rubric: Mapping[str, Any]) -> Dict[str, Any]:
    hits: List[Dict[str, str]] = []
    for sentence in _sentences(answer):
        if EXTERNAL_ACTION_PATTERN.search(sentence) and not _has_evidence(sentence):
            hits.append({"type": "unattested_external_action", "text": sentence[:240]})
        if TEST_SUCCESS_PATTERN.search(sentence) and not _has_evidence(sentence):
            hits.append({"type": "unattested_test_or_deploy_success", "text": sentence[:240]})
        if GUARANTEE_PATTERN.search(sentence):
            hits.append({"type": "guaranteed_outcome", "text": sentence[:240]})
        if UNSUPPORTED_UPLIFT_PATTERN.search(sentence) and not _has_hypothesis_boundary(sentence):
            hits.append({"type": "unsupported_numeric_uplift", "text": sentence[:240]})
        if COMPARATIVE_PATTERN.search(sentence) and not _has_evidence(sentence):
            hits.append({"type": "unsupported_comparative", "text": sentence[:240]})
    normalized = normalize_text(answer)
    for item in rubric.get("forbidden_terms", []):
        term = str(item)
        if _phrase_present(normalized, term):
            hits.append({"type": "forbidden_term", "text": term})
    for item in rubric.get("forbidden_patterns", []):
        pattern = re.compile(str(item), re.IGNORECASE)
        match = pattern.search(answer)
        if match:
            hits.append({"type": "task_forbidden_claim", "text": match.group(0)[:240]})
    return {"enabled": True, "score": 1.0 if not hits else 0.0, "hits": hits}


def _grade_completion(answer: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    words = WORD_PATTERN.findall(answer)
    minimum_words = max(1, int(spec.get("minimum_words", 300)))
    word_score = min(1.0, len(words) / minimum_words)
    headings = _grade_groups(answer, _requirement_groups(spec.get("required_sections", [])))
    placeholder_hits = [match.group(0) for match in PLACEHOLDER_PATTERN.finditer(answer)]
    stripped = answer.rstrip()
    proper_end = bool(
        stripped
        and (
            stripped.endswith((".", "!", "?", "]", "}", ")", "```"))
            and not re.search(r"\b(?:and|or|to|with|because|including|such as)\s*[.!?]?\s*$", stripped, re.I)
        )
    )
    score = (
        0.35 * word_score
        + 0.45 * float(headings["score"])
        + 0.10 * (1.0 if not placeholder_hits else 0.0)
        + 0.10 * (1.0 if proper_end else 0.0)
    )
    return {
        "enabled": True,
        "score": score,
        "word_count": len(words),
        "minimum_words": minimum_words,
        "word_score": word_score,
        "sections": headings,
        "placeholder_hits": placeholder_hits,
        "proper_end": proper_end,
    }


def validate_task(task: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in ("task_id", "suite", "split", "category", "prompt", "rubric"):
        if not _nonempty(task.get(field)):
            errors.append("missing or empty %s" % field)
    if task.get("suite") != SUITE_NAME:
        errors.append("suite must be %s" % SUITE_NAME)
    if task.get("split") != "evaluation" or task.get("training_eligible") is not False:
        errors.append("task must be evaluation-only and training_eligible=false")
    if not str(task.get("task_id", "")).startswith("ewf-v1-"):
        errors.append("task_id must use ewf-v1- prefix")
    canary = str(task.get("canary") or "")
    if not canary or canary not in str(task.get("prompt") or ""):
        errors.append("prompt must contain its unique canary")
    rubric = task.get("rubric")
    if isinstance(rubric, Mapping):
        for key in (
            "roles",
            "surfaces",
            "ui_states",
            "accessibility",
            "edge_cases",
            "constraints",
            "reuse_decision",
            "implementation",
            "tests",
            "deliverables",
        ):
            try:
                _requirement_groups(rubric.get(key, []))
            except ValueError as error:
                errors.append("%s: %s" % (key, error))
        if not isinstance(rubric.get("tool_schema", {}), Mapping):
            errors.append("tool_schema must be an object")
        if not isinstance(rubric.get("completion", {}), Mapping):
            errors.append("completion must be an object")
    elif rubric is not None:
        errors.append("rubric must be an object")
    return errors


def load_suite(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_fingerprints: Set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            errors = validate_task(task)
            if errors:
                raise ValueError("%s:%d: %s" % (path, line_number, "; ".join(errors)))
            task_id = str(task["task_id"])
            fingerprint = prompt_fingerprint(str(task["prompt"]))
            if task_id in seen_ids:
                raise ValueError("duplicate task_id %s" % task_id)
            if fingerprint in seen_fingerprints:
                raise ValueError("duplicate normalized prompt %s" % task_id)
            seen_ids.add(task_id)
            seen_fingerprints.add(fingerprint)
            tasks.append(task)
    if not tasks:
        raise ValueError("evaluation suite is empty")
    return tasks


def grade_answer(task: Mapping[str, Any], answer: str) -> Dict[str, Any]:
    errors = validate_task(task)
    if errors:
        raise ValueError("invalid task: %s" % "; ".join(errors))
    rubric = task["rubric"]
    assert isinstance(rubric, Mapping)
    dimensions: Dict[str, Dict[str, Any]] = {}
    dimensions["role_surface_coverage"] = _dimension_result(
        {
            "roles": _grade_groups(answer, _requirement_groups(rubric.get("roles", []))),
            "surfaces": _grade_groups(answer, _requirement_groups(rubric.get("surfaces", []))),
        }
    )
    dimensions["ui_states_accessibility"] = _dimension_result(
        {
            "ui_states": _grade_groups(answer, _requirement_groups(rubric.get("ui_states", []))),
            "accessibility": _grade_groups(
                answer, _requirement_groups(rubric.get("accessibility", []))
            ),
        }
    )
    dimensions["edge_cases"] = _dimension_result(
        {"edge_cases": _grade_groups(answer, _requirement_groups(rubric.get("edge_cases", [])))}
    )
    dimensions["constraints"] = _dimension_result(
        {"constraints": _grade_groups(answer, _requirement_groups(rubric.get("constraints", [])))}
    )
    dimensions["reuse_decision"] = _dimension_result(
        {
            "reuse_decision": _grade_groups(
                answer, _requirement_groups(rubric.get("reuse_decision", []))
            )
        }
    )
    dimensions["implementation_tests_deliverables"] = _dimension_result(
        {
            "implementation": _grade_groups(
                answer, _requirement_groups(rubric.get("implementation", []))
            ),
            "tests": _grade_groups(answer, _requirement_groups(rubric.get("tests", []))),
            "deliverables": _grade_groups(
                answer, _requirement_groups(rubric.get("deliverables", []))
            ),
        }
    )
    dimensions["tool_schema"] = _grade_tool_schema(answer, rubric.get("tool_schema", {}))
    dimensions["unsupported_claims"] = _grade_unsupported_claims(answer, rubric)
    dimensions["answer_completion"] = _grade_completion(answer, rubric.get("completion", {}))

    weights = {str(key): float(value) for key, value in rubric.get("weights", {}).items()}
    enabled = [name for name in DIMENSIONS if dimensions[name].get("enabled", True)]
    total_weight = sum(weights.get(name, 1.0) for name in enabled)
    overall = (
        sum(dimensions[name]["score"] * weights.get(name, 1.0) for name in enabled) / total_weight
        if total_weight
        else 0.0
    )
    pass_score = float(rubric.get("pass_score", DEFAULT_PASS_SCORE))
    dimension_floor = float(rubric.get("dimension_floor", DEFAULT_DIMENSION_FLOOR))
    critical = [
        str(value)
        for value in rubric.get(
            "critical_dimensions",
            ["constraints", "tool_schema", "unsupported_claims", "answer_completion"],
        )
    ]
    floor_failures = [
        name for name in enabled if float(dimensions[name]["score"]) < dimension_floor
    ]
    critical_failures = [
        name for name in critical if float(dimensions.get(name, {}).get("score", 0.0)) < dimension_floor
    ]
    passed = overall >= pass_score and not floor_failures and not critical_failures
    return {
        "suite": task["suite"],
        "task_id": task["task_id"],
        "category": task["category"],
        "passed": passed,
        "score": round(overall, 6),
        "pass_score": pass_score,
        "dimension_floor": dimension_floor,
        "floor_failures": floor_failures,
        "critical_failures": critical_failures,
        "dimensions": dimensions,
    }


def prompt_overlap_report(
    tasks: Sequence[Mapping[str, Any]],
    training_prompts: Iterable[Tuple[str, str]],
    *,
    shingle_size: int = 8,
    coverage_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Report exact or substantial prompt overlap with training seeds.

    Coverage is measured against the shorter shingle set.  This catches a
    training prompt copied into a longer evaluation prompt while tolerating
    ordinary shared phrases such as "implementation plan".
    """

    prepared_training = [
        (str(identifier), str(prompt), _shingles(str(prompt), shingle_size))
        for identifier, prompt in training_prompts
        if str(prompt).strip()
    ]
    flags: List[Dict[str, Any]] = []
    for task in tasks:
        prompt = str(task["prompt"])
        normalized = normalize_text(prompt)
        eval_shingles = _shingles(prompt, shingle_size)
        for identifier, training_prompt, training_shingles in prepared_training:
            training_normalized = normalize_text(training_prompt)
            exact = normalized == training_normalized
            intersection = len(eval_shingles & training_shingles)
            denominator = max(1, min(len(eval_shingles), len(training_shingles)))
            coverage = intersection / denominator
            if exact or (intersection >= 4 and coverage >= coverage_threshold):
                flags.append(
                    {
                        "task_id": task["task_id"],
                        "training_id": identifier,
                        "exact": exact,
                        "matching_shingles": intersection,
                        "shorter_prompt_coverage": round(coverage, 6),
                    }
                )
    return {
        "evaluation_tasks": len(tasks),
        "training_prompts": len(prepared_training),
        "flags": flags,
        "clean": not flags,
    }


def _load_answers(path: Path) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            answer = row.get("answer")
            if not task_id or not isinstance(answer, str):
                raise ValueError("%s:%d needs task_id and string answer" % (path, line_number))
            if task_id in answers:
                raise ValueError("duplicate answer for %s" % task_id)
            answers[task_id] = answer
    return answers


def grade_answers(tasks: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> Dict[str, Any]:
    results = [grade_answer(task, answers.get(str(task["task_id"]), "")) for task in tasks]
    passed = sum(bool(result["passed"]) for result in results)
    return {
        "suite": SUITE_NAME,
        "tasks": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_score": (
            sum(float(result["score"]) for result in results) / len(results) if results else 0.0
        ),
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grade held-out expert-workflow answers")
    parser.add_argument("--suite", type=Path, required=True, help="Evaluation tasks JSONL")
    parser.add_argument("--answers", type=Path, required=True, help="JSONL with task_id and answer")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    arguments = parser.parse_args(argv)
    tasks = load_suite(arguments.suite)
    report = grade_answers(tasks, _load_answers(arguments.answers))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] == report["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
