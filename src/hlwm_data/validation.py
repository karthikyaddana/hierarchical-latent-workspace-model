from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from jsonschema import Draft202012Validator

from .language import classify_language, contains_blocked_script


SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{20,}\b"),
]
BASE64_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")
PRIVATE_REASONING_PATTERNS = [
    re.compile(r"<\s*/?\s*\|?think\|?\s*>", re.IGNORECASE),
    re.compile(r"\bhidden chain[- ]of[- ]thought\b", re.IGNORECASE),
    re.compile(r"\binternal reasoning:\s", re.IGNORECASE),
]
LITERAL_SECRET_REQUIREMENT_PATTERN = re.compile(
    r"\b(?:strong|plausible|literal|actual)\s+(?:jwt[_ -]?secret|secret|password|api[_ -]?key|access[_ -]?token)\b"
    r"|\b(?:jwt[_ -]?secret|secret|password|api[_ -]?key|access[_ -]?token)\b.{0,60}"
    r"\bat least\s+\d+\s+characters\b",
    re.IGNORECASE,
)
NON_SOLVER_LANE_ID_PATTERN = re.compile(r"(?:^|-)(?:verifier|validator|reviewer|auditor)(?:-|$)")
REVIEW_ONLY_RESPONSIBILITY_PATTERN = re.compile(
    r"\b(?:review|inspect|audit|validate|verify|confirm|check|evaluate)\b",
    re.IGNORECASE,
)
CONSTRUCTIVE_RESPONSIBILITY_PATTERN = re.compile(
    r"\b(?:construct|derive|design|produce|develop|build|formulate|model|calculate|"
    r"diagnose|specify|write|create|identify|propose)\b",
    re.IGNORECASE,
)
UNATTESTED_OUTCOME_CLAIM_PATTERN = re.compile(
    r"\b(?:(?:compiles?|builds?|deploys?)\s+(?:successfully|cleanly|without (?:an? )?errors?)|"
    r"runs? successfully|tests? pass(?:es|ed)?|(?:is|are) valid json schema|"
    r"sla (?:is )?met|benchmark (?:shows?|proves?))\b",
    re.IGNORECASE,
)


def source_support_spans(chunk_id: str, text: str) -> List[Dict[str, str]]:
    """Create deterministic, exact normalized source windows for blueprint citations."""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    words = list(re.finditer(r"\S+", normalized))
    spans: List[Dict[str, str]] = []
    start = 0
    while start < len(words) and len(spans) < 24:
        end = min(len(words), start + 60)
        while end > start + 1 and words[end - 1].end() - words[start].start() > 600:
            end -= 1
        value = normalized[words[start].start() : words[end - 1].end()]
        if len(value) >= 20:
            spans.append(
                {
                    "span_id": "%s::span-%03d" % (chunk_id, len(spans) + 1),
                    "text": value,
                }
            )
        if end >= len(words):
            break
        start = max(start + 1, end - 10)
    return spans


def load_schema(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_errors(value: Dict[str, Any], schema_path: Path) -> List[str]:
    validator = Draft202012Validator(load_schema(schema_path))
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    return [
        "%s: %s" % (".".join(str(part) for part in error.absolute_path) or "$", error.message)
        for error in errors
    ]


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _collect_ids(value: Any) -> Set[str]:
    result: Set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if (key == "id" or key.endswith("_id")) and isinstance(child, str):
                result.add(child)
            result.update(_collect_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_ids(child))
    return result


def _contains_possible_secret(value: str) -> bool:
    if any(pattern.search(value) for pattern in SECRET_PATTERNS):
        return True
    for match in BASE64_TOKEN_PATTERN.finditer(value):
        token = match.group(0).rstrip("=")
        # Git object IDs, checksums, and sentinel values are identifiers, not
        # credentials. The old broad Base64 regex mislabeled forty-zero SHAs.
        if re.fullmatch(r"[0-9a-fA-F]{40,128}", token) or len(set(token)) < 8:
            continue
        has_lower = any(character.islower() for character in token)
        has_upper = any(character.isupper() for character in token)
        has_digit_or_symbol = any(character.isdigit() or character in "+/" for character in token)
        if has_lower and has_upper and has_digit_or_symbol:
            return True
    return False


def _episode_input_reference_ids(episode: Dict[str, Any]) -> Set[str]:
    episode_input = episode.get("input", {})
    result = {"input.user_request"}
    result.update(
        "input.context[%d]" % index for index, _ in enumerate(episode_input.get("context", []))
    )
    result.update(
        "input.constraints[%d]" % index
        for index, _ in enumerate(episode_input.get("constraints", []))
    )
    return result


def episode_invariant_errors(episode: Dict[str, Any], known_chunk_ids: Set[str]) -> List[str]:
    errors: List[str] = []
    strings = list(_all_strings(episode))
    combined = "\n".join(strings)
    if "\N{REPLACEMENT CHARACTER}" in combined:
        errors.append("corrupted Unicode replacement character found")
    if _contains_possible_secret(combined):
        errors.append("possible secret or credential found")
    for pattern in PRIVATE_REASONING_PATTERNS:
        if pattern.search(combined):
            errors.append("private chain-of-thought marker found")
            break
    if contains_blocked_script(combined):
        errors.append("non-English script found in an English-only episode")
    visible_language_sample = "\n".join(
        str(value)
        for value in (
            episode.get("input", {}).get("user_request", ""),
            episode.get("frame", {}).get("objective", ""),
            episode.get("integration", {}).get("published_answer", ""),
        )
        if value
    )
    if visible_language_sample:
        decision = classify_language(visible_language_sample, expected="en", minimum_confidence=0.70)
        if not decision.accepted:
            errors.append(
                "episode natural-language output is not English (%s, %.3f)"
                % (decision.language, decision.confidence)
            )

    selected = set(episode.get("routing", {}).get("selected_lane_ids", []))
    lane_ids = {lane.get("lane_id") for lane in episode.get("lanes", []) if lane.get("lane_id")}
    if selected != lane_ids:
        errors.append("selected_lane_ids must exactly match generated lane IDs")
    if len(lane_ids) < 2:
        errors.append("at least two distinct private lanes are required")

    referenced_chunks: Set[str] = set()
    for source_ref in episode.get("source_refs", []):
        referenced_chunks.update(source_ref.get("chunk_ids", []))
    unknown_chunks = referenced_chunks - known_chunk_ids
    if unknown_chunks:
        errors.append("unknown source chunk IDs: %s" % sorted(unknown_chunks)[:5])

    available_ids = _collect_ids(episode) | referenced_chunks | _episode_input_reference_ids(episode)
    claim_ids: Set[str] = set()
    for lane in episode.get("lanes", []):
        route = lane.get("route", [])
        if not route or route[0] not in {"root", "generalist", "root_generalist"}:
            errors.append("lane %s route must begin at the root/generalist" % lane.get("lane_id"))
        for claim in lane.get("claims", []):
            claim_id = claim.get("claim_id")
            if claim_id:
                claim_ids.add(claim_id)
            refs = set(claim.get("evidence_refs", []))
            if not refs:
                errors.append("claim %s has no evidence" % claim_id)
            missing = refs - available_ids
            if missing:
                errors.append("claim %s cites missing evidence %s" % (claim_id, sorted(missing)[:3]))

    verification_counts = Counter(
        item.get("claim_id") for item in episode.get("verification", []) if item.get("claim_id")
    )
    verified_ids = set(verification_counts)
    missing_verification = claim_ids - verified_ids
    if missing_verification:
        errors.append("claims missing verification: %s" % sorted(missing_verification)[:5])
    duplicate_verification = sorted(claim_id for claim_id, count in verification_counts.items() if count != 1)
    if duplicate_verification:
        errors.append("claims must be verified exactly once: %s" % duplicate_verification[:5])
    unknown_verification = verified_ids - claim_ids
    if unknown_verification:
        errors.append("verification references unknown claims: %s" % sorted(unknown_verification)[:5])

    blocking_verdicts = sorted(
        str(item.get("claim_id", "unknown"))
        for item in episode.get("verification", [])
        if item.get("verdict") != "supported"
    )
    if episode.get("commitment", {}).get("decision") == "publish" and blocking_verdicts:
        errors.append(
            "publish commitment contains refuted or insufficient claims: %s"
            % blocking_verdicts[:5]
        )

    summaries = episode.get("barrier", {}).get("lane_summaries", [])
    summary_ids = {item.get("lane_id") for item in summaries if isinstance(item, dict)}
    if summary_ids != lane_ids:
        errors.append("barrier must contain exactly one identified summary per lane")
    open_claims = [str(item) for item in episode.get("barrier", {}).get("open_claims", []) if str(item)]
    if episode.get("commitment", {}).get("decision") == "publish" and open_claims:
        errors.append("publish commitment contains open barrier claims: %s" % open_claims[:5])

    for run in episode.get("tool_runs", []):
        if run.get("status") in {"executed", "provided_evidence"} and not run.get("evidence_refs"):
            errors.append("tool run %s claims evidence without evidence_refs" % run.get("run_id"))
    for pair in episode.get("continuation_pairs", []):
        if pair.get("eligible") and not pair.get("evidence_refs"):
            errors.append("eligible continuation pair %s lacks evidence" % pair.get("pair_id"))

    return errors


def blueprint_invariant_errors(
    blueprint: Dict[str, Any],
    known_chunk_ids: Set[str],
    expected_identity: Optional[Dict[str, str]] = None,
    known_chunk_texts: Optional[Mapping[str, str]] = None,
) -> List[str]:
    errors: List[str] = []
    combined = "\n".join(_all_strings(blueprint))
    if "\N{REPLACEMENT CHARACTER}" in combined:
        errors.append("blueprint contains corrupted Unicode")
    if _contains_possible_secret(combined):
        errors.append("blueprint contains a possible secret or credential")
    task_text = "\n".join(_all_strings(blueprint.get("task", {})))
    if LITERAL_SECRET_REQUIREMENT_PATTERN.search(task_text):
        errors.append(
            "blueprint must not require a literal secret value; require a runtime-generation placeholder"
        )
    if any(pattern.search(combined) for pattern in PRIVATE_REASONING_PATTERNS):
        errors.append("blueprint contains a private chain-of-thought marker")
    if contains_blocked_script(combined):
        errors.append("blueprint contains a blocked non-English script")

    if expected_identity:
        for key, expected in expected_identity.items():
            if str(blueprint.get(key, "")) != str(expected):
                errors.append("blueprint %s does not match scheduled job" % key)

    task = blueprint.get("task", {})
    constraints = task.get("constraints", [])
    constraint_ids = [str(item.get("constraint_id", "")) for item in constraints]
    if len(constraint_ids) != len(set(constraint_ids)):
        errors.append("blueprint constraint IDs must be unique")

    premises = blueprint.get("premises", {})
    source_premises = premises.get("source_backed", [])
    scenario_premises = premises.get("scenario", [])
    source_ids = [str(item.get("premise_id", "")) for item in source_premises]
    scenario_ids = [str(item.get("premise_id", "")) for item in scenario_premises]
    premise_ids = source_ids + scenario_ids
    if len(premise_ids) != len(set(premise_ids)):
        errors.append("blueprint premise IDs must be unique")
    for premise in source_premises:
        refs = {str(item) for item in premise.get("evidence_refs", [])}
        unknown = refs - known_chunk_ids
        if unknown:
            errors.append(
                "source premise %s cites unknown chunks %s"
                % (premise.get("premise_id"), sorted(unknown)[:5])
            )
        if known_chunk_texts is not None and not unknown:
            quote = re.sub(r"\s+", " ", str(premise.get("support_quote", ""))).strip()
            span_id = str(premise.get("support_span_id", ""))
            spans = {
                item["span_id"]: (ref, item["text"])
                for ref in refs
                for item in source_support_spans(ref, str(known_chunk_texts.get(ref, "")))
            }
            supported = span_id in spans and spans[span_id][1] == quote
            if quote and not supported:
                errors.append(
                    "source premise %s support span is not an exact pipeline-issued span from its cited chunks"
                    % premise.get("premise_id")
                )

    lanes = blueprint.get("lanes", [])
    lane_ids = [str(item.get("lane_id", "")) for item in lanes]
    artifact_ids = [
        str(artifact_id)
        for item in lanes
        for artifact_id in item.get("artifact_ids", [])
    ]
    artifact_owner = {
        str(artifact_id): str(item.get("lane_id", ""))
        for item in lanes
        for artifact_id in item.get("artifact_ids", [])
    }
    if len(lane_ids) != len(set(lane_ids)):
        errors.append("blueprint lane IDs must be unique")
    solver_lane_ids = [
        lane_id for lane_id in lane_ids if not NON_SOLVER_LANE_ID_PATTERN.search(lane_id)
    ]
    if len(solver_lane_ids) < 2:
        errors.append(
            "blueprint requires at least two independent solver lanes before post-barrier verification"
        )
    non_solver_lane_ids = sorted(set(lane_ids) - set(solver_lane_ids))
    if non_solver_lane_ids:
        errors.append(
            "blueprint contains redundant verification-only lanes %s; use the post-barrier verifier"
            % non_solver_lane_ids[:5]
        )
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("blueprint lane artifact IDs must be unique")
    responsibilities = [str(item.get("responsibility", "")).strip().casefold() for item in lanes]
    if len(responsibilities) != len(set(responsibilities)):
        errors.append("blueprint lane responsibilities must be distinct")
    for lane in lanes:
        route = lane.get("specialist_route", [])
        if not route or str(route[0]) not in {"root", "generalist", "root_generalist"}:
            errors.append("blueprint lane %s must start at the root/generalist" % lane.get("lane_id"))
        responsibility = "%s %s" % (
            lane.get("responsibility", ""),
            lane.get("deliverable", ""),
        )
        if (
            REVIEW_ONLY_RESPONSIBILITY_PATTERN.search(responsibility)
            and not CONSTRUCTIVE_RESPONSIBILITY_PATTERN.search(responsibility)
        ):
            errors.append(
                "blueprint lane %s is review-only; post-barrier verification already owns review"
                % lane.get("lane_id")
            )

    claims = blueprint.get("claim_plan", [])
    claim_ids = [str(item.get("claim_id", "")) for item in claims]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("blueprint claim IDs must be unique")
    valid_premise_refs = set(premise_ids) | set(constraint_ids)
    valid_evidence_refs = valid_premise_refs | set(artifact_ids) | known_chunk_ids
    for claim in claims:
        claim_id = str(claim.get("claim_id", ""))
        owner = str(claim.get("owner_lane_id", ""))
        evidence_mode = str(claim.get("evidence_mode", ""))
        if owner not in set(lane_ids):
            errors.append("blueprint claim %s has an unknown owner lane" % claim_id)
        if evidence_mode == "supplied_execution":
            errors.append(
                "blueprint claim %s requires supplied execution but this pipeline has no execution attestation"
                % claim_id
            )
        if UNATTESTED_OUTCOME_CLAIM_PATTERN.search(str(claim.get("statement", ""))):
            errors.append(
                "blueprint claim %s asserts an outcome that requires unavailable execution evidence"
                % claim_id
            )
        unknown_premises = {str(item) for item in claim.get("premise_ids", [])} - valid_premise_refs
        if unknown_premises:
            errors.append(
                "blueprint claim %s cites unknown premise IDs %s"
                % (claim_id, sorted(unknown_premises)[:5])
            )
        unknown_evidence = {
            str(item) for item in claim.get("expected_evidence_refs", [])
        } - valid_evidence_refs
        if unknown_evidence:
            errors.append(
                "blueprint claim %s plans unknown evidence %s"
                % (claim_id, sorted(unknown_evidence)[:5])
            )
        cross_lane_artifacts = sorted(
            ref
            for ref in map(str, claim.get("expected_evidence_refs", []))
            if ref in artifact_owner and artifact_owner[ref] != owner
        )
        if cross_lane_artifacts:
            errors.append(
                "blueprint claim %s depends on sibling-lane artifacts %s before the barrier"
                % (claim_id, cross_lane_artifacts[:5])
            )

    owner_counts = Counter(str(item.get("owner_lane_id", "")) for item in claims)
    for lane_id in lane_ids:
        if owner_counts[lane_id] < 1:
            errors.append("blueprint lane %s must own at least one planned claim" % lane_id)

    verification = blueprint.get("verification_plan", {})
    if set(verification.get("target_claim_ids", [])) != set(claim_ids):
        errors.append("blueprint verification plan must cover every claim exactly once")

    trace = blueprint.get("constraint_trace", [])
    traced_constraints = [str(item.get("constraint_id", "")) for item in trace]
    if sorted(traced_constraints) != sorted(constraint_ids):
        errors.append("blueprint constraint trace must cover every constraint exactly once")
    for item in trace:
        unknown_artifacts = {
            str(value) for value in item.get("planned_artifact_ids", [])
        } - set(artifact_ids)
        if unknown_artifacts:
            errors.append(
                "constraint %s traces unknown artifacts %s"
                % (item.get("constraint_id"), sorted(unknown_artifacts)[:5])
            )
    return errors


def episode_blueprint_errors(episode: Dict[str, Any], blueprint: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    source_premises = blueprint.get("premises", {}).get("source_backed", [])
    scenario_premises = blueprint.get("premises", {}).get("scenario", [])
    expected_context = [
        "Source-backed premise [%s]: %s [%s]"
        % (
            item["premise_id"],
            item["statement"],
            ", ".join(str(ref) for ref in item.get("evidence_refs", [])),
        )
        for item in source_premises
    ] + [
        "Scenario premise [%s]: %s" % (item["premise_id"], item["statement"])
        for item in scenario_premises
    ]
    expected_constraints = [item["statement"] for item in blueprint.get("task", {}).get("constraints", [])]
    episode_input = episode.get("input", {})
    if episode_input.get("user_request") != blueprint.get("task", {}).get("user_request"):
        errors.append("episode user request diverges from approved blueprint")
    if episode_input.get("context") != expected_context:
        errors.append("episode premises diverge from approved blueprint")
    if episode_input.get("constraints") != expected_constraints:
        errors.append("episode constraints diverge from approved blueprint")

    expected_lane_ids = {str(item["lane_id"]) for item in blueprint.get("lanes", [])}
    episode_lanes = {str(item.get("lane_id")): item for item in episode.get("lanes", [])}
    if set(episode_lanes) != expected_lane_ids:
        errors.append("episode lanes diverge from approved blueprint")

    planned_claims = {str(item["claim_id"]): item for item in blueprint.get("claim_plan", [])}
    episode_claims: Dict[str, Dict[str, Any]] = {}
    duplicate_claims: Set[str] = set()
    for lane in episode.get("lanes", []):
        plan = next(
            (item for item in blueprint.get("lanes", []) if item.get("lane_id") == lane.get("lane_id")),
            None,
        )
        if plan:
            if lane.get("brief", {}).get("deliverable") != plan.get("deliverable"):
                errors.append("lane %s deliverable diverges from approved blueprint" % lane.get("lane_id"))
            lane_artifact_ids = {str(item.get("artifact_id")) for item in lane.get("artifacts", [])}
            planned_artifact_ids = {str(item) for item in plan.get("artifact_ids", [])}
            if not planned_artifact_ids.issubset(lane_artifact_ids):
                errors.append("lane %s omitted its planned artifact" % lane.get("lane_id"))
            if lane.get("route") != plan.get("specialist_route"):
                errors.append("lane %s specialist route diverges from approved blueprint" % lane.get("lane_id"))
        for claim in lane.get("claims", []):
            claim_id = str(claim.get("claim_id", ""))
            if claim_id in episode_claims:
                duplicate_claims.add(claim_id)
            episode_claims[claim_id] = claim
    if duplicate_claims:
        errors.append("episode duplicates blueprint claims: %s" % sorted(duplicate_claims)[:5])
    if set(episode_claims) != set(planned_claims):
        errors.append("episode claim set diverges from approved blueprint")

    premise_ref_map: Dict[str, str] = {}
    all_premises = source_premises + scenario_premises
    for index, premise in enumerate(all_premises):
        premise_ref_map[str(premise["premise_id"])] = "input.context[%d]" % index
    for index, constraint in enumerate(blueprint.get("task", {}).get("constraints", [])):
        premise_ref_map[str(constraint["constraint_id"])] = "input.constraints[%d]" % index
    for claim_id, plan in planned_claims.items():
        if claim_id not in episode_claims:
            continue
        expected_refs = {
            premise_ref_map.get(str(item), str(item))
            for item in list(plan.get("premise_ids", []))
            + list(plan.get("expected_evidence_refs", []))
        }
        actual_refs = {str(item) for item in episode_claims[claim_id].get("evidence_refs", [])}
        missing = expected_refs - actual_refs
        if missing:
            errors.append("claim %s omitted planned evidence %s" % (claim_id, sorted(missing)[:5]))

    verification_ids = [str(item.get("claim_id", "")) for item in episode.get("verification", [])]
    if sorted(verification_ids) != sorted(planned_claims):
        errors.append("episode must verify every blueprint claim exactly once")

    trace = episode.get("blueprint_trace", {})
    if trace.get("blueprint_hash") != _blueprint_hash(blueprint):
        errors.append("episode blueprint hash is missing or incorrect")
    trace_rows = trace.get("constraint_trace", [])
    trace_by_id = {str(item.get("constraint_id", "")): item for item in trace_rows}
    expected_constraint_ids = {
        str(item.get("constraint_id", "")) for item in blueprint.get("task", {}).get("constraints", [])
    }
    if set(trace_by_id) != expected_constraint_ids or len(trace_rows) != len(trace_by_id):
        errors.append("episode constraint trace must cover every blueprint constraint exactly once")
    selected_artifacts = {str(item) for item in episode.get("integration", {}).get("selected_artifacts", [])}
    episode_artifact_ids = {
        str(item.get("artifact_id"))
        for lane in episode.get("lanes", [])
        for item in lane.get("artifacts", [])
        if item.get("artifact_id")
    }
    blueprint_trace = {
        str(item.get("constraint_id")): item for item in blueprint.get("constraint_trace", [])
    }
    for constraint_id, plan in blueprint_trace.items():
        row = trace_by_id.get(constraint_id, {})
        required_artifacts = {str(item) for item in plan.get("planned_artifact_ids", [])}
        actual_artifacts = {str(item) for item in row.get("artifact_ids", [])}
        if not required_artifacts.issubset(actual_artifacts):
            errors.append("constraint %s omitted planned artifacts" % constraint_id)
        if not required_artifacts.issubset(episode_artifact_ids):
            errors.append("constraint %s references missing episode artifacts" % constraint_id)
        if required_artifacts and not required_artifacts.intersection(selected_artifacts):
            errors.append("constraint %s has no selected delivery artifact" % constraint_id)
        if not str(row.get("result", "")).strip():
            errors.append("constraint %s has no validation result" % constraint_id)
    return errors


def _blueprint_hash(blueprint: Dict[str, Any]) -> str:
    # Local import avoids a validation/util import cycle during package startup.
    from .util import stable_hash

    return stable_hash(blueprint, 32)


def review_errors(
    review: Dict[str, Any], schema_path: Path, threshold: float, expertise_threshold: float = 0.0
) -> List[str]:
    errors = schema_errors(review, schema_path)
    if review.get("verdict") != "accept":
        errors.append("judge rejected episode")
    if float(review.get("overall_score", 0.0)) < threshold:
        errors.append("judge score below %.2f" % threshold)
    expertise = float(review.get("scores", {}).get("expertise_uplift", 0.0))
    if expertise < expertise_threshold:
        errors.append("expertise uplift below %.2f" % expertise_threshold)
    if review.get("suspected_source_leakage"):
        errors.append("judge detected source or benchmark leakage")
    unsupported = [str(item) for item in review.get("unsupported_claim_ids", []) if str(item)]
    if unsupported:
        errors.append("judge identified unsupported claims: %s" % unsupported[:5])
    required_fixes = [str(item) for item in review.get("required_fixes", []) if str(item).strip()]
    if required_fixes:
        errors.append("judge required fixes before acceptance: %s" % required_fixes[:5])
    return errors
