from __future__ import annotations

import json
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .adversarial import adversarial_acceptance_errors
from .azure_client import AzureTeacherClient
from .language import classify_language, contains_blocked_script
from .util import append_jsonl, atomic_write_json, iter_jsonl, sanitize_unicode, stable_hash
from .validation import (
    blueprint_invariant_errors,
    episode_blueprint_errors,
    episode_invariant_errors,
    review_errors,
    schema_errors,
    source_support_spans,
)


@dataclass(frozen=True)
class EpisodeJob:
    episode_id: str
    domain: str
    objective: str
    source_group: str
    lineage_component_id: str
    chunks: Tuple[Dict[str, Any], ...]
    variation_seed: int


def _natural_key(value: Any) -> Tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def _contiguous_window(
    candidates: Sequence[Dict[str, Any]], count: int, rng: random.Random
) -> Tuple[Dict[str, Any], ...]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            _natural_key(item.get("part", "")),
            int(item.get("char_start", 0) or 0),
            str(item.get("chunk_id", "")),
        ),
    )
    if len(ordered) <= count:
        return tuple(ordered)
    anchor = rng.randrange(len(ordered))
    start = max(0, min(anchor - count // 2, len(ordered) - count))
    return tuple(ordered[start : start + count])


def _coherent_chunk_sample(
    candidates: Sequence[Dict[str, Any]], count: int, rng: random.Random
) -> Tuple[Dict[str, Any], ...]:
    """Select a local evidence window instead of unrelated passages from a whole source."""
    if not candidates or count <= 0:
        return tuple()
    source_group = str(candidates[0].get("source_group", ""))
    # Imported problem/solution rows and analytical packets are independent
    # records. Combining several creates accidental benchmark remixes.
    if source_group.startswith(("hf-", "uci-")):
        return (candidates[rng.randrange(len(candidates))],)

    by_component: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in candidates:
        component = str(chunk.get("lineage_component_id") or chunk.get("source_id") or chunk["chunk_id"])
        by_component.setdefault(component, []).append(chunk)
    component_names = sorted(by_component, key=_natural_key)
    local = by_component[component_names[rng.randrange(len(component_names))]]

    # A repository-level component may span unrelated files. Keep one file.
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in local:
        by_path.setdefault(str(chunk.get("path", "")), []).append(chunk)
    if len(by_path) > 1:
        path_names = sorted(by_path, key=_natural_key)
        local = by_path[path_names[rng.randrange(len(path_names))]]

    # PDF page blocks are coherent across nearby pages even though each page
    # has its own source_id. EPUB/HTML chapters instead stay within one part.
    if local and all(re.fullmatch(r"page-\d+", str(chunk.get("part", ""))) for chunk in local):
        return _contiguous_window(local, count, rng)
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in local:
        by_document.setdefault(str(chunk.get("source_id") or chunk["chunk_id"]), []).append(chunk)
    document_names = sorted(by_document, key=_natural_key)
    document = by_document[document_names[rng.randrange(len(document_names))]]
    return _contiguous_window(document, count, rng)


def _domain_counts(domains: Dict[str, Any], total: int) -> Dict[str, int]:
    if total <= 0:
        return {name: 0 for name in domains}
    raw = {name: total * float(spec.get("weight", 0.0)) for name, spec in domains.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(domains, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def build_jobs(
    chunks: Sequence[Dict[str, Any]], config: Dict[str, Any], count: int, offset: int = 0
) -> List[EpisodeJob]:
    generation = config["generation"]
    domains = generation["domains"]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.get("domain", "unassigned")), []).append(chunk)
    if not chunks:
        raise ValueError("No ingested chunks found. Add sources and run ingest first.")
    seed = int(config["project"].get("seed", 3407))
    rng = random.Random(seed)
    jobs: List[EpisodeJob] = []
    stable_indexed = bool(generation.get("stable_indexed_jobs", False))
    if stable_indexed:
        before = _domain_counts(domains, max(0, offset))
        after = _domain_counts(domains, max(0, offset) + count)
        per_domain = {name: after[name] - before[name] for name in domains}
    else:
        before = {name: 0 for name in domains}
        per_domain = _domain_counts(domains, count)
    chunks_per_episode = int(generation.get("chunks_per_episode", 2))
    for domain, domain_count in per_domain.items():
        pool = grouped.get(domain, [])
        if not pool:
            if generation.get("require_grounded_sources", True):
                continue
            pool = list(chunks)
        objectives = domains[domain].get("objectives", [domain])
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in pool:
            by_group.setdefault(str(chunk["source_group"]), []).append(chunk)
        group_names = sorted(by_group)
        if stable_indexed:
            group_rng = random.Random(int(stable_hash({"seed": seed, "domain": domain, "groups": group_names}, 16), 16))
            group_rng.shuffle(group_names)
        for index in range(before[domain], before[domain] + domain_count):
            group = group_names[index % len(group_names)]
            candidates = by_group[group]
            job_rng = random.Random(
                int(stable_hash({"seed": seed, "domain": domain, "index": index}, 16), 16)
            ) if stable_indexed else rng
            selected = _coherent_chunk_sample(candidates, chunks_per_episode, job_rng)
            objective = str(objectives[index % len(objectives)])
            episode_id = "%s-%s" % (
                domain.replace("_", "-"),
                stable_hash({"group": group, "index": index, "chunks": [c["chunk_id"] for c in selected]}, 16),
            )
            jobs.append(
                EpisodeJob(
                    episode_id=episode_id,
                    domain=domain,
                    objective=objective,
                    source_group=group,
                    lineage_component_id=str(selected[0].get("lineage_component_id", group)),
                    chunks=selected,
                    variation_seed=job_rng.randint(1, 2**31 - 1),
                )
            )
    rng.shuffle(jobs)
    return jobs


def _source_packet(job: EpisodeJob, max_chars: int) -> Dict[str, Any]:
    remaining = max_chars
    packet_chunks = []
    for chunk in job.chunks:
        text = str(chunk["text"])[:remaining]
        if not text:
            break
        data_role = str(chunk.get("data_role") or "visible_context")
        if str(chunk.get("source_group", "")).startswith("hf-"):
            data_role = "verification_material"
        packet_chunks.append(
            {
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk.get("title"),
                "author": chunk.get("author"),
                "version": chunk.get("version"),
                "license": chunk.get("license"),
                "language": chunk.get("language"),
                "lineage_component_id": chunk.get("lineage_component_id"),
                "part": chunk.get("part"),
                "data_role": data_role,
                "text": text,
                "support_spans": source_support_spans(str(chunk["chunk_id"]), text),
            }
        )
        remaining -= len(text)
    return {
        "source_group": job.source_group,
        "lineage_component_id": job.lineage_component_id,
        "usage_policy": (
            "Chunks marked verification_material may be used to check a newly constructed task, "
            "but their problem statement or reference answer must not be reused as the task template."
        ),
        "chunks": packet_chunks,
    }


def filter_generation_chunks_by_language(
    chunks: Sequence[Dict[str, Any]], generation: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], int]:
    allowed = tuple(str(item).lower() for item in generation.get("allowed_languages", []))
    if not allowed:
        return list(chunks), 0
    if allowed != ("en",):
        raise ValueError("Generation currently supports strict language filtering only for allowed_languages: [en]")
    minimum_confidence = float(generation.get("minimum_language_confidence", 0.78))
    accepted: List[Dict[str, Any]] = []
    removed = 0
    for chunk in chunks:
        text = sanitize_unicode(str(chunk.get("text", "")))
        if "\N{REPLACEMENT CHARACTER}" in text:
            removed += 1
            continue
        if contains_blocked_script(text):
            removed += 1
            continue
        decision = classify_language(
            text,
            expected="en",
            minimum_confidence=minimum_confidence,
        )
        if not decision.accepted:
            removed += 1
            continue
        value = dict(chunk)
        value["language"] = "en"
        value["language_confidence"] = round(decision.confidence, 6)
        value["language_decision"] = decision.reason
        accepted.append(value)
    if not accepted:
        raise ValueError("English-only filtering removed every source chunk; review the source manifest")
    return accepted, removed


_NON_SUBSTANTIVE_HEADING_RE = re.compile(
    r"^(?:acknowledg(?:e)?ments?|table of contents|contents|copyright|colophon|"
    r"about (?:the )?author|author biography|bibliography|references|index)\b",
    re.IGNORECASE,
)


def non_substantive_source_reason(chunk: Dict[str, Any]) -> Optional[str]:
    text = re.sub(r"\s+", " ", sanitize_unicode(str(chunk.get("text", "")))).strip()
    if not text:
        return "empty"
    heading_source = re.sub(r"^[\w.-]+\s+", "", text[:240]).strip()
    if _NON_SUBSTANTIVE_HEADING_RE.search(heading_source):
        return "front_matter_heading"
    lower = text.casefold()
    if re.search(
        r"(?:would(?:n['’]t| not) have been possible without|"
        r"particularly supportive|thanks? (?:to|you)|grateful to)",
        lower[:1600],
    ) and re.search(r"\b(?:support|help|encourag|feedback|thank)\w*\b", lower[:1600]):
        return "acknowledgments"
    if (
        ("all rights reserved" in lower[:1200] or "isbn" in lower[:1200])
        and "copyright" in lower[:1200]
    ):
        return "copyright_or_colophon"
    if (
        ("document outline" in lower[:1600] or "table of contents" in lower[:1600])
        and len(re.findall(r"\bchapter\s+\d+\b", lower[:2000])) >= 3
    ):
        return "table_of_contents"
    if "other books you may enjoy" in lower[:1200] and (
        "isbn" in lower[:2000] or len(re.findall(r"\b(?:book|guide|handbook)\b", lower[:2000])) >= 3
    ):
        return "promotional_bibliography"
    return None


def filter_generation_chunks_by_source_quality(
    chunks: Sequence[Dict[str, Any]], generation: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], int]:
    if not bool(generation.get("reject_non_substantive_source_chunks", False)):
        return list(chunks), 0
    accepted = [chunk for chunk in chunks if non_substantive_source_reason(chunk) is None]
    removed = len(chunks) - len(accepted)
    if not accepted:
        raise ValueError("Source-quality filtering removed every source chunk")
    return accepted, removed


def validate_job_chunks_by_language(
    jobs: Sequence[EpisodeJob], generation: Dict[str, Any]
) -> Tuple[List[EpisodeJob], int, int]:
    """Reclassify every evidence chunk that will cross the model boundary.

    The corpus audit covers the full file. Generation additionally validates
    the actual task packets, avoiding an expensive rescan of tens of thousands
    of unused chunks for every small resumable batch.
    """
    selected: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        for chunk in job.chunks:
            selected[str(chunk["chunk_id"])] = chunk
    checked, removed = filter_generation_chunks_by_language(list(selected.values()), generation)
    checked_by_id = {str(chunk["chunk_id"]): chunk for chunk in checked}
    missing = set(selected) - set(checked_by_id)
    if missing:
        raise ValueError(
            "Strict language validation rejected scheduled evidence chunks: %s"
            % sorted(missing)[:10]
        )
    validated_jobs = [
        EpisodeJob(
            episode_id=job.episode_id,
            domain=job.domain,
            objective=job.objective,
            source_group=job.source_group,
            lineage_component_id=job.lineage_component_id,
            chunks=tuple(checked_by_id[str(chunk["chunk_id"])] for chunk in job.chunks),
            variation_seed=job.variation_seed,
        )
        for job in jobs
    ]
    return validated_jobs, len(selected), removed


def _output_contract(
    config: Dict[str, Any], job: EpisodeJob, blueprint: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    generation = config["generation"]
    return {
        "schema_version": "1.0",
        "episode_id": job.episode_id,
        "domain": job.domain,
        "subdomain": job.objective,
        "difficulty": "integer 1-5",
        "source_group": job.source_group,
        "lineage_component_id": job.lineage_component_id,
        "source_refs": [{"source_id": "...", "chunk_ids": ["..."], "usage": "..."}],
        "input": {"user_request": "...", "context": ["..."], "constraints": ["..."]},
        "frame": {
            "objective": "...", "requirements": ["..."], "unknowns": ["..."],
            "failure_contract": ["..."]
        },
        "routing": {
            "candidate_briefs": [{"lane_id": "lane-a", "scope": "...", "bid": 0.0}],
            "selected_lane_ids": ["lane-a", "lane-b"],
            "budget": {"max_lanes": generation.get("lanes_max", 4), "max_checkpoints_per_lane": 4}
        },
        "lanes": [{
            "lane_id": "lane-a", "route": ["root", "domain", "specialist"],
            "route_windows": [{
                "window_id": "window-a1", "route": ["root", "domain", "specialist"],
                "requested_steps": 2, "admitted_steps": 2, "actual_steps": 2, "decision": "halt"
            }],
            "brief": {"scope": "...", "assumptions": ["..."], "deliverable": "...", "rejection_tests": ["..."]},
            "artifacts": [{"artifact_id": "artifact-a1", "type": "...", "content": "..."}],
            "claims": [{"claim_id": "claim-a1", "statement": "...", "evidence_refs": ["chunk-or-artifact-id"]}],
            "checkpoints": [{"step": 1, "observable_update": "...", "next_step_gain": 0.0, "decision": "halt"}],
            "summary": "..."
        }],
        "barrier": {"lane_summaries": [{"lane_id": "lane-a", "summary": "..."}], "conflicts": [], "open_claims": []},
        "verification": [{
            "claim_id": "claim-a1", "verdict": "supported|refuted|insufficient",
            "method": "...", "evidence_refs": ["..."], "result": "..."
        }],
        "tool_runs": [],
        "integration": {
            "decision": "select|merge|revise|abstain", "selected_artifacts": ["..."],
            "rejected_artifacts": [{"artifact_id": "...", "reason": "..."}], "published_answer": "..."
        },
        "commitment": {
            "decision": "publish|refine|fallback|abstain", "confidence": 0.0,
            "risks": ["..."], "additional_refinement_helped": False
        },
        "continuation_pairs": [{
            "pair_id": "pair-1", "kind": "lane_step", "added_operation": "...", "added_cost": 1.0,
            "before_utility": 0.0, "after_utility": 0.0, "target": "halt", "eligible": False,
            "evidence_refs": ["..."]
        }],
        "counterfactuals": [
            {"variant": "root-only", "outcome": "...", "utility": 0.0, "errors": ["..."]},
            {"variant": "single-lane", "outcome": "...", "utility": 0.0, "errors": ["..."]},
        ],
        "root_anchor_evaluations": [{
            "anchor_id": "anchor-1", "capability": "instruction clarity", "root_only_outcome": "...",
            "routed_outcome": "...", "retention_passed": True, "evidence_refs": ["..."]
        }],
        "blueprint_trace": {
            "blueprint_hash": stable_hash(blueprint, 32) if blueprint else "...",
            "constraint_trace": [{
                "constraint_id": "constraint-...", "artifact_ids": ["artifact-..."],
                "result": "Observable validation result"
            }]
        },
    }


def _blueprint_identity(job: EpisodeJob) -> Dict[str, str]:
    return {
        "episode_id": job.episode_id,
        "domain": job.domain,
        "objective": job.objective,
        "source_group": job.source_group,
        "lineage_component_id": job.lineage_component_id,
    }


def _normalize_blueprint(value: Dict[str, Any], job: EpisodeJob) -> Dict[str, Any]:
    # Repair only unambiguous serialization mistakes. Semantic defects still go
    # through the deterministic validator and independent quality judge.
    verification_plan = value.get("verification_plan")
    if (
        "constraint_trace" not in value
        and isinstance(verification_plan, dict)
        and isinstance(verification_plan.get("constraint_trace"), list)
    ):
        value["constraint_trace"] = verification_plan.pop("constraint_trace")

    def unwrap_rows(rows: Any, required_key: str) -> Any:
        if not isinstance(rows, list):
            return rows
        normalized = []
        for row in rows:
            if isinstance(row, dict) and required_key not in row and len(row) == 1:
                nested = next(iter(row.values()))
                if isinstance(nested, dict) and required_key in nested:
                    row = nested
            normalized.append(row)
        return normalized

    normalized_lanes = unwrap_rows(value.get("lanes"), "lane_id")
    normalized_claims = unwrap_rows(value.get("claim_plan"), "claim_id")
    if normalized_lanes is not None:
        value["lanes"] = normalized_lanes
    if normalized_claims is not None:
        value["claim_plan"] = normalized_claims
    for claim in value.get("claim_plan", []) if isinstance(value.get("claim_plan"), list) else []:
        if (
            isinstance(claim, dict)
            and claim.get("claim_type") == "logical_counterexample"
        ):
            claim["claim_type"] = "derived_result"
            claim["evidence_mode"] = "logical_counterexample"

    value["schema_version"] = "1.0"
    value.update(_blueprint_identity(job))
    task = value.get("task", {})
    criteria = [str(item).strip() for item in task.get("success_criteria", []) if str(item).strip()]
    if len(criteria) > 5:
        # Preserve every generated acceptance condition while fitting the hard schema.
        # The final item is one conjunction, not a silent truncation of distinct checks.
        criteria = criteria[:4] + [
            "Combined acceptance criterion: " + "; and ".join(criteria[4:])
        ]
    if criteria:
        task["success_criteria"] = criteria

    known_chunk_ids = {str(chunk["chunk_id"]) for chunk in job.chunks}
    source_chunks: Dict[str, List[str]] = {}
    source_id_by_chunk: Dict[str, str] = {}
    for chunk in job.chunks:
        source_id = str(chunk["source_id"])
        chunk_id = str(chunk["chunk_id"])
        source_chunks.setdefault(source_id, []).append(chunk_id)
        source_id_by_chunk[chunk_id] = source_id
    support_spans = {
        item["span_id"]: item["text"]
        for chunk in job.chunks
        for item in source_support_spans(str(chunk["chunk_id"]), str(chunk.get("text", "")))
    }
    spans_by_quote: Dict[str, List[str]] = {}
    for issued_span_id, issued_text in support_spans.items():
        spans_by_quote.setdefault(issued_text, []).append(issued_span_id)
    for premise in value.get("premises", {}).get("source_backed", []):
        span_id = str(premise.get("support_span_id", ""))
        span_chunk_id = span_id.split("::span-", 1)[0]
        normalized_refs: List[str] = []
        for raw_ref in premise.get("evidence_refs", []):
            ref = str(raw_ref)
            if "::span-" in ref and ref.split("::span-", 1)[0] in known_chunk_ids:
                candidates = [ref.split("::span-", 1)[0]]
            elif ref in known_chunk_ids:
                candidates = [ref]
            elif ref in source_chunks:
                # A model sometimes copies source_id from the packet. Resolve it to
                # the exact chunk selected by its authoritative support span when
                # possible; otherwise expose the selected chunks for that source.
                if (
                    span_chunk_id in known_chunk_ids
                    and source_id_by_chunk.get(span_chunk_id) == ref
                ):
                    candidates = [span_chunk_id]
                else:
                    candidates = source_chunks[ref]
            else:
                candidates = [ref]
            for candidate in candidates:
                if candidate not in normalized_refs:
                    normalized_refs.append(candidate)
        premise["evidence_refs"] = normalized_refs
        if span_id not in support_spans:
            exact_quote_matches = [
                candidate
                for candidate in spans_by_quote.get(str(premise.get("support_quote", "")), [])
                if candidate.split("::span-", 1)[0] in normalized_refs
            ]
            if len(exact_quote_matches) == 1:
                span_id = exact_quote_matches[0]
                premise["support_span_id"] = span_id
        if span_id in support_spans:
            premise["support_quote"] = support_spans[span_id]
    return value


def _blueprint_errors(
    blueprint: Dict[str, Any], job: EpisodeJob, schema_path: Path
) -> List[str]:
    known_chunk_ids = {str(chunk["chunk_id"]) for chunk in job.chunks}
    errors = schema_errors(blueprint, schema_path)
    errors.extend(
        blueprint_invariant_errors(
            blueprint,
            known_chunk_ids,
            expected_identity=_blueprint_identity(job),
            known_chunk_texts={str(chunk["chunk_id"]): str(chunk.get("text", "")) for chunk in job.chunks},
        )
    )
    return errors


def _blueprint_quality_errors(
    review: Dict[str, Any], threshold: float, expertise_threshold: float, issue_limit: int = 8
) -> List[str]:
    errors: List[str] = []
    if str(review.get("verdict", "")).lower() != "accept":
        errors.append("independent blueprint quality judge rejected the candidate")
    try:
        overall = float(review.get("overall_score", 0.0))
        expertise = float(review.get("expertise_uplift", 0.0))
    except (TypeError, ValueError):
        return ["independent blueprint quality judge returned invalid numeric scores"]
    if not 0.0 <= overall <= 1.0 or overall < threshold:
        errors.append("blueprint quality score below %.2f" % threshold)
    if not 0.0 <= expertise <= 1.0 or expertise < expertise_threshold:
        errors.append("blueprint expertise uplift below %.2f" % expertise_threshold)
    issues = [str(item).strip() for item in review.get("issues", []) if str(item).strip()]
    if issues:
        errors.extend(
            "blueprint quality issue: %s" % item for item in issues[: max(1, issue_limit)]
        )
    return errors


_CONSTRUCTION_CRITIC_KEYS = {
    "verdict",
    "overall_score",
    "expertise_uplift",
    "blocking_defects",
}
_CONSTRUCTION_DEFECT_KEYS = {
    "category",
    "location",
    "contract_reference",
    "defect",
    "required_correction",
}
_NON_BLOCKING_LANGUAGE = (
    "not a blocking defect",
    "not blocking",
    "not harmful",
    "not clearly a violation",
    "not clearly a defect",
    "may be acceptable",
    "might be acceptable",
    "could be acceptable",
    "ambiguous whether",
    "potential omission",
)


def _construction_critic_protocol_errors(
    review: Dict[str, Any], threshold: float, expertise_threshold: float
) -> List[str]:
    """Validate critic output independently from the candidate it reviewed."""
    errors: List[str] = []
    if set(review) != _CONSTRUCTION_CRITIC_KEYS:
        missing = sorted(_CONSTRUCTION_CRITIC_KEYS - set(review))
        extra = sorted(set(review) - _CONSTRUCTION_CRITIC_KEYS)
        if missing:
            errors.append("missing critic fields: %s" % ", ".join(missing))
        if extra:
            errors.append("unexpected critic fields: %s" % ", ".join(extra))

    verdict = str(review.get("verdict", "")).lower()
    if verdict not in {"accept", "reject"}:
        errors.append("critic verdict must be accept or reject")
    try:
        overall = float(review.get("overall_score"))
        expertise = float(review.get("expertise_uplift"))
    except (TypeError, ValueError):
        errors.append("critic scores must be numeric")
        overall = expertise = -1.0
    if not 0.0 <= overall <= 1.0:
        errors.append("critic overall_score must be between 0 and 1")
    if not 0.0 <= expertise <= 1.0:
        errors.append("critic expertise_uplift must be between 0 and 1")

    defects = review.get("blocking_defects")
    if not isinstance(defects, list):
        errors.append("blocking_defects must be a list")
        defects = []
    elif len(defects) > 3:
        errors.append("critic returned more than three blocking defects")

    for index, defect in enumerate(defects):
        prefix = "blocking_defects.%d" % index
        if not isinstance(defect, dict):
            errors.append("%s must be an object" % prefix)
            continue
        if set(defect) != _CONSTRUCTION_DEFECT_KEYS:
            errors.append("%s must contain exactly the required defect fields" % prefix)
            continue
        for field in sorted(_CONSTRUCTION_DEFECT_KEYS):
            value = str(defect.get(field, "")).strip()
            if not value:
                errors.append("%s.%s must not be empty" % (prefix, field))
            limit = 160 if field in {"category", "location", "contract_reference"} else 420
            if len(value) > limit:
                errors.append("%s.%s exceeds %d characters" % (prefix, field, limit))
        allegation = "%s %s" % (defect.get("defect", ""), defect.get("required_correction", ""))
        lowered = allegation.lower()
        if any(phrase in lowered for phrase in _NON_BLOCKING_LANGUAGE):
            errors.append("%s explicitly describes a non-blocking or ambiguous difference" % prefix)

    if verdict == "accept":
        if defects:
            errors.append("an accepted critic review must have no blocking defects")
        if overall < threshold:
            errors.append("an accepted critic review is below the quality threshold")
        if expertise < expertise_threshold:
            errors.append("an accepted critic review is below the expertise threshold")
    elif verdict == "reject":
        if not defects:
            errors.append("a rejected critic review must identify a blocking defect")
        if overall >= threshold and expertise >= expertise_threshold:
            errors.append("a rejected critic review must score below at least one acceptance threshold")
    return errors


def _construction_critic_candidate_errors(
    review: Dict[str, Any], threshold: float, expertise_threshold: float
) -> List[str]:
    """Convert a protocol-valid critic rejection into bounded editor feedback."""
    if str(review.get("verdict", "")).lower() == "accept":
        return []
    errors = ["construction critic rejected the candidate"]
    overall = float(review["overall_score"])
    expertise = float(review["expertise_uplift"])
    if overall < threshold:
        errors.append("construction quality score below %.2f" % threshold)
    if expertise < expertise_threshold:
        errors.append("construction expertise uplift below %.2f" % expertise_threshold)
    for defect in review["blocking_defects"]:
        errors.append(
            "construction critic: %s | %s | contract: %s | %s | correction: %s"
            % (
                defect["category"],
                defect["location"],
                defect["contract_reference"],
                defect["defect"],
                defect["required_correction"],
            )
        )
    return errors


def construction_critic_calibration_errors(config: Dict[str, Any], root: Path) -> List[str]:
    """Require a prompt/deployment-bound critic calibration before episode generation."""
    generation = config.get("generation", {})
    if not bool(generation.get("require_construction_critic_calibration", False)):
        return []
    report_path = root / str(
        generation.get(
            "construction_critic_calibration_report",
            "reports/reasoning9000-construction-critic-calibration.json",
        )
    )
    dataset_path = root / str(
        generation.get(
            "construction_critic_calibration_dataset",
            "data/reasoning9000/critic-calibration/cases.jsonl",
        )
    )
    if not report_path.exists():
        return ["construction critic calibration report is missing"]
    if not dataset_path.exists():
        return ["construction critic calibration dataset is missing"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cases = list(iter_jsonl(dataset_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ["construction critic calibration evidence is unreadable: %s" % exc]

    errors: List[str] = []
    if not bool(report.get("passed", False)):
        errors.append("construction critic calibration did not pass")
    expected_deployment = generation_model_roles(config)["episode_construction_critic"]
    if report.get("deployment") != expected_deployment:
        errors.append("construction critic calibration deployment does not match configuration")
    prompt_text = (root / "prompts/episode_critic_system.md").read_text(encoding="utf-8")
    if report.get("prompt_hash") != stable_hash(prompt_text, 32):
        errors.append("construction critic calibration is stale for the current prompt")
    if report.get("dataset_hash") != stable_hash(cases, 32):
        errors.append("construction critic calibration is stale for the current dataset")
    minimum_cases = int(generation.get("minimum_construction_critic_calibration_cases", 12))
    if int(report.get("case_count", 0)) < minimum_cases:
        errors.append("construction critic calibration has fewer than %d cases" % minimum_cases)
    confusion = report.get("confusion", {})
    if int(confusion.get("false_blocking", -1)) != 0:
        errors.append("construction critic calibration contains false blocking decisions")
    if int(confusion.get("missed_blocking", -1)) != 0:
        errors.append("construction critic calibration missed blocking defects")
    if int(report.get("invalid_predictions", -1)) != 0:
        errors.append("construction critic calibration contains invalid predictions")
    return errors


def _approved_blueprint_context(blueprint: Dict[str, Any]) -> List[str]:
    source_premises = blueprint.get("premises", {}).get("source_backed", [])
    scenario_premises = blueprint.get("premises", {}).get("scenario", [])
    return [
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


def _apply_blueprint_contract(value: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, Any]:
    task = blueprint.get("task", {})
    source_premises = blueprint.get("premises", {}).get("source_backed", [])
    scenario_premises = blueprint.get("premises", {}).get("scenario", [])
    premise_ref_map = {
        str(item["premise_id"]): "input.context[%d]" % index
        for index, item in enumerate(source_premises + scenario_premises)
    }
    premise_ref_map.update(
        {
            str(item.get("support_span_id")): "input.context[%d]" % index
            for index, item in enumerate(source_premises)
            if str(item.get("support_span_id", "")).strip()
        }
    )
    blueprint_chunk_ids = {
        str(ref)
        for item in source_premises
        for ref in item.get("evidence_refs", [])
    }
    premise_ref_map.update(
        {
            str(item["constraint_id"]): "input.constraints[%d]" % index
            for index, item in enumerate(task.get("constraints", []))
        }
    )

    def normalize_evidence_refs(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key == "evidence_refs" and isinstance(child, list):
                    normalized: List[str] = []
                    for item in child:
                        ref = str(item)
                        mapped = premise_ref_map.get(ref)
                        if mapped is None and "::span-" in ref:
                            chunk_id = ref.split("::span-", 1)[0]
                            if chunk_id in blueprint_chunk_ids:
                                mapped = chunk_id
                        candidate = mapped if mapped is not None else ref
                        if candidate not in normalized:
                            normalized.append(candidate)
                    node[key] = normalized
                else:
                    normalize_evidence_refs(child)
        elif isinstance(node, list):
            for child in node:
                normalize_evidence_refs(child)

    normalize_evidence_refs(value)
    value["input"] = {
        "user_request": task.get("user_request", ""),
        "context": _approved_blueprint_context(blueprint),
        "constraints": [str(item["statement"]) for item in task.get("constraints", [])],
    }
    lane_plans = {str(item["lane_id"]): item for item in blueprint.get("lanes", [])}
    for lane in value.get("lanes", []):
        plan = lane_plans.get(str(lane.get("lane_id")))
        if not plan:
            continue
        lane["route"] = list(plan.get("specialist_route", []))
        if isinstance(lane.get("brief"), dict):
            lane["brief"]["deliverable"] = str(plan.get("deliverable", ""))
    claim_plans = {
        str(item["claim_id"]): item for item in blueprint.get("claim_plan", [])
    }
    for lane in value.get("lanes", []):
        for claim in lane.get("claims", []):
            plan = claim_plans.get(str(claim.get("claim_id")))
            if not plan:
                continue
            refs = [str(item) for item in claim.get("evidence_refs", [])]
            planned_refs = list(plan.get("premise_ids", [])) + list(
                plan.get("expected_evidence_refs", [])
            )
            for planned_ref in planned_refs:
                mapped_ref = premise_ref_map.get(str(planned_ref), str(planned_ref))
                if mapped_ref not in refs:
                    refs.append(mapped_ref)
            claim["evidence_refs"] = refs
    trace = value.get("blueprint_trace")
    if not isinstance(trace, dict):
        trace = {}
    existing_rows = {
        str(item.get("constraint_id", "")): item
        for item in trace.get("constraint_trace", [])
        if isinstance(item, dict)
    }
    trace["blueprint_hash"] = stable_hash(blueprint, 32)
    trace["constraint_trace"] = [
        {
            "constraint_id": str(item.get("constraint_id", "")),
            "artifact_ids": [str(value) for value in item.get("planned_artifact_ids", [])],
            "result": str(
                existing_rows.get(str(item.get("constraint_id", "")), {}).get("result", "")
            ).strip(),
        }
        for item in blueprint.get("constraint_trace", [])
    ]
    value["blueprint_trace"] = trace
    for verification in value.get("verification", []):
        if isinstance(verification, dict) and isinstance(verification.get("result"), str):
            verification["result"] = verification["result"].replace("\\n", "\n")
    return value


def _episode_write_errors(
    episode: Dict[str, Any],
    known_chunk_ids: Set[str],
    episode_schema: Path,
    blueprint: Optional[Dict[str, Any]] = None,
    source_packet: Optional[Dict[str, Any]] = None,
) -> List[str]:
    errors = schema_errors(episode, episode_schema)
    errors.extend(episode_invariant_errors(episode, known_chunk_ids))
    if blueprint is not None:
        errors.extend(episode_blueprint_errors(episode, blueprint))
        errors.extend(
            adversarial_acceptance_errors(
                episode,
                blueprint=blueprint,
                source_packet=source_packet,
                trusted_execution_run_ids=set(),
            )
        )
    return errors


def _normalize_episode(value: Dict[str, Any], job: EpisodeJob) -> Dict[str, Any]:
    value["schema_version"] = "1.0"
    value["episode_id"] = job.episode_id
    value["domain"] = job.domain
    value["subdomain"] = value.get("subdomain") or job.objective
    value["source_group"] = job.source_group
    value["lineage_component_id"] = job.lineage_component_id
    refs: Dict[str, List[str]] = {}
    for chunk in job.chunks:
        refs.setdefault(str(chunk["source_id"]), []).append(str(chunk["chunk_id"]))
    value["source_refs"] = [
        {"source_id": source_id, "chunk_ids": chunk_ids, "usage": "grounding and task construction"}
        for source_id, chunk_ids in sorted(refs.items())
    ]
    return value


def _normalize_repaired_episode(
    value: Dict[str, Any], original: Dict[str, Any], request_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep lineage/provenance fields authoritative when a teacher repairs an episode."""
    for key in (
        "schema_version",
        "episode_id",
        "domain",
        "subdomain",
        "source_group",
        "lineage_component_id",
        "source_refs",
    ):
        if key in original:
            value[key] = original[key]
    if "generation_metadata" in original:
        value["generation_metadata"] = original["generation_metadata"]
    previous = original.get("repair_metadata", {})
    value["repair_metadata"] = {
        **request_metadata,
        "round": int(previous.get("round", 0)) + 1,
        "parent_episode_hash": stable_hash(original, 32),
    }
    return value


def _normalize_review(value: Dict[str, Any], episode_id: str, threshold: float) -> Dict[str, Any]:
    raw_issues = value.get("issues", [])
    issues: List[str] = []
    fixes: List[str] = [str(item) for item in value.get("required_fixes", []) if str(item).strip()]
    severities: List[str] = []
    for item in raw_issues:
        if isinstance(item, dict):
            issues.append(str(item.get("description") or item.get("issue") or item))
            if item.get("required_fix"):
                fixes.append(str(item["required_fix"]))
            severities.append(str(item.get("severity", "")).lower())
        else:
            issues.append(str(item))
    score_breakdown = value.get("score_breakdown", {})
    supplied_scores = value.get("scores", {})
    overall = float(value.get("overall_score", score_breakdown.get("overall", 0.0)) or 0.0)

    def score(name: str, *fallbacks: str) -> float:
        if name in supplied_scores:
            return max(0.0, min(1.0, float(supplied_scores[name])))
        for fallback in fallbacks:
            if fallback in score_breakdown:
                return max(0.0, min(1.0, float(score_breakdown[fallback])))
        return max(0.0, min(1.0, overall))

    verdict = str(value.get("verdict", "")).lower()
    if verdict not in {"accept", "reject"}:
        has_blocking_issue = any(level in {"major", "critical", "blocker"} for level in severities)
        verdict = "accept" if overall >= threshold and not has_blocking_issue else "reject"
    return {
        "episode_id": episode_id,
        "verdict": verdict,
        "overall_score": max(0.0, min(1.0, overall)),
        "scores": {
            "groundedness": score("groundedness", "evidence_support"),
            "correctness": score("correctness", "answer_support"),
            "lane_quality": score("lane_quality", "no_duplication_penalty"),
            "verification": score("verification", "evidence_support"),
            "usefulness": score("usefulness", "overall"),
            "style": score("style", "overall"),
            "expertise_uplift": score("expertise_uplift", "usefulness", "overall"),
        },
        "issues": issues,
        "required_fixes": fixes,
        "defect_scope": (
            "blueprint" if str(value.get("defect_scope", "episode")).lower() == "blueprint" else "episode"
        ),
        "suspected_source_leakage": bool(value.get("suspected_source_leakage", False)),
        "unsupported_claim_ids": [str(item) for item in value.get("unsupported_claim_ids", [])],
    }


def _rejection_review(episode_id: str, issues: Sequence[str]) -> Dict[str, Any]:
    messages = [str(item) for item in issues if str(item).strip()]
    return {
        "episode_id": episode_id,
        "verdict": "reject",
        "overall_score": 0.0,
        "scores": {
            "groundedness": 0.0,
            "correctness": 0.0,
            "lane_quality": 0.0,
            "verification": 0.0,
            "usefulness": 0.0,
            "style": 0.0,
            "expertise_uplift": 0.0,
        },
        "issues": messages,
        "required_fixes": messages,
        "defect_scope": "episode",
        "suspected_source_leakage": False,
        "unsupported_claim_ids": [],
    }


def _consensus_review(
    primary: Dict[str, Any], secondary: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if secondary is None:
        return primary
    score_names = set(primary.get("scores", {})) | set(secondary.get("scores", {}))
    scores = {
        name: min(
            float(primary.get("scores", {}).get(name, 0.0)),
            float(secondary.get("scores", {}).get(name, 0.0)),
        )
        for name in score_names
    }
    both_accept = primary.get("verdict") == "accept" and secondary.get("verdict") == "accept"
    return {
        "episode_id": str(primary.get("episode_id") or secondary.get("episode_id") or ""),
        "verdict": "accept" if both_accept else "reject",
        "overall_score": min(
            float(primary.get("overall_score", 0.0)),
            float(secondary.get("overall_score", 0.0)),
        ),
        "scores": scores,
        "issues": ["primary: %s" % item for item in primary.get("issues", [])]
        + ["secondary: %s" % item for item in secondary.get("issues", [])],
        "required_fixes": ["primary: %s" % item for item in primary.get("required_fixes", [])]
        + ["secondary: %s" % item for item in secondary.get("required_fixes", [])],
        "defect_scope": (
            "blueprint"
            if "blueprint" in {primary.get("defect_scope"), secondary.get("defect_scope")}
            else "episode"
        ),
        "suspected_source_leakage": bool(primary.get("suspected_source_leakage"))
        or bool(secondary.get("suspected_source_leakage")),
        "unsupported_claim_ids": sorted(
            set(map(str, primary.get("unsupported_claim_ids", [])))
            | set(map(str, secondary.get("unsupported_claim_ids", [])))
        ),
    }


def archived_episode_id_collisions(
    root: Path, generated_dir: Path, jobs: Sequence[EpisodeJob]
) -> List[str]:
    pilot_root = generated_dir.parent / "pilots"
    if not pilot_root.exists():
        return []
    archived_ids = {
        path.stem
        for path in pilot_root.rglob("generated/*.json")
        if path.is_file()
    }
    return sorted({job.episode_id for job in jobs} & archived_ids)


def generation_model_roles(config: Dict[str, Any]) -> Dict[str, str]:
    azure = config["azure"]
    return {
        "blueprint_planner": str(azure.get("blueprint_deployment") or azure.get("deployment") or ""),
        "blueprint_quality_critic": str(
            azure.get("blueprint_judge_deployment") or azure.get("judge_deployment") or ""
        ),
        "episode_constructor": str(azure.get("deployment") or ""),
        "episode_editor": str(
            azure.get("episode_editor_deployment") or azure.get("blueprint_deployment") or ""
        ),
        "episode_construction_critic": str(azure.get("episode_critic_deployment") or ""),
        "primary_episode_judge": str(azure.get("judge_deployment") or ""),
        "secondary_episode_judge": str(azure.get("secondary_judge_deployment") or ""),
    }


def model_role_separation_errors(config: Dict[str, Any]) -> List[str]:
    if not bool(config.get("generation", {}).get("require_distinct_model_roles", False)):
        return []
    roles = generation_model_roles(config)
    errors = ["missing deployment for role %s" % role for role, value in roles.items() if not value]
    by_deployment: Dict[str, List[str]] = {}
    for role, deployment in roles.items():
        if deployment:
            by_deployment.setdefault(deployment, []).append(role)
    allowed_shared_roles = {frozenset({"blueprint_planner", "episode_editor"})}
    for deployment, assigned_roles in by_deployment.items():
        if len(assigned_roles) > 1:
            if frozenset(assigned_roles) in allowed_shared_roles:
                continue
            errors.append(
                "deployment %s is assigned to multiple required roles: %s"
                % (deployment, ", ".join(sorted(assigned_roles)))
            )
    blueprint_judge = roles["blueprint_quality_critic"]
    if blueprint_judge == roles["blueprint_planner"]:
        errors.append("the blueprint planner cannot be its own blueprint quality judge")
    return errors


def episode_attempt_role(config: Dict[str, Any], attempt: int) -> Tuple[str, str]:
    roles = generation_model_roles(config)
    if attempt <= 1:
        return roles["episode_constructor"], "first_draft_constructor"
    return roles["episode_editor"], "episode_editor_reconstructor"


def generate_episodes(
    config: Dict[str, Any], root: Path, count: int, resume: bool = True, offset: int = 0
) -> Dict[str, int]:
    role_errors = model_role_separation_errors(config)
    if role_errors:
        raise ValueError("Invalid model-role separation: %s" % "; ".join(role_errors))
    calibration_errors = construction_critic_calibration_errors(config, root)
    if calibration_errors:
        raise ValueError(
            "Construction critic is not calibrated: %s" % "; ".join(calibration_errors)
        )
    output = config["output"]
    all_chunks = list(iter_jsonl(root / output["chunks_file"]))
    eligible_chunks, source_quality_filtered = filter_generation_chunks_by_source_quality(
        all_chunks, config["generation"]
    )
    jobs = build_jobs(eligible_chunks, config, count, offset=offset)
    generated_dir = root / output["generated_dir"]
    collisions = archived_episode_id_collisions(root, generated_dir, jobs)
    if collisions:
        raise ValueError(
            "scheduled episode IDs already exist in archived pilots; choose an unused offset: %s"
            % collisions[:10]
        )
    jobs, checked_job_chunks, language_filtered = validate_job_chunks_by_language(
        jobs, config["generation"]
    )
    generated_dir.mkdir(parents=True, exist_ok=True)
    blueprint_dir = root / output.get(
        "blueprints_dir", str(Path(output["generated_dir"]).parent / "blueprints")
    )
    blueprint_dir.mkdir(parents=True, exist_ok=True)
    blueprint_system = (root / "prompts/blueprint_system.md").read_text(encoding="utf-8")
    episode_system = (root / "prompts/episode_system.md").read_text(encoding="utf-8")
    episode_editor_system = (
        (root / "prompts/episode_editor_system.md").read_text(encoding="utf-8")
        + "\n\n"
        + episode_system
    )
    require_episode_quality = bool(
        config["generation"].get("require_episode_quality_critic", False)
    )
    episode_critic_system = (
        (root / "prompts/episode_critic_system.md").read_text(encoding="utf-8")
        if require_episode_quality
        else ""
    )
    require_blueprint_quality = bool(
        config["generation"].get("require_blueprint_quality_judge", False)
    )
    blueprint_judge_system = (
        (root / "prompts/blueprint_judge_system.md").read_text(encoding="utf-8")
        if require_blueprint_quality
        else ""
    )
    blueprint_schema = root / "schemas/task-blueprint.schema.json"
    episode_schema = root / "schemas/episode.schema.json"
    episode_config = dict(config["azure"])
    episode_config["deployment"] = generation_model_roles(config)["episode_constructor"]
    episode_client = AzureTeacherClient(episode_config, root / output["request_log"])
    blueprint_config = dict(config["azure"])
    blueprint_config["deployment"] = generation_model_roles(config)["blueprint_planner"]
    blueprint_client = AzureTeacherClient(blueprint_config, root / output["request_log"])
    episode_editor_config = dict(config["azure"])
    episode_editor_config["deployment"] = generation_model_roles(config)["episode_editor"]
    if episode_editor_config["deployment"] == blueprint_config["deployment"]:
        episode_editor_client = blueprint_client
        owns_episode_editor_client = False
    else:
        episode_editor_client = AzureTeacherClient(
            episode_editor_config, root / output["request_log"]
        )
        owns_episode_editor_client = True
    episode_critic_client: Optional[AzureTeacherClient] = None
    episode_critic_config: Optional[Dict[str, Any]] = None
    if require_episode_quality:
        episode_critic_config = dict(config["azure"])
        episode_critic_config["deployment"] = config["azure"].get(
            "episode_critic_deployment"
        )
        if not episode_critic_config["deployment"]:
            raise ValueError("Episode quality critic is required but no deployment is configured")
        episode_critic_client = AzureTeacherClient(
            episode_critic_config, root / output["request_log"]
        )
    blueprint_judge_client: Optional[AzureTeacherClient] = None
    blueprint_judge_config: Optional[Dict[str, Any]] = None
    if require_blueprint_quality:
        blueprint_judge_config = dict(config["azure"])
        blueprint_judge_config["deployment"] = config["azure"].get(
            "blueprint_judge_deployment",
            config["azure"].get("judge_deployment", config["azure"]["deployment"]),
        )
        blueprint_judge_client = AzureTeacherClient(
            blueprint_judge_config, root / output["request_log"]
        )
    max_chars = int(config["generation"].get("max_source_chars", 18000))
    max_blueprint_attempts = int(config["generation"].get("max_blueprint_attempts", 3))
    max_episode_attempts = int(config["generation"].get("max_episode_attempts", 2))
    stats = {
        "source_chunks": len(all_chunks),
        "source_quality_filtered": source_quality_filtered,
        "validated_job_chunks": checked_job_chunks,
        "job_chunks_filtered": language_filtered,
        "scheduled": len(jobs),
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "blueprints_generated": 0,
        "blueprints_reused": 0,
        "blueprint_rejected_attempts": 0,
        "blueprint_quality_rejected_attempts": 0,
        "episode_rejected_attempts": 0,
        "episode_quality_rejected_attempts": 0,
    }

    def run(job: EpisodeJob) -> Tuple[str, Dict[str, int]]:
        local_stats = {
            "blueprints_generated": 0,
            "blueprints_reused": 0,
            "blueprint_rejected_attempts": 0,
            "blueprint_quality_rejected_attempts": 0,
            "episode_rejected_attempts": 0,
            "episode_quality_rejected_attempts": 0,
        }
        path = generated_dir / (job.episode_id + ".json")
        if resume and path.exists():
            return "skipped", local_stats
        packet = _source_packet(job, max_chars)
        approved_path = blueprint_dir / (job.episode_id + ".json")
        blueprint: Optional[Dict[str, Any]] = None
        if resume and approved_path.exists():
            candidate = json.loads(approved_path.read_text(encoding="utf-8"))
            if not _blueprint_errors(candidate, job, blueprint_schema):
                blueprint = candidate
                local_stats["blueprints_reused"] += 1

        blueprint_feedback: List[str] = []
        rejected_blueprint: Optional[Dict[str, Any]] = None
        if blueprint is None:
            blueprint_schema_value = json.loads(blueprint_schema.read_text(encoding="utf-8"))
            for attempt in range(1, max_blueprint_attempts + 1):
                planner_action = (
                    "Design one expert task blueprint from the evidence."
                    if rejected_blueprint is None
                    else (
                        "Critique and fully rewrite the rejected blueprint. Correct every listed defect; "
                        "do not merely score, annotate, or cosmetically edit it."
                    )
                )
                blueprint_prompt = (
                    "%s\n\n"
                    "TARGET DOMAIN:\n%s\n\nTARGET OBJECTIVE:\n%s\n\nVARIATION SEED:\n%d\n\n"
                    "SOURCE PACKET:\n%s\n\nSTRICT BLUEPRINT JSON SCHEMA:\n%s\n\n"
                    "REJECTED BLUEPRINT TO CRITIQUE AND REWRITE:\n%s\n\n"
                    "PREVIOUS VALIDATION ERRORS TO CORRECT:\n%s"
                    % (
                        planner_action,
                        job.domain,
                        job.objective,
                        job.variation_seed,
                        json.dumps(packet, ensure_ascii=False, indent=2),
                        json.dumps(blueprint_schema_value, ensure_ascii=False, indent=2),
                        json.dumps(rejected_blueprint, ensure_ascii=False, indent=2),
                        json.dumps(blueprint_feedback, ensure_ascii=False, indent=2),
                    )
                )
                completion = blueprint_client.chat_json(
                    blueprint_system, blueprint_prompt, "blueprint-plan-or-rewrite", job.episode_id
                )
                candidate = _normalize_blueprint(completion.value, job)
                errors = _blueprint_errors(candidate, job, blueprint_schema)
                if not errors and blueprint_judge_client is not None:
                    quality_prompt = (
                        "SOURCE PACKET:\n%s\n\nCANDIDATE BLUEPRINT:\n%s"
                        % (
                            json.dumps(packet, ensure_ascii=False, indent=2),
                            json.dumps(candidate, ensure_ascii=False, indent=2),
                        )
                    )
                    quality_completion = blueprint_judge_client.chat_json(
                        blueprint_judge_system,
                        quality_prompt,
                        "blueprint-judge",
                        job.episode_id,
                    )
                    quality_review = quality_completion.value
                    quality_errors = _blueprint_quality_errors(
                        quality_review,
                        float(config["generation"].get("blueprint_quality_threshold", 0.84)),
                        float(
                            config["generation"].get(
                                "minimum_blueprint_expertise_uplift",
                                config["generation"].get("minimum_expertise_uplift", 0.0),
                            )
                        ),
                    )
                    if quality_errors:
                        local_stats["blueprint_quality_rejected_attempts"] += 1
                        errors.extend(quality_errors)
                    atomic_write_json(
                        root / "data/raw/azure/blueprint-judge" / (
                            "%s-attempt-%d.json" % (job.episode_id, attempt)
                        ),
                        {
                            "response": quality_review,
                            "validation_errors": quality_errors,
                            "provenance": {
                                "deployment": blueprint_judge_config["deployment"]
                                if blueprint_judge_config
                                else None,
                                "request_id": quality_completion.request_id,
                                "prompt_template_hash": stable_hash(blueprint_judge_system, 32),
                                "blueprint_hash": stable_hash(candidate, 32),
                                "source_packet_hash": stable_hash(packet, 32),
                                "prompt_tokens": quality_completion.prompt_tokens,
                                "completion_tokens": quality_completion.completion_tokens,
                                "elapsed_seconds": quality_completion.elapsed_seconds,
                                "attempt": attempt,
                            },
                        },
                    )
                raw_attempt_path = root / "data/raw/azure/blueprint" / (
                    "%s-attempt-%d.json" % (job.episode_id, attempt)
                )
                atomic_write_json(
                    raw_attempt_path,
                    {
                        "response": candidate,
                        "validation_errors": errors,
                        "provenance": {
                            "deployment": blueprint_config["deployment"],
                            "role": "blueprint_planner_critic_rewriter",
                            "request_id": completion.request_id,
                            "prompt_template_hash": stable_hash(blueprint_system, 32),
                            "source_packet_hash": stable_hash(packet, 32),
                            "prompt_tokens": completion.prompt_tokens,
                            "completion_tokens": completion.completion_tokens,
                            "elapsed_seconds": completion.elapsed_seconds,
                            "attempt": attempt,
                        },
                    },
                )
                if not errors:
                    blueprint = candidate
                    atomic_write_json(approved_path, blueprint)
                    local_stats["blueprints_generated"] += 1
                    break
                local_stats["blueprint_rejected_attempts"] += 1
                blueprint_feedback = errors
                rejected_blueprint = candidate
        if blueprint is None:
            failure = ValueError(
                "blueprint failed validation after %d attempts: %s"
                % (max_blueprint_attempts, blueprint_feedback[:8])
            )
            failure.local_stats = local_stats  # type: ignore[attr-defined]
            raise failure

        known_chunk_ids = {str(chunk["chunk_id"]) for chunk in job.chunks}
        episode_feedback: List[str] = []
        rejected_episode: Optional[Dict[str, Any]] = None
        for attempt in range(1, max_episode_attempts + 1):
            constructor_action = (
                "Build one complete episode from the approved blueprint."
                if rejected_episode is None
                else (
                    "Critique and fully rewrite the rejected episode. Preserve the approved blueprint, "
                    "but correct every local and construction-critic defect. Do not merely annotate it."
                )
            )
            prompt = (
                "%s Do not change its task, "
                "premises, constraints, lanes, planned artifacts, claims, or verification coverage.\n\n"
                "APPROVED BLUEPRINT:\n%s\n\nSOURCE PACKET:\n%s\n\n"
                "REQUIRED OUTPUT SHAPE:\n%s\n\nREJECTED EPISODE TO CRITIQUE AND REWRITE:\n%s\n\n"
                "PREVIOUS VALIDATION ERRORS TO CORRECT:\n%s"
                % (
                    constructor_action,
                    json.dumps(blueprint, ensure_ascii=False, indent=2),
                    json.dumps(packet, ensure_ascii=False, indent=2),
                    json.dumps(_output_contract(config, job, blueprint), ensure_ascii=False, indent=2),
                    json.dumps(rejected_episode, ensure_ascii=False, indent=2),
                    json.dumps(episode_feedback, ensure_ascii=False, indent=2),
                )
            )
            active_deployment, active_role = episode_attempt_role(config, attempt)
            active_client = episode_client if attempt == 1 else episode_editor_client
            active_system = episode_system if attempt == 1 else episode_editor_system
            operation = "generate-first-draft" if attempt == 1 else "edit-reconstruct-episode"
            completion = active_client.chat_json(
                active_system, prompt, operation, job.episode_id
            )
            episode = _normalize_episode(completion.value, job)
            episode = _apply_blueprint_contract(episode, blueprint)
            episode["generation_metadata"] = {
                "provider": "azure",
                "deployment": active_deployment,
                "role": active_role,
                "first_draft_deployment": episode_config["deployment"],
                "editor_deployment": (
                    episode_editor_config["deployment"] if attempt > 1 else None
                ),
                "request_id": completion.request_id,
                "prompt_template_hash": stable_hash(active_system, 32),
                "blueprint_prompt_template_hash": stable_hash(blueprint_system, 32),
                "blueprint_hash": stable_hash(blueprint, 32),
                "source_packet_hash": stable_hash(packet, 32),
                "variation_seed": job.variation_seed,
            }
            errors = _episode_write_errors(
                episode,
                known_chunk_ids,
                episode_schema,
                blueprint=blueprint,
                source_packet=packet,
            )
            if not errors and episode_critic_client is not None:
                critic_prompt = (
                    "SOURCE PACKET:\n%s\n\nAPPROVED BLUEPRINT:\n%s\n\nCANDIDATE EPISODE:\n%s"
                    % (
                        json.dumps(packet, ensure_ascii=False, indent=2),
                        json.dumps(blueprint, ensure_ascii=False, indent=2),
                        json.dumps(episode, ensure_ascii=False, indent=2),
                    )
                )
                critic_completion = episode_critic_client.chat_json(
                    episode_critic_system,
                    critic_prompt,
                    "episode-construction-critic",
                    job.episode_id,
                )
                critic_review = critic_completion.value
                critic_threshold = float(
                    config["generation"].get("episode_quality_threshold", 0.84)
                )
                critic_expertise_threshold = float(
                    config["generation"].get(
                        "minimum_episode_expertise_uplift",
                        config["generation"].get("minimum_expertise_uplift", 0.0),
                    )
                )
                critic_protocol_errors = _construction_critic_protocol_errors(
                    critic_review,
                    critic_threshold,
                    critic_expertise_threshold,
                )
                critic_errors = (
                    []
                    if critic_protocol_errors
                    else _construction_critic_candidate_errors(
                        critic_review,
                        critic_threshold,
                        critic_expertise_threshold,
                    )
                )
                if critic_errors:
                    local_stats["episode_quality_rejected_attempts"] += 1
                    errors.extend(critic_errors)
                atomic_write_json(
                    root / "data/raw/azure/episode-critic" / (
                        "%s-attempt-%d.json" % (job.episode_id, attempt)
                    ),
                    {
                        "response": critic_review,
                        "validation_errors": critic_errors,
                        "protocol_errors": critic_protocol_errors,
                        "provenance": {
                            "deployment": episode_critic_config["deployment"]
                            if episode_critic_config
                            else None,
                            "request_id": critic_completion.request_id,
                            "prompt_template_hash": stable_hash(episode_critic_system, 32),
                            "episode_hash": stable_hash(episode, 32),
                            "blueprint_hash": stable_hash(blueprint, 32),
                            "prompt_tokens": critic_completion.prompt_tokens,
                            "completion_tokens": critic_completion.completion_tokens,
                            "elapsed_seconds": critic_completion.elapsed_seconds,
                            "attempt": attempt,
                        },
                    },
                )
                if critic_protocol_errors:
                    raise ValueError(
                        "construction critic returned an invalid review: %s"
                        % "; ".join(critic_protocol_errors)
                    )
            raw_value = {
                "response": episode,
                "validation_errors": errors,
                "provenance": {
                    "deployment": active_deployment,
                    "role": active_role,
                    "first_draft_deployment": episode_config["deployment"],
                    "editor_deployment": (
                        episode_editor_config["deployment"] if attempt > 1 else None
                    ),
                    "request_id": completion.request_id,
                    "prompt_template_hash": stable_hash(active_system, 32),
                    "blueprint_hash": stable_hash(blueprint, 32),
                    "source_packet_hash": stable_hash(packet, 32),
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "elapsed_seconds": completion.elapsed_seconds,
                    "attempt": attempt,
                },
            }
            raw_attempt_path = root / "data/raw/azure/generate" / (
                "%s-attempt-%d.json" % (job.episode_id, attempt)
            )
            atomic_write_json(raw_attempt_path, raw_value)
            if not errors:
                atomic_write_json(
                    root / "data/raw/azure/generate" / (job.episode_id + ".json"), raw_value
                )
                atomic_write_json(path, episode)
                return "generated", local_stats
            local_stats["episode_rejected_attempts"] += 1
            episode_feedback = errors
            rejected_episode = episode
        failure = ValueError(
            "episode failed local validation after %d attempts: %s"
            % (max_episode_attempts, episode_feedback[:8])
        )
        failure.local_stats = local_stats  # type: ignore[attr-defined]
        raise failure

    try:
        with ThreadPoolExecutor(max_workers=int(config["azure"].get("max_workers", 8))) as executor:
            futures = {executor.submit(run, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    status, local_stats = future.result()
                    stats[status] += 1
                    for key, value in local_stats.items():
                        stats[key] += value
                except Exception as exc:
                    stats["failed"] += 1
                    for key, value in getattr(exc, "local_stats", {}).items():
                        stats[key] += value
                    append_jsonl(
                        root / output["run_log"],
                        {"stage": "generate", "episode_id": job.episode_id, "status": "failed", "error": str(exc)[:1000]},
                    )
    finally:
        episode_client.close()
        if owns_episode_editor_client:
            episode_editor_client.close()
        blueprint_client.close()
        if episode_critic_client is not None:
            episode_critic_client.close()
        if blueprint_judge_client is not None:
            blueprint_judge_client.close()
    return stats


def judge_episodes(config: Dict[str, Any], root: Path, resume: bool = True) -> Dict[str, int]:
    role_errors = model_role_separation_errors(config)
    if role_errors:
        raise ValueError("Invalid model-role separation: %s" % "; ".join(role_errors))
    output = config["output"]
    generated_dir = root / output["generated_dir"]
    reviewed_dir = root / output["reviewed_dir"]
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    blueprint_dir = root / output.get(
        "blueprints_dir", str(Path(output["generated_dir"]).parent / "blueprints")
    )
    chunks = list(iter_jsonl(root / output["chunks_file"]))
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    known_ids = set(chunks_by_id)
    episode_schema = root / "schemas/episode.schema.json"
    review_schema = root / "schemas/review.schema.json"
    threshold = float(config["generation"].get("quality_threshold", 0.82))
    system = (root / "prompts/judge_system.md").read_text(encoding="utf-8")
    primary_config = dict(config["azure"])
    primary_config["deployment"] = config["azure"].get(
        "judge_deployment", config["azure"]["deployment"]
    )
    require_dual = bool(config["generation"].get("require_dual_judge", False))
    secondary_deployment = config["azure"].get("secondary_judge_deployment")
    if require_dual and not secondary_deployment:
        raise ValueError("Dual-judge acceptance is required but no secondary judge deployment is configured")
    if require_dual and str(secondary_deployment) == str(primary_config["deployment"]):
        raise ValueError("Primary and secondary judge deployments must be distinct")
    secondary_config: Optional[Dict[str, Any]] = None
    if secondary_deployment:
        secondary_config = dict(config["azure"])
        secondary_config["deployment"] = secondary_deployment

    primary_client = AzureTeacherClient(primary_config, root / output["request_log"])
    secondary_client = (
        AzureTeacherClient(secondary_config, root / output["request_log"])
        if require_dual and secondary_config is not None
        else None
    )
    paths = sorted(generated_dir.glob("*.json"))
    stats = {
        "scheduled": len(paths),
        "accepted": 0,
        "rejected": 0,
        "skipped": 0,
        "failed": 0,
        "adversarial_rejected": 0,
        "secondary_judged": 0,
        "judge_disagreements": 0,
    }

    def save_wrapper(
        reviewed_path: Path,
        episode: Dict[str, Any],
        review: Dict[str, Any],
        *,
        accepted: bool,
        static_errors: Sequence[str],
        adversarial_errors: Sequence[str],
        judge_errors: Sequence[str],
        primary_review: Optional[Dict[str, Any]] = None,
        secondary_review: Optional[Dict[str, Any]] = None,
        primary_metadata: Optional[Dict[str, Any]] = None,
        secondary_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        atomic_write_json(
            reviewed_path,
            {
                "episode": episode,
                "review": review,
                "primary_review": primary_review,
                "secondary_review": secondary_review,
                "accepted": accepted,
                "static_errors": list(static_errors),
                "adversarial_errors": list(adversarial_errors),
                "judge_errors": list(judge_errors),
                "judge_consensus": bool(
                    accepted
                    and primary_review
                    and (
                        not require_dual
                        or (
                            secondary_review
                            and primary_review.get("verdict") == "accept"
                            and secondary_review.get("verdict") == "accept"
                        )
                    )
                ),
                "judge_metadata": primary_metadata
                or {"provider": "local", "deployment": None, "request_id": None},
                "secondary_judge_metadata": secondary_metadata,
            },
        )

    def model_judge(
        client: AzureTeacherClient,
        deployment: str,
        prompt: str,
        episode: Dict[str, Any],
        path: Path,
        operation: str,
        raw_directory: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        completion = client.chat_json(
            system,
            prompt,
            operation,
            str(episode.get("episode_id", path.stem)),
        )
        provenance = {
            "deployment": deployment,
            "request_id": completion.request_id,
            "prompt_template_hash": stable_hash(system, 32),
            "episode_hash": stable_hash(episode, 32),
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "elapsed_seconds": completion.elapsed_seconds,
        }
        atomic_write_json(
            root / "data/raw/azure" / raw_directory / path.name,
            {"response": completion.value, "provenance": provenance},
        )
        review = _normalize_review(
            completion.value,
            str(episode.get("episode_id", path.stem)),
            threshold,
        )
        return review, {
            "provider": "azure",
            "deployment": deployment,
            "request_id": completion.request_id,
            "prompt_template_hash": stable_hash(system, 32),
        }

    def run(path: Path) -> Tuple[str, Dict[str, int]]:
        local_stats = {"adversarial_rejected": 0, "secondary_judged": 0, "judge_disagreements": 0}
        reviewed_path = reviewed_dir / path.name
        if resume and reviewed_path.exists():
            return "skipped", local_stats
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(episode.get("episode_id", path.stem))
        static_errors = schema_errors(episode, episode_schema)
        static_errors.extend(episode_invariant_errors(episode, known_ids))
        blueprint: Optional[Dict[str, Any]] = None
        blueprint_path = blueprint_dir / (episode_id + ".json")
        if episode.get("generation_metadata", {}).get("blueprint_hash"):
            if not blueprint_path.exists():
                static_errors.append("approved blueprint is missing")
            else:
                blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
                static_errors.extend(
                    schema_errors(blueprint, root / "schemas/task-blueprint.schema.json")
                )
                static_errors.extend(
                    blueprint_invariant_errors(
                        blueprint,
                        known_ids,
                        known_chunk_texts={
                            chunk_id: str(chunk.get("text", ""))
                            for chunk_id, chunk in chunks_by_id.items()
                        },
                    )
                )
                static_errors.extend(episode_blueprint_errors(episode, blueprint))
        if static_errors:
            review = _rejection_review(
                episode_id, ["Static validation: %s" % item for item in static_errors]
            )
            save_wrapper(
                reviewed_path,
                episode,
                review,
                accepted=False,
                static_errors=static_errors,
                adversarial_errors=[],
                judge_errors=["static validation failed; model judges not called"],
            )
            return "rejected", local_stats
        chunk_ids = {
            chunk_id
            for ref in episode.get("source_refs", [])
            for chunk_id in ref.get("chunk_ids", [])
        }
        packet = {
            "chunks": [
                {
                    "source_id": chunks_by_id[item]["source_id"],
                    "chunk_id": chunks_by_id[item]["chunk_id"],
                    "title": chunks_by_id[item].get("title"),
                    "version": chunks_by_id[item].get("version"),
                    "license": chunks_by_id[item].get("license"),
                    "source_group": chunks_by_id[item].get("source_group"),
                    "data_role": chunks_by_id[item].get("data_role"),
                    "text": chunks_by_id[item]["text"],
                }
                for item in sorted(chunk_ids)
                if item in chunks_by_id
            ]
        }
        deterministic_errors = adversarial_acceptance_errors(
            episode,
            blueprint=blueprint,
            source_packet=packet,
            trusted_execution_run_ids=set(),
        )
        if deterministic_errors:
            local_stats["adversarial_rejected"] = 1
            review = _rejection_review(
                episode_id,
                ["Adversarial acceptance gate: %s" % item for item in deterministic_errors],
            )
            save_wrapper(
                reviewed_path,
                episode,
                review,
                accepted=False,
                static_errors=[],
                adversarial_errors=deterministic_errors,
                judge_errors=["adversarial acceptance gate failed; model judges not called"],
            )
            return "rejected", local_stats

        prompt = "APPROVED BLUEPRINT:\n%s\n\nSOURCE PACKET:\n%s\n\nEPISODE:\n%s\n\nSTATIC VALIDATION ISSUES:\n%s" % (
            json.dumps(blueprint or {}, ensure_ascii=False, indent=2),
            json.dumps(packet, ensure_ascii=False, indent=2),
            json.dumps(episode, ensure_ascii=False, indent=2),
            json.dumps(static_errors, ensure_ascii=False),
        )
        try:
            primary_review, primary_metadata = model_judge(
                primary_client,
                str(primary_config["deployment"]),
                prompt,
                episode,
                path,
                "judge-primary",
                "judge",
            )
        except Exception as exc:
            message = "primary judge failed closed: %s" % str(exc)[:500]
            review = _rejection_review(episode_id, [message])
            save_wrapper(
                reviewed_path,
                episode,
                review,
                accepted=False,
                static_errors=[],
                adversarial_errors=[],
                judge_errors=[message],
            )
            return "rejected", local_stats

        primary_errors = review_errors(
            primary_review,
            review_schema,
            threshold,
            float(config["generation"].get("minimum_expertise_uplift", 0.0)),
        )
        if primary_errors:
            save_wrapper(
                reviewed_path,
                episode,
                primary_review,
                accepted=False,
                static_errors=[],
                adversarial_errors=[],
                judge_errors=primary_errors,
                primary_review=primary_review,
                primary_metadata=primary_metadata,
            )
            return "rejected", local_stats

        secondary_review: Optional[Dict[str, Any]] = None
        secondary_metadata: Optional[Dict[str, Any]] = None
        secondary_errors: List[str] = []
        if require_dual:
            local_stats["secondary_judged"] = 1
            try:
                assert secondary_client is not None and secondary_config is not None
                secondary_review, secondary_metadata = model_judge(
                    secondary_client,
                    str(secondary_config["deployment"]),
                    prompt,
                    episode,
                    path,
                    "judge-secondary",
                    "judge-secondary",
                )
                secondary_errors = review_errors(
                    secondary_review,
                    review_schema,
                    threshold,
                    float(config["generation"].get("minimum_expertise_uplift", 0.0)),
                )
            except Exception as exc:
                secondary_errors = ["secondary judge failed closed: %s" % str(exc)[:500]]

        consensus = _consensus_review(primary_review, secondary_review)
        disagreement = bool(
            require_dual
            and (
                secondary_review is None
                or secondary_review.get("verdict") != primary_review.get("verdict")
                or bool(secondary_errors)
            )
        )
        if disagreement:
            local_stats["judge_disagreements"] = 1
        combined_errors = list(primary_errors) + [
            "secondary judge: %s" % item for item in secondary_errors
        ]
        if disagreement and not combined_errors:
            combined_errors.append("independent judges disagreed; acceptance failed closed")
        accepted = not combined_errors and (not require_dual or not disagreement)
        save_wrapper(
            reviewed_path,
            episode,
            consensus,
            accepted=accepted,
            static_errors=[],
            adversarial_errors=[],
            judge_errors=combined_errors,
            primary_review=primary_review,
            secondary_review=secondary_review,
            primary_metadata=primary_metadata,
            secondary_metadata=secondary_metadata,
        )
        return ("accepted" if accepted else "rejected"), local_stats

    try:
        with ThreadPoolExecutor(max_workers=int(config["azure"].get("max_workers", 8))) as executor:
            futures = {executor.submit(run, path): path for path in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    status, local_stats = future.result()
                    stats[status] += 1
                    for key, value in local_stats.items():
                        stats[key] += value
                except Exception as exc:
                    stats["failed"] += 1
                    append_jsonl(
                        root / output["run_log"],
                        {"stage": "judge", "episode_id": path.stem, "status": "failed", "error": str(exc)[:1000]},
                    )
    finally:
        primary_client.close()
        if secondary_client is not None:
            secondary_client.close()
    return stats


def repair_rejected_episodes(
    config: Dict[str, Any], root: Path, resume: bool = True
) -> Dict[str, int]:
    """Revise rejected episodes from judge feedback while preserving source lineage."""
    output = config["output"]
    reviewed_dir = root / output["reviewed_dir"]
    generated_dir = root / output["generated_dir"]
    chunks = list(iter_jsonl(root / output["chunks_file"]))
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    known_chunk_ids = set(chunks_by_id)
    system = (root / "prompts/repair_system.md").read_text(encoding="utf-8")
    episode_schema = root / "schemas/episode.schema.json"
    blueprint_schema = root / "schemas/task-blueprint.schema.json"
    blueprint_dir = root / output.get(
        "blueprints_dir", str(Path(output["generated_dir"]).parent / "blueprints")
    )
    client = AzureTeacherClient(config["azure"], root / output["request_log"])
    max_rounds = int(config["generation"].get("max_repair_rounds", 1))
    max_validation_attempts = int(config["generation"].get("max_repair_validation_attempts", 2))
    paths = sorted(reviewed_dir.glob("*.json"))
    stats = {
        "scheduled": 0,
        "repaired": 0,
        "blueprint_fatal": 0,
        "skipped": 0,
        "failed": 0,
    }

    def run(path: Path) -> str:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if wrapper.get("accepted"):
            return "skipped"
        original = wrapper.get("episode", {})
        if resume and int(original.get("repair_metadata", {}).get("round", 0)) >= max_rounds:
            return "skipped"
        episode_id = str(original.get("episode_id", path.stem))
        review_scopes = {
            str(review.get("defect_scope", "episode"))
            for review in (
                wrapper.get("review") or {},
                wrapper.get("primary_review") or {},
                wrapper.get("secondary_review") or {},
            )
        }
        if "blueprint" in review_scopes:
            quarantine_path = generated_dir.parent / "history" / "blueprint-quarantine" / path.name
            atomic_write_json(
                quarantine_path,
                {
                    "episode_id": episode_id,
                    "reason": "independent review classified the defect as blueprint-fatal",
                    "blueprint": (
                        json.loads((blueprint_dir / (episode_id + ".json")).read_text(encoding="utf-8"))
                        if (blueprint_dir / (episode_id + ".json")).exists()
                        else None
                    ),
                    "primary_review": wrapper.get("primary_review") or wrapper.get("review"),
                    "secondary_review": wrapper.get("secondary_review"),
                },
            )
            append_jsonl(
                root / output["run_log"],
                {
                    "stage": "repair",
                    "episode_id": episode_id,
                    "status": "blueprint_fatal",
                    "error": "blueprint must be regenerated; episode repair was not attempted",
                },
            )
            return "blueprint_fatal"
        blueprint_path = blueprint_dir / (episode_id + ".json")
        blueprint: Optional[Dict[str, Any]] = None
        if blueprint_path.exists():
            blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
            blueprint_errors = schema_errors(blueprint, blueprint_schema)
            blueprint_errors.extend(
                blueprint_invariant_errors(
                    blueprint,
                    known_chunk_ids,
                    known_chunk_texts={
                        chunk_id: str(chunk.get("text", ""))
                        for chunk_id, chunk in chunks_by_id.items()
                    },
                )
            )
            if blueprint_errors:
                raise ValueError("approved blueprint is invalid: %s" % blueprint_errors[:8])
        elif original.get("generation_metadata", {}).get("blueprint_hash"):
            raise ValueError("approved blueprint is missing; refusing to repair blueprint-bound episode")
        chunk_ids = {
            str(chunk_id)
            for ref in original.get("source_refs", [])
            for chunk_id in ref.get("chunk_ids", [])
        }
        packet = {
            "chunks": [
                {
                    "source_id": chunks_by_id[item]["source_id"],
                    "chunk_id": chunks_by_id[item]["chunk_id"],
                    "title": chunks_by_id[item].get("title"),
                    "version": chunks_by_id[item].get("version"),
                    "license": chunks_by_id[item].get("license"),
                    "source_group": chunks_by_id[item].get("source_group"),
                    "data_role": chunks_by_id[item].get("data_role"),
                    "text": chunks_by_id[item]["text"],
                }
                for item in sorted(chunk_ids)
                if item in chunks_by_id
            ]
        }
        validation_feedback: List[str] = []
        repaired: Optional[Dict[str, Any]] = None
        final_raw: Optional[Dict[str, Any]] = None
        repair_job = EpisodeJob(
            episode_id=episode_id,
            domain=str(original.get("domain", "general_reasoning_and_learning")),
            objective=str(original.get("subdomain", "general synthesis")),
            source_group=str(original.get("source_group", "unknown")),
            lineage_component_id=str(original.get("lineage_component_id", "unknown")),
            chunks=tuple(packet["chunks"]),
            variation_seed=0,
        )
        repair_contract = _output_contract(config, repair_job, blueprint)
        for attempt in range(1, max_validation_attempts + 1):
            prompt = (
                "APPROVED BLUEPRINT:\n%s\n\nSOURCE PACKET:\n%s\n\nREJECTED EPISODE:\n%s\n\n"
                "STATIC VALIDATION ERRORS:\n%s\n\nADVERSARIAL ACCEPTANCE ERRORS:\n%s\n\n"
                "PRIMARY INDEPENDENT REVIEW:\n%s\n\nSECONDARY INDEPENDENT REVIEW:\n%s\n\n"
                "JUDGE CONSENSUS ERRORS:\n%s\n\n"
                "FAILED REPAIR VALIDATION TO CORRECT:\n%s\n\nREQUIRED OUTPUT CONTRACT:\n%s"
                % (
                    json.dumps(blueprint or {}, ensure_ascii=False, indent=2),
                    json.dumps(packet, ensure_ascii=False, indent=2),
                    json.dumps(original, ensure_ascii=False, indent=2),
                    json.dumps(wrapper.get("static_errors", []), ensure_ascii=False, indent=2),
                    json.dumps(wrapper.get("adversarial_errors", []), ensure_ascii=False, indent=2),
                    json.dumps(wrapper.get("primary_review") or wrapper.get("review", {}), ensure_ascii=False, indent=2),
                    json.dumps(wrapper.get("secondary_review") or {}, ensure_ascii=False, indent=2),
                    json.dumps(wrapper.get("judge_errors", []), ensure_ascii=False, indent=2),
                    json.dumps(validation_feedback, ensure_ascii=False, indent=2),
                    json.dumps(repair_contract, ensure_ascii=False, indent=2),
                )
            )
            completion = client.chat_json(system, prompt, "repair", episode_id)
            candidate = _normalize_repaired_episode(
                completion.value,
                original,
                {
                    "provider": "azure",
                    "deployment": config["azure"]["deployment"],
                    "request_id": completion.request_id,
                    "prompt_template_hash": stable_hash(system, 32),
                    "review_hash": stable_hash(wrapper.get("review", {}), 32),
                    "validation_attempt": attempt,
                },
            )
            if blueprint is not None:
                candidate = _apply_blueprint_contract(candidate, blueprint)
            validation_feedback = _episode_write_errors(
                candidate,
                known_chunk_ids,
                episode_schema,
                blueprint=blueprint,
                source_packet=packet,
            )
            raw_value = {
                "response": candidate,
                "validation_errors": validation_feedback,
                "provenance": {
                    "deployment": config["azure"]["deployment"],
                    "request_id": completion.request_id,
                    "prompt_template_hash": stable_hash(system, 32),
                    "source_packet_hash": stable_hash(packet, 32),
                    "blueprint_hash": stable_hash(blueprint, 32) if blueprint else None,
                    "rejected_episode_hash": stable_hash(original, 32),
                    "review_hash": stable_hash(wrapper.get("review", {}), 32),
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "elapsed_seconds": completion.elapsed_seconds,
                    "attempt": attempt,
                },
            }
            atomic_write_json(
                root / "data/raw/azure/repair" / ("%s-attempt-%d.json" % (episode_id, attempt)),
                raw_value,
            )
            if not validation_feedback:
                repaired = candidate
                final_raw = raw_value
                break
        if repaired is None or final_raw is None:
            raise ValueError(
                "repair failed closed after %d validation attempts: %s"
                % (max_validation_attempts, validation_feedback[:8])
            )
        atomic_write_json(root / "data/raw/azure/repair" / path.name, final_raw)
        generated_path = generated_dir / path.name
        archive_path = generated_dir.parent / "history" / "pre-repair" / path.name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        if generated_path.exists() and not archive_path.exists():
            shutil.copy2(str(generated_path), str(archive_path))
        atomic_write_json(generated_path, repaired)
        path.unlink()
        return "repaired"

    candidates = []
    for path in paths:
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            original = wrapper.get("episode", {})
            if wrapper.get("accepted") or (
                resume and int(original.get("repair_metadata", {}).get("round", 0)) >= max_rounds
            ):
                stats["skipped"] += 1
            else:
                candidates.append(path)
        except Exception:
            candidates.append(path)
    stats["scheduled"] = len(candidates)
    try:
        with ThreadPoolExecutor(max_workers=int(config["azure"].get("max_workers", 8))) as executor:
            futures = {executor.submit(run, path): path for path in candidates}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    stats[future.result()] += 1
                except Exception as exc:
                    stats["failed"] += 1
                    append_jsonl(
                        root / output["run_log"],
                        {"stage": "repair", "episode_id": path.stem, "status": "failed", "error": str(exc)[:1000]},
                    )
    finally:
        client.close()
    return stats
