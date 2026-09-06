#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Optional

from hlwm_data.language import classify_language, contains_blocked_script
from hlwm_data.util import normalize_text, stable_hash


PRIVATE_REASONING_PATTERNS = (
    re.compile(r"<\s*/?\s*\|?think\|?\s*>", re.IGNORECASE),
    re.compile(r"\bhidden chain[- ]of[- ]thought\b", re.IGNORECASE),
    re.compile(r"\binternal reasoning:\s", re.IGNORECASE),
)


def _visible_episode_text(episode: dict) -> str:
    return "\n".join(
        str(value)
        for value in (
            episode.get("input", {}).get("user_request", ""),
            episode.get("frame", {}).get("objective", ""),
            episode.get("integration", {}).get("published_answer", ""),
        )
        if value
    )


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _content_failures(episode: dict) -> tuple[bool, bool]:
    combined = "\n".join(_all_strings(episode))
    private_reasoning = any(pattern.search(combined) for pattern in PRIVATE_REASONING_PATTERNS)
    visible = _visible_episode_text(episode)
    non_english = contains_blocked_script(combined)
    if visible and not non_english:
        decision = classify_language(visible, expected="en", minimum_confidence=0.70)
        non_english = not decision.accepted
    return non_english, private_reasoning


def accepted_corpus_hash(wrappers: List[dict]) -> str:
    accepted_rows = [
        {
            "episode_id": str(item.get("episode", {}).get("episode_id", "")),
            "episode_hash": stable_hash(item.get("episode", {}), 32),
            "primary_review_hash": stable_hash(item.get("primary_review", {}), 32),
            "secondary_review_hash": stable_hash(item.get("secondary_review", {}), 32),
        }
        for item in wrappers
        if item.get("accepted")
    ]
    return stable_hash(sorted(accepted_rows, key=lambda item: item["episode_id"]), 32)


def manual_audit_valid(wrappers: List[dict], audit: Optional[dict]) -> bool:
    if not isinstance(audit, dict):
        return False
    accepted_ids = {
        str(item.get("episode", {}).get("episode_id", ""))
        for item in wrappers
        if item.get("accepted")
    }
    if not accepted_ids:
        return False
    inspections = audit.get("inspections", [])
    inspected_ids = {
        str(item.get("episode_id", ""))
        for item in inspections
        if isinstance(item, dict) and item.get("verdict") == "pass"
    }
    return bool(
        audit.get("decision") == "pass"
        and str(audit.get("reviewer", "")).strip()
        and str(audit.get("completed_at", "")).strip()
        and audit.get("accepted_corpus_hash") == accepted_corpus_hash(wrappers)
        and inspected_ids == accepted_ids
        and len(inspections) == len(accepted_ids)
        and not audit.get("false_accept_episode_ids")
    )


def assess(
    reviewed_dir: Path,
    minimum_reviewed: int,
    minimum_acceptance: float,
    minimum_expertise: float,
    minimum_first_pass_acceptance: float = 0.50,
    minimum_domains: int = 12,
    minimum_overall: float = 0.84,
    minimum_accepted_expertise: float = 0.80,
    manual_audit: Optional[dict] = None,
) -> dict:
    wrappers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(reviewed_dir.glob("*.json"))]
    accepted = [item for item in wrappers if item.get("accepted")]
    expertise = [float(item.get("review", {}).get("scores", {}).get("expertise_uplift", 0.0)) for item in accepted]
    overall_scores = [float(item.get("review", {}).get("overall_score", 0.0)) for item in accepted]
    domains = Counter(str(item.get("episode", {}).get("domain", "unknown")) for item in accepted)
    static_failures = sum(bool(item.get("static_errors")) for item in wrappers)
    reported_language_failures = sum(
        any("not english" in str(error).lower() or "non-english" in str(error).lower() for error in item.get("static_errors", []))
        for item in wrappers
    )
    reported_private_reasoning_failures = sum(
        any("private chain-of-thought" in str(error).lower() for error in item.get("static_errors", []))
        for item in wrappers
    )
    direct_failures = [_content_failures(item.get("episode", {})) for item in wrappers]
    language_failures = sum(failure[0] for failure in direct_failures)
    private_reasoning_failures = sum(failure[1] for failure in direct_failures)
    language_failures = max(language_failures, reported_language_failures)
    private_reasoning_failures = max(private_reasoning_failures, reported_private_reasoning_failures)
    acceptance_rate = len(accepted) / len(wrappers) if wrappers else 0.0
    first_pass_accepted = sum(
        int(item.get("episode", {}).get("repair_metadata", {}).get("round", 0)) == 0
        for item in accepted
    )
    first_pass_acceptance_rate = first_pass_accepted / len(wrappers) if wrappers else 0.0
    mean_expertise = statistics.mean(expertise) if expertise else 0.0
    static_failure_rate = static_failures / len(wrappers) if wrappers else 1.0
    accepted_adversarial_failures = sum(bool(item.get("adversarial_errors")) for item in accepted)
    accepted_dual_judge_failures = sum(
        not item.get("judge_consensus")
        or not isinstance(item.get("primary_review"), dict)
        or not isinstance(item.get("secondary_review"), dict)
        or item.get("primary_review", {}).get("verdict") != "accept"
        or item.get("secondary_review", {}).get("verdict") != "accept"
        or not item.get("judge_metadata", {}).get("deployment")
        or not item.get("secondary_judge_metadata", {}).get("deployment")
        or item.get("judge_metadata", {}).get("deployment")
        == item.get("secondary_judge_metadata", {}).get("deployment")
        for item in accepted
    )
    accepted_review_inconsistencies = sum(
        bool(item.get("static_errors"))
        or bool(item.get("adversarial_errors"))
        or bool(item.get("judge_errors"))
        or not item.get("judge_consensus")
        or not isinstance(item.get("primary_review"), dict)
        or not isinstance(item.get("secondary_review"), dict)
        or item.get("primary_review", {}).get("verdict") != "accept"
        or item.get("secondary_review", {}).get("verdict") != "accept"
        or item.get("review", {}).get("verdict") != "accept"
        or float(item.get("review", {}).get("overall_score", 0.0)) < minimum_overall
        or float(item.get("review", {}).get("scores", {}).get("expertise_uplift", 0.0))
        < minimum_accepted_expertise
        for item in accepted
    )
    signatures = [
        stable_hash(
            {
                "request": normalize_text(str(item.get("episode", {}).get("input", {}).get("user_request", ""))).lower(),
                "answer": normalize_text(str(item.get("episode", {}).get("integration", {}).get("published_answer", ""))).lower(),
            },
            32,
        )
        for item in accepted
    ]
    duplicate_accepted = len(signatures) - len(set(signatures))
    corpus_hash = accepted_corpus_hash(wrappers)
    manual_inspection_passed = manual_audit_valid(wrappers, manual_audit)
    gates = {
        "enough_reviewed": len(wrappers) >= minimum_reviewed,
        "acceptance_rate": acceptance_rate >= minimum_acceptance,
        "first_pass_acceptance_rate": first_pass_acceptance_rate >= minimum_first_pass_acceptance,
        "expertise_uplift": mean_expertise >= minimum_expertise,
        "accepted_expertise_floor": bool(expertise) and min(expertise) >= minimum_accepted_expertise,
        "accepted_score_floor": bool(overall_scores) and min(overall_scores) >= minimum_overall,
        "accepted_review_consistency": accepted_review_inconsistencies == 0,
        "adversarial_acceptance": accepted_adversarial_failures == 0,
        "dual_judge_consensus": accepted_dual_judge_failures == 0,
        "manual_adversarial_inspection": manual_inspection_passed,
        "static_validity": static_failure_rate <= 0.05,
        "english_only": language_failures == 0,
        "private_reasoning_free": private_reasoning_failures == 0,
        "no_exact_accepted_duplicates": duplicate_accepted == 0,
        "domain_coverage": len(domains) >= minimum_domains,
    }
    return {
        "reviewed": len(wrappers),
        "accepted": len(accepted),
        "rejected": len(wrappers) - len(accepted),
        "acceptance_rate": round(acceptance_rate, 4),
        "first_pass_accepted": first_pass_accepted,
        "repaired_accepted": len(accepted) - first_pass_accepted,
        "first_pass_acceptance_rate": round(first_pass_acceptance_rate, 4),
        "accepted_mean_expertise_uplift": round(mean_expertise, 4),
        "accepted_minimum_expertise_uplift": round(min(expertise), 4) if expertise else 0.0,
        "accepted_minimum_overall_score": round(min(overall_scores), 4) if overall_scores else 0.0,
        "static_failure_rate": round(static_failure_rate, 4),
        "language_failures": language_failures,
        "private_reasoning_failures": private_reasoning_failures,
        "accepted_review_inconsistencies": accepted_review_inconsistencies,
        "accepted_adversarial_failures": accepted_adversarial_failures,
        "accepted_dual_judge_failures": accepted_dual_judge_failures,
        "duplicate_accepted_episodes": duplicate_accepted,
        "accepted_corpus_hash": corpus_hash,
        "manual_adversarial_inspection_passed": manual_inspection_passed,
        "accepted_domain_counts": dict(sorted(domains.items())),
        "gates": gates,
        "scale_allowed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply quality gates to an HLWM reasoning pilot")
    parser.add_argument("--reviewed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-reviewed", type=int, default=50)
    parser.add_argument("--minimum-acceptance", type=float, default=0.75)
    parser.add_argument("--minimum-first-pass-acceptance", type=float, default=0.50)
    parser.add_argument("--minimum-expertise", type=float, default=0.85)
    parser.add_argument("--minimum-domains", type=int, default=12)
    parser.add_argument("--minimum-overall", type=float, default=0.84)
    parser.add_argument("--minimum-accepted-expertise", type=float, default=0.80)
    parser.add_argument("--manual-audit")
    args = parser.parse_args()
    manual_audit = (
        json.loads(Path(args.manual_audit).read_text(encoding="utf-8"))
        if args.manual_audit
        else None
    )
    report = assess(
        Path(args.reviewed_dir),
        args.minimum_reviewed,
        args.minimum_acceptance,
        args.minimum_expertise,
        minimum_first_pass_acceptance=args.minimum_first_pass_acceptance,
        minimum_domains=args.minimum_domains,
        minimum_overall=args.minimum_overall,
        minimum_accepted_expertise=args.minimum_accepted_expertise,
        manual_audit=manual_audit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["scale_allowed"] else 2)


if __name__ == "__main__":
    main()
