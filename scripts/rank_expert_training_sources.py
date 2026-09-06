#!/usr/bin/env python3
"""Rank discovered sources for manual audit; never auto-approve training data."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TRUSTED_PREFIXES = {
    "allenai", "apple", "argilla", "bigcode", "deepmind", "eleutherai", "evalplus",
    "google", "google-research", "google-research-datasets", "glaiveai", "huggingface",
    "microsoft", "nvidia", "open-thoughts", "openai", "princeton-nlp", "salesforce",
    "servicenow", "swe-bench", "xlang-ai", "18f", "rust-lang", "exercism",
}

NON_ENGLISH_MARKERS = {
    "arabic", "bengali", "chinese", "danish", "dutch", "french", "german", "hindi",
    "indonesian", "italian", "japanese", "korean", "portuguese", "romanian", "russian",
    "spanish", "turkish", "urdu", "vietnamese", "multilingual",
}

LOW_TRUST_MARKERS = {
    "mirror", "converted", "results", "leaderboard-results", "model-comparison",
    "benchmark-tracker", "scraped", "tweets", "synthetic-code-generations",
}

TEXT_INAPPLICABLE_MARKERS = {
    "image", "images", "multimodal", "ocr", "optical-flow", "vision", "blip", "audio", "video",
}

BENCHMARK_MARKERS = {"benchmark", "leaderboard", "humaneval", "swe-bench", "livecodebench", "gpqa", "mmlu"}

CATEGORY_TERMS: List[Tuple[str, Tuple[str, ...]]] = [
    ("coding_debugging", ("code generation", "code debugging", "code review", "software engineering", "unit testing", "python", "javascript", "repository")),
    ("tool_agents", ("tool calling", "function calling", "agent trajectories", "browser automation", "computer use", "api tool")),
    ("data_sql_analytics", ("sql", "data science", "data analysis", "spreadsheet", "visualization", "business intelligence", "analytics", "forecasting", "time series", "ab testing")),
    ("business_product_operations", ("project management", "product management", "requirements", "business process", "operations management", "supply chain", "decision making", "research planning")),
    ("finance_marketing_sales", ("financial", "investment", "marketing", "sales", "copywriting", "pitch deck", "startup", "customer", "ecommerce")),
    ("ui_accessibility", ("ui ux", "accessibility", "dashboard", "web development")),
    ("systems_devops", ("cloud", "devops", "incident response", "linux")),
    ("reasoning_instruction", ("reasoning", "mathematics", "communication", "technical writing", "large language models", "machine learning")),
]

CONVERSION = {
    "coding_debugging": "retain only runnable problems or patches; execute tests in a pinned container",
    "tool_agents": "normalize typed tools; replay calls and observations; reject invented state transitions",
    "data_sql_analytics": "create questions from immutable tables; recompute SQL, formulas and charts deterministically",
    "business_product_operations": "create source-grounded briefs and artifact checklists; one teacher plus human sample audit",
    "finance_marketing_sales": "retain evidence ledger and calculations; reject unsupported performance or investment claims",
    "ui_accessibility": "render artifacts at fixed viewports; run keyboard and accessibility checks",
    "systems_devops": "use executable configs or documented procedures with validation and rollback",
    "reasoning_instruction": "retain final verifiable artifacts; exclude hidden traces and benchmark evaluation items",
}

CATEGORY_QUOTAS = {
    "coding_debugging": 80,
    "tool_agents": 70,
    "data_sql_analytics": 100,
    "business_product_operations": 60,
    "finance_marketing_sales": 70,
    "ui_accessibility": 20,
    "systems_devops": 40,
    "reasoning_instruction": 60,
}


def category(row: Dict[str, Any]) -> Tuple[str, bool]:
    identifier = str(row.get("id") or row.get("name") or "")
    source_name = identifier.split("/", 1)[-1]
    direct_haystack = "%s %s" % (source_name, row.get("name", ""))
    lowered = direct_haystack.lower()
    for label, terms in CATEGORY_TERMS:
        if any(term in lowered for term in terms):
            return label, True
    search_term = str(row.get("search_term") or "").lower()
    for label, terms in CATEGORY_TERMS:
        if any(term in search_term for term in terms):
            return label, False
    return "manual_other", False


def numeric_popularity(row: Dict[str, Any]) -> float:
    for key in ("downloads", "stars", "popularity", "likes"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def source_prefix(row: Dict[str, Any]) -> str:
    value = str(row.get("id") or row.get("name") or "")
    return value.split("/", 1)[0].lower()


def rank(row: Dict[str, Any], direct_relevance: bool) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    provider = row.get("provider")
    if provider in {"huggingface", "github"}:
        score += 2.0
        reasons.append("direct technical source")
    elif provider == "kaggle":
        score += 1.0
        reasons.append("Kaggle candidate; uploader provenance still required")
    if row.get("english_status") == "declared_english":
        score += 3.0
        reasons.append("declares English")
    prefix = source_prefix(row)
    if prefix in TRUSTED_PREFIXES:
        score += 4.0
        reasons.append("recognized upstream organization")
    popularity = numeric_popularity(row)
    if popularity:
        score += min(4.0, math.log10(popularity + 1.0))
        reasons.append("nonzero adoption signal")
    haystack = "%s %s" % (row.get("id", ""), row.get("name", ""))
    normalized = re.sub(r"[^a-z0-9]+", "-", haystack.lower())
    if any(marker in normalized for marker in NON_ENGLISH_MARKERS):
        score -= 8.0
        reasons.append("language marker requires rejection or English subset proof")
    if any(marker in normalized for marker in LOW_TRUST_MARKERS):
        score -= 4.0
        reasons.append("mirror/result/scrape risk")
    if bool(row.get("gated")):
        score -= 1.0
        reasons.append("gated access")
    if direct_relevance:
        score += 2.0
        reasons.append("source name directly matches capability")
    else:
        score -= 4.0
        reasons.append("only search-query relevance; inspect before retaining")
    if any(marker in normalized for marker in TEXT_INAPPLICABLE_MARKERS):
        score -= 7.0
        reasons.append("appears image/audio/video focused and may not fit the text student")
    if row.get("disposition") == "manual_license_review":
        score -= 6.0
        reasons.append("licence is not yet acceptable; research only until resolved")
    return round(score, 3), reasons


def counts(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    selected: List[Dict[str, Any]] = []
    for row in payload.get("datasets", []):
        if row.get("disposition") not in {"training_candidate_needs_content_audit", "manual_license_review"}:
            continue
        normalized_id = re.sub(r"[^a-z0-9]+", "-", str(row.get("id") or row.get("name") or "").lower())
        if any(marker in normalized_id for marker in BENCHMARK_MARKERS):
            continue
        label, direct_relevance = category(row)
        score, reasons = rank(row, direct_relevance)
        selected.append({
            **row,
            "capability_bucket": label,
            "priority_score": score,
            "ranking_reasons": reasons,
            "conversion_and_verification": CONVERSION.get(label, "manual source and content audit"),
            "approval": (
                "not_approved_manual_audit_required"
                if row.get("disposition") == "training_candidate_needs_content_audit"
                else "not_approved_license_research_only"
            ),
        })
    selected.sort(key=lambda row: (-float(row["priority_score"]), str(row.get("provider")), str(row.get("id"))))
    requested_limit = max(1, args.limit)
    balanced: List[Dict[str, Any]] = []
    used = set()
    if requested_limit >= sum(CATEGORY_QUOTAS.values()):
        scale = requested_limit / float(sum(CATEGORY_QUOTAS.values()))
        scaled_quotas = {key: int(round(value * scale)) for key, value in CATEGORY_QUOTAS.items()}
        quota_difference = requested_limit - sum(scaled_quotas.values())
        first_capability = next(iter(scaled_quotas))
        scaled_quotas[first_capability] += quota_difference
        for capability, quota in scaled_quotas.items():
            for row in (item for item in selected if item["capability_bucket"] == capability):
                key = (row.get("provider"), row.get("id") or row.get("url"))
                if key in used:
                    continue
                balanced.append(row)
                used.add(key)
                if sum(item["capability_bucket"] == capability for item in balanced) >= quota:
                    break
    for row in selected:
        if len(balanced) >= requested_limit:
            break
        key = (row.get("provider"), row.get("id") or row.get("url"))
        if key not in used:
            balanced.append(row)
            used.add(key)
    selected = balanced
    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_catalog": str(Path(args.input)),
        "policy": {
            "ranking_is_not_approval": True,
            "english_content_must_be_sampled": True,
            "kaggle_uploader_license_does_not_prove_upstream_rights": True,
            "benchmark_evaluation_rows_are_excluded": True,
            "one_teacher_only_for_subjective_targets": True,
        },
        "counts": {
            "shortlist": len(selected),
            "by_provider": counts(selected, "provider"),
            "by_capability": counts(selected, "capability_bucket"),
        },
        "candidates": selected,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **output_payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
