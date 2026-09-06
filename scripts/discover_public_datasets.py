#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SEARCH_TERMS = [
    "reasoning",
    "mathematics",
    "code generation",
    "software engineering",
    "repository level software engineering",
    "code debugging",
    "code review",
    "unit testing",
    "python programming",
    "javascript web development",
    "sql reasoning",
    "data science",
    "data analysis",
    "spreadsheet analysis",
    "data visualization",
    "business intelligence",
    "project management",
    "product management",
    "requirements engineering",
    "business process analysis",
    "operations management",
    "supply chain analytics",
    "marketing analytics",
    "sales",
    "sales analytics",
    "customer analytics",
    "copywriting",
    "pitch deck",
    "startup analysis",
    "financial analysis",
    "financial statements",
    "investment analysis",
    "forecasting",
    "time series",
    "ab testing",
    "cloud computing",
    "devops",
    "incident response",
    "linux administration",
    "machine learning",
    "large language models",
    "tool calling",
    "function calling",
    "agent trajectories",
    "browser automation",
    "computer use agents",
    "api tool use",
    "communication skills",
    "technical writing",
    "ecommerce",
    "business analytics",
    "decision making",
    "research planning",
    "ui ux design",
    "web accessibility",
    "dashboard design",
]

BENCHMARK_SEARCH_TERMS = [
    "coding benchmark",
    "code generation benchmark",
    "software engineering benchmark",
    "repository debugging benchmark",
    "code review benchmark",
    "unit testing benchmark",
    "sql benchmark",
    "data science benchmark",
    "data analysis benchmark",
    "spreadsheet benchmark",
    "business benchmark",
    "business analytics benchmark",
    "finance benchmark",
    "financial reasoning benchmark",
    "marketing benchmark",
    "product management benchmark",
    "project management benchmark",
    "requirements benchmark",
    "tool use benchmark",
    "tool calling benchmark",
    "function calling benchmark",
    "agent benchmark",
    "browser agent benchmark",
    "computer use benchmark",
    "instruction following benchmark",
    "reasoning benchmark",
    "planning benchmark",
    "long context benchmark",
    "truthfulness benchmark",
    "hallucination benchmark",
    "web development benchmark",
    "ui generation benchmark",
    "accessibility benchmark",
]

ALLOW_LICENSE_FRAGMENTS = {
    "mit", "apache", "bsd", "isc", "cc0", "cc-by", "creative commons attribution",
    "open government", "unlicense", "public domain",
}
BLOCK_LICENSE_FRAGMENTS = {
    "non-commercial", "noncommercial", "no derivatives", "research only", "unknown", "other",
    "cc-by-nc", "cc by-nc", "cc-by-nd", "cc by-nd", "gfdl",
}
BENCHMARK_TERMS = {
    "benchmark", "evaluation", "eval", "humaneval", "swe-bench", "mmlu", "arc-agi", "gpqa",
    "livecodebench", "truthfulqa", "hellaswag", "winogrande", "bigbench", "bbh", "ifeval",
}


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "hlwm-dataset-discovery/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def normalized_license(value: Any) -> str:
    return str(value or "unknown").strip()


def disposition(name: str, license_name: str) -> str:
    searchable = (name + " " + license_name).lower()
    if any(term in searchable for term in BENCHMARK_TERMS):
        return "benchmark_only_or_split_audit"
    if any(term in searchable for term in BLOCK_LICENSE_FRAGMENTS):
        return "quarantine_license"
    if any(term in searchable for term in ALLOW_LICENSE_FRAGMENTS):
        return "training_candidate_needs_content_audit"
    return "manual_license_review"


def language_status(tags: Iterable[str]) -> str:
    normalized = {str(tag).lower() for tag in tags}
    if "language:en" in normalized or "language:eng" in normalized:
        return "declared_english"
    return "needs_content_language_audit"


def kaggle(term: str, *, per_provider: int, pages: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in range(1, pages + 1):
        query = urllib.parse.urlencode({"search": term, "sortBy": "hottest", "page": page})
        try:
            page_rows = fetch_json("https://www.kaggle.com/api/v1/datasets/list?" + query)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break
            raise
        rows.extend(page_rows[:per_provider])
    result = []
    for row in rows:
        name = row.get("titleNullable") or row.get("ref") or "untitled"
        license_name = normalized_license(row.get("licenseNameNullable"))
        result.append({
            "provider": "kaggle",
            "search_term": term,
            "name": name,
            "id": row.get("ref") or row.get("datasetRef"),
            "url": row.get("urlNullable") or row.get("url"),
            "license": license_name,
            "size_bytes": row.get("totalBytesNullable"),
            "popularity": row.get("voteCountNullable") or row.get("usabilityRatingNullable"),
            "english_status": "needs_content_language_audit",
            "disposition": disposition(name, license_name),
        })
    return result


def huggingface(term: str, *, per_provider: int, pages: int) -> List[Dict[str, Any]]:
    del pages
    query = urllib.parse.urlencode({"search": term, "limit": per_provider, "sort": "downloads", "direction": -1})
    rows = fetch_json("https://huggingface.co/api/datasets?" + query)
    result = []
    for row in rows:
        tags = [str(tag) for tag in row.get("tags", [])]
        license_tags = [tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")]
        license_name = normalized_license(",".join(license_tags) if license_tags else "unknown")
        name = row.get("id") or row.get("_id") or "untitled"
        result.append({
            "provider": "huggingface",
            "search_term": term,
            "name": name,
            "id": row.get("id"),
            "url": "https://huggingface.co/datasets/%s" % row.get("id") if row.get("id") else None,
            "license": license_name,
            "downloads": row.get("downloads"),
            "likes": row.get("likes"),
            "gated": bool(row.get("gated", False)),
            "language_tags": [tag.split(":", 1)[1] for tag in tags if tag.startswith("language:")],
            "english_status": language_status(tags),
            "disposition": disposition(name, license_name),
        })
    return result


def github(term: str, *, per_provider: int, pages: int) -> List[Dict[str, Any]]:
    del pages
    command = [
        "gh", "api", "-X", "GET", "search/repositories",
        "-f", "q=%s dataset in:name,description,readme" % term,
        "-f", "sort=stars", "-f", "order=desc", "-f", "per_page=%d" % per_provider,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "GitHub search failed")[:500])
    rows = json.loads(completed.stdout).get("items", [])
    result = []
    for row in rows:
        license_name = normalized_license((row.get("license") or {}).get("spdx_id"))
        name = row.get("full_name") or row.get("name") or "untitled"
        result.append({
            "provider": "github",
            "search_term": term,
            "name": name,
            "id": row.get("full_name"),
            "url": row.get("html_url"),
            "license": license_name,
            "stars": row.get("stargazers_count"),
            "forks": row.get("forks_count"),
            "archived": bool(row.get("archived", False)),
            "default_branch": row.get("default_branch"),
            "revision": row.get("pushed_at"),
            "english_status": "needs_content_language_audit",
            "disposition": disposition(name, license_name),
        })
    return result


def deduplicate(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = "%s:%s" % (row["provider"], row.get("id") or row.get("url") or row["name"])
        if key not in selected:
            selected[key] = row
        else:
            terms = set(str(selected[key].get("search_term", "")).split(" | "))
            terms.add(str(row.get("search_term", "")))
            selected[key]["search_term"] = " | ".join(sorted(term for term in terms if term))
    order = {
        "training_candidate_needs_content_audit": 0,
        "benchmark_only_or_split_audit": 1,
        "manual_license_review": 2,
        "quarantine_license": 3,
    }
    return sorted(selected.values(), key=lambda row: (order.get(row["disposition"], 9), row["provider"], row["name"].casefold()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover public dataset candidates without downloading content")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-provider", type=int, default=25)
    parser.add_argument("--kaggle-pages", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile", choices=("training", "benchmarks", "all"), default="training")
    args = parser.parse_args()
    if args.profile == "training":
        search_terms = SEARCH_TERMS
    elif args.profile == "benchmarks":
        search_terms = BENCHMARK_SEARCH_TERMS
    else:
        search_terms = SEARCH_TERMS + BENCHMARK_SEARCH_TERMS
    jobs = [(provider, term) for term in search_terms for provider in ("kaggle", "huggingface", "github")]
    rows: List[Dict[str, Any]] = []
    errors = []
    functions = {"kaggle": kaggle, "huggingface": huggingface, "github": github}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                functions[provider],
                term,
                per_provider=max(1, min(args.per_provider, 100)),
                pages=max(1, min(args.kaggle_pages, 10)),
            ): (provider, term)
            for provider, term in jobs
        }
        for future in as_completed(futures):
            provider, term = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append({"provider": provider, "search_term": term, "error": str(exc)[:500]})
    rows = deduplicate(rows)
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "training_candidates_are_not_approved_until_license_and_content_are_audited": True,
            "benchmark_test_and_validation_splits_must_not_enter_training": True,
            "unknown_research_only_noncommercial_and_no_derivatives_licenses_are_quarantined": True,
            "english_only_requires_declared_language_or_content_audit": True,
            "official_benchmark_training_splits_require_separate_split_and_contamination_policy": True,
        },
        "profile": args.profile,
        "search_terms": search_terms,
        "counts": counts,
        "errors": errors,
        "datasets": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "datasets": len(rows), "counts": counts, "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
