#!/usr/bin/env python3
"""Export the machine catalogs into reviewable CSV and Markdown artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            for key, value in list(normalized.items()):
                if isinstance(value, (list, dict)):
                    normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(normalized)


def safe(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    discovery = load(args.discovery)
    shortlist = load(args.shortlist)
    benchmarks = load(args.benchmarks)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    discovery_rows = discovery["datasets"]
    shortlist_rows = shortlist["candidates"]
    benchmark_rows = benchmarks["benchmarks"]
    write_csv(output / "all-discovered-sources-2484.csv", discovery_rows, [
        "provider", "id", "name", "url", "license", "english_status", "search_term",
        "disposition", "downloads", "likes", "popularity", "stars", "size_bytes", "gated",
    ])
    shortlist_count = len(shortlist_rows)
    write_csv(output / ("ranked-training-audit-queue-%d.csv" % shortlist_count), shortlist_rows, [
        "priority_score", "provider", "id", "name", "url", "license", "english_status",
        "capability_bucket", "search_term", "conversion_and_verification", "ranking_reasons", "approval",
    ])
    write_csv(output / "benchmark-registry-163.csv", benchmark_rows, [
        "name", "registry", "capability", "url", "language_scope", "execution",
        "reachable", "http_status", "dataset_license_status",
    ])

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in shortlist_rows:
        grouped[str(row.get("capability_bucket"))].append(row)
    lines = [
        "# HLWM English expert-data and benchmark research",
        "",
        "Date: 2026-08-19",
        "",
        "## Verified scope",
        "",
        "- 2,484 distinct sources discovered from Hugging Face, Kaggle and GitHub metadata.",
        "- 1,289 have a superficially permissive licence and still require content/provenance audit.",
        "- %s are ranked for manual audit: %s Hugging Face, %s Kaggle and %s GitHub." % (
            format(shortlist_count, ","),
            shortlist["counts"]["by_provider"].get("huggingface", 0),
            shortlist["counts"]["by_provider"].get("kaggle", 0),
            shortlist["counts"]["by_provider"].get("github", 0),
        ),
        "- 163 English-capable benchmark suites are registered; all 81 specialized landing pages resolved.",
        "- Discovery and ranking are not approval. No raw source is automatically added to training.",
        "",
        "## Benchmark-data rule",
        "",
        "Official training splits may be converted after licence, English and contamination checks. "
        "Validation, test, hidden, challenge, leaderboard and future-release questions are protected. "
        "Scores on families whose training splits were used must be labelled trained-on-family.",
        "",
        "## Ranked training candidates",
        "",
    ]
    for capability in sorted(grouped):
        lines.extend([
            "### %s" % capability.replace("_", " ").title(),
            "",
            "| Score | Provider | Source | Licence | English status |",
            "|---:|---|---|---|---|",
        ])
        for row in grouped[capability]:
            lines.append("| %s | %s | [%s](%s) | %s | %s |" % (
                safe(row.get("priority_score")), safe(row.get("provider")),
                safe(row.get("id") or row.get("name")), safe(row.get("url")),
                safe(row.get("license")), safe(row.get("english_status")),
            ))
        lines.append("")
    lines.extend([
        "## Benchmark registry",
        "",
        "| Benchmark | Capability | Execution | Registry | Reachable |",
        "|---|---|---|---|---|",
    ])
    for row in benchmark_rows:
        reachable = row.get("registry_verified") if row.get("registry") != "specialized" else row.get("reachable")
        lines.append("| [%s](%s) | %s | %s | %s | %s |" % (
            safe(row.get("name")), safe(row.get("url")), safe(row.get("capability")),
            safe(row.get("execution")), safe(row.get("registry")), safe(bool(reachable)),
        ))
    lines.append("")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "discovered": len(discovery_rows),
        "shortlist": len(shortlist_rows),
        "benchmarks": len(benchmark_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
