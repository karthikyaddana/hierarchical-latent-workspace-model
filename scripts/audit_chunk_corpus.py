#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from hlwm_data.language import classify_language, contains_blocked_script
from hlwm_data.util import atomic_write_json, iter_jsonl


PRIVATE_REASONING_PATTERNS = (
    re.compile(r"<\s*/?\s*\|?think\|?\s*>", re.IGNORECASE),
    re.compile(r"<\|?(?:begin|end)_of_thought\|?>", re.IGNORECASE),
    re.compile(r"\bhidden chain[- ]of[- ]thought\b", re.IGNORECASE),
    re.compile(r"\binternal reasoning:\s", re.IGNORECASE),
)
REQUIRED_FIELDS = (
    "chunk_id",
    "source_id",
    "source_group",
    "lineage_component_id",
    "domain",
    "license",
    "license_evidence",
    "language",
    "language_confidence",
    "text",
)


def audit(path: Path, minimum_confidence: float = 0.78) -> Dict[str, Any]:
    total = 0
    missing_fields = 0
    blocked_script_chunks = 0
    corrupted_unicode_chunks = 0
    private_reasoning_chunks = 0
    non_english_chunks = 0
    invalid_language_metadata = 0
    duplicate_chunk_ids = 0
    duplicate_content_hashes = 0
    samples = []
    chunk_ids = set()
    content_hashes = set()
    sources = Counter()
    domains = Counter()
    licenses = Counter()

    def sample(chunk_id: str, reason: str) -> None:
        if len(samples) < 20:
            samples.append({"chunk_id": chunk_id, "reason": reason})

    for row in iter_jsonl(path):
        total += 1
        chunk_id = str(row.get("chunk_id") or "row-%d" % total)
        missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
        if missing:
            missing_fields += 1
            sample(chunk_id, "missing fields: %s" % ", ".join(missing))

        if chunk_id in chunk_ids:
            duplicate_chunk_ids += 1
            sample(chunk_id, "duplicate chunk_id")
        chunk_ids.add(chunk_id)

        content_hash = str(row.get("chunk_content_hash") or "")
        if content_hash:
            if content_hash in content_hashes:
                duplicate_content_hashes += 1
                sample(chunk_id, "duplicate chunk_content_hash")
            content_hashes.add(content_hash)

        text = str(row.get("text") or "")
        try:
            declared_confidence = float(row.get("language_confidence") or 0.0)
        except (TypeError, ValueError):
            declared_confidence = 0.0
        if str(row.get("language") or "").lower() != "en" or declared_confidence < minimum_confidence:
            invalid_language_metadata += 1
            sample(chunk_id, "invalid English language metadata")
        if "\N{REPLACEMENT CHARACTER}" in text:
            corrupted_unicode_chunks += 1
            sample(chunk_id, "Unicode replacement character")
        if contains_blocked_script(text):
            blocked_script_chunks += 1
            sample(chunk_id, "blocked non-English script")
        if any(pattern.search(text) for pattern in PRIVATE_REASONING_PATTERNS):
            private_reasoning_chunks += 1
            sample(chunk_id, "private-reasoning marker")

        decision = classify_language(text, expected="en", minimum_confidence=minimum_confidence)
        if not decision.accepted:
            non_english_chunks += 1
            sample(
                chunk_id,
                "language=%s confidence=%.3f reason=%s"
                % (decision.language, decision.confidence, decision.reason),
            )

        sources[str(row.get("source_group") or "unknown")] += 1
        domains[str(row.get("domain") or "unknown")] += 1
        licenses[str(row.get("license") or "unknown")] += 1

    violations = {
        "missing_required_fields": missing_fields,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "duplicate_content_hashes": duplicate_content_hashes,
        "non_english_chunks": non_english_chunks,
        "invalid_language_metadata": invalid_language_metadata,
        "blocked_script_chunks": blocked_script_chunks,
        "corrupted_unicode_chunks": corrupted_unicode_chunks,
        "private_reasoning_marker_chunks": private_reasoning_chunks,
    }
    return {
        "input": str(path.resolve()),
        "chunks": total,
        "source_groups": len(sources),
        "domains": dict(sorted(domains.items())),
        "licenses": dict(sorted(licenses.items())),
        "source_group_counts": dict(sorted(sources.items())),
        "violations": violations,
        "violation_samples": samples,
        "valid": total > 0 and not any(violations.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a final English-only chunk corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--minimum-confidence", type=float, default=0.78)
    args = parser.parse_args()
    report = audit(Path(args.input).expanduser().resolve(), args.minimum_confidence)
    if args.output:
        atomic_write_json(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__":
    main()
