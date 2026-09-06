#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from hlwm_data.language import classify_language, contains_blocked_script
from hlwm_data.util import normalize_text, stable_hash, write_jsonl


API_ROOT = "https://huggingface.co/api"
DATASET_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "hlwm-data-factory/0.1 source-audit"
PRIVATE_REASONING_MARKERS = ("<think>", "</think>", "<|begin_of_thought|>", "<|end_of_thought|>")
LAST_DATASET_REQUEST_AT = 0.0
LICENSE_LABELS = {"apache-2.0": "Apache-2.0", "mit": "MIT", "cc-by-4.0": "CC-BY-4.0"}


def get_json(url: str, parameters: Optional[Dict[str, Any]] = None, retries: int = 5) -> Dict[str, Any]:
    if parameters:
        url += "?" + urllib.parse.urlencode(parameters)
    error: Optional[Exception] = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object from %s" % url)
                return value
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            if attempt + 1 < retries:
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 8.0 * (2**attempt))
                else:
                    delay = min(20.0, 1.5 * (2**attempt))
                time.sleep(delay)
    raise RuntimeError("Request failed after %d attempts: %s (%s)" % (retries, url, error))


def dataset_size(dataset_id: str, config: str, split: str) -> int:
    payload = get_json(DATASET_SERVER + "/size", {"dataset": dataset_id})
    for item in payload.get("size", {}).get("splits", []):
        if item.get("config") == config and item.get("split") == split:
            return int(item["num_rows"])
    raise ValueError("No size metadata for %s/%s/%s" % (dataset_id, config, split))


def dataset_rows(dataset_id: str, config: str, split: str, offset: int, length: int) -> List[Dict[str, Any]]:
    global LAST_DATASET_REQUEST_AT
    elapsed = time.monotonic() - LAST_DATASET_REQUEST_AT
    if elapsed < 1.25:
        time.sleep(1.25 - elapsed)
    payload = get_json(
        DATASET_SERVER + "/rows",
        {"dataset": dataset_id, "config": config, "split": split, "offset": offset, "length": length},
    )
    LAST_DATASET_REQUEST_AT = time.monotonic()
    return [item for item in payload.get("rows", []) if isinstance(item, dict)]


def compact(value: Any, maximum_string: int = 3000, maximum_items: int = 8) -> Any:
    if isinstance(value, str):
        return value[:maximum_string]
    if isinstance(value, list):
        return [compact(item, maximum_string, maximum_items) for item in value[:maximum_items]]
    if isinstance(value, dict):
        return {
            str(key): compact(child, maximum_string, maximum_items)
            for key, child in list(value.items())[:maximum_items]
        }
    return value


def compact_tests(value: Any, maximum_chars: int = 14000) -> str:
    if value is None or value == "":
        return ""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    rendered = json.dumps(compact(parsed), ensure_ascii=False, sort_keys=True)
    return rendered[:maximum_chars]


def normalize_open_thoughts(row: Dict[str, Any], feed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    domain = str(row.get("domain") or "").lower()
    source = str(row.get("source") or "").lower()
    allowed_domains = {str(item).lower() for item in feed.get("allowed_record_domains", [])}
    prefixes = tuple(str(item).lower() for item in feed.get("allowed_source_prefixes", []))
    if allowed_domains and domain not in allowed_domains:
        return None
    if prefixes and not source.startswith(prefixes):
        return None
    problem = str(row.get("problem") or "").strip()
    ground_truth = str(row.get("ground_truth_solution") or "").strip()
    answer = ground_truth or str(row.get("deepseek_solution") or "").strip()
    tests = compact_tests(row.get("test_cases"))
    if not problem or not answer or not tests:
        return None
    return {
        "task_family": "software-reasoning",
        "problem": problem[:16000],
        "reference_answer": answer[:9000],
        "reference_origin": "ground_truth_solution" if ground_truth else "verified_final_solution_without_private_reasoning",
        "verification_material": tests,
        "source_subset": source,
        "source_domain": domain,
    }


def normalize_deepcoder(row: Dict[str, Any], feed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    problem = str(row.get("problem") or "").strip()
    solutions = row.get("solutions") or []
    if isinstance(solutions, str):
        solutions = [solutions]
    solution = next((str(item).strip() for item in solutions if str(item).strip()), "")
    tests = compact_tests(row.get("tests"))
    if not problem or not solution or not tests:
        return None
    return {
        "task_family": "software-reasoning",
        "problem": problem[:18000],
        "reference_answer": solution[:14000],
        "verification_material": tests,
        "source_subset": str(feed["config"]),
        "source_domain": "coding",
    }


NORMALIZERS = {"open_thoughts": normalize_open_thoughts, "deepcoder": normalize_deepcoder}


def sample_offsets(start: int, end: int, desired: int, batch_length: int, oversample: float, seed: str) -> List[int]:
    windows = max(1, int(math.ceil(desired * oversample / batch_length)))
    last = max(start, end - batch_length)
    if windows == 1 or last == start:
        offsets = [start]
    else:
        offsets = sorted({round(start + index * (last - start) / (windows - 1)) for index in range(windows)})
    random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)).shuffle(offsets)
    return offsets


def materialize_feed(
    feed: Dict[str, Any],
    output_dir: Path,
    requested_override: Optional[int],
    seen_problems: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_id = str(feed["dataset_id"])
    config = str(feed["config"])
    split = str(feed["split"])
    total = dataset_size(dataset_id, config, split)
    start = max(0, int(feed.get("sample_start", 0)))
    end_value = feed.get("sample_end")
    end = min(total, int(end_value)) if end_value is not None else total
    if end <= start:
        raise ValueError("Invalid sample range for %s" % feed["name"])
    requested = int(requested_override or feed.get("records", 100))
    batch_length = int(feed.get("batch_length", 8))
    offsets = sample_offsets(
        start,
        end,
        requested,
        batch_length,
        float(feed.get("oversample_factor", 1.5)),
        "%s:%s:%s:%s" % (dataset_id, config, split, feed["expected_revision"]),
    )
    normalizer = NORMALIZERS[str(feed["normalizer"])]
    accepted: List[Dict[str, Any]] = []
    rejected_language = rejected_policy = duplicates = 0
    for offset in offsets:
        for wrapper in dataset_rows(dataset_id, config, split, offset, min(batch_length, end - offset)):
            source_row = wrapper.get("row")
            if not isinstance(source_row, dict):
                continue
            normalized = normalizer(source_row, feed)
            if normalized is None:
                rejected_policy += 1
                continue
            natural_language = "%s\n%s" % (normalized["problem"], normalized["reference_answer"])
            decision = classify_language(natural_language, expected="en", minimum_confidence=0.78)
            if not decision.accepted or contains_blocked_script(natural_language):
                rejected_language += 1
                continue
            serialized = json.dumps(normalized, ensure_ascii=False).lower()
            if any(marker in serialized for marker in PRIVATE_REASONING_MARKERS):
                rejected_policy += 1
                continue
            problem_hash = stable_hash(normalize_text(normalized["problem"]).lower(), 32)
            if problem_hash in seen_problems:
                duplicates += 1
                continue
            seen_problems.add(problem_hash)
            accepted.append(
                {
                    "schema_version": "source-packet-v1",
                    "language": "en",
                    "language_confidence": round(decision.confidence, 6),
                    "source": {
                        "dataset_id": dataset_id,
                        "revision": feed["expected_revision"],
                        "config": config,
                        "split": split,
                        "row_index": int(wrapper.get("row_idx", offset)),
                    },
                    **normalized,
                    "curation": {
                        "raw_private_reasoning_included": False,
                        "benchmark_splits_included": False,
                        "transform": "problem-ground-truth-and-compact-verification-v1",
                    },
                }
            )
            if len(accepted) >= requested:
                break
        if len(accepted) >= requested:
            break
    if len(accepted) < max(1, int(requested * 0.60)):
        raise RuntimeError(
            "Feed %s produced only %d/%d acceptable English records" % (feed["name"], len(accepted), requested)
        )
    accepted.sort(key=lambda item: int(item["source"]["row_index"]))
    output_path = output_dir / (str(feed["name"]) + ".jsonl")
    write_jsonl(output_path, accepted)
    manifest_entry = {
        "path": str(output_path.resolve()),
        "title": feed["title"],
        "author": feed["author"],
        "domain": feed["target_domain"],
        "source_group": "hf-" + str(feed["name"]),
        "lineage_component_id": "hf-" + str(feed["name"]),
        "lineage_mode": "source",
        "version": "hf:%s:%s:%s:%s" % (feed["expected_revision"], config, split, "curation-v1"),
        "license": LICENSE_LABELS.get(
            str(feed["expected_license"]).lower(), str(feed["expected_license"])
        ),
        "license_evidence": "https://huggingface.co/datasets/%s/blob/%s/README.md"
        % (dataset_id, feed["expected_revision"]),
        "allowed_for_training": True,
        "data_role": "visible_context",
        "language": "en",
        "url": "https://huggingface.co/datasets/%s" % dataset_id,
        "split_policy": "%s only; benchmark and excluded collection splits are not materialized" % split,
        "transformation": "Private reasoning removed; only English problem, ground truth/reference answer, and compact verification material retained",
    }
    report = {
        "name": feed["name"],
        "dataset_id": dataset_id,
        "revision": feed["expected_revision"],
        "config": config,
        "split": split,
        "requested": requested,
        "accepted": len(accepted),
        "rejected_language": rejected_language,
        "rejected_policy": rejected_policy,
        "duplicates_removed": duplicates,
        "output": str(output_path.resolve()),
    }
    return manifest_entry, report


def collection_snapshot(slug: str) -> Dict[str, Any]:
    payload = get_json(API_ROOT + "/collections/" + slug)
    return {
        "slug": payload.get("slug"),
        "title": payload.get("title"),
        "items": [
            {"id": item.get("id"), "type": item.get("type"), "position": item.get("position")}
            for item in payload.get("items", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and materialize safe English reasoning collection subsets")
    parser.add_argument("--policy", default="config/reasoning-collection-policy.yaml")
    parser.add_argument("--output-dir", default="sources/public/huggingface-curated")
    parser.add_argument("--manifest", default="config/sources.reasoning-collection.yaml")
    parser.add_argument("--audit", default="catalog/reasoning-collection-audit-2026-08-17.json")
    parser.add_argument("--records-per-feed", type=int)
    parser.add_argument("--allow-revision-drift", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    policy_path = (root / args.policy).resolve()
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_audits: Dict[str, Any] = {}
    for feed in policy.get("training_feeds", []):
        dataset_id = str(feed["dataset_id"])
        if dataset_id in dataset_audits:
            continue
        metadata = get_json(API_ROOT + "/datasets/" + dataset_id)
        actual_revision = str(metadata.get("sha") or "")
        actual_license = str((metadata.get("cardData") or {}).get("license") or "").lower()
        if not args.allow_revision_drift and actual_revision != str(feed["expected_revision"]):
            raise RuntimeError("Revision drift for %s: expected %s, found %s" % (dataset_id, feed["expected_revision"], actual_revision))
        if actual_license != str(feed["expected_license"]).lower():
            raise RuntimeError("License drift for %s: expected %s, found %s" % (dataset_id, feed["expected_license"], actual_license))
        dataset_audits[dataset_id] = {
            "revision": actual_revision,
            "license": actual_license,
            "gated": metadata.get("gated"),
            "private": metadata.get("private"),
            "last_modified": metadata.get("lastModified"),
        }

    entries: List[Dict[str, Any]] = []
    feed_reports: List[Dict[str, Any]] = []
    seen_problems: Set[str] = set()
    for feed in policy.get("training_feeds", []):
        entry, report = materialize_feed(feed, output_dir, args.records_per_feed, seen_problems)
        entries.append(entry)
        feed_reports.append(report)

    manifest_path = (root / args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "# Audited English-only subsets from the linked reasoning collection.\n"
        + yaml.safe_dump(
            {"language_policy": policy["language_policy"], "sources": entries},
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )

    collections = {
        name: collection_snapshot(str(slug)) for name, slug in policy.get("collections", {}).items()
    }
    audit = {
        "schema_version": "1.0",
        "created_date": "2026-08-17",
        "policy": str(policy_path),
        "language_policy": policy["language_policy"],
        "collections": collections,
        "discovery_catalogs": policy.get("discovery_catalogs", []),
        "dataset_audits": dataset_audits,
        "materialized_feeds": feed_reports,
        "quarantine": policy.get("quarantine", {}),
        "excluded_splits": policy.get("excluded_splits", {}),
        "raw_private_reasoning_included": False,
    }
    audit_path = (root / args.audit).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "audit": str(audit_path),
                "feeds": feed_reports,
                "records": sum(item["accepted"] for item in feed_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
