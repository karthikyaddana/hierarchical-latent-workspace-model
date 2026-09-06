#!/usr/bin/env python3
"""Materialize official TRAIN splits of coding benchmarks as training packets.

Policy (config/benchmarks.reasoning.yaml): evaluation items never train, but
"official_training_splits_allowed_after_audit" is true. This script therefore
downloads ONLY designated training splits, records the exact dataset revision
and license at fetch time, re-checks English, excludes anything overlapping a
protected split (MBPP test ID range; SWE-bench test-split repositories), and
writes source packets compatible with the combined-bundle builder.

Every source is fetched independently; a failure quarantines that source in
the manifest instead of aborting the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlwm_data.language import classify_language  # noqa: E402
from hlwm_data.util import atomic_write_json, stable_hash, write_jsonl  # noqa: E402

OUTPUT_DIR = ROOT / "sources" / "public" / "benchmark-train-splits"
MAX_PROBLEM_CHARS = 12_000
MAX_ANSWER_CHARS = 14_000
MAX_VERIFICATION_CHARS = 5_000


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated for packet size]"


def _packet(
    dataset_id: str,
    revision: str,
    split: str,
    license_id: str,
    task_family: str,
    problem: str,
    reference_answer: str,
    verification_material: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    problem = str(problem).strip()
    reference_answer = str(reference_answer).strip()
    if not problem or not reference_answer:
        return None
    serialized = "\n".join((problem, reference_answer, str(verification_material)))
    if not classify_language(serialized).accepted:
        return None
    row: Dict[str, Any] = {
        "schema_version": "benchmark-train-packet-v1",
        "language": "en",
        "source": {"dataset_id": dataset_id, "revision": revision, "split": split},
        "license": license_id,
        "task_family": task_family,
        "problem": _clip(problem, MAX_PROBLEM_CHARS),
        "reference_answer": _clip(reference_answer, MAX_ANSWER_CHARS),
        "verification_material": _clip(str(verification_material), MAX_VERIFICATION_CHARS),
        "curation": {
            "method": "official_train_split",
            "evaluation_split_excluded": True,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    if extra:
        row["curation"].update(dict(extra))
    return row


def _revision(dataset_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id)
    return str(info.sha)


def _stream(dataset_id: str, revision: str, split: str, config: Optional[str] = None):
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {"split": split, "streaming": True, "revision": revision}
    if config:
        return load_dataset(dataset_id, config, **kwargs)
    return load_dataset(dataset_id, **kwargs)


# --------------------------------------------------------------------------
# Per-source fetchers. Each yields packet dicts.
# --------------------------------------------------------------------------

def fetch_mbpp(limit: int) -> Iterable[Dict[str, Any]]:
    dataset_id = "google-research-datasets/mbpp"
    revision = _revision(dataset_id)
    count = 0
    # Official MBPP protocol: task_ids 11-510 are the TEST set and 1-10 are
    # few-shot prompts; only 511-600 (validation) and 601-974 (train) may train.
    for split in ("train", "validation"):
        for row in _stream(dataset_id, revision, split, config="full"):
            task_id = int(row.get("task_id", 0))
            if task_id <= 510:
                continue
            packet = _packet(
                dataset_id,
                revision,
                split,
                "cc-by-4.0",
                "python-function",
                "%s\n\nYour solution must pass these tests:\n%s"
                % (row.get("text", ""), "\n".join(row.get("test_list") or [])),
                row.get("code", ""),
                json.dumps({"test_list": row.get("test_list"), "task_id": task_id}),
                {"protected_ids_excluded": "task_id 1-510"},
            )
            if packet:
                yield packet
                count += 1
                if count >= limit:
                    return


def fetch_apps(limit: int) -> Iterable[Dict[str, Any]]:
    # codeparrot/apps is a legacy script dataset; datasets>=3 refuses those, so
    # read the Hub's auto-converted parquet branch directly.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset_id = "codeparrot/apps"
    revision = _revision(dataset_id)
    files = HfApi().list_repo_files(
        dataset_id, repo_type="dataset", revision="refs/convert/parquet"
    )
    train_files = [
        "hf://datasets/%s@refs/convert/parquet/%s" % (dataset_id, name)
        for name in files
        if name.endswith(".parquet") and "train" in name and name.startswith("all/")
    ] or [
        "hf://datasets/%s@refs/convert/parquet/%s" % (dataset_id, name)
        for name in files
        if name.endswith(".parquet") and "train" in name
    ]
    if not train_files:
        raise RuntimeError("no parquet train shards found for %s" % dataset_id)
    stream = load_dataset(
        "parquet", data_files={"train": train_files}, split="train", streaming=True
    )
    count = 0
    for row in stream:
        solutions = row.get("solutions") or "[]"
        try:
            solution_list = json.loads(solutions) if isinstance(solutions, str) else list(solutions)
        except json.JSONDecodeError:
            continue
        if not solution_list:
            continue
        packet = _packet(
            dataset_id,
            revision,
            "train",
            "mit",
            "competitive-programming",
            row.get("question", ""),
            str(solution_list[0]),
            str(row.get("input_output") or "")[:MAX_VERIFICATION_CHARS],
            {"difficulty": row.get("difficulty")},
        )
        if packet:
            yield packet
            count += 1
            if count >= limit:
                return


def fetch_taco(limit: int) -> Iterable[Dict[str, Any]]:
    last_error: Optional[BaseException] = None
    for dataset_id, license_id in (
        ("BAAI/TACO", "apache-2.0"),
        ("likaixin/TACO-verified", "apache-2.0"),
    ):
        try:
            revision = _revision(dataset_id)
            count = 0
            for row in _stream(dataset_id, revision, "train"):
                solutions = row.get("solutions") or "[]"
                try:
                    solution_list = (
                        json.loads(solutions) if isinstance(solutions, str) else list(solutions)
                    )
                except json.JSONDecodeError:
                    continue
                if not solution_list:
                    continue
                packet = _packet(
                    dataset_id,
                    revision,
                    "train",
                    license_id,
                    "competitive-programming",
                    row.get("question", ""),
                    str(solution_list[0]),
                    str(row.get("input_output") or "")[:MAX_VERIFICATION_CHARS],
                    {"difficulty": row.get("difficulty")},
                )
                if packet:
                    yield packet
                    count += 1
                    if count >= limit:
                        return
            return
        except Exception as exc:  # try the fallback mirror
            last_error = exc
            continue
    raise RuntimeError("all TACO candidates failed: %s" % last_error)


def fetch_code_contests(limit: int) -> Iterable[Dict[str, Any]]:
    dataset_id = "deepmind/code_contests"
    revision = _revision(dataset_id)
    count = 0
    for row in _stream(dataset_id, revision, "train"):
        solutions = row.get("solutions") or {}
        languages = list(solutions.get("language") or [])
        bodies = list(solutions.get("solution") or [])
        python_solution = ""
        for language, body in zip(languages, bodies):
            if int(language) == 3:  # PYTHON3 in the dataset's language enum
                python_solution = str(body)
                break
        if not python_solution:
            continue
        public_tests = row.get("public_tests") or {}
        packet = _packet(
            dataset_id,
            revision,
            "train",
            "cc-by-4.0",
            "competitive-programming",
            row.get("description", ""),
            python_solution,
            json.dumps(
                {
                    "public_test_inputs": list(public_tests.get("input") or [])[:8],
                    "public_test_outputs": list(public_tests.get("output") or [])[:8],
                }
            ),
            {"name": row.get("name")},
        )
        if packet:
            yield packet
            count += 1
            if count >= limit:
                return


def fetch_swebench_train(limit: int) -> Iterable[Dict[str, Any]]:
    dataset_id = "princeton-nlp/SWE-bench"
    revision = _revision(dataset_id)
    protected_repos = set()
    for row in _stream(dataset_id, revision, "test"):
        protected_repos.add(str(row.get("repo", "")))
    count = 0
    for row in _stream(dataset_id, revision, "train"):
        repo = str(row.get("repo", ""))
        if repo in protected_repos:
            continue
        packet = _packet(
            dataset_id,
            revision,
            "train",
            "mit (dataset); underlying repos carry their own OSS licenses",
            "repository-debugging",
            "Repository: %s (commit %s)\n\nIssue:\n%s"
            % (repo, row.get("base_commit", ""), row.get("problem_statement", "")),
            str(row.get("patch", "")),
            _clip(str(row.get("test_patch", "")), MAX_VERIFICATION_CHARS),
            {"protected_repos_excluded": sorted(protected_repos)},
        )
        if packet:
            yield packet
            count += 1
            if count >= limit:
                return


def fetch_spider(limit: int) -> Iterable[Dict[str, Any]]:
    dataset_id = "xlangai/spider"
    revision = _revision(dataset_id)
    count = 0
    for row in _stream(dataset_id, revision, "train"):
        packet = _packet(
            dataset_id,
            revision,
            "train",
            "cc-by-sa-4.0 (attribution preserved via provenance record)",
            "text-to-sql",
            "Database: %s\nWrite a single SQL query answering: %s"
            % (row.get("db_id", ""), row.get("question", "")),
            str(row.get("query", "")),
            json.dumps({"db_id": row.get("db_id"), "gold_sql": row.get("query")}),
        )
        if packet:
            yield packet
            count += 1
            if count >= limit:
                return


SOURCES = {
    "mbpp-train": (fetch_mbpp, 474),
    "apps-train": (fetch_apps, 1200),
    "taco-train": (fetch_taco, 800),
    "code-contests-train": (fetch_code_contests, 400),
    "swebench-train": (fetch_swebench_train, 400),
    "spider-train": (fetch_spider, 1200),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Comma-separated source slugs")
    parser.add_argument("--force", action="store_true", help="Refetch existing outputs")
    args = parser.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()} or set(SOURCES)
    manifest: Dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": "official train splits only; protected splits excluded at source",
        "sources": {},
    }
    manifest_path = OUTPUT_DIR / "fetch-manifest.json"
    if manifest_path.exists():
        manifest["sources"] = json.loads(manifest_path.read_text())["sources"]
    for slug, (fetcher, limit) in SOURCES.items():
        if slug not in wanted:
            continue
        target = OUTPUT_DIR / ("%s.jsonl" % slug)
        if target.exists() and not args.force:
            print("skip (exists):", slug)
            continue
        try:
            rows = list(fetcher(limit))
            deduped: List[Dict[str, Any]] = []
            seen = set()
            for row in rows:
                key = stable_hash(row["problem"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            count = write_jsonl(target, deduped)
            manifest["sources"][slug] = {
                "status": "ok",
                "rows": count,
                "dataset_id": deduped[0]["source"]["dataset_id"] if deduped else None,
                "revision": deduped[0]["source"]["revision"] if deduped else None,
                "license": deduped[0]["license"] if deduped else None,
            }
            print("ok:", slug, count, "rows")
        except Exception as exc:
            manifest["sources"][slug] = {"status": "quarantined", "error": str(exc)[:400]}
            print("quarantined:", slug, "->", str(exc)[:200])
        atomic_write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
