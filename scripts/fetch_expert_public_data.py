#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from hlwm_data.language import classify_language, contains_blocked_script
from hlwm_data.util import normalize_text, stable_hash, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
HF_API = "https://huggingface.co/api/datasets/"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_SIZE = "https://datasets-server.huggingface.co/size"
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets"
USER_AGENT = "hlwm-expert-public-data/1.0"
PRIVATE_MARKERS = ("<think>", "</think>", "<|begin_of_thought|>", "<|end_of_thought|>")
REPOSITORY_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mdx",
    ".njk",
    ".py",
    ".rs",
    ".rst",
    ".scss",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
REPOSITORY_EXCLUDED_PARTS = {
    ".git",
    ".next",
    "coverage",
    "dist",
    "generated",
    "node_modules",
    "snapshots",
    "target",
    "vendor",
}


def get_json(url: str, parameters: Optional[Mapping[str, Any]] = None, retries: int = 5) -> Any:
    if parameters:
        url += "?" + urllib.parse.urlencode(parameters)
    error: Optional[BaseException] = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise RuntimeError("request failed for %s: %s" % (url, error))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_size(dataset_id: str, config: str, split: str) -> int:
    value = get_json(HF_SIZE, {"dataset": dataset_id})
    for row in (value.get("size") or {}).get("splits") or []:
        if row.get("config") == config and row.get("split") == split:
            return int(row["num_rows"])
    raise ValueError("size unavailable for %s/%s/%s" % (dataset_id, config, split))


def dataset_rows(dataset_id: str, config: str, split: str, offset: int, length: int) -> List[Dict[str, Any]]:
    value = get_json(
        HF_ROWS,
        {
            "dataset": dataset_id,
            "config": config,
            "split": split,
            "offset": offset,
            "length": min(100, length),
        },
    )
    rows = value.get("rows") or []
    return [dict(item.get("row") or {}) for item in rows]


def render_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return normalize_text(str(messages or ""))
    parts = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = normalize_text(str(message.get("role") or message.get("from") or "user"))
        content = normalize_text(str(message.get("content") or message.get("value") or ""))
        if content:
            parts.append("%s: %s" % (role, content))
    return "\n\n".join(parts)


def base_pair(
    *,
    feed: Mapping[str, Any],
    source_index: int,
    prompt: str,
    chosen: str,
    rejected: str,
    defect: str,
    domain: str,
    task_type: str,
    difficulty: int,
    response_mode: str,
) -> Optional[Dict[str, Any]]:
    prompt = normalize_text(prompt)
    chosen = normalize_text(chosen)
    rejected = normalize_text(rejected)
    defect = normalize_text(defect)
    if not (24 <= len(prompt) <= 16000 and 8 <= len(chosen) <= 14000):
        return None
    if not rejected or chosen.lower() == rejected.lower() or not defect:
        return None
    serialized = (prompt + "\n" + chosen + "\n" + rejected).lower()
    if any(marker in serialized for marker in PRIVATE_MARKERS):
        return None
    natural = prompt + "\n" + re.sub(r"```[\s\S]*?```", "", chosen)
    decision = classify_language(natural[:12000], expected="en", minimum_confidence=0.70)
    if not decision.accepted or contains_blocked_script(natural):
        return None
    pair_id = "public-%s-%s" % (feed["name"], stable_hash({"index": source_index, "prompt": prompt}, 18))
    return {
        "id": pair_id,
        "prompt": prompt,
        "context": [],
        "constraints": [],
        "chosen": chosen,
        "rejected": rejected,
        "rejected_defect": defect,
        "domain": domain,
        "difficulty": int(difficulty),
        "response_mode": response_mode,
        "source_group": "hf-%s" % feed["name"],
        "lineage_component_id": pair_id,
        "mean_judge_score": 1.0,
        "judge_models": ["source-ground-truth-and-deterministic-policy"],
        "task_type": task_type,
        "policy_labels": {
            "chosen_publish": 1,
            "chosen_risk": 0,
            "rejected_publish": 0,
            "rejected_risk": 1,
        },
        "source": {
            "provider": "huggingface",
            "dataset_id": feed["dataset_id"],
            "revision": feed["revision"],
            "config": feed["config"],
            "split": feed["split"],
            "row_index": source_index,
            "license": feed["license"],
        },
        "provenance_sha256": stable_hash(
            {"feed": feed["name"], "revision": feed["revision"], "index": source_index, "chosen": chosen},
            64,
        ),
    }


def normalize_helpsteer(row: Mapping[str, Any], feed: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    prompt = render_messages(row.get("context"))
    defect_value = row.get("change_summary") or row.get("feedback") or "The original response required an expert edit."
    if isinstance(defect_value, list):
        defect_value = "; ".join(str(item) for item in defect_value[:6])
    return base_pair(
        feed=feed,
        source_index=index,
        prompt=prompt,
        chosen=str(row.get("edited_response") or ""),
        rejected=str(row.get("original_response") or ""),
        defect=str(defect_value),
        domain="software_debugging" if str(row.get("domain")) == "code" else "expert_communication",
        task_type="edited_response",
        difficulty=4,
        response_mode="brief",
    )


def normalize_hermes(row: Mapping[str, Any], feed: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    conversations = row.get("conversations") or []
    if not isinstance(conversations, list):
        return None
    system = next((str(item.get("value") or "") for item in conversations if item.get("from") == "system"), "")
    human_position = next((i for i, item in enumerate(conversations) if item.get("from") == "human"), -1)
    if human_position < 0:
        return None
    user = str(conversations[human_position].get("value") or "")
    assistant = next(
        (str(item.get("value") or "") for item in conversations[human_position + 1 :] if item.get("from") == "gpt"),
        "",
    )
    if "<tool_call>" not in assistant:
        return None
    prompt = "Available tool contract:\n%s\n\nUser request:\n%s" % (system[:10000], user)
    rejected = re.sub(
        r'(\"name\"\s*:\s*\")[^\"]+',
        r'\1unavailable_tool',
        assistant,
        count=1,
    )
    if rejected == assistant:
        rejected = assistant.replace("<tool_call>", "<tool_call>\n{\"name\":\"unavailable_tool\",\"arguments\":{}}\n", 1)
    return base_pair(
        feed=feed,
        source_index=index,
        prompt=prompt,
        chosen=assistant,
        rejected=rejected,
        defect="The rejected call selects a tool that is not present in the supplied tool contract.",
        domain="tool_using_agents",
        task_type="tool_call",
        difficulty=3,
        response_mode="brief",
    )


def normalize_swe_gym(row: Mapping[str, Any], feed: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    patch = str(row.get("patch") or "")
    problem = str(row.get("problem_statement") or "")
    if not (80 <= len(patch) <= 8500 and 50 <= len(problem) <= 10000):
        return None
    tests = [str(item) for item in (row.get("FAIL_TO_PASS") or [])[:8]]
    prompt = (
        "Repository: %s\nBase commit: %s\n\nIssue:\n%s\n\nTests that must pass:\n%s"
        % (row.get("repo"), row.get("base_commit"), problem, "\n".join(tests) or "Use the supplied regression test patch.")
    )
    chosen = "Apply this focused patch, then run the named regression tests:\n```diff\n%s\n```" % patch
    return base_pair(
        feed=feed,
        source_index=index,
        prompt=prompt,
        chosen=chosen,
        rejected="Modify unrelated documentation and close the issue without reproducing the failure or running regression tests.",
        defect="The rejected response does not address the reported code path and provides no executable verification.",
        domain="software_debugging",
        task_type="repository_patch",
        difficulty=5,
        response_mode="expert_workflow",
    )


def normalize_websight(row: Mapping[str, Any], feed: Mapping[str, Any], index: int) -> Optional[Dict[str, Any]]:
    idea = str(row.get("llm_generated_idea") or "")
    html = str(row.get("text") or "")
    if not (40 <= len(idea) <= 1800 and 300 <= len(html) <= 8500):
        return None
    html = re.sub(r"https?://[^\"')\s]+", "/assets/placeholder", html)
    prompt = (
        "Implement an original responsive interface from this brief. Use semantic HTML, keyboard-accessible controls, "
        "clear empty states, and local placeholder assets only.\n\nBrief: %s" % idea
    )
    return base_pair(
        feed=feed,
        source_index=index,
        prompt=prompt,
        chosen=html,
        rejected="<html><body><div>Website</div></body></html>",
        defect="The rejected page ignores the requested layout, responsive behavior, semantics, accessibility, and visual hierarchy.",
        domain="product_and_ui_engineering",
        task_type="ui_implementation",
        difficulty=4,
        response_mode="expert_workflow",
    )


NORMALIZERS: Dict[str, Callable[[Mapping[str, Any], Mapping[str, Any], int], Optional[Dict[str, Any]]]] = {
    "helpsteer_edit": normalize_helpsteer,
    "hermes_tool": normalize_hermes,
    "swe_gym": normalize_swe_gym,
    "websight": normalize_websight,
}


def materialize_feed(feed: Mapping[str, Any], output_dir: Path, record_override: Optional[int]) -> Dict[str, Any]:
    metadata = get_json(HF_API + str(feed["dataset_id"]))
    actual_revision = str(metadata.get("sha") or "")
    actual_license = str((metadata.get("cardData") or {}).get("license") or "").lower()
    if actual_revision != str(feed["revision"]):
        raise RuntimeError("revision drift for %s: %s" % (feed["dataset_id"], actual_revision))
    if actual_license != str(feed["license"]).lower():
        raise RuntimeError("license drift for %s: %s" % (feed["dataset_id"], actual_license))
    total = split_size(str(feed["dataset_id"]), str(feed["config"]), str(feed["split"]))
    target = int(record_override or feed.get("pilot_records") or 128)
    rng = random.Random(int(stable_hash({"feed": feed["name"], "revision": feed["revision"]}, 16), 16))
    offsets = list(range(0, total, 100))
    rng.shuffle(offsets)
    normalizer = NORMALIZERS[str(feed["normalizer"])]
    accepted: List[Dict[str, Any]] = []
    seen = set()
    examined = 0
    for offset in offsets:
        for local_index, row in enumerate(dataset_rows(str(feed["dataset_id"]), str(feed["config"]), str(feed["split"]), offset, 100)):
            examined += 1
            pair = normalizer(row, feed, offset + local_index)
            if pair is None:
                continue
            key = stable_hash(pair["prompt"].lower(), 32)
            if key in seen:
                continue
            seen.add(key)
            accepted.append(pair)
            if len(accepted) >= target:
                break
        if len(accepted) >= target or examined >= max(1000, target * 12):
            break
    if len(accepted) < max(1, int(target * 0.50)):
        raise RuntimeError("feed %s yielded only %d/%d rows" % (feed["name"], len(accepted), target))
    path = output_dir / (str(feed["name"]) + ".jsonl")
    write_jsonl(path, accepted)
    return {
        "name": feed["name"],
        "dataset_id": feed["dataset_id"],
        "revision": feed["revision"],
        "license": feed["license"],
        "requested": target,
        "examined": examined,
        "accepted": len(accepted),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def download_file(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("unsafe archive member %s" % member.filename)
        bundle.extractall(destination)


def download_kaggle(spec: Mapping[str, Any], destination: Path, evaluation_only: bool) -> Dict[str, Any]:
    ref = str(spec["ref"])
    metadata = get_json(KAGGLE_API + "/view/" + ref)
    actual_version = int(metadata.get("currentVersionNumberNullable") or metadata.get("currentVersionNumber") or 0)
    declared = str(metadata.get("licenseNameNullable") or metadata.get("licenseName") or "unknown")
    if actual_version != int(spec["version"]):
        raise RuntimeError("Kaggle version drift for %s: %d" % (ref, actual_version))
    canonical_licenses = {
        "cc0: public domain": "CC0-1.0",
        "apache 2.0": "Apache-2.0",
        "attribution 4.0 international (cc by 4.0)": "CC-BY-4.0",
    }
    canonical_declared = canonical_licenses.get(declared.lower(), declared)
    if canonical_declared.lower() != str(spec["declared_license"]).lower():
        raise RuntimeError(
            "Kaggle licence drift for %s: expected %s, found %s"
            % (ref, spec["declared_license"], declared)
        )
    slug = ref.replace("/", "__")
    target = destination / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    archive = target / "dataset.zip"
    download_file(
        KAGGLE_API + "/download/" + ref + "?datasetVersionNumber=%d" % actual_version,
        archive,
    )
    extracted = target / "files"
    safe_extract(archive, extracted)
    record = {
        "ref": ref,
        "version": actual_version,
        "declared_license": canonical_declared,
        "expected_license": spec["declared_license"],
        "evaluation_only": evaluation_only,
        "use": spec["use"],
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "files": [
            {"path": str(path.relative_to(target)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(extracted.rglob("*"))
            if path.is_file()
        ],
    }
    (target / "source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned expert public data with train/evaluation separation")
    parser.add_argument("--policy", default="config/expert-public-data.yaml")
    parser.add_argument("--output-dir", default="sources/public/expert-curated")
    parser.add_argument("--records-per-feed", type=int)
    parser.add_argument("--skip-kaggle", action="store_true")
    args = parser.parse_args()
    policy_path = (ROOT / args.policy).resolve()
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feeds = [materialize_feed(feed, output_dir, args.records_per_feed) for feed in policy["training_feeds"]]
    kaggle_train: List[Dict[str, Any]] = []
    kaggle_eval: List[Dict[str, Any]] = []
    if not args.skip_kaggle:
        for spec in policy.get("kaggle_training_candidates") or []:
            kaggle_train.append(download_kaggle(spec, ROOT / "sources/public/kaggle", False))
        for spec in policy.get("kaggle_evaluation_only") or []:
            kaggle_eval.append(download_kaggle(spec, ROOT / "sources/evaluation/kaggle", True))
    report = {
        "schema_version": "1.0",
        "policy": str(policy_path),
        "created_unix": time.time(),
        "training_feeds": feeds,
        "training_records": sum(row["accepted"] for row in feeds),
        "kaggle_training_candidates": kaggle_train,
        "kaggle_evaluation_only": kaggle_eval,
        "deferred_training_feeds": policy.get("deferred_training_feeds") or [],
        "evaluation_only": policy.get("evaluation_only") or [],
        "private_reasoning_included": False,
        "benchmark_rows_in_training": False,
    }
    report_path = ROOT / "catalog" / "expert-public-data-audit-2026-08-19.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "training_records": report["training_records"], "feeds": feeds}, indent=2))


if __name__ == "__main__":
    main()
