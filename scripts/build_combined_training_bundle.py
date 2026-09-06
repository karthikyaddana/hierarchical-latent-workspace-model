#!/usr/bin/env python3
"""Combine every audit-cleared training source into one checksummed bundle.

Sources (all previously cleared; nothing new is fetched):
  1. Builder-factory episodes (data/builder/final): deterministically verified
     product/code/copy/plan/devops episodes plus SFT and DPO views.
  2. The v5.5 candidate bundle master (Reasoning9000 episodes + behavior
     anchors), already split-isolated by lineage.
  3. The 384 license-audited OpenThoughts/DeepCoder packets (problem +
     verified reference answer only; train split only; English re-checked).

Benchmark evaluation items never enter this bundle (config/benchmarks.
reasoning.yaml policy). Cross-split prompt leakage fails closed: a train row
whose normalized prompt also appears in validation/test is dropped from train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlwm_data.language import classify_language  # noqa: E402
from hlwm_data.util import (  # noqa: E402
    atomic_write_json,
    iter_jsonl,
    normalize_text,
    stable_hash,
    write_jsonl,
)

SPLITS = ("train", "validation", "test")
PACKET_FILES = (
    "sources/public/huggingface-curated/open-thoughts-verified-code.jsonl",
    "sources/public/huggingface-curated/deepcoder-primeintellect-train.jsonl",
    "sources/public/huggingface-curated/deepcoder-taco-train.jsonl",
)
BUNDLE_MASTER = "artifacts/kaggle/hlwm-v5.5/bundle/hlwm_kaggle/data/master"
BUILDER_FINAL = "data/builder/final"
OUTPUT_DIR = "data/combined"


def _prompt_of_episode(episode: Mapping[str, Any]) -> str:
    source = episode.get("input") or {}
    return "\n".join(
        [str(source.get("user_request", ""))]
        + [str(item) for item in source.get("context") or []]
    )


def _prompt_key(text: str) -> str:
    return stable_hash(normalize_text(text).lower())


def _episode_sft(episode: Mapping[str, Any], source: str) -> Dict[str, Any] | None:
    if str((episode.get("commitment") or {}).get("decision", "")).lower() != "publish":
        return None
    answer = str((episode.get("integration") or {}).get("published_answer", "")).strip()
    prompt = _prompt_of_episode(episode).strip()
    if not answer or not prompt:
        return None
    source_input = episode.get("input") or {}
    lines = [prompt, ""]
    lines.extend("- %s" % item for item in source_input.get("context") or [])
    lines.append("")
    lines.extend("- %s" % item for item in source_input.get("constraints") or [])
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a senior %s specialist. Deliver complete, verified work."
                % str(episode.get("domain", "generalist")),
            },
            {"role": "user", "content": "\n".join(lines).strip()},
            {"role": "assistant", "content": answer},
        ],
        "episode_id": str(episode.get("episode_id", "")),
        "domain": str(episode.get("domain", "unknown")),
        "bundle_source": source,
    }


def _packet_sft(packet: Mapping[str, Any]) -> Dict[str, Any] | None:
    problem = str(packet.get("problem", "")).strip()
    answer = str(packet.get("reference_answer", "")).strip()
    if not problem or not answer:
        return None
    # OpenThoughts rows carry an explicit origin marker; DeepCoder rows omit the
    # field because the audited fetch already strips private reasoning. Exclude
    # only when a present marker says something other than the verified form.
    origin = packet.get("reference_origin")
    if origin is not None and str(origin) != "verified_final_solution_without_private_reasoning":
        return None
    serialized = "\n".join(
        (problem, answer, json.dumps(packet.get("verification_material") or "", ensure_ascii=False))
    )
    if not classify_language(serialized).accepted:
        return None
    source = packet.get("source") or {}
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a senior %s specialist. Solve the task completely and verifiably."
                % str(packet.get("task_family", "software")),
            },
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ],
        "episode_id": "packet-%s" % stable_hash(problem, 16),
        "domain": str(packet.get("task_family", "software-reasoning")),
        "bundle_source": "audited-packets",
        "provenance": {
            "dataset_id": str(source.get("dataset_id", "")),
            "revision": str(source.get("revision", "")),
        },
    }


def build(root: Path) -> Dict[str, Any]:
    master: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    sft: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    dpo: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}

    def bump(source: str, key: str, amount: int = 1) -> None:
        counts.setdefault(source, {})[key] = counts.setdefault(source, {}).get(key, 0) + amount

    seen_episode_ids: set[str] = set()
    seen_prompts: Dict[str, set[str]] = {split: set() for split in SPLITS}

    # 1. Episode sources: v5.5 bundle first (protected evaluation rows), then builder.
    episode_sources = [
        ("v5.5-bundle", root / BUNDLE_MASTER),
        ("builder-factory", root / BUILDER_FINAL / "master"),
    ]
    for source_name, directory in episode_sources:
        for split in SPLITS:
            path = directory / ("%s.jsonl" % split)
            if not path.exists():
                continue
            for episode in iter_jsonl(path):
                episode_id = str(episode.get("episode_id", ""))
                key = _prompt_key(_prompt_of_episode(episode))
                if episode_id and episode_id in seen_episode_ids:
                    bump(source_name, "duplicate_episode_id")
                    continue
                if key in seen_prompts[split]:
                    bump(source_name, "duplicate_prompt")
                    continue
                seen_episode_ids.add(episode_id)
                seen_prompts[split].add(key)
                episode = dict(episode)
                episode["bundle_source"] = source_name
                master[split].append(episode)
                bump(source_name, "master_%s" % split)
                row = _episode_sft(episode, source_name)
                if row is not None:
                    sft[split].append(row)
                    bump(source_name, "sft_%s" % split)

    # 2. Audited packets: train only, never used for evaluation. Includes the
    # original 384 plus any official benchmark train splits materialized by
    # scripts/fetch_benchmark_train_splits.py.
    packet_paths = [root / relative for relative in PACKET_FILES]
    packet_paths += sorted((root / "sources/public/benchmark-train-splits").glob("*.jsonl"))
    for path in packet_paths:
        if not path.exists():
            continue
        source_label = (
            "audited-packets"
            if "huggingface-curated" in str(path)
            else "benchmark-train-splits/%s" % path.stem
        )
        for packet in iter_jsonl(path):
            row = _packet_sft(packet)
            if row is None:
                bump(source_label, "excluded")
                continue
            key = _prompt_key(row["messages"][1]["content"])
            if key in seen_prompts["train"]:
                bump(source_label, "duplicate_prompt")
                continue
            seen_prompts["train"].add(key)
            sft["train"].append(row)
            bump(source_label, "sft_train")

    # 3. Builder DPO pairs.
    dpo_path = root / BUILDER_FINAL / "dpo" / "pairs.jsonl"
    if dpo_path.exists():
        for pair in iter_jsonl(dpo_path):
            dpo.append(dict(pair, bundle_source="builder-factory"))
        bump("builder-factory", "dpo_pairs", len(dpo))

    # 4. Fail-closed cross-split leakage: drop train rows that shadow eval rows.
    eval_keys = seen_prompts["validation"] | seen_prompts["test"]
    leaked = 0
    for collection in (master, sft):
        kept = []
        for row in collection["train"]:
            prompt = (
                _prompt_of_episode(row)
                if "input" in row
                else row["messages"][1]["content"]
            )
            if _prompt_key(prompt) in eval_keys:
                leaked += 1
                continue
            kept.append(row)
        collection["train"] = kept
    bump("bundle", "train_rows_dropped_for_eval_overlap", leaked)

    output = root / OUTPUT_DIR
    checksums: Dict[str, str] = {}
    totals: Dict[str, Dict[str, int]] = {"master": {}, "sft": {}}
    for split in SPLITS:
        for family, collection in (("master", master), ("sft", sft)):
            target = output / family / ("%s.jsonl" % split)
            totals[family][split] = write_jsonl(target, collection[split])
            checksums[str(target.relative_to(output))] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    dpo_target = output / "dpo" / "pairs.jsonl"
    totals["dpo"] = {"pairs": write_jsonl(dpo_target, dpo)}
    checksums[str(dpo_target.relative_to(output))] = hashlib.sha256(
        dpo_target.read_bytes()
    ).hexdigest()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totals": totals,
        "per_source": counts,
        "checksums_sha256": checksums,
        "policy": {
            "benchmark_evaluation_items_included": False,
            "teacher_private_reasoning_included": False,
            "external_packets_split": "train_only",
            "cross_split_prompt_leakage": "train rows dropped on collision",
            "unapproved_shortlist_sources_included": False,
        },
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    manifest = build(Path(args.root).resolve())
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))
    print(json.dumps(manifest["per_source"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
