from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .util import atomic_write_json, canonical_json, stable_hash, write_jsonl
from .validation import episode_invariant_errors, schema_errors
from .contamination import contamination_report


STAGE_INSTRUCTIONS = {
    "framing": "Convert grounded context into an objective, requirements, unknowns, and failure contract.",
    "decomposition": "Propose complementary briefs and select a budget-feasible set of private lanes.",
    "private_solving": "Produce only this lane's observable artifacts, atomic claims, checkpoints, and summary.",
    "barrier": "Create identified lane summaries and expose conflicts without leaking private lane state.",
    "verification": "Independently adjudicate claims using only supplied evidence and rejection tests.",
    "synthesis": "Integrate only barrier-approved and verified material into one coherent answer.",
    "commitment": "Choose publish, refine, fallback, or abstain for the exact candidate and evidence snapshot.",
    "counterfactual": "Predict the verified outcome and errors for the declared controlled intervention.",
    "continuation": "Estimate whether one declared additional operation improved verified utility enough to continue.",
    "root_retention": "Compare root-only and routed observable capability without using a fixed mixture percentage.",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def simhash64(text: str) -> int:
    words = _words(text)
    shingles = [" ".join(words[index : index + 4]) for index in range(max(1, len(words) - 3))]
    if not shingles:
        shingles = [text.lower().strip()]
    vector = [0] * 64
    for shingle in shingles:
        value = int(hashlib.sha256(shingle.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")


def _deduplicate(episodes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    accepted: List[Dict[str, Any]] = []
    removed: List[Dict[str, str]] = []
    exact: Dict[str, str] = {}
    buckets: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for episode in sorted(episodes, key=lambda item: str(item["episode_id"])):
        text = "%s\n%s" % (
            episode.get("input", {}).get("user_request", ""),
            episode.get("integration", {}).get("published_answer", ""),
        )
        exact_key = stable_hash(re.sub(r"\s+", " ", text.lower()).strip(), 32)
        if exact_key in exact:
            removed.append({"episode_id": episode["episode_id"], "duplicate_of": exact[exact_key], "type": "exact"})
            continue
        fingerprint = simhash64(text)
        bucket = fingerprint >> 48
        near_match = next(
            (episode_id for other, episode_id in buckets[bucket] if hamming_distance(fingerprint, other) <= 3),
            None,
        )
        if near_match:
            removed.append({"episode_id": episode["episode_id"], "duplicate_of": near_match, "type": "near"})
            continue
        exact[exact_key] = episode["episode_id"]
        buckets[bucket].append((fingerprint, episode["episode_id"]))
        accepted.append(episode)
    return accepted, removed


def _balanced_quality_cap(
    episodes: Sequence[Dict[str, Any]],
    score_by_id: Dict[str, float],
    target: int,
    domain_specs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int]:
    """Cap a release by configured domain mix while rotating across source groups."""
    if target <= 0 or len(episodes) <= target:
        return list(episodes), 0
    weights = {name: max(0.0, float(spec.get("weight", 0.0))) for name, spec in domain_specs.items()}
    total_weight = sum(weights.values()) or 1.0
    raw = {name: target * value / total_weight for name, value in weights.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    for name in sorted(weights, key=lambda item: (raw[item] - quotas[item], item), reverse=True)[: target - sum(quotas.values())]:
        quotas[name] += 1

    by_domain_group: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        by_domain_group[str(episode.get("domain"))][str(episode.get("source_group"))].append(episode)
    for groups in by_domain_group.values():
        for rows in groups.values():
            rows.sort(key=lambda item: (-score_by_id.get(str(item["episode_id"]), 0.0), str(item["episode_id"])))

    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    for domain in sorted(quotas):
        groups = by_domain_group.get(domain, {})
        domain_selected = 0
        while groups and domain_selected < quotas[domain]:
            group_order = sorted(
                groups,
                key=lambda name: (
                    -score_by_id.get(str(groups[name][0]["episode_id"]), 0.0),
                    name,
                ),
            )
            progressed = False
            for group in group_order:
                if domain_selected >= quotas[domain]:
                    break
                rows = groups.get(group, [])
                if not rows:
                    groups.pop(group, None)
                    continue
                episode = rows.pop(0)
                selected.append(episode)
                selected_ids.add(str(episode["episode_id"]))
                domain_selected += 1
                progressed = True
                if not rows:
                    groups.pop(group, None)
            if not progressed:
                break

    if len(selected) < target:
        remaining = sorted(
            (item for item in episodes if str(item["episode_id"]) not in selected_ids),
            key=lambda item: (-score_by_id.get(str(item["episode_id"]), 0.0), str(item["episode_id"])),
        )
        selected.extend(remaining[: target - len(selected)])
    selected = sorted(selected[:target], key=lambda item: str(item["episode_id"]))
    return selected, len(episodes) - len(selected)


def _assign_split(lineage_component_id: str, splits: Dict[str, float]) -> str:
    value = int(hashlib.sha256(lineage_component_id.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    cumulative = 0.0
    for name in ("train", "validation", "test"):
        cumulative += float(splits[name])
        if value < cumulative:
            return name
    return "test"


def _envelope(episode: Dict[str, Any], split: str, stage: str, suffix: str) -> Dict[str, Any]:
    return {
        "view_version": "1.0",
        "sample_id": "%s-%s-%s" % (episode["episode_id"], stage, suffix),
        "episode_id": episode["episode_id"],
        "split": split,
        "lineage_component_id": episode["lineage_component_id"],
        "domain": episode["domain"],
        "subdomain": episode["subdomain"],
        "stage": stage,
        "source_provenance_ids": [ref["source_id"] for ref in episode.get("source_refs", [])],
        "loss_mask": True,
    }


def materialize_episode(episode: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    public_context = {
        "source_refs": episode["source_refs"],
        "input": episode["input"],
    }
    rows: List[Dict[str, Any]] = []

    row = _envelope(episode, split, "framing", "0")
    row.update({"input": public_context, "target": episode["frame"]})
    rows.append(row)

    row = _envelope(episode, split, "decomposition", "0")
    row.update({"input": {**public_context, "frame": episode["frame"]}, "target": episode["routing"]})
    rows.append(row)

    for lane in episode["lanes"]:
        row = _envelope(episode, split, "private_solving", str(lane["lane_id"]))
        row.update(
            {
                "input": {
                    **public_context,
                    "frame": episode["frame"],
                    "lane_id": lane["lane_id"],
                    "route": lane["route"],
                    "route_windows": lane["route_windows"],
                    "brief": lane["brief"],
                    "budget": episode["routing"]["budget"],
                },
                "target": {
                    "artifacts": lane["artifacts"],
                    "claims": lane["claims"],
                    "checkpoints": lane["checkpoints"],
                    "summary": lane["summary"],
                },
            }
        )
        rows.append(row)

    row = _envelope(episode, split, "barrier", "0")
    row.update(
        {
            "input": {
                "episode_id": episode["episode_id"],
                "lane_terminals": [
                    {"lane_id": lane["lane_id"], "artifacts": lane["artifacts"], "claims": lane["claims"], "summary": lane["summary"]}
                    for lane in episode["lanes"]
                ],
            },
            "target": episode["barrier"],
        }
    )
    rows.append(row)

    row = _envelope(episode, split, "verification", "0")
    row.update(
        {
            "input": {
                **public_context,
                "barrier": episode["barrier"],
                "lane_claims": [lane["claims"] for lane in episode["lanes"]],
                "tool_runs": episode["tool_runs"],
            },
            "target": episode["verification"],
        }
    )
    rows.append(row)

    for index, pair in enumerate(episode["continuation_pairs"]):
        row = _envelope(episode, split, "continuation", str(index))
        row.update(
            {
                "input": {
                    "frame": episode["frame"],
                    "routing": episode["routing"],
                    "kind": pair["kind"],
                    "added_operation": pair["added_operation"],
                    "added_cost": pair["added_cost"],
                    "evidence_refs": pair["evidence_refs"],
                },
                "target": pair,
                "loss_mask": bool(pair.get("eligible", False)),
            }
        )
        rows.append(row)

    row = _envelope(episode, split, "synthesis", "0")
    row.update(
        {
            "input": {**public_context, "barrier": episode["barrier"], "verification": episode["verification"]},
            "target": episode["integration"],
        }
    )
    rows.append(row)

    row = _envelope(episode, split, "commitment", "0")
    row.update(
        {
            "input": {
                "candidate": episode["integration"],
                "verification": episode["verification"],
                "constraints": episode["input"]["constraints"],
            },
            "target": episode["commitment"],
        }
    )
    rows.append(row)

    for index, counterfactual in enumerate(episode["counterfactuals"]):
        row = _envelope(episode, split, "counterfactual", str(index))
        row.update(
            {
                "input": {
                    "frame": episode["frame"],
                    "baseline_routing": episode["routing"],
                    "intervention": counterfactual["variant"],
                },
                "target": counterfactual,
            }
        )
        rows.append(row)
    for index, anchor in enumerate(episode["root_anchor_evaluations"]):
        row = _envelope(episode, split, "root_retention", str(index))
        row.update(
            {
                "input": {"frame": episode["frame"], "capability": anchor["capability"], "source_refs": episode["source_refs"]},
                "target": anchor,
            }
        )
        rows.append(row)
    return rows


def to_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are one root-preserving model operating in %s mode. %s "
        "Return only the requested observable structured output; do not reveal private chain-of-thought."
        % (row["stage"], STAGE_INSTRUCTIONS[row["stage"]])
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical_json(row["input"])},
            {"role": "assistant", "content": canonical_json(row["target"])},
        ],
        "metadata": {
            "sample_id": row["sample_id"],
            "episode_id": row["episode_id"],
            "stage": row["stage"],
            "domain": row["domain"],
            "lineage_component_id": row["lineage_component_id"],
        },
    }


def to_dpo(
    repaired: Dict[str, Any], original: Dict[str, Any], review: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = {
        "source_refs": repaired.get("source_refs", []),
        "input": repaired.get("input", {}),
        "frame": repaired.get("frame", {}),
        "instruction": "Produce a grounded, constraint-compliant integration decision and published answer from observable evidence only.",
    }
    return {
        "input": {
            "messages": [
                {"role": "system", "content": "Prefer verified synthesis over plausible unsupported claims. Do not reveal private chain-of-thought."},
                {"role": "user", "content": canonical_json(prompt)},
            ]
        },
        "preferred_output": [{"role": "assistant", "content": canonical_json(repaired.get("integration", {}))}],
        "non_preferred_output": [{"role": "assistant", "content": canonical_json(original.get("integration", {}))}],
        "metadata": {
            "episode_id": repaired["episode_id"],
            "domain": repaired["domain"],
            "lineage_component_id": repaired["lineage_component_id"],
            "judge_score": review.get("overall_score"),
            "judge_issues": review.get("issues", []),
            "repair_round": repaired.get("repair_metadata", {}).get("round"),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    reviewed_dir = root / config["output"]["reviewed_dir"]
    final_dir = root / config["output"]["final_dir"]
    reviewed_paths = sorted(reviewed_dir.glob("*.json"))
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(reviewed_paths)))) as executor:
        wrappers = list(
            executor.map(
                lambda path: json.loads(path.read_text(encoding="utf-8")),
                reviewed_paths,
            )
        )
    accepted_wrappers = [wrapper for wrapper in wrappers if wrapper.get("accepted")]
    accepted = [wrapper["episode"] for wrapper in accepted_wrappers]
    score_by_id = {
        str(wrapper["episode"]["episode_id"]): float(wrapper.get("review", {}).get("overall_score", 0.0))
        for wrapper in accepted_wrappers
    }
    review_by_id = {
        str(wrapper["episode"]["episode_id"]): wrapper.get("review", {})
        for wrapper in accepted_wrappers
    }
    episodes, duplicates = _deduplicate(accepted)
    target = int(config.get("generation", {}).get("target_accepted_episodes", 0) or 0)
    episodes, quality_capped = _balanced_quality_cap(
        episodes,
        score_by_id,
        target,
        config.get("generation", {}).get("domains", {}),
    )
    split_config = config["splits"]
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for episode in episodes:
        lineage = str(episode.get("lineage_component_id") or episode.get("source_group"))
        episode["lineage_component_id"] = lineage
        split = _assign_split(lineage, split_config)
        by_split[split].append(episode)

    written: List[Path] = []
    stage_counts: Counter = Counter()
    for split, split_episodes in by_split.items():
        master_path = final_dir / "master" / (split + ".jsonl")
        write_jsonl(master_path, split_episodes)
        written.append(master_path)
        native_rows = []
        for episode in split_episodes:
            rows = materialize_episode(episode, split)
            native_rows.extend(rows)
            stage_counts.update(row["stage"] for row in rows)
        native_path = final_dir / "native" / (split + ".jsonl")
        sft_path = final_dir / "sft" / (split + ".jsonl")
        write_jsonl(native_path, native_rows)
        write_jsonl(sft_path, (to_sft(row) for row in native_rows if row.get("loss_mask", False)))
        history_dir = reviewed_dir.parent / "history" / "pre-repair"
        dpo_rows = []
        for episode in split_episodes:
            original_path = history_dir / (str(episode["episode_id"]) + ".json")
            if not original_path.exists() or not episode.get("repair_metadata"):
                continue
            original = json.loads(original_path.read_text(encoding="utf-8"))
            if canonical_json(original.get("integration", {})) == canonical_json(episode.get("integration", {})):
                continue
            dpo_rows.append(to_dpo(episode, original, review_by_id.get(str(episode["episode_id"]), {})))
        dpo_path = final_dir / "dpo" / (split + ".jsonl")
        write_jsonl(dpo_path, dpo_rows)
        written.extend([native_path, sft_path, dpo_path])

    duplicate_path = final_dir / "reports" / "duplicates.json"
    atomic_write_json(duplicate_path, duplicates)
    written.append(duplicate_path)
    manifest = {
        "release_version": "dataset-v1",
        "schema_version": config["project"].get("schema_version", "1.0"),
        "project": config["project"].get("name"),
        "accepted_before_deduplication": len(accepted),
        "episodes": len(episodes),
        "duplicates_removed": len(duplicates),
        "quality_capped": quality_capped,
        "target_accepted_episodes": target or None,
        "split_episode_counts": {key: len(value) for key, value in by_split.items()},
        "stage_counts": dict(sorted(stage_counts.items())),
        "files": {},
    }
    for path in written:
        manifest["files"][str(path.relative_to(root))] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    manifest_path = final_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_built_dataset(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    final_dir = root / config["output"]["final_dir"]
    report: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "counts": {}}
    lineage_splits: Dict[str, Set[str]] = defaultdict(set)
    episode_splits: Dict[str, Set[str]] = defaultdict(set)
    sample_ids: Set[str] = set()
    episode_schema = root / "schemas/episode.schema.json"
    all_master_episodes: List[Dict[str, Any]] = []
    chunks_path = root / config["output"]["chunks_file"]
    known_chunk_ids: Set[str] = set()
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as handle:
            known_chunk_ids = {
                str(json.loads(line)["chunk_id"])
                for line in handle
                if line.strip()
            }
    for split in ("train", "validation", "test"):
        master_path = final_dir / "master" / (split + ".jsonl")
        sft_path = final_dir / "sft" / (split + ".jsonl")
        dpo_path = final_dir / "dpo" / (split + ".jsonl")
        masters = []
        if master_path.exists():
            with master_path.open("r", encoding="utf-8") as handle:
                masters = [json.loads(line) for line in handle if line.strip()]
        report["counts"]["master_%s" % split] = len(masters)
        all_master_episodes.extend(masters)
        for episode in masters:
            lineage_splits[str(episode["lineage_component_id"])].add(split)
            episode_splits[str(episode["episode_id"])].add(split)
            for error in schema_errors(episode, episode_schema):
                report["errors"].append("episode %s schema: %s" % (episode["episode_id"], error))
            for error in episode_invariant_errors(episode, known_chunk_ids):
                report["errors"].append("episode %s invariant: %s" % (episode["episode_id"], error))
        sft_count = 0
        if sft_path.exists():
            with sft_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    roles = [message.get("role") for message in row.get("messages", [])]
                    if roles != ["system", "user", "assistant"]:
                        report["errors"].append("%s:%d has invalid message roles" % (sft_path, line_number))
                    sample_id = str(row.get("metadata", {}).get("sample_id", ""))
                    if not sample_id:
                        report["errors"].append("%s:%d is missing sample_id" % (sft_path, line_number))
                    elif sample_id in sample_ids:
                        report["errors"].append("duplicate materialized sample_id %s" % sample_id)
                    sample_ids.add(sample_id)
                    sft_count += 1
        report["counts"]["sft_%s" % split] = sft_count
        dpo_count = 0
        if dpo_path.exists():
            with dpo_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    input_roles = [message.get("role") for message in row.get("input", {}).get("messages", [])]
                    preferred_roles = [message.get("role") for message in row.get("preferred_output", [])]
                    rejected_roles = [message.get("role") for message in row.get("non_preferred_output", [])]
                    if input_roles != ["system", "user"] or preferred_roles != ["assistant"] or rejected_roles != ["assistant"]:
                        report["errors"].append("%s:%d has invalid DPO message roles" % (dpo_path, line_number))
                    dpo_count += 1
        report["counts"]["dpo_%s" % split] = dpo_count
    for lineage, splits in lineage_splits.items():
        if len(splits) > 1:
            report["errors"].append("lineage %s appears in multiple splits: %s" % (lineage, sorted(splits)))
    for episode_id, splits in episode_splits.items():
        if len(splits) > 1:
            report["errors"].append("episode %s appears in multiple splits: %s" % (episode_id, sorted(splits)))
    evaluation = config.get("evaluation", {})
    if evaluation.get("benchmark_dir"):
        contamination = contamination_report(
            all_master_episodes,
            root / str(evaluation["benchmark_dir"]),
            float(evaluation.get("contamination_coverage_threshold", 0.45)),
        )
        report["benchmark_contamination"] = contamination
        for flag in contamination["flags"]:
            report["errors"].append(
                "benchmark contamination: episode %s overlaps %s"
                % (flag["episode_id"], flag["benchmark_item"])
            )
    if sum(report["counts"].get("master_%s" % split, 0) for split in ("train", "validation", "test")) >= 20:
        for split in ("train", "validation", "test"):
            if report["counts"].get("master_%s" % split, 0) == 0:
                report["warnings"].append("%s split is empty; provide more independent lineage groups" % split)

    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, metadata in manifest.get("files", {}).items():
            path = root / relative
            if not path.exists():
                report["errors"].append("manifest file missing: %s" % relative)
            elif _sha256_file(path) != metadata.get("sha256"):
                report["errors"].append("manifest checksum mismatch: %s" % relative)
    report["valid"] = not report["errors"]
    atomic_write_json(final_dir / "validation-report.json", report)
    return report
