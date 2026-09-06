import json
from pathlib import Path

import yaml

from hlwm_data.language import contains_blocked_script


ROOT = Path(__file__).resolve().parents[1]


def test_every_reasoning_collection_dataset_is_approved_or_quarantined():
    policy = yaml.safe_load((ROOT / "config/reasoning-collection-policy.yaml").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "catalog/reasoning-collection-audit-2026-08-17.json").read_text(encoding="utf-8"))
    collection_datasets = {
        item["id"]
        for item in audit["collections"]["training_discovery"]["items"]
        if item["type"] == "dataset"
    }
    approved = {item["dataset_id"] for item in policy["training_feeds"]}
    quarantined = set(policy["quarantine"])
    assert collection_datasets == approved | quarantined
    assert approved.isdisjoint(quarantined)


def test_reference_collection_datasets_are_evaluation_only():
    audit = json.loads((ROOT / "catalog/reasoning-collection-audit-2026-08-17.json").read_text(encoding="utf-8"))
    benchmarks = yaml.safe_load((ROOT / "config/benchmarks.reasoning.yaml").read_text(encoding="utf-8"))
    benchmark_urls = "\n".join(str(item["url"]) for item in benchmarks["benchmarks"])
    reference_ids = {
        item["id"]
        for name in ("model_and_research_reference", "evaluation_reference")
        for item in audit["collections"][name]["items"]
        if item["type"] == "dataset"
    }
    assert reference_ids
    assert all(dataset_id in benchmark_urls for dataset_id in reference_ids)
    assert benchmarks["policy"]["allowed_for_training"] is False


def test_materialized_collection_has_no_private_reasoning_fields_or_markers():
    manifest = yaml.safe_load((ROOT / "config/sources.reasoning-collection.yaml").read_text(encoding="utf-8"))
    forbidden_keys = {"deepseek_reasoning", "conversations", "system"}
    forbidden_markers = ("<think>", "</think>", "<|begin_of_thought|>", "<|end_of_thought|>")
    count = 0
    for source in manifest["sources"]:
        with Path(source["path"]).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                count += 1
                assert record["language"] == "en"
                assert forbidden_keys.isdisjoint(record)
                serialized = line.lower()
                assert not any(marker in serialized for marker in forbidden_markers)
                assert not contains_blocked_script(serialized)
                assert record["curation"]["raw_private_reasoning_included"] is False
                assert record["curation"]["benchmark_splits_included"] is False
    assert count == 384
