#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hlwm_data.config import load_pipeline_config
from hlwm_data.generate import generate_episodes, judge_episodes, repair_rejected_episodes
from hlwm_data.materialize import build_dataset, validate_built_dataset
from hlwm_data.util import atomic_write_json


REQUIRED_SCALE_GATES = {
    "enough_reviewed",
    "acceptance_rate",
    "first_pass_acceptance_rate",
    "expertise_uplift",
    "accepted_expertise_floor",
    "accepted_score_floor",
    "accepted_review_consistency",
    "adversarial_acceptance",
    "dual_judge_consensus",
    "manual_adversarial_inspection",
    "static_validity",
    "english_only",
    "private_reasoning_free",
    "no_exact_accepted_duplicates",
    "domain_coverage",
}


def scale_gate_errors(gate: dict) -> list[str]:
    errors = []
    if gate.get("scale_allowed") is not True:
        errors.append("scale_allowed")
    errors.extend(
        sorted(name for name in REQUIRED_SCALE_GATES if gate.get("gates", {}).get(name) is not True)
    )
    return errors


def reviewed_counts(path: Path) -> dict:
    accepted = rejected = 0
    for item in path.glob("*.json"):
        value = json.loads(item.read_text(encoding="utf-8"))
        if value.get("accepted"):
            accepted += 1
        else:
            rejected += 1
    return {"accepted": accepted, "rejected": rejected, "reviewed": accepted + rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate until an exact accepted-episode target is materialized")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--start-offset", type=int, default=50)
    parser.add_argument("--max-generated", type=int, default=18000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pilot-gate", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_pipeline_config(root / args.config)
    config["azure"]["max_workers"] = args.workers
    config["generation"]["target_accepted_episodes"] = args.target
    gate = json.loads((root / args.pilot_gate).read_text(encoding="utf-8"))
    failed_or_missing = scale_gate_errors(gate)
    if failed_or_missing:
        raise SystemExit(
            "Pilot gate failed, is stale, or is incomplete (%s); refusing high-cost scale generation"
            % ", ".join(failed_or_missing)
        )

    output = config["output"]
    reviewed_dir = root / output["reviewed_dir"]
    progress_path = root / "data/reasoning9000/progress.json"
    stop_path = root / "data/reasoning9000/STOP"
    offset = args.start_offset
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        offset = max(offset, int(previous.get("next_offset", offset)))

    while offset < args.max_generated:
        if stop_path.exists():
            raise SystemExit("Graceful stop requested by %s" % stop_path)
        counts = reviewed_counts(reviewed_dir)
        if counts["accepted"] >= args.target:
            manifest = build_dataset(config, root)
            if int(manifest.get("episodes", 0)) >= args.target:
                report = validate_built_dataset(config, root)
                atomic_write_json(progress_path, {"status": "complete", "next_offset": offset, "counts": counts, "manifest": manifest, "validation": report})
                print(json.dumps({"status": "complete", "counts": counts, "episodes": manifest["episodes"], "validation": report}, indent=2), flush=True)
                raise SystemExit(0 if report.get("valid") else 1)

        batch = min(args.batch_size, args.max_generated - offset)
        print(json.dumps({"stage": "generate", "offset": offset, "count": batch}), flush=True)
        generation = generate_episodes(config, root, batch, resume=True, offset=offset)
        print(json.dumps({"stage": "judge", "offset": offset}), flush=True)
        first_judge = judge_episodes(config, root, resume=True)
        repair = repair_rejected_episodes(config, root, resume=True) if config["generation"].get("repair_rejected") else {}
        final_judge = judge_episodes(config, root, resume=True)
        counts = reviewed_counts(reviewed_dir)
        offset += batch
        atomic_write_json(
            progress_path,
            {
                "status": "running",
                "next_offset": offset,
                "counts": counts,
                "last_batch": {"generate": generation, "first_judge": first_judge, "repair": repair, "final_judge": final_judge},
            },
        )
        print(json.dumps({"stage": "batch_complete", "next_offset": offset, "counts": counts}), flush=True)

    manifest = build_dataset(config, root)
    report = validate_built_dataset(config, root)
    atomic_write_json(progress_path, {"status": "max_generated_reached", "next_offset": offset, "counts": reviewed_counts(reviewed_dir), "manifest": manifest, "validation": report})
    raise SystemExit("Maximum generation limit reached before accepted target")


if __name__ == "__main__":
    main()
