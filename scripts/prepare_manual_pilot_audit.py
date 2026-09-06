#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hlwm_data.util import atomic_write_json
from scripts.assess_reasoning_pilot import accepted_corpus_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a corpus-bound checklist for manual adversarial inspection"
    )
    parser.add_argument("--reviewed-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reviewed_dir = Path(args.reviewed_dir)
    wrappers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(reviewed_dir.glob("*.json"))
    ]
    accepted_ids = sorted(
        str(item.get("episode", {}).get("episode_id", ""))
        for item in wrappers
        if item.get("accepted")
    )
    template = {
        "decision": "pending" if accepted_ids else "not_applicable",
        "reviewer": "",
        "completed_at": "",
        "accepted_corpus_hash": accepted_corpus_hash(wrappers),
        "false_accept_episode_ids": [],
        "inspections": [
            {"episode_id": episode_id, "verdict": "pending", "notes": ""}
            for episode_id in accepted_ids
        ],
    }
    output = Path(args.output)
    atomic_write_json(output, template)
    print(json.dumps({"output": str(output), "accepted_to_inspect": len(accepted_ids)}, indent=2))


if __name__ == "__main__":
    main()
