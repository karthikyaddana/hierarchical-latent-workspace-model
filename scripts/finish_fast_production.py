#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hlwm_data.config import load_pipeline_config
from hlwm_data.fast_generate import _count_fast_accepted, fast_generate_episodes
from hlwm_data.materialize import build_dataset
from hlwm_data.util import atomic_write_json


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finish a no-judge fast generation run and materialize at least 4,000 episodes"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--target", type=int, default=4300)
    parser.add_argument("--minimum-final", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_pipeline_config(root / args.config)
    config["azure"]["max_workers"] = args.workers
    progress_path = root / "data/reasoning9000/fast-progress.json"
    report_path = root / "reports/reasoning9000-fast-production.json"

    while process_exists(args.wait_pid):
        time.sleep(30)

    target = args.target
    while True:
        accepted = _count_fast_accepted(root / config["output"]["reviewed_dir"])
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
        next_offset = int(progress.get("next_offset", 335))
        if accepted < target:
            before = accepted
            result = fast_generate_episodes(
                config,
                root,
                target=target,
                resume=True,
                offset=next_offset,
            )
            accepted = int(result.get("accepted", 0))
            if accepted <= before and result.get("status") != "complete":
                raise SystemExit("Fast generation made no progress; inspect Azure request logs")
            continue

        config["generation"]["target_accepted_episodes"] = target
        manifest = build_dataset(config, root)
        final_episodes = int(manifest.get("episodes", 0))
        if final_episodes >= args.minimum_final:
            report = {
                "status": "complete",
                "generation_mode": "direct_fast_no_judge",
                "accepted_checkpoints": accepted,
                "final_episodes_after_deduplication": final_episodes,
                "minimum_final": args.minimum_final,
                "manifest": str(root / config["output"]["final_dir"] / "manifest.json"),
            }
            atomic_write_json(report_path, report)
            print(json.dumps(report, indent=2), flush=True)
            return

        target += (args.minimum_final - final_episodes) + 200


if __name__ == "__main__":
    main()
