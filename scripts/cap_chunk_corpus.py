#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List


def seed_for(group: str, seed: int) -> int:
    return int(hashlib.sha256((str(seed) + ":" + group).encode("utf-8")).hexdigest()[:16], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically reservoir-cap chunks per source group")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.cap <= 0:
        raise ValueError("cap must be positive")
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    reservoirs: Dict[str, List[dict]] = {}
    seen: Counter = Counter()
    random_by_group: Dict[str, random.Random] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            group = str(row["source_group"])
            seen[group] += 1
            bucket = reservoirs.setdefault(group, [])
            if len(bucket) < args.cap:
                bucket.append(row)
                continue
            rng = random_by_group.setdefault(group, random.Random(seed_for(group, args.seed)))
            replacement = rng.randrange(seen[group])
            if replacement < args.cap:
                bucket[replacement] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    written = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for group in sorted(reservoirs):
                for row in sorted(reservoirs[group], key=lambda item: str(item["chunk_id"])):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = {
        "input": str(source),
        "output": str(output),
        "cap_per_source_group": args.cap,
        "source_groups": len(reservoirs),
        "input_chunks": sum(seen.values()),
        "output_chunks": written,
        "removed_chunks": sum(seen.values()) - written,
        "original_counts": dict(sorted(seen.items())),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "original_counts"}, indent=2))


if __name__ == "__main__":
    main()
