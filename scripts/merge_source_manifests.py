#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge provenance manifests without weakening source permissions")
    parser.add_argument("--output", required=True)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    entries = []
    seen = set()
    language_policy = None
    for name in args.manifests:
        path = Path(name).expanduser().resolve()
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        candidate_policy = payload.get("language_policy")
        if candidate_policy:
            if language_policy is not None and candidate_policy != language_policy:
                raise ValueError("Conflicting language_policy values in source manifests")
            language_policy = candidate_policy
        for entry in payload.get("sources", []):
            key = (str(entry.get("path")), str(entry.get("version")))
            if key in seen:
                continue
            seen.add(key)
            value = dict(entry)
            value["manifest_origin"] = str(path)
            entries.append(value)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Combined provenance manifest. Individual source permissions remain authoritative.\n"
        + yaml.safe_dump(
            {
                "language_policy": language_policy
                or {"required": True, "allowed_languages": ["en"], "minimum_confidence": 0.78},
                "sources": entries,
            },
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ),
        encoding="utf-8",
    )
    print("merged_sources=%d output=%s" % (len(entries), output))


if __name__ == "__main__":
    main()
