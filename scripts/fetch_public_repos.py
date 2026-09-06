#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def run(command: List[str], cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and pin audited public training repositories")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []
    report_rows = []

    for spec in plan.get("repositories", []):
        repo = str(spec["repo"])
        owner, name = repo.split("/", 1)
        target = destination / (owner + "__" + name)
        metadata = json.loads(run(["gh", "api", "repos/%s" % repo]))
        detected_license = str((metadata.get("license") or {}).get("spdx_id") or "unknown")
        expected = {str(item) for item in spec.get("expected_licenses", [])}
        if not target.exists():
            run(["git", "clone", "--depth", "1", "--filter=blob:none", "https://github.com/%s.git" % repo, str(target)])
        else:
            run(["git", "fetch", "--depth", "1", "origin", str(metadata.get("default_branch") or "HEAD")], cwd=target)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
        commit = run(["git", "rev-parse", "HEAD"], cwd=target)
        license_filename = "LICENSE"
        if detected_license in {"NOASSERTION", "unknown"}:
            cc0_path = target / "LICENSE.md"
            if cc0_path.exists() and "CC0 1.0" in cc0_path.read_text(encoding="utf-8", errors="replace"):
                detected_license = "CC0-1.0"
                license_filename = "LICENSE.md"
            elif (target / "LICENSE-APACHE").exists() and (target / "LICENSE-MIT").exists():
                detected_license = "Apache-2.0 OR MIT"
                license_filename = "COPYRIGHT"
        license_ok = detected_license in expected or (
            detected_license == "Apache-2.0 OR MIT" and {"Apache-2.0", "MIT"}.issubset(expected)
        )
        license_url = "https://github.com/%s/blob/%s/%s" % (repo, commit, license_filename)
        entry = {
            "path": str(target),
            "title": repo,
            "author": owner,
            "domain": spec["domain"],
            "source_group": "github-%s-%s" % (owner.lower(), name.lower()),
            "lineage_component_id": "github-%s-%s" % (owner.lower(), name.lower()),
            "lineage_mode": "source",
            "version": "git:%s" % commit,
            "license": detected_license,
            "license_evidence": license_url,
            "allowed_for_training": license_ok,
            "url": "https://github.com/%s" % repo,
            "use": spec.get("use"),
        }
        if not license_ok:
            entry["quarantine_reason"] = "GitHub SPDX license %s not in expected allowlist %s" % (detected_license, sorted(expected))
        manifest_entries.append(entry)
        report_rows.append({"repo": repo, "commit": commit, "license": detected_license, "enabled": license_ok})

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump({"sources": manifest_entries}, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    report = {
        "repositories": report_rows,
        "enabled": sum(1 for row in report_rows if row["enabled"]),
        "quarantined": sum(1 for row in report_rows if not row["enabled"]),
        "manifest": str(manifest_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
