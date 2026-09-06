#!/usr/bin/env python3
"""Create Zenodo deposits for the HLWM record set.

Mints one record per target and prints the reserved DOI. Nothing is published
automatically: deposits are left in draft so you can review them in the Zenodo UI
before hitting publish.

    export ZENODO_TOKEN=...            # deposit:actions + deposit:write scopes
    python tools/zenodo_deposit.py --target postmortem
    python tools/zenodo_deposit.py --target hlwm-paper
    python tools/zenodo_deposit.py --target checkpoints --apply

Use --sandbox against sandbox.zenodo.org first; it is a separate account and token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

CONCEPT_DOI = "10.5281/zenodo.22343152"
REPO_URL = "https://github.com/karthikyaddana/hierarchical-latent-workspace-model"

AUTHORS = [{"name": "Yaddanapudi, Karthik", "affiliation": "Independent Researcher"}]

TARGETS = {
    "postmortem": {
        "upload_type": "publication",
        "publication_type": "preprint",
        "title": (
            "Readable but Not Usable: A Preregistered Post-Mortem of a "
            "Latent-Workspace Language Model, Including Its Own Novelty Audit"
        ),
        "description": (
            "With the operative premise deleted from the decoder's input token ids, "
            "generation through the latent channel scored 0.000 on all 99 masked rows "
            "on both seeds, while a linear probe recovered roughly 0.30 of the withheld "
            "premise. The channel was readable but not usable. Includes a novelty audit "
            "of the work's own six architectural claims and an instrument-failure "
            "taxonomy covering ~30 documented defects in five classes."
        ),
        "files": ["paper/postmortem-paper.pdf"],
        "license": "cc-by-4.0",
    },
    "hlwm-paper": {
        "upload_type": "publication",
        "publication_type": "preprint",
        "title": (
            "The Hierarchical Latent Workspace Model: Root-Preserving Variable-Depth "
            "Expertise for Parallel Diffusion Reasoning"
        ),
        "description": (
            "The full technical report on a latent-workspace language model built on a "
            "frozen Qwen3-0.6B decoder, and the ten preregistered prototype experiments "
            "that falsified it at this scale. Routing failed causal liveness on 10 of 10 "
            "seed-runs; ablating the read-out's 34 memory positions changed graded "
            "accuracy by 0.000 on one seed and improved it by 0.022 on the other. "
            "Parity was reached by irrelevance."
        ),
        "files": ["paper/hlwm-paper.pdf"],
        "license": "cc-by-4.0",
    },
    "checkpoints": {
        "upload_type": "dataset",
        "title": (
            "Hierarchical Latent Workspace Model: checkpoints and full run bundles "
            "(17 experiment versions)"
        ),
        "description": (
            "Model checkpoints and complete Kaggle run bundles for the ten-experiment "
            "HLWM program, v4 through v10.0. Approximately 9.6 GB. The stripped "
            "evidence tree (gate verdicts, per-step metrics, training logs) is committed "
            "to the code repository; this record holds the weights and the bundled "
            "training corpora."
        ),
        "files": [],  # set with --file, these are too large to enumerate here
        "license": "cc-by-4.0",
    },
}

KEYWORDS = [
    "machine learning",
    "language models",
    "latent reasoning",
    "interpretability",
    "negative results",
    "preregistration",
]


def api(base: str, token: str, method: str, path: str, payload=None, raw=None, ctype=None):
    url = path if path.startswith("http") else f"{base}{path}"
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}access_token={token}"
    data = raw if raw is not None else (json.dumps(payload).encode() if payload else None)
    headers = {"Content-Type": ctype or "application/json"} if data else {}
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        sys.exit(f"zenodo {exc.code}: {exc.read().decode()[:600]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--file", action="append", default=[], help="extra file to attach")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--apply", action="store_true", help="actually create the deposit")
    args = ap.parse_args()

    spec = TARGETS[args.target]
    files = [Path(p) for p in (spec["files"] + args.file)]

    missing = [p for p in files if not p.exists()]
    if missing:
        sys.exit("missing files: " + ", ".join(str(p) for p in missing))
    if not files:
        sys.exit(f"no files for target {args.target}; pass --file")

    metadata = {
        "title": spec["title"],
        "upload_type": spec["upload_type"],
        "description": spec["description"],
        "creators": AUTHORS,
        "keywords": KEYWORDS,
        "license": spec["license"],
        "access_right": "open",
        "related_identifiers": [
            {"identifier": REPO_URL, "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": CONCEPT_DOI, "relation": "isPartOf", "scheme": "doi"},
        ],
    }
    if "publication_type" in spec:
        metadata["publication_type"] = spec["publication_type"]

    print(f"target   : {args.target}")
    print(f"type     : {spec['upload_type']}")
    print(f"title    : {spec['title']}")
    for p in files:
        print(f"  file   : {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    if not args.apply:
        print("\ndry run — re-run with --apply to create the draft deposit.")
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN (scopes: deposit:write, deposit:actions)")
    base = (
        "https://sandbox.zenodo.org/api"
        if args.sandbox
        else "https://zenodo.org/api"
    )

    dep = api(base, token, "POST", "/deposit/depositions", {"metadata": metadata})
    bucket = dep["links"]["bucket"]
    for p in files:
        print(f">>> uploading {p.name}")
        api(base, token, "PUT", f"{bucket}/{p.name}", raw=p.read_bytes(),
            ctype="application/octet-stream")

    print(f"\ndraft   : {dep['links']['html']}")
    print(f"DOI     : {dep['metadata'].get('prereserve_doi', {}).get('doi', '(reserved on publish)')}")
    print("\nReview the draft in the Zenodo UI, then publish there.")


if __name__ == "__main__":
    main()
