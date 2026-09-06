#!/usr/bin/env python3
"""Build the HLWM Version 10.0 candidate bundle: dense supervision.

Version 10.0 is a NEW program (plan: reports/hlwm-v10-plan-2026-09-03.md,
amendments section 10 BINDING).  The v5-v9 signature — components training
never forces to be useful collapse — is answered with ~100+ dense CE/L1
supervision terms per row.  This builder owns the DATA side:

  1. regenerates the behavior anchors through ``behavior_anchor_rows_v9``
     (SAME key arithmetic, imported, never forked) at the scaled counts
     train 4096 / validation 768 / test 1152, and attaches to every anchor a
     ``v10`` payload derived from ``scripts/hlwm_v10_traces.py``: the gold
     compact trace, the teacher-step exclusion, the single-step-corrupted
     process negative, and the family routing label;
  2. asserts, per emitted row, the four data-contract guarantees under the
     REAL pinned Qwen tokenizer (trace >= 8 tokens; zero pad/eos ids inside
     every CoLaR compression window; no withheld literal in any masked
     prompt; every trace step <= 30 tokens);
  3. merges the anchors with the non-anchor rows of the Version 8.0 master;
  4. runs the offline bundle test suite and refuses to build on failure;
  5. writes the manifest and a deterministic zip (fixed timestamps).

Teacher-exclusion encoding (documented contract, mirrored in
``data.V10AnchorCollator``): the payload records the padded step list
(``steps``) and ``teacher_excluded_steps`` = 1 + pad_repeats; the excluded
span is always the trailing segments, whose token extent any consumer
re-derives exactly via the shared compositional ``encode_trace_segments``.
``teacher_excluded_tokens`` records that extent under the real pinned
tokenizer for the audit trail.  Short traces are textually padded by
repeating the final step's equation (compression_windows' documented "pad
the TRACE, never the windows" rule); measured floor under the pinned
tokenizer is 8 tokens (the abstention step), so pad_repeats is 0 in
practice and the machinery is defensive.

The dual-T4 notebook is OWNED BY A LATER SESSION: ``notebook_document`` is
a deliberate stub and this builder never writes an .ipynb.

Trace derivation helpers (operands/operators/values from the row key) are
imported from ``scripts/test_hlwm_v10_traces.py``: they are the verbatim,
consistency-tested copies of the v9 generator arithmetic — importing them
keeps a single source instead of a third fork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUNDLE_ROOT = ROOT / "artifacts" / "kaggle" / "hlwm-v10.0"
PROJECT = BUNDLE_ROOT / "bundle" / "hlwm_kaggle"
V8_MASTER = ROOT / "artifacts" / "kaggle" / "hlwm-v8.0" / "bundle" / "hlwm_kaggle" / "data" / "master"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import hlwm_v10_traces as traces
from scripts.build_hlwm_v90_bundle import (
    behavior_anchor_rows_v9,
    sha256,
    stable_key,
    write_jsonl,
)
from scripts.test_hlwm_v10_traces import build_trace_for_row

from data import encode_trace_segments  # noqa: E402  (bundle module, path above)

PACKAGE_NAME = "hlwm-dense-supervision-v10-0-candidate"
PACKAGE_VERSION = "10.0"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

ANCHOR_COUNTS = {"train": 4096, "validation": 768, "test": 1152}
SMOKE_ANCHOR_COUNTS = {"train": 32, "validation": 8, "test": 12}
SMOKE_NON_ANCHOR_CAP = 64

LATENT_THOUGHTS = 6
TRACE_MIN_TOKENS = 8
TRACE_STEP_TOKEN_BUDGET = 30
TRACE_TOKEN_BUDGET = 96

TEST_FILES = (
    "test_modeling_hlwm.py",
    "test_selective_stats.py",
    "test_evaluate_v10.py",
    "test_data_v10.py",
)

V10_CHANGES = [
    "Dense-supervision channel (plan section 2): K=6 autoregressive continuous "
    "thoughts, per-layer gated KV-prefix slots, CODI all-layer smooth-L1 "
    "distillation, CoLaR compressed-gold per-window CE, SIM-CoT per-latent "
    "decode CE, continued reconstruction; every anchor row now ships a gold "
    "compact trace with the answer-producing final step excluded from teacher CE.",
    "A1 training-side information asymmetry: the student channel reads the "
    "masked prompt on masked rows (v9 machinery retained); latents are produced "
    "from the full premise.",
    "A2 producer supervision: per-window compression targets "
    "(trace_window_targets) support the producer MSE and per-latent decode CE; "
    "windows computed over VALID trace length with zero pad/eos ids inside any "
    "window, asserted at build time under the real tokenizer.",
    "Process negatives: one single-step-corrupted trace per non-abstention "
    "anchor (digit-count-preserving, cascaded re-derivation, key-derived step "
    "pick) with the corrupted step index — free Math-Shepherd labels for the "
    "verifier head.  Abstention rows use the documented sentinel (empty "
    "corrupt trace, index -1).",
    "Deterministic label routing (plan section 3): family labels "
    "numeric/unit/ordering/abstention -> family_index 0..3, expert map "
    "{numeric, unit}->0, {ordering, abstention}->1; balance comes from the "
    "stratified dataloader, never a balance loss.",
    "Anchor counts scaled to train 4096 / validation 768 / test 1152 for the "
    "25k-row spec (section 5); difficulty knobs, masked variants and premise "
    "literals unchanged from the v9 generator (same key arithmetic, imported).",
]

TRAINING_DEFAULTS = {
    "steps": "smoke 101 + warm 600 + main wall-clock-bounded (1,400 planned, 1,200 floor) + w/o-L1 branch",
    "latent_thoughts": LATENT_THOUGHTS,
    "kv_prefix_slots": 16,
    "kv_prefix_rank": 64,
    "prefix_attn_gate_init": 0.08,
    "mlp_expert_count": 2,
    "mlp_expert_rank": 16,
    "router_families": 4,
    "router_probe_weight": 0.1,
    "gamma": "smoke-calibrated in {5, 10, 20}",
    "colar_weight": 0.5,
    "recon_weight": 0.3,
    "producer_weight": 0.5,
    "step_decode_weight": 0.5,
    "trace_token_budget": TRACE_TOKEN_BUDGET,
    "lora": "r64 all layers (r128 only on smoke-measured T4 headroom)",
    "optimizer": "AdamW wd 0.1 (prefix_attn_gate exempt), lr 8e-4 cosine, 3% warmup, bf16",
}


# --------------------------------------------------------------------------
# v10 payload


def _pad_steps_to_floor(
    steps: List[str], tokenizer: Any, floor: int
) -> tuple[List[str], int]:
    """Repeat the final step's equation until the compositional token count
    reaches ``floor`` (the documented pad method: the repeated text is the
    answer-producing step, so the whole padded suffix stays inside the
    teacher-excluded span)."""

    padded = list(steps)
    ids, _ = encode_trace_segments(tokenizer, padded, 1)
    repeats = 0
    while len(ids) < floor:
        padded.append(padded[-1])
        repeats += 1
        ids, _ = encode_trace_segments(tokenizer, padded, 1)
    return padded, repeats


def v10_payload(
    row: Dict[str, Any],
    key: int,
    tokenizer: Optional[Any],
    latent_thoughts: int = LATENT_THOUGHTS,
) -> Dict[str, Any]:
    """Attachable ``v10`` dict for one v9 anchor row.

    With ``tokenizer=None`` (offline unit tests) the token-count fields are
    None, pad_repeats is 0 and no token assertions run; the real build always
    passes the pinned tokenizer, which pads and asserts.
    """

    trace_dict = build_trace_for_row(row, key)
    kind = str(trace_dict["kind"])
    gold_steps = [str(step) for step in trace_dict["steps"]]

    pad_repeats = 0
    steps = list(gold_steps)
    excluded_tokens: Optional[int] = None
    trace_tokens: Optional[int] = None
    if tokenizer is not None:
        floor = max(TRACE_MIN_TOKENS, latent_thoughts)
        steps, pad_repeats = _pad_steps_to_floor(gold_steps, tokenizer, floor)
        ids, excluded_tokens = encode_trace_segments(
            tokenizer, steps, 1 + pad_repeats
        )
        trace_tokens = len(ids)

    if kind == "abstention":
        corrupt_steps: List[str] = []
        corrupt_trace = ""
        corrupt_index = -1
    else:
        corrupt = traces.corrupt_one_step(trace_dict, key)
        corrupt_steps = [str(step) for step in corrupt["steps"]]
        if tokenizer is not None:
            corrupt_steps, _ = _pad_steps_to_floor(
                corrupt_steps, tokenizer, max(TRACE_MIN_TOKENS, latent_thoughts)
            )
        corrupt_trace = " ".join(corrupt_steps)
        corrupt_index = int(corrupt["corrupted_step_index"])

    return {
        "family": kind,
        "trace": " ".join(steps),
        "steps": steps,
        "gold_steps": gold_steps,
        "pad_repeats": pad_repeats,
        "teacher_excluded_steps": 1 + pad_repeats,
        "teacher_excluded_tokens": excluded_tokens,
        "trace_tokens": trace_tokens,
        "corrupt_trace": corrupt_trace,
        "corrupt_steps": corrupt_steps,
        "corrupt_step_index": corrupt_index,
    }


def _verify_row(
    row: Dict[str, Any],
    payload: Dict[str, Any],
    tokenizer: Any,
    latent_thoughts: int,
) -> None:
    """The four per-row data-contract guarantees plus corruption sanity."""

    episode_id = str(row.get("episode_id"))
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or eos_id)
    forbidden = {eos_id, pad_id}
    steps = [str(step) for step in payload["steps"]]

    ids, excluded = encode_trace_segments(
        tokenizer, steps, payload["teacher_excluded_steps"]
    )
    # (a) the trace always has enough tokens for K windows.
    if len(ids) < max(TRACE_MIN_TOKENS, latent_thoughts):
        raise SystemExit(
            "guarantee (a) failed: %s trace has %d tokens" % (episode_id, len(ids))
        )
    if len(ids) > TRACE_TOKEN_BUDGET:
        raise SystemExit(
            "trace budget exceeded: %s needs %d > %d tokens"
            % (episode_id, len(ids), TRACE_TOKEN_BUDGET)
        )
    if payload["trace_tokens"] != len(ids) or payload["teacher_excluded_tokens"] != excluded:
        raise SystemExit(
            "recorded token accounting drifted on %s: %r vs (%d, %d)"
            % (episode_id, (payload["trace_tokens"], payload["teacher_excluded_tokens"]), len(ids), excluded)
        )
    # (b) zero pad/eos ids inside every compression window (raises on a
    # planted forbidden id, and on len < k by construction of (a)).
    traces.compression_windows(ids, latent_thoughts, forbidden)
    # (d) every trace step fits the 30-token step budget.
    for step in steps:
        step_tokens = len(tokenizer.encode(step, add_special_tokens=False))
        if step_tokens > TRACE_STEP_TOKEN_BUDGET:
            raise SystemExit(
                "guarantee (d) failed: %s step %r is %d tokens"
                % (episode_id, step, step_tokens)
            )
    # Parse/structure validation (generous whole-trace budget: multi-step
    # numeric traces legitimately exceed 30 whole-trace tokens).
    verdict = traces.validate_trace(
        {"steps": steps, "trace": payload["trace"]},
        tokenizer,
        max_tokens=TRACE_TOKEN_BUDGET,
        max_steps=3 + int(payload["pad_repeats"]),
    )
    if not verdict["ok"]:
        raise SystemExit("trace validation failed on %s: %r" % (episode_id, verdict))

    # (c) the masked prompt never contains any withheld literal (v9 scan).
    evaluation = row.get("evaluation") or {}
    if evaluation.get("masked"):
        masked_request = row["input"]["user_request_masked"]
        for literal in evaluation.get("withheld_literals") or []:
            if str(literal) in masked_request:
                raise SystemExit(
                    "build-time leak: %r in masked request of %s"
                    % (literal, episode_id)
                )

    # Corruption sanity: sentinel for abstention, one differing keyed step
    # (with an untouched prefix) plus the token-side guarantees otherwise.
    corrupt_steps = [str(step) for step in payload["corrupt_steps"]]
    if payload["family"] == "abstention":
        if corrupt_steps or payload["corrupt_step_index"] != -1:
            raise SystemExit("abstention corruption sentinel broken on %s" % episode_id)
    else:
        corrupt_index = int(payload["corrupt_step_index"])
        gold_steps = [str(step) for step in payload["gold_steps"]]
        if not 0 <= corrupt_index < len(gold_steps):
            raise SystemExit("corrupt_step_index out of range on %s" % episode_id)
        if corrupt_steps[corrupt_index] == gold_steps[corrupt_index]:
            raise SystemExit("corruption did not change step %d of %s" % (corrupt_index, episode_id))
        if corrupt_steps[:corrupt_index] != gold_steps[:corrupt_index]:
            raise SystemExit("corruption touched the pre-index prefix of %s" % episode_id)
        bad_ids, _ = encode_trace_segments(tokenizer, corrupt_steps, 1)
        if len(bad_ids) < max(TRACE_MIN_TOKENS, latent_thoughts) or len(bad_ids) > TRACE_TOKEN_BUDGET:
            raise SystemExit("corrupt trace token count out of range on %s" % episode_id)
        if set(bad_ids) & forbidden:
            raise SystemExit("pad/eos id inside the corrupt trace of %s" % episode_id)


def behavior_anchor_rows_v10(
    split: str,
    count: int,
    seed: int,
    tokenizer: Optional[Any] = None,
    latent_thoughts: int = LATENT_THOUGHTS,
) -> List[Dict[str, Any]]:
    """v9 anchor rows (same key arithmetic, called not forked) + v10 payloads.

    When a tokenizer is supplied every emitted row is asserted against the
    four data-contract guarantees; ``tokenizer=None`` is the offline test
    mode (token-count fields None, no padding needed in practice).
    """

    rows = behavior_anchor_rows_v9(split, count, seed)
    for index, row in enumerate(rows):
        key = stable_key(seed, split, index)
        payload = v10_payload(row, key, tokenizer, latent_thoughts)
        if tokenizer is not None:
            _verify_row(row, payload, tokenizer, latent_thoughts)
        row["v10"] = payload
    return rows


# --------------------------------------------------------------------------
# master rebuild


def rebuild_master(
    seed: int,
    tokenizer: Any,
    anchor_counts: Dict[str, int],
    non_anchor_cap: Optional[int] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source = V8_MASTER / ("%s.jsonl" % split)
        non_anchor: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("domain")) != "behavior-anchor":
                    non_anchor.append(row)
                if non_anchor_cap is not None and len(non_anchor) >= non_anchor_cap:
                    break
        anchors = behavior_anchor_rows_v10(
            split, anchor_counts[split], seed, tokenizer=tokenizer
        )
        rows = anchors + non_anchor
        write_jsonl(PROJECT / "data" / "master" / ("%s.jsonl" % split), rows)
        counts[split] = len(rows)

        maskable = [row for row in anchors if row["evaluation"]["withheld_literals"]]
        flagged = [row for row in maskable if row["evaluation"]["masked"]]
        kinds = Counter(row["v10"]["family"] for row in anchors)
        pad_repeats = Counter(row["v10"]["pad_repeats"] for row in anchors)
        print(
            json.dumps(
                {
                    "split": split,
                    "rows": len(rows),
                    "anchors": len(anchors),
                    "non_anchor": len(non_anchor),
                    "maskable": len(maskable),
                    "masked": len(flagged),
                    "masked_fraction_of_anchors": round(len(flagged) / max(1, len(anchors)), 4),
                    "kinds": dict(kinds),
                    "pad_repeats": dict(pad_repeats),
                }
            )
        )
    return counts


# --------------------------------------------------------------------------
# Notebook: owned by a later session.


def notebook_document(
    expected_counts: Dict[str, int], code_digest: str = ""
) -> Dict[str, Any]:
    """Dual-T4 executable-preregistration notebook (see hlwm_v100_notebook)."""

    from hlwm_v100_notebook import notebook_document as build_document

    return build_document(expected_counts, code_digest)


def manifest_code_digest() -> str:
    """Content address of the shipped code, as the notebook pins it.

    A sha256 over the manifest's ``code_sha256`` map: deterministic (no
    timestamps, so the zip stays reproducible) and different for any change
    to any shipped file, which is exactly what cell 3 needs to tell this
    bundle apart from an older one attached for resume.
    """

    manifest = json.loads((PROJECT / "manifest.json").read_text())
    return hashlib.sha256(
        json.dumps(manifest.get("code_sha256") or {}, sort_keys=True).encode()
    ).hexdigest()


def write_notebook(counts: Dict[str, int], code_digest: str) -> str:
    document = notebook_document(counts, code_digest)
    sources = ["".join(cell["source"]) for cell in document["cells"]]
    # Build-time invariant: an unpinned notebook can silently run a stale
    # code tree attached for resume, which is a whole wasted session.
    if not any(code_digest in source for source in sources):
        raise SystemExit(
            "refusing to write a notebook that does not pin the code digest"
        )
    if len(code_digest) != 64:
        raise SystemExit(f"code digest is not a sha256: {code_digest!r}")
    path = BUNDLE_ROOT / "embel-hlwm-v10.0-kaggle-2xt4.ipynb"
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def run_bundle_tests() -> None:
    """Run the full bundle suite; refuse to build anything on failure."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *TEST_FILES, "-q"], cwd=PROJECT
    )
    if result.returncode != 0:
        raise SystemExit("v10.0 bundle tests failed; refusing to build a bundle")


def run_notebook_tests() -> None:
    """The notebook's launch-blocking invariants (cells 3, 9, 11)."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "scripts/test_hlwm_v100_notebook.py", "-q"],
        cwd=Path(__file__).resolve().parent.parent,
    )
    if result.returncode != 0:
        raise SystemExit("notebook invariant tests failed; refusing to build")


def run_local_rehearsal() -> None:
    """Blocking audit rehearsal on the REAL shipped data (session I-5 tier).

    Unit tests run on tiny fixtures; the two launch defects that unit tests
    could not see (legacy-family rows in the shipped test split, the fp16
    naked teacher-force entry) both live at the real-data / real-regime
    boundary this rehearsal exercises.
    """

    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "hlwm_v100_local_rehearsal.py")]
    )
    if result.returncode != 0:
        raise SystemExit("local audit rehearsal failed; refusing to build a bundle")


def manifest_code_files() -> List[str]:
    names = sorted(path.name for path in PROJECT.glob("*.py"))
    return names + ["README.md"]


def write_manifest(counts: Dict[str, int], anchor_counts: Dict[str, int], seed: int, smoke: bool) -> None:
    manifest = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "behavior_anchor_counts": anchor_counts,
        "build_mode": "smoke" if smoke else "full",
        "code_sha256": {name: sha256(PROJECT / name) for name in manifest_code_files()},
        "counts": counts,
        "name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "purpose": (
            "dense supervision: every anchor row carries a gold compact trace "
            "(answer step excluded from teacher CE), CoLaR compression windows, "
            "a single-step-corrupted process negative, and a deterministic "
            "family routing label; ~100+ supervision terms per row so no "
            "component's usefulness is left to emergence; sized for a Kaggle "
            "dual-T4 two-seed session"
        ),
        "review_status": (
            "Reasoning9000 rows remain unreviewed and policy-masked. The masked-"
            "row setting is a constructed mechanism demonstration, not a "
            "capability claim. Not authorized as production or factual-quality "
            "evidence until the preregistered gate passes."
        ),
        "seed": seed,
        "supersedes": "hlwm-necessity-v9-0-candidate",
        "target_hardware": "Kaggle T4 x2, one independent seed per GPU",
        "training_defaults": TRAINING_DEFAULTS,
        "v10_0_changes": V10_CHANGES,
    }
    (PROJECT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_zip(archive_path: Path) -> str:
    """Deterministic bundle zip: sorted members, fixed timestamps, fixed mode."""

    if archive_path.exists():
        archive_path.unlink()
    members: List[Path] = []
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        members.append(path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(
                str(Path("hlwm_kaggle") / path.relative_to(PROJECT)),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return sha256(archive_path)


def verify_zip_determinism(archive_path: Path) -> str:
    """Build the zip twice around an mtime touch; byte-identical or die.

    Touching a member between the builds is what catches an mtime leak (the
    v6.0-era defect class the fixed ZipInfo timestamps exist to kill)."""

    first = build_zip(archive_path)
    (PROJECT / "data.py").touch()
    second = build_zip(archive_path)
    if first != second:
        raise SystemExit(
            "zip is not deterministic: %s != %s (mtime leaked into the archive)"
            % (first, second)
        )
    print("zip determinism verified:", first)
    return first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny anchor counts + capped non-anchor merge; writes a clearly "
        "named -smoke zip; the real build must be rerun before any launch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PROJECT.exists():
        raise SystemExit(f"v10.0 bundle sources not found at {PROJECT}")
    missing = [name for name in TEST_FILES + ("data.py", "train_kaggle.py", "modeling_hlwm.py", "README.md") if not (PROJECT / name).exists()]
    if missing:
        raise SystemExit(f"bundle is missing files: {missing}")
    if not V8_MASTER.exists():
        raise SystemExit(f"v8.0 master data not found at {V8_MASTER}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False
    )

    anchor_counts = SMOKE_ANCHOR_COUNTS if args.smoke else ANCHOR_COUNTS
    non_anchor_cap = SMOKE_NON_ANCHOR_CAP if args.smoke else None
    counts = rebuild_master(args.seed, tokenizer, anchor_counts, non_anchor_cap)

    if not args.skip_tests:
        run_bundle_tests()
        run_notebook_tests()
        if not args.smoke:
            run_local_rehearsal()

    write_manifest(counts, anchor_counts, args.seed, args.smoke)

    archive_name = (
        "hlwm-v10.0-candidate-bundle-smoke.zip"
        if args.smoke
        else "hlwm-v10.0-candidate-bundle.zip"
    )
    archive_path = BUNDLE_ROOT / archive_name
    bundle_sha = verify_zip_determinism(archive_path)

    code_digest = manifest_code_digest()
    notebook_sha = write_notebook(counts, code_digest)
    print("notebook written:", BUNDLE_ROOT / "embel-hlwm-v10.0-kaggle-2xt4.ipynb")
    print("pinned code digest:", code_digest)

    report = {
        "build_mode": "smoke" if args.smoke else "full",
        "package_version": PACKAGE_VERSION,
        "bundle_sha256": bundle_sha,
        "code_digest": code_digest,
        "counts": counts,
        "anchor_counts": anchor_counts,
        "code_sha256": {name: sha256(PROJECT / name) for name in manifest_code_files()},
        "notebook_sha256": notebook_sha,
    }
    report_name = "build-report-smoke.json" if args.smoke else "build-report.json"
    (BUNDLE_ROOT / report_name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
