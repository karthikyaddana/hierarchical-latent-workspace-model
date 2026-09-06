"""Local pre-launch rehearsal for the HLWM v10.0 audit (session I-5 lesson).

The blocking on-Kaggle preflight catches CUDA-class defects, but two launch
failures were catchable locally and were not caught: the io_tests legacy
row crash lived in the shipped test.jsonl (session I-4) and the fp16 naked
teacher-force crash lived in an entry point the dtype sweep missed
(session I-5). This script is the missing tier: it exercises the audit code
path against the REAL shipped data files with the REAL Qwen tokenizer, on a
tiny real-vocab model forced into the audit's actual mixed regime (fp16
frozen base + fp32 trainables, no autocast). It certifies crash-safety, not
science: every number it produces is meaningless except "did it run".

Usage: .venv/bin/python scripts/hlwm_v100_local_rehearsal.py [--full]

--full runs the audit at PRODUCTION shape (all kept test rows, core 160,
expert arms 128/family, probe 512, 8-candidate pool, 48 new tokens) on the
tiny real-vocab model. That exercises every n-dependent branch (conformal
quantile index, e-process over ~1,150 records, DeLong at scale, probe
training at 512 rows) that the fast stratified pass cannot reach. Expect
roughly 30-60 minutes on CPU.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "artifacts" / "kaggle" / "hlwm-v10.0" / "bundle" / "hlwm_kaggle"
sys.path.insert(0, str(BUNDLE))

import evaluate_v10 as ev  # noqa: E402
from data import Reasoning9000Dataset  # noqa: E402
from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def tiny_real_vocab_model(tokenizer) -> HLWMForConditionalGeneration:
    """Tiny stack over the REAL vocabulary, forced into the audit regime."""

    cue = tokenizer.encode("\n### Response\n", add_special_tokens=False)
    config = HLWMConfig.tiny(
        vocab_size=max(len(tokenizer), 151_936),
        latent_thoughts=6,
        kv_prefix_slots=4,
        kv_prefix_rank=8,
        response_cue_ids=tuple(cue),
        lora_rank=4,
        lora_tail_layers=2,
        mlp_expert_count=2,
        mlp_expert_rank=4,
        max_position_embeddings=1024,
        num_lanes=1,
        pad_token_id=tokenizer.pad_token_id or 0,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(11)
    model = HLWMForConditionalGeneration(config)
    # Mirror main()'s freeze so the regime matches the shipped run exactly.
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    # The audit regime (load_hlwm): fp16 storage for the frozen substrate,
    # fp32 trainables, and NO autocast anywhere in evaluate_v10.
    for parameter in model.parameters():
        parameter.data = (
            parameter.data.float()
            if parameter.requires_grad
            else parameter.data.to(torch.float16)
        )
    return model.eval()


def main() -> int:
    full = "--full" in sys.argv[1:]
    started = time.time()
    tokenizer = real_tokenizer()
    print("tokenizer ok | bos", tokenizer.bos_token_id, "| eos", tokenizer.eos_token_id)

    failures = []
    for split in ("test", "validation"):
        rows = Reasoning9000Dataset(
            BUNDLE / "data" / "master" / f"{split}.jsonl", num_lanes=1
        ).rows
        kept, excluded = ev.v10_audit_rows(rows)
        families = Counter(ev.family_of_row(row) for row in kept)
        print(
            f"{split}: {len(rows)} rows -> {len(kept)} kept, {excluded} legacy excluded |",
            dict(families),
        )
        if sorted(families) != sorted(ev.FAMILY_ORDER):
            failures.append(f"{split}: kept families {sorted(families)} != FAMILY_ORDER")
        if split == "test":
            test_kept = kept

    model = tiny_real_vocab_model(tokenizer)

    # ---- 1. Row-prep landmine sweep over EVERY audit row (the io_tests
    # class: any row the real audit will touch must survive preparation).
    prepared = ev.prepare_rows(
        model, tokenizer, ev.sorted_by_anchor(test_kept), context_tokens=320, device=None
    )
    print(f"prepare_rows survived all {len(prepared)} audit rows")

    # ---- 2. The REAL masked core (production masked_core_rows=160) must be
    # derangement-feasible: a singleton family in the core crashes
    # arm_shuffled ~40 minutes into the on-Kaggle audit.
    real_core = ev.masked_core_selection(ev.sorted_by_anchor(test_kept), 160)
    core_families = Counter(ev.family_of_row(row) for row in real_core)
    print("real 160-row masked core composition:", dict(core_families))
    try:
        ev._within_family_derangement(
            [ev.family_of_row(row) for row in real_core], seed=18
        )
        print("real masked core is within-family-shuffle feasible")
    except ValueError as error:
        failures.append(f"real masked core cannot shuffle: {error}")

    # ---- 3. Audit through the exact notebook cell-9 sequence incl. gate
    # dict. Fast mode: stratified subset (core needs >=2 masked rows per
    # represented family for the shuffle arm). Full mode: every kept row at
    # production parameters — the shipped notebook call, verbatim shape.
    if full:
        subset = list(test_kept)
        audit_kwargs = dict(masked_core_rows=160, expert_rows_per_family=128,
                            probe_latent_rows=512, max_new_tokens=48)
    else:
        ordered = ev.sorted_by_anchor(test_kept)
        subset, masked_seen, unmasked_seen = [], Counter(), Counter()
        for row in ordered:
            family = ev.family_of_row(row)
            if bool(row.get("masked")) and masked_seen[family] < 2:
                subset.append(row)
                masked_seen[family] += 1
            elif not bool(row.get("masked")) and unmasked_seen[family] < 1:
                subset.append(row)
                unmasked_seen[family] += 1
        audit_kwargs = dict(masked_core_rows=sum(masked_seen.values()),
                            expert_rows_per_family=2, probe_latent_rows=8,
                            max_new_tokens=8)
    audit = ev.v10_audit(
        model,
        subset,
        tokenizer,
        model.config,
        seed=17,
        context_tokens=320,
        target_coverage=0.40,
        **audit_kwargs,
    )
    # The gold-answer CE instrument runs inside v10_audit; it must have
    # scored real rows (a silently-empty instrument is the session I-5
    # failure class) and its unmasked exposure contrast must be exactly 0.
    gold = audit.get("gold_forced_ce") or {}
    print(
        "gold CE: scored", gold.get("n_scored"), "of", gold.get("n_selected"),
        "| overall", {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in (gold.get("overall") or {}).items()
        },
    )
    if not gold.get("n_scored"):
        failures.append("gold_forced_ce scored no rows on the real corpus")
    if (gold.get("unmasked") or {}).get("n") and gold["unmasked"]["exposure_gap"] != 0.0:
        failures.append(
            "unmasked exposure_gap is %r, must be exactly 0"
            % gold["unmasked"]["exposure_gap"]
        )

    gates = ev.v10_gate_dict(
        audit,
        training_complete=False,
        zero_skipped_updates=True,
        warm_gate_passed=False,
        infrastructure_ok=True,
        wo_l1=None,
    )
    json.dumps(audit, default=str)
    json.dumps(gates, default=str)
    rung = ev.binding_rung({17: gates, 29: gates})
    print(
        f"mini-audit ok on {len(subset)} rows | n_rows {audit['n_rows']} |"
        f" legacy_excluded {audit['legacy_rows_excluded']} |"
        f" records {len(audit['records'])} | pool {audit['records'][0]['pool_size']} |"
        f" gate dict + binding rung serialize (rung {rung})"
    )

    elapsed = time.time() - started
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        print(f"REHEARSAL FAILED in {elapsed:.0f}s")
        return 1
    print(f"REHEARSAL PASSED in {elapsed:.0f}s — audit path crash-safe on real data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
