# HLWM V5.6.2 Preregistered Plan — the reduction study

Date: 2026-08-31
Governing results: `reports/hlwm-v5.6.1-session-b-results-2026-08-31.md`
Inherits everything not listed below from `reports/hlwm-v5.6-plan-2026-08-30.md`
(data byte-identical to v5.6; token budgets, steps 4,224, LR, balance weight,
label smoothing, audit design, gate thresholds all frozen).

## Why a reduction study

Session B triggered the preregistered ladder: joint-phase outcomes remained a
seed lottery **after** the two targeted fixes (seed 17 collapsed to one expert
under two different initializations; seed 29's diversity decayed 0.418→0.113
toward the gate line; causal liveness 0-for-8 across four studies). The
paper's failure-mode table prescribes "simplify the workspace" for exactly
this condition, and the routing-collapse row prescribes "cut the expert
count" for homogeneous experts. v5.6.2 executes both prescriptions at once,
plus the two mechanism-level repairs Session B identified. This is a step
toward production in the only honest sense available: shrinking the
architecture to the largest configuration that can pass its full gate
reliably, which is the precondition for the plain-LoRA control, which is the
precondition for any capability claim.

## The four changes

| Change | Mechanism | Traces to |
|---|---|---|
| `--num-experts 2` (from 6) | With 2 experts the balance loss constrains the router's entire decision space; collapse is a single measurable bit, entropy normalization is over 2 routes, and per-expert traffic (and hence gradient) is 3× denser at fixed step count. The strongest observed routing (seed 29 A) concentrated on ~3 experts of 6; two is the smallest count that still tests "which expertise" at all. | Seed 17's 2-for-2 collapse; ladder rung (ii); failure-mode rows "cut the expert count" / "simplify the workspace" |
| Validity-gated family balance in the head phase: resample per family until **≥16 valid positives per family** (grader-verified), cap 512 emission attempts, fail loudly if a family cannot reach floor | Session B showed 85–87 of 128 emissions invalid, so round-robin over *requests* left valid positives at 41–43 vs ~342 negatives — the same 1:8 imbalance as before the "fix". Balance must bind after validity filtering. | Seed 29 commit accuracy frozen at 0.625; policy-head-training.json invalid_emissions |
| Generation-time canary every `eval_every` (512 steps): generate 8 stratified anchors mid-training; log prompt-leak flags, format-marker rate, per-canary route entropy. Report-only this run (no abort), but recorded in metrics.jsonl | Both Session A collapse and Session B routing decay were invisible to every training-time signal; the paper's failure-mode table already schedules this canary. Report-only first so the canary itself is validated before it gates anything. | Study 5/6 finding "collapse visible only at generation time" |
| Commitment-threshold non-degeneracy check: declared diagnostic, threshold ∈ [0.02, 0.98] after calibration | Fourth recurrence of the degenerate threshold, now on both seeds. Diagnostic (not a new hard gate) to avoid post-hoc gate inflation; the underlying fix (de-correlating policy labels) stays scheduled as its own revision per Known Limitations item 3. | Seed 17 B threshold 0.0 |

Kept from v5.6.1: `--expert-init-scale 0.01` (it demonstrably rescued the
degeneration mode) and `--policy-records 128` (subsumed by the validity floor).

## Session and endpoints

Session C: seeds 17 + 29 (same seeds, third measurement — the reduction claim
is only meaningful against the same lottery). Fresh training, no resumables
attached; artifact names `hlwm-v5.6.2-*`. Est. 6–7 h; telemetry must be
complete this time — **do not resume past the training window**, since Session
B's metrics.jsonl shipped empty of telemetry and left diagnostic (a)
unmeasured.

- **Primary endpoint (unchanged in form):** both seeds pass the complete
  18-gate capability gate, with routing gates evaluated at num_experts 2
  (entropy normalized over 2 routes; second-route load gate unchanged at
  ≥0.10; causal liveness unchanged at ≥0.01).
- **Declared diagnostic sub-questions:**
  (a) does `expert_diversity` telemetry engage (>0.01 sustained) now that
  init is nonzero and traffic per expert is 3× denser — this re-asks Session
  B's unmeasured question;
  (b) do both seeds hold second-route load ≥0.10 — i.e. does the reduction
  end the lottery;
  (c) does the causal-liveness intervention clear 0.01 for the first time;
  (d) does validity-gated balancing lift seed 29's commit accuracy ≥0.70 and
  ordering-family accuracy above its 0.625 floor;
  (e) canary validation: would the canary have fired before step 2,048 on any
  collapsing run (retrospective check against whichever seed, if any, fails).
- **Fallback ladder for v5.6.2:**
  1. If either seed still collapses at num_experts 2 → the expert graph is
     removed from the small-scale claim entirely: v5.7 runs workspace-only
     (lanes + commitment, no routed experts), and the paper's routing section
     is re-scoped to proposal-plus-negative-evidence.
  2. If routing passes on both seeds but commit accuracy still fails with a
     satisfied validity floor → the ordering grader and its anchor family are
     audited before any further head-phase changes (data before mechanism).
  3. If the full gate passes on both seeds → the plain-LoRA control
     (36/36 tests, postponed 2026-08-31) runs **immediately** as Session D on
     the same seeds, before any benchmark authorization. This ordering is
     binding.

## Paper state

`paper/hlwm-paper.tex` updated 2026-08-31 with Study 6 (v5.6.1 Session B):
abstract, provenance row, §9.7 results table and findings, failure-mode
annotations (reduction rule now invoked), Known Limitations items 3 and 6
updated, conclusion revised. Compiled clean; installed at repo root as
`hierarchical-latent-workspace-model.pdf` (arXiv-ready source is the single
self-contained .tex).
