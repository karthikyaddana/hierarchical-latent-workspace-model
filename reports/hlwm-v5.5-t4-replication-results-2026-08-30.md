# HLWM V5.5 Dual-T4 Replication Results — Session A verdict

Date: 2026-08-30
Plan: `reports/hlwm-v5.5-t4-replication-plan-2026-08-30.md`
Artifacts: `artifacts/kaggle/hlwm-v5.5/session-a/` (seed-17/, seed-29/, verdict JSON)
Session: 206.9 minutes wall clock, both seeds concurrent, ~26.5 h quota remaining.

## Verdict

**The preregistered primary endpoint failed: neither seed passed the full
gate.** Both seeds failed the *same three sub-gates* — the routing cluster —
and passed all fifteen others. Two independent seeds agreeing exactly on
which gates fail is a replicated structural finding about the architecture,
not noise. The v5.4 calibration failure is fixed and replicated; the v5.4
routing failure is improved but not fixed, and the causal-liveness test says
the remaining routing is close to decorative.

Per the preregistration, a failed primary endpoint does **not** authorize
the external-benchmark step, and running session B (seeds 41/73) on
unchanged v5.5 would spend ~7 h of quota reproducing a known failure.

## Gate table (identical pattern across seeds)

| Sub-gate | Seed 17 | Seed 29 | Gate |
|---|---|---|---|
| training_complete (4,224 steps, 0 skips) | PASS | PASS | required |
| no_prompt_leak | PASS (0.0) | PASS (0.0) | = 0 |
| quality_pass_rate | PASS 0.938 | PASS 0.906 | ≥ 0.75 |
| complete_answer_rate | PASS 0.594 | PASS 0.563 | ≥ 0.50 |
| semantic_probe_accuracy | PASS 0.875 | PASS 0.844 | ≥ 0.75 |
| safe_abstention_accuracy / commit rate | PASS 1.0 / 1.0 | PASS 1.0 / 1.0 | ≥ 0.75 / 0.70 |
| probe_commit_accuracy | PASS 0.844 | PASS 0.906 | ≥ 0.70 |
| validation_calibration (on-policy heads) | PASS | PASS | all ≥ 0.70 |
| clean/corrupt margins (3) | PASS | PASS | > 0 |
| lanes_materially_distinct | PASS 0.860 | PASS 0.831 | < 0.90 |
| candidate_f1 ≥ 0.8× base | PASS 0.571 vs 0.218 | PASS 0.505 vs 0.218 | — |
| **second_route_load** | **FAIL 0.0625** | **FAIL 0.0938** | ≥ 0.10 |
| **route_entropy_normalized** | **FAIL 0.130** | **FAIL 0.174** | ≥ 0.25 |
| **routing_causally_live** | **FAIL 0.0035** | **FAIL 0.0008** | ≥ 0.01 |

## Finding 1 — the v5.4 calibration failure is fixed, and it replicated

v5.4 rejected only 0.583 of negatives (gate 0.70) with heads that had never
seen the model's own emissions. With the v5.5 on-policy head phase:

- negative rejection **0.889 (seed 17)** and **0.917 (seed 29)**;
- positive accept 1.0 / 0.982; balanced accuracy 0.944 / 0.949;
- head-phase separation went 0.84 → 1.0 (before → after) on both seeds, and
  the improvement held on *held-out* generated validation emissions — this
  is not the head memorizing its 288 training records.

Two caveats, recorded so the number is not over-read:

1. **The heads are saturated.** After 400 epochs, negative commit
   probability sits at ~5×10⁻⁵ and positive at 0.99994 (both seeds), and on
   the 64 test outputs 39/64 (seed 17) and 60/64 (seed 29) commit
   probabilities are at the rails (< 0.01 or > 0.99). Ranking is perfect on
   head-phase data because the data is memorized; the honest figure is the
   validation 0.89–0.92.
2. **The fitted commitment threshold is 0.0 on both seeds** — the joint
   threshold fit effectively disabled the commitment head as a filter and
   assigned all rejection work to the risk and verifier thresholds
   (0.973/0.973 on seed 17; 0.883/0.762 on seed 29). v5.4's threshold-0 was
   flagged as degeneracy; this one co-occurs with strong held-out rejection,
   so it is an *operating-point* oddity rather than a failure, but a
   three-head system whose calibrated policy uses two heads is carrying dead
   weight, and this pattern replicated across seeds.

## Finding 2 — routing failed the same way twice, and the regularizer reveals why

Expert usage across all lane/depth decisions in the audits:

- seed 17: expert 0 = 480/512 (93.8%), expert 1 = 32, experts 2–5 = **0**;
- seed 29: expert 0 = 464/512 (90.6%), expert 1 = 48, experts 2–5 = **0**.

The training metrics explain the mechanism. The v5.5 marginal-entropy
regularizer did its literal job: the KL of mean routing *probabilities* to
uniform fell to ~0.01–0.03 through training on both seeds. But hard route
*decisions* stayed concentrated (route_unique ≈ 1.8 of 6 throughout). Soft
marginals near uniform with argmax decisions concentrated means every
decision carries a small consistent tilt toward expert 0 — the regularizer
moved the probabilities, not the choices.

Worse, near-uniform soft mixing during training feeds every expert nearly
identical gradients, which homogenizes them. The causal intervention
confirms it: pinning all routing to the least-used expert moved mean
|commit probability| by only 0.0035 (seed 17) and 0.0008 (seed 29) against
a 0.01 gate — swapping experts barely changes the computation, because the
experts have converged toward the same function. (n=8 intervention records,
so treat magnitudes as coarse; both are an order of magnitude under the
gate.) The saturated policy heads also compress these deltas: scores pinned
at 0.9999/0.00005 leave little room to move, so causal deadness and head
saturation are confounded in this measurement.

Improvement over v5.4 for the record: one-expert share fell from 98.4% to
90.6–93.8%, and second-route load roughly doubled. Direction right,
magnitude insufficient, and the intervention says the surviving diversity
does no work.

## Finding 3 — the gates agree across seeds, but operating behavior does not

Seed 17 committed on 48/64 outputs (75% coverage, 71.9% commit accuracy);
seed 29 committed on 26/64 (40.6% coverage, 45.3% commit accuracy). Both
pass every behavioral gate, because the gated quantities (probe commit
accuracy, abstention rates) are robust to this difference — but the two
seeds are materially different deciders at the same thresholds' gate
values. Seed 29 is far more conservative (median behavior visible in its
first audit record: a garbled candidate correctly caught and not
published). This coverage spread is exactly the kind of quantity the
external benchmark's selective-accuracy metric will surface; it should be
reported alongside any benchmark number, not averaged away.

## Session economics

3.45 h charged for two full seeds (training 3.0 h each in parallel, audits,
packaging) — well under the 6–9 h estimate. Peak memory 4.77 GB of 14.56.
BF16 on Turing (deviation D1) ran without incident: zero skipped updates,
finite gradients throughout.

## Decision

1. **Do not run session B (seeds 41/73) on unchanged v5.5.** Two seeds
   already replicate the routing failure; two more would purchase nothing.
2. **Benchmarks are not authorized** by the preregistered rule (full gate
   required). The calibration/abstention machinery — the part benchmarks
   would showcase — passed everywhere, but the rule is the rule; amending
   it post hoc to fit the result would defeat the point of preregistering.
3. **Build v5.6 around the routing mechanism**, the third consecutive
   version where routing is the failing subsystem. Candidate directions,
   in order of preference:
   - replace the marginal-KL regularizer with a **per-decision
     load-balancing objective on hard assignments** (Switch-style auxiliary
     loss on actual route counts, optionally with router z-loss), which
     penalizes the thing that failed (concentrated decisions) rather than
     the thing that didn't (concentrated probabilities);
   - add **expert-diversity pressure** (e.g., orthogonality or output-
     decorrelation penalty between expert outputs on the same input) so
     experts cannot satisfy the objective by converging;
   - if diversity still fails to emerge, **cut num-experts from 6 to 2** and
     re-gate honestly at the capacity the data can support — five of six
     experts received zero test traffic on both seeds.
   Plus one head-phase change either way: reduce policy-head epochs or add
   label smoothing so calibration operates on an unsaturated score scale,
   which also unconfounds the causal-liveness measurement.
4. Quota after this session (~26.5 h) comfortably covers a v5.6 dual-seed
   session (~3.5 h), its seed-41/73 extension if it passes (~3.5 h), and
   benchmarks (~2–3 h).

## Claim boundary

Unchanged. This session establishes: the on-policy calibration fix works
and replicates; the routing subsystem does not meet its preregistered
gates and its remaining diversity is not causally load-bearing. No
production-readiness, benchmark, or Qwen-superiority claim is made or
implied.
