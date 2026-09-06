# HLWM V6.0 Session E Results — fan-in loses to free ensembling; the selector is near-oracle but the workspace channel starves it

Date: 2026-09-02
Plan: `reports/hlwm-v6.0-plan-2026-09-01.md` (preregistered Study 8)
Bundle: hotfixed v6.0 candidate (autocast fix in `causal_mean_logprob`), zip
sha256 39477eb5…
Artifacts: `artifacts/kaggle/hlwm-v6.0/session-e/seed-17/` and `seed-29/`,
plus `session-e/hlwm-v6.0-replication-verdict.json` (Kaggle kernel
`karthikyaddanapudi/notebook92a54c792f`, script version 346509490, run on the
second Kaggle account). Adapter sha256: s17 fb2f2ec4…, s29 b0174da4….
Config verified from summaries: num_experts 1, num_lanes 2, 16-window
read-out memory, candidate temperatures {0.0, 0.7, 0.9, 1.1}, validity floor
16/family cap 1024, scalar publish rule, 4224 steps, bf16, seeds 17/29.

## Session D is vindicated: this is the same run, completed

Train and policy phases reproduced session D bit-for-bit: seed 17's
`commitment-calibration.json` and seed 29's `policy-head-training.json` are
byte-identical to the crashed session's archives, and all six policy
thresholds match to full float precision (s17 commit 0.4368 / risk 0.1330 /
verifier 0.4102; s29 0.2056 / 0.0515 / 0.8700). Everything session D
established stands (floor met 16×4 on both seeds, triple thresholds
non-degenerate, calibration balanced accuracy 0.884/0.877, publish rule
fitted sanely). What follows is the audit session D never reached.

## Verdict

**Replication FAILED on both seeds (seed 17: 9 of 20 gates failed; seed 29:
11 of 20). The headline gate `fan_in_beats_self_consistency` failed
decisively on both seeds, while `fan_in_no_worse_than_greedy` passed on
both — this is exactly the preregistered fallback ladder's rung 2: the claim
re-scopes to calibrated abstention only.** The selective-prediction bake-off
that rung 2 says decides the abstention claim then **split**: seed 17 passed
(heads 0.818 vs logprob 0.636 at matched coverage 0.306), seed 29 failed
(0.333 vs 1.0 at coverage 0.083, three committed rows). One seed of two: the
abstention-only claim is supported but not replicated. Rungs 3 and 4 are
**not** triggered — the read-out ablation is decisively non-null and the
publish threshold stayed non-degenerate on both seeds.

| Gate (of 20) | Seed 17 | Seed 29 |
|---|---|---|
| training_complete | PASS | PASS |
| no_prompt_leak (=0) | **FAIL 0.1875** | **FAIL 0.0156** |
| quality_pass_rate ≥0.75 | FAIL 0.5625 | FAIL 0.6875 |
| complete_answer_rate ≥0.50 | PASS 0.781 | PASS 0.719 |
| semantic_probe_accuracy ≥0.75 | FAIL 0.250 | FAIL 0.500 |
| safe_abstention_accuracy ≥0.75 | FAIL 0.625 | **FAIL 0.000** |
| safe_abstention_commit_rate ≥0.70 | FAIL 0.375 | **FAIL 0.000** |
| probe_commit_accuracy ≥0.70 | FAIL 0.656 | FAIL 0.406 |
| validation_calibration | PASS 0.884 | PASS 0.877 |
| policy_heads_trained_on_policy | PASS | PASS |
| clean_commit_ranked_above_corrupt | PASS | PASS |
| corrupt_risk_ranked_above_clean | PASS | PASS |
| corrupt_verifier_ranked_above_clean | PASS | FAIL (margin −0.109) |
| lanes_materially_distinct (<0.90) | FAIL 0.945 | FAIL 0.968 |
| candidate_f1 ≥ 0.8×base | PASS | PASS |
| publish_threshold_non_degenerate | **PASS** | **PASS** |
| fan_in_no_worse_than_greedy | **PASS +0.222** | **PASS +0.028** |
| fan_in_beats_self_consistency | **FAIL 0.455 vs 0.758** | **FAIL 0.515 vs 0.818** |
| fan_in_beats_logprob_bon | FAIL 0.455 vs 0.758 | FAIL 0.515 vs 0.818 |
| calibrated_abstention_beats_logprob | **PASS 0.818 vs 0.636** | FAIL 0.333 vs 1.000 |

## Finding 1 — the selector is near-oracle; the generation channel is the bottleneck

The fan-in loss to self-consistency is **not** a selection failure. On seed
17 the publish-score selection hit its oracle exactly (selected 0.417 =
oracle-any-valid 0.417, +0.222 over its own greedy); on seed 29 it recovered
half the available headroom (0.472 of oracle 0.556, greedy 0.444). The
problem is what the selector gets to choose from: the workspace-conditioned
oracle itself (0.417/0.556) sits ~30 points below plain causal greedy
generation from the *same adapter on the same rows* (0.758/0.818). Token-F1
tells the same story: causal 0.560/0.570 vs workspace candidates
0.289/0.454. SC majority vote, max-logprob BoN, and causal greedy all tie at
0.758/0.818 because they all draw from the healthy causal channel; fan-in
draws from the degraded workspace channel and no selection rule can recover
content that was never generated. Study 6's content levels (0.75–0.78) were
never re-attained through the workspace path (probe content 0.25/0.50).

## Finding 2 — the read-out memory is live, which makes the channel failure sharper

Rung 3 (null ablation) is cleanly rejected: zeroing the 43-token latent
read-out costs 0.167/0.306 graded accuracy and 0.104/0.232 token-F1
(s17/s29). The lanes *do* inform the answer — the channel carries real
information and simultaneously corrupts the generation register. This is the
most useful diagnostic of the session: the latent-workspace hypothesis
fails at the *interface*, not at the *content*. v6.0's negative result is
"conditioning on the workspace prefix costs more than the workspace
knows," not "the workspace knows nothing."

## Finding 3 — the corruption is visible: the latent prefix knocks the model into a new dialogue turn

First-ever look at the audit emission path (session D crashed before it):
12/64 seed-17 and 1/64 seed-29 published answers open with role scaffold —
`"Human: 46920 seconds."`, `"Human: ascending order: …"`,
`"Assistant: The software version was not provided…"`. The same episodes'
causal generations are clean (`"The result is 84325."` vs workspace
`"Human: 84325."`). The 43-token synthesis prefix reads to the model like
end-of-turn, so it begins a fresh one. The training canary showed leak 0.0
at every checkpoint on both seeds — this is emission-path-specific and
invisible to the canary. But the leak is a symptom, not the binding failure:
counterfactually forgiving every leak-only quality failure lifts pass rates
only to 0.609/0.703, still under the 0.75 gate. Semantic content loss is the
main event; the scaffold is the visible edge of it.

## Finding 4 — the scalar publish rule held; seed 29's is calibrated but strangled

The 6× degenerate-threshold streak stays broken: thresholds non-degenerate
on both seeds, head separation healthy (positive vs negative commit
probability 0.80 vs 0.05, pairwise ranking 0.999/0.998). But the rule can
only be as good as its operating point. Seed 29's fitted rule
(w [2.52, −1.80, −1.00], τ = 0.689, verifier threshold 0.870 against a mean
verifier-error probability of 0.843) commits on 7.8% of audit rows and 0% of
the abstention battery's commit-expected probes — safe-abstention accuracy
0.000 is over-conservatism, not miscalibration (its corrupt rejection rate
is 0.922). Seed 17 commits 17.2% and posts the program's first
selective-prediction win over matched-coverage logprob abstention (0.818 vs
0.636 at 0.306 coverage). Same architecture, same rule — the difference is
where calibration landed the operating point. Coverage control (fit τ to a
target coverage, not the margin midpoint) is the obvious v6.1 candidate fix.

## Finding 5 — small negatives, disclosed

Lanes are not distinct (cosine 0.945/0.968; seed 29's session-C distinctness
did not carry into the workspace-only architecture). Seed 29's verifier head
ranks corrupt candidates *below* clean on the audit (margin −0.109) despite
correct ranking at policy-fit time. Temperature selection differs by seed
(s29 picks T=0 on 34/64 rows; s17 spreads 18/19/15/12). The domain-canary
n=1 grading gap from session D persists (1/8 canary rows graded). All
baseline comparisons remain small-n (33–36 graded rows/seed) as disclosed in
the plan's claim boundary.

## Preregistered consequences

1. **Rung 2 is invoked.** The verified fan-in claim is dead at this scale:
   Study 8 reports the headline gate as failed on both seeds, and the fan-in
   mechanism joins routing as proposal-plus-negative-evidence. The
   plain-LoRA control and external benchmarks remain **unauthorized** (they
   were gated behind a full pass).
2. **The paper's claim re-scopes to calibrated abstention only**, with the
   bake-off split disclosed as-is: supported on seed 17 (0.818 vs 0.636 at
   matched coverage), refuted on seed 29 (over-conservative operating
   point), not a replicated claim at n=2 seeds.
3. Rungs 3 and 4 explicitly did not fire: the read-out ablation is non-null
   on both seeds and the scalar publish rule stayed non-degenerate — both
   structural fixes from the v6.0 design are confirmed working.

## Not preregistered — decisions that need an owner

- **v6.1 (interface repair):** the diagnosis is unusually actionable — keep
  the read-out (it carries 0.17–0.31 of graded accuracy) and fix the
  register: ban role-scaffold tokens at decode, repair the prefix splice so
  it doesn't read as end-of-turn, and calibrate τ to a coverage target so
  seed-29-style strangulation can't happen. That is a targeted ~7h two-seed
  session against the sharpest failure surface the program has had.
- **Or stop mechanism work** and write up Study 8 as the terminal negative
  result with the abstention split as the residual open question.
- Budget note: the second account's ~30h no longer owes 5h to the control
  (not authorized); the old account retains ~2.5h.
