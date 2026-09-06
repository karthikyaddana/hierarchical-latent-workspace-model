# HLWM V6.0 Preregistered Plan — verified fan-in

Date: 2026-09-01
Base: v5.7 workspace-only (inherits everything not listed: data byte-identical
to v5.6, budgets 256/128/96/1024, 4,224 steps, LR 8e-5, de-correlated head
labels, validity-gated head phase, domain-stratified canary, all behavioral
gate thresholds).
Artifacts: `artifacts/kaggle/hlwm-v6.0/` (bundle sha 195940ac…, 46/46 tests).
Supersedes running v5.7 as-is: v6.0 is v5.7 plus the three quality-directed
mechanisms below, each carrying its own causal test or killer baseline.

## Why these three changes

The program's honest position after seven studies: calibrated commitment
works and replicates; routing exited on negative evidence; no mechanism has
ever improved content. The three known reasons content never moved, and the
v6.0 answer to each:

1. **The decoder could not read the workspace.** The synthesis prefix is a
   linear reshape of exactly three pooled vectors (~3×1024 floats), so lane
   computation was information-bottlenecked out of the answer regardless of
   what the lanes learned (paper Known Limitations item 2). → **Latent
   read-out memory**: 16 windowed canvas hidden-state tokens + 1 summary
   token per lane join the prefix (8 → 43 tokens, +9% decode sequence).
   Zero-initialized, input-normalized projections open the channel only as
   gradients demand (ControlNet-style). The privacy invariant holds: hidden
   states cross the barrier, never token identities, and publication stays
   gated. The audit re-decodes every row with memory sliced off — the
   narrow-interface ablation is measured in the same run (the lesson of the
   routing era: no mechanism without its causal test).
2. **The heads only ever braked; they never steered.** The replicated
   strength (negative rejection 0.89–0.98) scored one candidate take-it-or-
   leave-it. → **Verified fan-in**: one workspace pass, four candidates
   (greedy + T∈{0.7, 0.9, 1.1}), the heads score each through the identical
   feature path used in training (reading private workspace state — pooled
   candidate hidden, global state, private verifier error, entropy — not
   just text), the best calibrated score is published if it clears the rule.
   Head-phase collection harvests all four candidates per anchor with
   within-anchor ranking pairs, so the heads train on exactly the judgment
   inference asks of them. Verifier-selection beating self-consistency at
   low N is established at 8B+ (Calibrated Reasoning, arXiv:2509.19681);
   whether a frozen 0.6B with sidecar heads can do it is open — that is the
   headline gate.
3. **The publication rule was structurally degenerate.** Six recurrences of
   a rail-pinned threshold under joint triple-threshold sweeps. → **Scalar
   rule**: regularized logistic combiner over the three de-correlated head
   probabilities (Platt-style), one threshold chosen at the margin midpoint
   nearest 0.5 among balanced-accuracy ties. One scalar does selection and
   abstention; `publish_weights/bias/threshold` live in the model config and
   ship in the checkpoint.

## The killer baselines (in the same audit, same budgets, same rows)

| Baseline | What it answers |
|---|---|
| Self-consistency majority vote over 4 causal-path samples (Wang et al. 2022) | Is latent-workspace fan-in better than the free ensembling anyone would try first? Gate: `fan_in_beats_self_consistency`. |
| Max-mean-logprob best-of-4 (training-free, Self-Certainty-style) | Do the trained heads beat the model's own logits as a selector? Gate: `fan_in_beats_logprob_bon`. |
| Greedy single candidate (fan-in's own candidate 0) | Does selection help at all? Gate: `fan_in_no_worse_than_greedy`. |
| Coverage-matched mean-logprob abstention (post-hoc threshold, favors baseline) | Is calibrated abstention more than a training-pipeline curiosity? Gate: `calibrated_abstention_beats_logprob`. |
| Narrow-interface re-decode (per-row ablation) | Is the widened read-out causally load-bearing? Report-only this study; declared gate margin (≥0.02 greedy F1 delta) from Study 9. |

Baseline comparisons run on lightweight-grader rows (numeric / unit /
ordering / abstention / multiple-choice / SQL) so voting and per-candidate
grading are exact; code-executing graders still grade the published output.

## Configuration deltas vs v5.7

`--workspace-memory-windows 16`, `--candidate-temperatures 0.0,0.7,0.9,1.1`
(trainer, collection, calibration, and audit all share the list),
`--policy-max-attempts 1024` (candidates, ≈256 anchors worst case; floor 16
valid/family unchanged and now counted over candidates, which sampled decodes
make easier to reach). Trainable parameters 76,627,079 (+1.9% vs Study 6's
75,174,348; +2.8% vs v5.7) — the plain-LoRA control is budget-matched at its
own rank when it runs. Calibration fits on 64 validation anchors × 4
candidates + 64 corrupts (~320 rows). Gate battery 15 → 20.

## Session and endpoints

Session: seeds 17 + 29, fresh training, one per T4. Estimated 7–7.5 h
(training +9% sequence, audit ≈2× for the extra decodes) against the 8.5 h
trainer cap and 12 h session ceiling; fits the ~9 h weekly quota.

- **Primary endpoint:** both seeds pass all 20 gates.
- **Headline quality question:** `fan_in_beats_self_consistency` on both
  seeds, with `fan_in.selected_accuracy` also above Study 6 content levels
  (0.75–0.78) — quality recovered *and* attributable to the mechanism, not
  just to sampling.
- **Declared diagnostics:** (a) read-out ablation delta (is the memory
  load-bearing); (b) publish threshold ∈ [0.05, 0.95] on both seeds — the
  structural fix's direct test; (c) selection gain over greedy on validation
  vs audit (generalization of the selector); (d) oracle-any-valid vs
  selected (how much headroom the selector leaves); (e) canary trajectories
  unchanged from v5.7 expectations.
- **Fallback ladder:**
  1. Fan-in ≥ SC on both seeds and full gate pass → plain-LoRA control runs
     immediately (budget-matched, with the same N-sample SC/logprob
     machinery — the control now doubles as the "any adapter + SC" test),
     then external benchmarks per the standing draft prereg.
  2. Fan-in < SC but > greedy → the workspace verifier adds value but less
     than free ensembling: re-scope the claim to calibrated abstention only;
     the selective-prediction bake-off (gate 4) decides whether that claim
     survives.
  3. Read-out ablation ≈ 0 and content ≈ v5.7 → the lanes do not inform the
     answer even with an open channel; the latent-workspace quality
     hypothesis is falsified at this scale and the paper's lane sections are
     re-scoped to proposal-plus-negative-evidence, mirroring the routing
     precedent.
  4. Publish threshold degenerate again despite the scalar rule → the
     failure is upstream in the features, not the rule; v6.1 revisits
     candidate features before any further calibration work.

## Claim boundary

No production claim. A full pass authorizes the matched control, nothing
else. All baseline comparisons are disclosed as small-n (≤64 rows/seed,
lightweight-grader subset for voting comparisons); the benchmark kit's
official-split suites remain the external test, gated behind control + full
pass, per the standing rules.
