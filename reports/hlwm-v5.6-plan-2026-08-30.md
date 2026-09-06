# HLWM V5.6 Preregistered Plan — routing redesign on verified multi-domain data

Date: 2026-08-30
Package: `artifacts/kaggle/hlwm-v5.6/hlwm-v5.6-candidate-bundle.zip`
(manifest `5.6.0`, sha256 in `artifacts/kaggle/hlwm-v5.6/build-report.json`)
Notebook: `artifacts/kaggle/hlwm-v5.6/embel-hlwm-v5.6-kaggle-2xt4.ipynb`
Dataset: `data/hlwm-v5.6/` built by `scripts/build_hlwm_v56_dataset.py`
(manifest with per-source execution results in `data/hlwm-v5.6/manifest.json`)
Data review this plan implements: `reports/hlwm-v5.6-data-review-2026-08-30.md`
Supersedes: v5.5 plans remain the record for Studies 1–4; this plan governs
the v5.6 Kaggle dual-T4 sessions. Quota at time of writing: ~25 h.

## What v5.6 changes, and why (each traces to a Study 4 finding or review gap)

| Change | Mechanism | Traces to |
|---|---|---|
| Remove marginal-KL router regularizer (`--router-entropy-weight 0.0`) | KL-to-uniform moved probabilities, not decisions, and homogenized experts via near-uniform soft mixing | Study 4 Finding 2 |
| Strengthen hard-assignment balance (`--router-aux-weight 0.01→0.05`) | Penalizes concentrated *decisions* (the thing that failed) | Study 4 Finding 2 |
| Expert-output diversity penalty (`--expert-diversity-weight 0.05`) | All experts evaluated on shared probe rows; pairwise squared-cosine penalty applies pressure on the expert functions directly, independent of routing traffic | Study 4 homogenization mechanism |
| Policy-head de-saturation (`--policy-epochs 400→120`, `--policy-label-smoothing 0.05`) | v5.5 fitted rails (~5e-5 / 0.9999); saturation memorized phase data and compressed intervention deltas | Study 4 Findings 1–2 (confound) |
| Verified multi-domain data | 1,699 execution-verified code episodes + 1,152 Spider + 349 audited + 11 adjudicated builder + 3,263 R9000 (train, post-dedup) | Review G4; routing had no domain diversity to specialize on |
| Budgets context 256 / canvas 128 / brief 96 / causal 1024 | v5.5's causal 384 fit ~25% of the code data; 1024 fits 86–100% | Review G2 |
| Anchor ratio 0.75→0.50 | External verified rows need real batch exposure; anchors still guarantee the abstention class | Review G6 |
| Domain-stratified 64-output audit, 288 generation tokens, execution/SQL graders, per-domain metrics | Code answers cannot complete in 96 tokens; audits must cover every source | Review G5 |

Architecture, optimizer, LoRA configuration, lanes, diffusion steps, the
on-policy head-phase design (96 emissions + hard negatives), calibration
(`validation_generated_candidate_joint_threshold_v2`, test split untouched),
and every gate threshold are unchanged from the v5.5 preregistration.

## Dataset (frozen at build time)

`data/hlwm-v5.6/master` + regenerated behavior anchors, packaged with
cross-split prompt-collision drops (5) and cross-source train dedup (48 —
the known APPS/TACO overlap). Final counts: train 7,498 / validation 945 /
test 841, of which anchors are 1,024/256/256 and policy-eligible rows
2,734/313/348.

Policy-supervision boundary (fail-closed, enforced in `data.py`):
- eligible: behavior anchors (programmatic), 11 builder episodes
  (`independently_adjudicated`, executed pytest), and benchmark episodes
  whose **official reference passed its own executed checks at build time**
  (MBPP 463, APPS 704, TACO 387, CodeContests 334, pre-dedup; 166 references
  that failed execution were dropped entirely, not shipped unverified);
- causal/workspace only: Spider (gold SQL not executed against databases),
  audited OpenThoughts/DeepCoder packets, Reasoning9000 (policy-masked as
  always).
- `is_behavior_anchor` is now strictly narrower than policy eligibility:
  verified code rows do not enter anchor-only probe gates, the anchor
  oversampler, or the head-phase emission pool.

Negative supervision for verified code rows: deterministic cross-problem
reference swaps (fluent, wrong-by-construction), feeding the joint-phase
paired commitment objective under the eligibility mask.

Held out: 5%/3% per-source dev slices (test/validation) carved from official
*train* material by stable hash; official eval splits of every benchmark
remain untouched and the 143-entry registry stays `allowed_for_training:
false`. SWE-bench is deferred to the 8B candidate.

## Sessions and quota ledger (~25 h available)

| Session | Content | Est. charge |
|---|---|---|
| A | v5.6 seeds 17 + 29 concurrently, full pipeline, per-seed audits | 6–8 h |
| B | matched plain-LoRA control (decisive paper control) + seeds 41/73 if A passes | 6–8 h |
| C | external benchmarks (eval-only registry) — only if the full gate passes | 2–3 h |
| reserve | resumes for truncated seeds | remainder |

Timing note (declared estimate, not a gate): v5.5 trained 4,224 steps in
~3.0 h/seed; the larger causal budget is expected to cost ~1.6–1.9×, i.e.
~5–6 h/seed in parallel plus ~0.5–0.8 h audits at 288 tokens. The 8.5 h cap
with exact resumable checkpoints absorbs the upside; a truncated seed is
resumed before any gate claim.

## Preregistered endpoints (thresholds unchanged from v5.5)

Per-seed gate, evaluated once on 64 fresh-process domain-stratified outputs:
leakage 0; quality ≥ 0.75; complete answers ≥ 0.50; anchor semantic ≥ 0.75;
safe abstention ≥ 0.75 with commit rate ≥ 0.70; anchor commit accuracy
≥ 0.70; validation calibration ≥ 0.70/0.70/0.70 fitted after the on-policy
head phase; positive clean/corrupt margins (3); second-route load ≥ 0.10;
normalized route entropy ≥ 0.25; least-used-expert intervention |Δ commit|
≥ 0.01; lane cosine < 0.90; candidate F1 ≥ 0.8× frozen base;
`training_complete` with zero skipped updates.

- **Primary endpoint:** both session-A seeds (17, 29) pass the full gate.
- **Secondary endpoint:** ≥ 3 of 4 seeds across A and B pass, with the three
  routing gates passing on every completed seed.
- **Report-only (not gated, declared before the run):** per-domain
  dev-slice metrics — python-function pass-by-execution,
  competitive-programming pass-by-execution, text-to-sql exact match,
  per-domain commit coverage/accuracy — recorded to set v5.7 baselines and
  to pair coverage with risk per seed (Study 4 Finding 3).
- **Declared fallback:** if both seeds fail the three routing gates again,
  the next session runs `--num-experts 2` with the same objective, and the
  paper's simplification rule (reduce to the strongest ablation) is invoked
  for the routing subsystem.

A passing primary endpoint authorizes the external-benchmark step
(eval-only registry suites vs same-size instruct models, standard plus
selective/abstention accuracy). It is not a production-readiness claim.
Session B's plain-LoRA control runs regardless of A's outcome: no capability
claim is valid without it.

## Code deltas from v5.5 (all tested; 33/33 suite green)

- `modeling_hlwm.py`: `expert_diversity_weight` config +
  `RoutedAdapterBank.output_diversity` (all-expert probe evaluation, pairwise
  squared cosine) wired into local and joint losses and metrics.
- `train_kaggle.py`: `--router-aux-weight`, `--expert-diversity-weight`,
  `--policy-label-smoothing` (smoothed [1−s, s, s]/[s, 1−s, 1−s] targets in
  the head phase; smoothing recorded in the phase report).
- `semantic_grading.py`: `python_tests` (executed assert lists), `io_tests`
  (executed stdin/stdout pairs), `sql_exact` (normalized match) graders +
  isolated-subprocess runner with hard timeouts; shared by the build-time
  verifier and the audit.
- `evaluate_checkpoint.py`: domain-stratified sampling; per-domain metrics in
  the aggregate.
- `data.py`: `is_behavior_anchor` restricted to the anchor domain (see
  boundary above).
- New: `scripts/build_hlwm_v56_dataset.py` (converter + build-time execution
  verification), `scripts/build_hlwm_v56_bundle.py` (bundle + notebook).

## Deviations ledger

D1 (BF16 on Turing) carries over from the v5.5 T4 plan: `--precision auto`
resolves to BF16 on these T4s; numerically safe, `skipped_optimizer_updates
== 0` remains asserted but is weak evidence there. No other deviations at
plan time; any mid-run deviation is recorded here before results are read.
