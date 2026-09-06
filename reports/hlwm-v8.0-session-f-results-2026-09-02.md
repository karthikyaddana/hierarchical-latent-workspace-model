# HLWM v8.0 Session F results — 2026-09-02

**Run:** Kaggle script version 346663543 (`karthikyaddanapudi/notebooke353b0f99d`), 2×T4,
seeds 17 and 29 concurrent, bundle sha `3139b6bb…`, notebook sha `8a3bb5ee…`,
base `Qwen/Qwen3-0.6B-Base` @ `da87bfb6…`. 58/58 unit tests passed in-session.
Training 4,224/4,224 microsteps both seeds, zero skipped updates, ~6.1 h; audits
320 rows/seed at ~18–19 s/row, ~1.7 h each, concurrent. Total ≈ 8 h of the planned 9–10.

**Session defect (disclosed):** the notebook crashed in a *debug print* in the gates
cell (`KeyError: 'hlwm_candidate'` — a v6.0 record field name that v8.0 renamed) after
seed 17's gates were computed and written but before seed 29's gate block, the pooled
McNemar, the verdict JSON, and the deliverable zips. All of those are pure
post-processing; seed 29's gates were already inside its `checkpoint-evaluation.json`
(written by `evaluate_checkpoint.py`), and the pooled statistic is recomputed locally
below with the notebook cell-17 logic verbatim. No GPU work was lost.

**Provenance:** `metrics.jsonl` for both seeds fetched directly from the run's signed
URLs (byte sizes match the training heartbeat exactly: 4,101,232 / 4,057,942); all
other numbers transcribed from the run log's JSON blocks
(`summary.json`, `commitment-calibration.json`, `policy-head-training.json`,
`checkpoint-evaluation.json` aggregates + gates). Full archive fetch:
`kaggle kernels output karthikyaddanapudi/notebooke353b0f99d -p <dest>` (checkpoints
were not cleaned up by the crashed packaging cell, so the pull is large).
Local verdict: `artifacts/kaggle/hlwm-v8.0/session-f/hlwm-v8.0-replication-verdict-local.json`.

---

## 1. Verdict

**Both seeds FAILED the 24-gate battery: 17/24 gates passed on each seed, 7 failed
on each, 6 failures shared.** The six shared failures are one coherent result:

> Every v8.0 *repair* worked and replicated. Every *mechanism* the repairs were
> built to enable measured null or lost to its training-free baseline, and that
> replicated too.

| | seed 17 | seed 29 |
|---|---|---|
| Gates passed | 17/24 | 17/24 |
| `passed` | false | false |
| Unique failure | `weighted_sc_beats_sc_medium` | `coverage_in_band` (0.1875 < 0.20) |

Shared failures: `readout_live`, `selection_headroom_present`,
`abstention_beats_logprob_augrc`, `quality_pass_rate_at_least_70pct`,
`corrupt_verifier_ranked_above_clean`, `lanes_materially_distinct`.

## 2. The repairs are vindicated (the v6.0 root cause is confirmed)

| Measure | v6.0 Session E (s17/s29) | v8.0 Session F (s17/s29) |
|---|---|---|
| Scaffold leak (audit) | 12/64, 1/64 | **0/320, 0/320** |
| Canary marker rate (training) | blind by construction | **0.000 at 17/18 boundaries** (one 1/8 blip, s29 step 1024, local phase, gone after) |
| Workspace vs causal accuracy | ≈ −30 pts | **+0.0125 / −0.0031 gap** (0.666 vs 0.678; 0.669 vs 0.666) |
| Workspace/causal F1 ratio | 0.52 / 0.80 | **1.015 / 1.025** (workspace above causal both) |
| Calibrated coverage | 0.172 / 0.078 | **0.244 / 0.1875** (conformal k=83, tie mass 0.8% both) |
| Threshold degeneracy | rail-degenerate 6× streak broken | non-degenerate both, τ from held-out half B |

The BOS/eos diagnosis (no BOS in Qwen3 → `bos_token_id` aliases to `<|endoftext|>` →
mid-sequence document boundary) is now confirmed at 320-row scale on two seeds:
remove the seed token and the ~30-point channel gap vanishes entirely. R1/R2
(no seed id; cue as appended embeddings, verified in-session:
`bos_token None`, cue `[198, 14374, 5949, 198]` absent from prompt) are validated.
The conformal publish rule (R6) structurally killed both the degenerate-threshold
pathology (7th consecutive run would have been expected under the old midpoint rule)
and the v6.0 coverage strangle.

## 3. The mechanisms are dead, replicated

### 3.1 Claim G — parity PASS, liveness FAIL (both seeds) → the read-out is inert

- `readout_ablation.graded_accuracy_delta`: **0.000** (s17), **−0.0219** (s29).
  F1 deltas 0.0003 / −0.0029.
- **Instrument correction (post-session code audit, 2026-09-02):** the shipped
  ablation did NOT remove the whole prefix. `disable_workspace_memory` slices
  `prefix[:, :synthesis_prefix_tokens]` (modeling_hlwm.py:2670-2671), so the 8
  gated synthesis tokens stayed in BOTH arms and only the 34 memory tokens were
  ablated. Full-prefix inertness is instead bounded by **channel parity** (the
  causal arm is literally prefix=None): gap +0.0125/−0.0031. The inertness
  conclusion stands on parity; the ablation as shipped measured only the memory
  tokens.
- **Second instrument correction:** the flat `prefix_gate_tanh` ([0.0498, 0.0507]
  all run) is NOT evidence that training rejected the prefix. Layer-0 RMSNorm is
  scale-invariant, so the attention-visible content of the prefix is independent
  of α for any α>0, and the gate's gradient through that (dominant) branch is ~0
  by homogeneity *regardless of whether the prefix is useful*. The gate is an
  inert instrument, not a closed valve: the channel was open at tanh(0.05) and the
  decoder ignored it via attention. The evidence for inertness is parity + the
  memory-token ablation, not α.
- A third design fact sharpens the diagnosis: the KL anchor's reference is this
  same model with `workspace_prefix=None` on identical inputs
  (modeling_hlwm.py:2435-2441), so at CE convergence the only persistent gradient
  on the prefix pathway points exactly at "change the logits by nothing." Combined
  with both channels conditioning on the same `input_ids` (data.py:489-496), the
  prefix was redundant by construction: textbook posterior collapse under a strong
  decoder.
- Interpretation discipline: v6.0's non-null ablation (−0.167/−0.306) measured the
  removal of a *corrupting* prefix, not an informative one. With the corruption
  repaired, the honest measurement is: at 0.6B, with matched-pair channels and a
  redundant latent, the workspace read-out contributes **nothing** to answer
  quality. Parity was achieved by irrelevance.
- Pre-mortem P1 (attenuated prefix / KL anchor nulls mechanism) fired as declared,
  with `readout_live` as its named detector — though per the corrections above,
  the "attenuation" framing was wrong: the channel was open and unneeded.

### 3.2 Claim S — VOID for headroom (both seeds), and the weighting was inert

- `exactly_one_valid_fraction` on the medium bin: **0.0 and 0.0** (gate ≥ 0.10).
  Preregistered outcome: Claim S is **void for lack of headroom** — a design result
  about row difficulty, not a model failure.
- Difficulty is bimodal, not banded: easy 187/189 rows (every arm 1.000), hard 91/98
  (oracle 0.143/0.122), medium only 42/33. What headroom exists lives in the hard
  bin (exactly-one-valid 0.143/0.122), exactly where the prereg's fallback points
  (move the selection question to harder rows).
- Discordant pairs weighted-SC vs plurality-SC: s17 **0 in 320 rows**; s29 2 wins /
  0 losses. Pooled McNemar mid-p = **0.125** (2W/0L over 75 medium rows). s29's
  per-seed gate `weighted_sc_beats_sc_medium` is nominally true (0.727 vs 0.667) but
  void under the headroom gate and nowhere near the replication p<0.05.
- Signal worth recording: on medium rows, `heads_argmax` beat greedy on both seeds
  (0.786 vs 0.643; 0.818 vs 0.697) and `bon_logprob` did comparably (0.810/0.788) —
  n=42/33, report-only. The heads rank; there is just nothing for ranking to win at
  this difficulty mix.

### 3.3 Claim A — FAILED as preregistered (both seeds); operationally flawless

- AUGRC: heads **0.0637 / 0.0654** vs logprob **0.0594 / 0.0623** — the heads *lost*
  to mean-logprob abstention on both seeds. Bootstrap win fraction 0.13 / 0.179
  (needed ≥ 0.90). `abstention_beats_logprob_augrc` FAILED both.
- Coverage: 0.244 (in band) / 0.1875 (1.25 pts below the 0.20 floor). The pre-mortem
  explicitly predicted "a coverage band missed on one seed"; the conformal guarantee
  held on its own calibration distribution (0.359 empirical, 0.357 guaranteed) and
  the shortfall is validation→test shift.
- The operational result, replicated on BOTH seeds for the first time in the program:
  **every published answer was correct** — 78/78 (s17) and 60/60 (s29) at coverage
  0.244/0.1875, selective-risk UCB95 0.038/0.049, safe-abstention probes 1.000 both,
  and **commit coverage 0.000 on text-to-SQL** on both seeds — the abstention system
  refused to publish anything on the one domain where the model is incompetent
  (quality 0.078/0.109).
- Why perfect-precision heads lose AUGRC: 58–59% of audit rows are easy (all-8-valid),
  where logprob's smooth ranking saturates the low-coverage region of the curve just
  as well; AUGRC integrates the full curve, and the heads' score mass is coarser.
  At the matched-coverage *point* the heads win on both seeds (1.000 vs 0.987; 1.000
  vs 0.983) — `selective_accuracy_matched` passed both. `heads_not_logprob` also
  passed both (Spearman 0.830/0.821): the heads are not a logprob re-derivation;
  they are simply not better than one on this distribution.

### 3.4 Lanes — failures #8 and #9; the unconditional exit fires

Mean lane-summary cosine **0.9396 / 0.9751** against the 0.90 gate, with R5's
structurally asymmetric context slices and live orthogonality gradients in place.
Per the preregistered unconditional exit: **lanes are demoted to a single lane**, and
the paper's lane-diversity sections move to proposal + negative evidence alongside
routing and fan-in. No further lane fixes are authorized.

### 3.5 Other failures

- `corrupt_verifier_ranked_above_clean` FAILED both (margins −0.233/−0.297) while the
  same head ranked 0.944/0.947 on teacher-forced training pairs: train→generated
  distribution shift inverted the verifier head at audit. Commit and risk margins
  stayed positive both seeds.
- `quality_pass_rate_at_least_70pct` FAILED both (0.663/0.666) — entirely
  text-to-SQL: behavior anchors 0.809/0.805 vs SQL 0.078/0.109. The domain-stratified
  audit did what the v6.0 anchor-only audit could not: it priced the incompetent
  domain into the headline gate.
- Policy validity floor NOT met (both): valid positives {abstention 16, numeric 2/1,
  ordering 5/2, unit 2/5} after the 1,024-attempt cap. **Prereg-vs-code gap
  (disclosed):** the plan describes `policy_heads_trained_on_policy` as a 16/family
  floor, but the shipped gate only requires `trained` (≥8 positives and negatives
  total), so the gate shows *true* on both seeds while the floor status is false.
  The heads' non-abstention supervision is ~9/8 valid positives; every head result
  above must be read under that limit.
  **Root-cause reclassification (post-session code audit, 2026-09-02):** the floor
  miss is a train/audit input-surface BUG, not model incapacity. The harvest feeds
  `generate_hlwm_nbest` the collator's fixed-length right-padded context (pad to
  256 with pad=eos; anchors are ~84-105 real tokens, so ~150-170 masked pad
  positions sit between the prompt and the response cue, and generation starts
  after the gap; data.py `_fixed`, train_kaggle.py:1390-1406), while the audit
  encodes prompts UNPADDED (`encode_preserving_ends`,
  evaluate_checkpoint.py:253-266). Audit-time causal validity on anchors was 0.81,
  vs ~1-3% valid harvest emissions on generative families: the model was decoding
  on a surface it never sees anywhere else. The calibration pools share the padded
  surface, which also plausibly contributes to the seed-29 validation→test
  coverage shift (0.359 → 0.1875). Fix is bug-class (one shared encode primitive)
  and is scheduled in the v9.0 plan.

## 4. Ladder application

The bundle notebook's preregistered ladder governs the run. Its rung 1 enumerates
**dead read-out** among Claim G failures: *"Claim G fails (parity, leak, or dead
read-out) — the repair did not take. Stop; no selection or abstention result from
this run is interpretable."* `readout_live` failed on both seeds → **rung 1**.

Two honesty notes, stated rather than smoothed over:

1. **Wording gap between preregistrations.** The paper's Study 9 section words the
   generative-falsification rung as failure "on parity or leak"; a liveness-only
   failure is not explicitly assigned there. The notebook (shipped inside the frozen
   bundle, sha-verified before training) is the operative document and it puts dead
   read-out in rung 1. The paper write-up must quote both and follow the notebook.
2. **Rung 1's rationale is partially stale for this outcome.** Its "not
   interpretable" clause was written for a *corrupt* channel contaminating the S/A
   pools. What happened instead is an *inert* prefix over a healthy causal pool
   (Claim S/A were measured on causal candidates by design), and Claim A's failure is
   independently decisive: it lost its own preregistered comparison on both seeds.
   So the conservative summary does not depend on the clause: **G failed (liveness,
   replicated), S is void (headroom, replicated), A failed (AUGRC, replicated).**

**Consequences (all preregistered):**

- The latent-workspace **read-out mechanism is retired at 0.6B** — the third and
  last of the three proposed mechanisms to be measured and fail (routing: Study 7;
  fan-in selection: Study 8; read-out liveness: Study 9).
- **Lanes are demoted to a single lane** (failures #8 and #9 of the cosine gate).
- The **plain-LoRA control remains UNAUTHORIZED** (no full pass; 9 studies, 0 passes)
  and the external benchmarks remain gated behind it.
- The abstention claim as preregistered ("conformal heads beat logprob abstention")
  is **dead on both seeds**. What survives is narrower and is stated as such: the
  conformal publication machinery is the first mechanism in the program to
  *replicate its operational contract on both seeds* (coverage controlled to within
  1.25 pts of band on the worse seed, zero selective errors on 138 published answers
  across both seeds, correct total refusal on the incompetent domain) while failing
  to beat the training-free baseline that any deployment would get for free.

## 5. What the paper says now

Study 9 results to be written against the frozen preregistration in
`paper/hlwm-paper.tex` §Study 9 (currently prereg-only, no results — deliberate):

- Abstract, Study 9 results subsection, Known Limitations, failure-modes table
  (lane exit fires; verifier inversion), and Conclusion all update.
- The sharpened primary claim from Study 8 ("calibrated selective prediction,
  supported on one seed of two") is **re-sharpened downward**: the selective
  machinery is real, replicated, and *not better than mean-logprob abstention* at
  this scale on this distribution. The paper's honest headline is a completed
  negative-results program: three mechanisms proposed, three measured to null or
  worse under preregistered gates, with the calibration infrastructure (conformal
  coverage control, de-correlated labels, validity-gated harvesting, domain-stratified
  audits) as the transferable positive contribution.
- Disclose in the Study 9 results: the notebook debug-print crash (post-science,
  post-hoc recomputation documented here), the prereg-vs-code floor-gate gap, the
  ladder wording gap, and the s29 coverage-band miss having been a named pre-mortem
  prediction.

## 6. Session accounting

- 2nd Kaggle account: ~8 h consumed this session; ≈22 h of the weekly 30 remain.
  Old account: 2.5 h. No further GPU run is authorized by the ladder.
- Notebook fix for the archive (not a rerun): the gates-cell debug print must read
  the v8.0 record fields; guard with `row.get(...)`.
