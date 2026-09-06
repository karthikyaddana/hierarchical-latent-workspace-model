# HLWM v8.0 preregistered plan — "repair the register, decouple the claims"

Date: 2026-09-02 (preregistered before any v8.0 code was written)
Lineage: v6.0 (Study 8, replication failed, rung 2 invoked) → v8.0. The
v6.1/v7 designations are skipped at the owner's direction; v8.0 is the
interface-repair + claim-decoupling redesign. Session E evidence:
`reports/hlwm-v6.0-session-e-results-2026-09-02.md`.

## 0. Why v6.0 failed, in one paragraph each

**Bug, not just design flaw.** The workspace answer channel teacher-forces
and decodes from `decoder_ids[:,0] = bos_token_id` — and
`HLWMConfig.from_hf_config` maps `bos = eos` because Qwen3 has no BOS token.
Every workspace-channel answer therefore begins with `<|endoftext|>`
mid-sequence: an end-of-document marker. A base model that sees
end-of-document starts a new document — chat transcripts are a large slice
of pretraining, so it opens "Human:". The causal channel never inserts this
token, which is why it stayed clean (0.758–0.818) while the workspace
channel collapsed (0.417/0.556 oracle). The scaffold leak (12/64, 1/64) was
the visible edge; the ~30-point content loss was the mass of the iceberg.

**The headline gate was structurally unwinnable.** With N=4 candidates at
per-sample accuracy p, a selector's edge over plurality voting lives almost
entirely in P(exactly one correct) = N·p·(1−p)^(N−1). At the causal
channel's p ≈ 0.76–0.82 this is 2–5 accuracy points — 1–2 rows out of 36
graded. Published verifier-beats-voting results (Cobbe 2021; Lightman 2023;
DIVERSE) operate at N = 100–1860 and/or low-p regimes. v6.0 asked a
question its own audit could not answer even if everything had worked.

**Calibration had no coverage control.** τ maximized balanced accuracy on a
majority-negative pool; nothing prevented the seed-29 operating point
(coverage 0.078, safe-abstention 0.000). Fitting weights and threshold on
the same rows also breaks the exchangeability needed for any finite-sample
guarantee.

**Lanes were symmetric, so they converged.** Two lanes fed identical inputs
under a shared loss receive near-identical gradients ("symmetry lock",
learner collusion); the 0.05 squared-cosine penalty was live but
outmatched, and the zero-init projections kept most diversity gradients
dead. Cosine 0.945/0.968 is the expected default, not a surprise.

## 1. Research base (four commissioned memos, 2026-09-02)

1. **Register corruption.** Every working latent-injection system ends the
   prompt with *hard* tokens (Coconut's `<bot>`/`<eot>` delimiters,
   arXiv:2412.06769; gist tokens mid-prompt, arXiv:2304.08467; BLIP-2's
   soft-tokens-then-text-prompt, arXiv:2301.12597). KL/fluency anchoring of
   the conditioned distribution to the unconditioned one preserves fluency
   at fixed information content (Wingate et al., arXiv:2210.03162; Plug &
   Play with Prompts, arXiv:2404.05143). Zero-init scalar gates on injected
   pathways (Flamingo tanh gates, arXiv:2204.14198; LLaMA-Adapter zero-init
   attention, arXiv:2303.16199) protect the base model, with a documented
   "silent phase" pathology (ControlNeXt, arXiv:2408.06070). Qwen has no
   BOS; attention sinks form on the physically-first token (StreamingLLM,
   arXiv:2309.17453) — keep hard tokens at position 0.
2. **Selection headroom.** Verifier-argmax vs plain SC at N=4, p≈0.8 is
   undetectable; the winnable claims are (a) verifier-weighted SC ≥ plain
   SC (DIVERSE's voting verifier, arXiv:2206.02336), and (b) selection on a
   medium-difficulty stratum (pass@1 ∈ [0.25, 0.625]; Snell et al.,
   arXiv:2408.03314). Power: McNemar mid-p needs ~250 paired rows for a
   7–8 point effect. Controls: max-logprob baseline (self-preference is
   largely a perplexity effect), score↔length and score↔vote-count
   correlations.
3. **Coverage-controlled calibration.** Split conformal: fit the score
   function on split A, choose τ on split B as an order statistic —
   coverage ≥ c guaranteed in expectation with k = ⌊(nB+1)(1−c)⌋; realized
   coverage ~ Beta, so target nominal ~0.35 to guarantee ≥0.25 with ~95%
   probability at nB=128 (SSBC, arXiv:2509.15349; Angelopoulos & Bates,
   arXiv:2107.07511). Gate on the risk–coverage curve (AUGRC, NeurIPS 2024
   arXiv:2407.01032), not a single matched point; McNemar mid-p for points.
   One score per anchor (candidates are dependent). Logistic stacker is
   the right combiner; add mean-logprob as a fourth feature.
4. **Interface & lanes.** Queries should only ever originate from real text
   tokens (KV-style injection; prefix as attended memory, never as the
   position the model continues from — Petrov et al., arXiv:2310.19698
   show a prefix biases attention without re-patterning it). Lane
   diversity must be structural: asymmetric inputs and/or winner-take-
   gradient (sMCL, arXiv:1606.07839); penalties on zero-init outputs are
   provably inert.

## 2. The v8.0 design

### 2.1 Claim decoupling (the deepest change)

v6.0 entangled generation quality with selection quality: heads could only
select among workspace-channel candidates, so a broken channel doomed the
selection claim. v8.0 separates three claims with three surfaces:

- **Claim S (selection):** workspace-informed heads improve candidate
  *selection/weighted voting* over self-consistency **on the causal-channel
  pool** — the same healthy pool SC votes over. Channel quality cannot
  contaminate this claim; heads score arbitrary token sequences via
  teacher-forcing under the workspace prefix.
- **Claim A (abstention):** conformally calibrated heads beat mean-logprob
  abstention on the risk–coverage curve, same causal pool.
- **Claim G (generation):** the *repaired* workspace channel reaches causal
  parity (non-inferiority) with zero scaffold leak, while the latent
  read-out remains causally live (ablation non-null). No "beats" claim —
  parity + liveness only, at this scale.

### 2.2 Channel repair (Claim G machinery)

- **R1 — remove the mid-sequence EOS.** The answer channel never begins
  with `bos_token_id`. Teacher-forcing predicts answer token 0 from the
  last hard cue token; decoding starts from the cue.
- **R2 — sandwich layout.** Both channels share one prompt builder:
  context loses its trailing `### Response` header at encode time, and the
  model appends hard **cue tokens** (`\n### Response\n`) itself. Causal
  channel: `[context][cue] → answer`. Workspace channel:
  `[context][prefix][cue] → answer`. The channels are now a matched pair
  differing *only* by the latent prefix; position 0 stays a hard token
  (sink insurance); the prompt always ends in hard tokens (register
  insurance); the prefix sits adjacent to the cue (influence).
- **R3 — one register.** Context, cue, and answer tokens all carry
  MODE_CAUSAL. Only prefix tokens carry MODE_SYNTHESIS/MODE_PRIVATE. The
  causal channel is literally the workspace channel with the prefix
  removed — which is also the read-out ablation.
- **R4 — gated prefix.** The whole prefix is scaled by tanh(α), α
  initialized at 0.05 (small positive: the ControlNeXt silent-phase
  contingency, pre-applied), logged at every eval step. Memory/summary
  projections move from zero-init to small random init (0.02) — the zero
  lives in the gate now, so diversity and orthogonality gradients are live
  from step 0.
- **R5 — KL-to-causal anchor.** On anchor steps, auxiliary loss
  β·KL(workspace-conditioned answer logits ‖ detached causal answer
  logits), β = 0.05. Pre-registered tension: too-strong β nulls the
  mechanism; the read-out-liveness gate is the canary.
- **R6 — decode hygiene, not fix-of-record.** Scaffold stop-check at
  decode is *not* added; the leak gate stays a pure measurement so the
  repair is falsifiable. (Banning tokens would hide a failed repair.)

### 2.3 Candidate machinery (Claims S and A)

- Pool: N=8 causal-channel candidates — 1 greedy + 7 samples at T=0.8
  (single moderate temperature; v6.0's ladder added zero new valids on
  s17 and complicates head calibration).
- Heads score every candidate: teacher-force under the (gated) prefix →
  features → 3 head probabilities; publish score =
  σ(w·[commit, risk, verifier, exp(mean-logprob)] + b) — the fourth
  feature makes the logprob baseline nested in the combiner.
- **Weighted SC:** candidates cluster by answer signature; each cluster's
  weight is the sum of its members' publish scores; argmax cluster wins.
  Plain SC = unit weights. Ties break by summed publish score, then mean
  logprob (tie rates reported).
- Policy-head training and calibration both harvest from the SAME pool
  configuration (causal channel, T ∈ {0, 0.8}) — on-policy with the audit,
  answering the verifier-OOD warning.

### 2.4 Conformal calibration (Claim A machinery)

- 256 validation anchors, split A=128 / B=128 by hash of episode id.
- Fit combiner (w, b) on split-A candidate rows (class-balanced logistic,
  L2, as v6.0).
- On split B: one score per anchor — the publish score of the
  greedy-selected candidate under the fitted rule. τ = k-th smallest with
  k = ⌊(nB+1)(1−c_nominal)⌋, c_nominal = 0.35 (SSBC-inflated so realized
  coverage ≥ 0.25 with ~95% probability). If >20% of split-B scores tie at
  the k-th order statistic, use the pre-registered randomized tie-break.
- Fallback: if the combiner cannot fit (class floor), the run reports
  calibration failure; no midpoint fallback (that road led to seed 29).

### 2.5 Lanes (last chance, structural)

- Asymmetric inputs: lane 0's brief pools the first half of context
  positions, lane 1's the second half (structural, certain, cheap).
- Random-init lane projections at distinct scales + live orthogonality
  pressure (the existing 0.05 squared-cosine penalty now has gradient).
- Pre-registered exit: if mean lane-summary cosine ≥ 0.90 again (would be
  failure #8), lanes are demoted to single-lane in v8.1 and the paper's
  lane-diversity sections go negative-evidence. No further lane fixes.

### 2.6 Infrastructure

- **KV cache** in the backbone decode paths (cache post-RoPE keys;
  absolute position ids threaded; training paths unchanged). Required: it
  buys the 320-row audit. CI: cached-vs-uncached logit equivalence and
  greedy-token identity tests.
- **Audit n = 320**: all 256 test behavior anchors (64/family) + 64
  text-to-SQL rows (sql_exact grader). Difficulty bins from the same 8
  samples: easy pass@1 > 5/8, medium ∈ [2/8, 5/8], hard < 2/8 —
  label-dependent stratification, disclosed; headline on the medium bin,
  all bins + pooled reported.
- **Canary** now exercises the workspace decode path and counts scaffold
  leaks during training (v6.0's canary was blind to the emission-path
  leak); all 8 canary rows graded with their own family grader.
- Manifest code shas regenerated at zip time (v6.0 shipped stale shas
  after the hotfix).
- Budget: train ≤ 6.5 h + audit ≤ 3 h per seed, two seeds concurrent on
  2×T4; hard cap unchanged (8.5 h train runtime guard). Estimated total
  wall clock ≈ 9–10 h, one Kaggle session, ~10 of the fresh 30 h.
- Trainable-parameter delta vs v6.0: +1 scalar (gate α) + re-inits; ≤0.1%
  — disclosed, budget-matched.

## 3. Gate battery (24 gates; v6.0 had 20)

Training & heads (6): training_complete · calibration_fitted (combiner
fitted AND τ strictly inside the B-split score range with tie mass ≤20%) ·
policy_heads_trained_on_policy (16/family floor) ·
clean_commit_ranked_above_corrupt · corrupt_risk_ranked_above_clean ·
corrupt_verifier_ranked_above_clean.

Claim G (6): no_scaffold_leak (= 0 on workspace-channel greedy, n=320) ·
channel_parity_accuracy (workspace greedy graded acc ≥ causal greedy −
0.05) · channel_parity_f1 (workspace greedy F1 ≥ 0.9 × causal) ·
readout_live (prefix ablation drops workspace greedy graded acc ≥ 0.05) ·
gate_alpha_open (|tanh α| ≥ 0.01 at final checkpoint) ·
quality_pass_rate ≥ 0.70 (workspace greedy).

Claim S (5): **weighted_sc_beats_sc (HEADLINE)** — weighted SC > plain SC
on the medium bin per seed; replication verdict additionally requires
pooled McNemar mid-p < 0.05 (one-sided) · weighted_sc_no_worse_overall
(all-rows weighted SC ≥ SC − 0.02) · argmax_selection_ge_greedy (≥ greedy
− 0.02) · selection_headroom_present (≥10% of medium-bin rows have
exactly-one-correct among N=8; if this fails the S-claim is void-for-
headroom, not failed) · heads_not_logprob (|Spearman(publish score, mean
logprob)| < 0.95 on audit candidates).

Claim A (5): coverage_in_band (audit coverage ∈ [0.20, 0.50]) ·
abstention_beats_logprob_augrc (AUGRC heads < AUGRC logprob in ≥90% of
1000 paired bootstrap resamples, per seed) · selective_accuracy_matched
(heads ≥ logprob at matched coverage, direction; mid-p reported) ·
risk_bound (Clopper–Pearson 95% UCB on selective risk ≤ 0.45) ·
safe_abstention_probes (commit-expected commit rate ≥ 0.50 AND abstention
accuracy ≥ 0.70).

General (2): complete_answer_rate ≥ 0.50 · lanes_materially_distinct
(cosine < 0.90; failure triggers the lane exit, not run failure).

Probe content (semantic_probe_accuracy ≥ 0.75) rides inside
channel_parity: parity at causal levels implies it; reported separately.

## 4. Pre-mortem — predicted failure modes and preregistered responses

| # | Predicted problem | Detector | Preregistered response |
|---|---|---|---|
| P1 | Cue-strip fails on some prompts (no trailing `### Response`) | build-time assertion over every row (local test, all splits) | fixed before ship; cue appended unconditionally if strip misses |
| P2 | Prefix influence attenuates in sandwich position | readout_live gate | v8.1: per-layer gates / prefix adjacency shift; not patched mid-run |
| P3 | KL anchor nulls the mechanism | readout_live fails while parity passes | v8.1 drops β to 0; KL was the dispensable half of the repair |
| P4 | Gate α silent phase outlasts 4224 steps | gate_alpha_open + per-eval α logging | α init 0.05 pre-applies the fix; if still closed, v8.1 cross-normalizes instead of gating |
| P5 | No selection headroom even on medium bin (SC ≈ oracle) | selection_headroom_present | S-claim declared void-for-headroom (a design result, not a model failure); program re-scopes to A+G |
| P6 | Heads OOD on causal-pool candidates | score↔logprob corr + calibration bal. acc | policy/calibration already harvest the audit pool (on-policy); residual = monitored |
| P7 | Conformal τ lands in a tie mass | calibration_fitted tie check | randomized tie-break, preregistered |
| P8 | Coverage band missed on one seed (Beta draw) | coverage_in_band per seed | one-seed miss = noise (documented); two-seed miss = calibration transfer failure → shift investigation, never re-tuned τ |
| P9 | Lanes converge again (#8) | lanes gate | terminal: single-lane v8.1, lane sections go negative-evidence |
| P10 | KV cache numerical drift (bf16) | CI equivalence tests + audit spot-check | audit falls back to uncached decode with a logged flag |
| P11 | Audit overruns 3 h | per-row timing log | preregistered truncation order: keep all 256 anchors, drop SQL tail; truncation disclosed |
| P12 | SQL rows distort the medium bin | per-family bin composition report | headline also computed anchors-only as a sensitivity check |
| P13 | Repaired channel shifts head-feature distribution vs v6.0 | — | no carryover risk: heads retrained on-policy from scratch |
| P14 | fp16/loss-scale instability recurrence (T4 history) | training guards | same bf16-auto + loss-scale guards + max_skipped_updates as v6.0 |
| P15 | Weighted SC wins via length bias, not verification | score↔length correlation report | if corr > 0.5 and SC gate passed, claim demoted to "selection signal confounded with length"; length-controlled re-analysis in report |

## 5. Fallback ladder (binding)

1. **Full pass (G + S + A cores):** the plain-LoRA control runs
   immediately (budget-matched, same N-sample machinery), then external
   benchmarks per the standing draft prereg. Nothing else authorizes them.
2. **G passes, S fails with headroom present:** heads don't out-select
   voting — selection re-scopes to negative evidence; A-claim stands on
   its own if its gates pass.
3. **G passes, S void-for-headroom (P5):** selection question moves to the
   benchmark kit (harder rows) — gated behind an A+G pass, not rerun at
   this difficulty.
4. **G fails (parity or leak):** the latent-workspace generation channel
   is falsified at this scale even with a repaired interface; the program's
   generative claim ends (paper: negative evidence); heads/abstention work
   continues on the causal channel only.
5. **A fails on both seeds with G passing:** abstention claim dies; v8.0
   reports as the terminal negative for calibrated selective prediction at
   0.6B.

## 6. Claim boundary

No production claim. Small-n disclosed everywhere (320 audit rows/seed;
128-anchor conformal split). The medium-bin headline is a conditional
claim, conditioned on model-estimated difficulty computed identically for
all arms from the same 8 samples. Two seeds (17, 29), fixed data, fixed
base revision; nothing tuned on test rows; τ and w never touch audit data.
External benchmarks remain gated behind the ladder's rung 1.

## 7. Build record (2026-09-02)

The bundle is built and packaged; the run has not been launched.

| Item | Value |
| --- | --- |
| Bundle | `artifacts/kaggle/hlwm-v8.0/hlwm-v8.0-candidate-bundle.zip` (24,057,834 bytes) |
| Bundle sha256 | `3139b6bb2ba138dd6c9b6624820edb1e3287a829994f59d7624ce7cae08cc234` |
| Notebook | `artifacts/kaggle/hlwm-v8.0/embel-hlwm-v8.0-kaggle-2xt4.ipynb` |
| Notebook sha256 | `8a3bb5ee79153d72b7a8a70125bd9fb7ca7fd8ba866a5579526f577fab089f93` |
| Builder | `scripts/build_hlwm_v80_bundle.py` |
| Tests | 58/58 |
| Supersedes | v6.0 bundle `39477eb5…` (both seeds failed Session E) |

Shipped `code_sha256`:

- `__init__.py` — `15eee54eb55eb69558a57fe5b0141d6d555c010166ed5f82c9bff1b7e26c6028`
- `data.py` — `69044bc2b58aae8bc20761c05121a1fcc38576c429c802364a71bc1299a30eb0`
- `evaluate_checkpoint.py` — `c0b9273d7a42fe28904a90aaa8708b373f1a910b5c664954e0c562ab3424581e`
- `inference_hlwm.py` — `5fc80eefd30d783120b0cd16e08fdd7ca55c7da3d989d2e9e804d478a7ef43b7`
- `modeling_hlwm.py` — `df4ba821bfe8ad4dc64e51681408c82d2425bcc6f67836b5caeba32c4854bcbc`
- `semantic_grading.py` — `87c0061f9f5f37c8e75efdfe89ca20db40b9d2237f90852e5cfee8b58670b0fc`
- `test_modeling_hlwm.py` — `d5b8eb7c9e847aba80d65e2ea6dcf5d20bd6cde3221a191b274d7ac30bb6aaf6`
- `train_kaggle.py` — `14901478e7eb673eb8a07a041fac7ed86068d730c291fc3db47cd4318e592bf0`

### Changes made during packaging, beyond the design above

Four things surfaced while building that the design sections did not cover.
They are recorded here because each one changes what the run will measure.

1. **The leak gate could have certified a broken channel.** `output_quality`
   tested for `Human:`/`Assistant:` scaffold only at the *start* of the answer,
   but v6.0's scaffold followed real answer text — it appeared after the
   injected document boundary, not before it. A partially-working R1 could
   therefore have produced mid-answer scaffold and still passed
   `no_scaffold_leak`, which is the gate that certifies the repair. Detection
   now matches a turn marker opening any line (`TURN_SCAFFOLD`, shared verbatim
   between the audit and the trainer), anchored to line starts so prose that
   merely mentions a role is not flagged. Two tests pin both directions.
2. **`multiple_choice` removed from the light-grader filter.** It was listed in
   `LIGHT_GRADERS` and `CANARY_LIGHT_GRADERS`, but the shipped
   `semantic_grading.py` has no such grader and no row in the corpus uses that
   type. Had either become true, the rows would have been admitted to the audit
   population and graded incorrect unconditionally.
3. **The bundle archive is now deterministic** (fixed zip entry timestamps).
   Previously a no-op rebuild produced a different `bundle_sha256`, which made
   the recorded hash useless for confirming that Kaggle received the reviewed
   bytes. Verified by building twice and comparing.
4. **The notebook verifies `manifest.json` against the shipped files before
   training starts.** This is the v6.0 stale-sha defect turned into a
   precondition: v6.0's manifest described pre-hotfix code, so the recorded
   hashes did not identify what ran. The check costs seconds and runs before
   the ~7 GPU-hour commitment.

### Source-of-record note

`artifacts/kaggle/hlwm-v8.0/bundle/hlwm_kaggle/` is the v8.0 source of record.
`experiments/kaggle_hlwm/` was deliberately **not** overwritten: it holds the
pre-hotfix v6.0 tree plus an unshipped `multiple_choice` grader, and the v6.0
hotfix already went into the bundle rather than back into it. Merging v8.0 over
it would have destroyed that divergence silently. `scripts/build_hlwm_v80_bundle.py`
reads from the bundle directory for this reason.

### Not done

- `paper/hlwm-paper.tex` still lacks the Study 8 (v6.0 Session E) results and a
  preregistered Study 9 for v8.0.
- The plain-LoRA control remains unrun; it is rung 5 of the ladder and is the
  decisive comparison for any positive result here.
