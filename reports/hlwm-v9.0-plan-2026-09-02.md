# HLWM v9.0 plan — "Necessity by construction" — 2026-09-02

Synthesized from a 22-agent research workflow (4 code-diagnosis agents, 6 literature
agents, 3 competing designs, 9 adversarial red-team reviews; 2 red-team agents
stalled and returned nothing — their lenses were covered by the surviving reviews of
the other designs). All three candidate designs FAILED red-team as written; this plan
is the merge that repairs every fatal flaw. Full evidence:
`/tmp/hlwm_v9_research.json` (424 KB); workflow journal under the session dir.

## 0. New root-cause findings (verified by hand against the shipped v8.0 bundle)

These change the Session F interpretation and are corrected in the Session F report:

**RC-1. The v8.0 liveness instruments were structurally broken; the inertness
conclusion survives only via channel parity.**
- The "read-out ablation" never removed the whole prefix: `disable_workspace_memory`
  slices `prefix[:, :synthesis_prefix_tokens]` (modeling_hlwm.py:2670-2671), keeping
  the 8 gated synthesis tokens in BOTH arms; only the 34 memory tokens were removed.
  Full-prefix inertness is bounded instead by channel parity (causal == prefix=None),
  gap ~0.01. Conclusion unchanged, instrument mislabeled.
- The prefix gate alpha COULD NOT have moved even if the prefix were useful: layer-0
  RMSNorm is scale-invariant (RMSNorm(g·u)=RMSNorm(u) up to eps/(2g²)≈2e-4 at
  g=0.05), so the attention-visible K/V content of the prefix is independent of
  alpha, and by Euler's homogeneity identity that branch contributes ~0 to dL/dα.
  The observed flat alpha is an inert instrument, not a closed valve. The channel was
  OPEN; the decoder ignored it via attention.
- The KL anchor's reference is this same model with `workspace_prefix=None` on the
  SAME inputs (modeling_hlwm.py:2435-2441), so its unique minimum over the prefix
  pathway is "the prefix changes the logits by exactly nothing." At CE convergence it
  is the only persistent gradient on the prefix and it points at no-op. (Doc bug: the
  docstring states the reverse KL direction.)

**RC-2. The validity-floor failure (Failure 6) is a BUG, not model incapacity.**
The harvest feeds `generate_hlwm_nbest` the training collator's fixed-length
right-padded context (data.py `_fixed`: pad to 256 with pad=eos, anchors are ~84-105
real tokens → ~150-170 pad positions inserted between prompt and response cue,
cue at RoPE ~256-259, generation from ~260 across a masked gap;
train_kaggle.py:1390-1406), while the audit encodes every prompt UNPADDED
(`encode_preserving_ends`, evaluate_checkpoint.py:253-266). The model decoded on an
input surface it never sees at audit — audit-time causal validity on anchors was
0.82 while harvest emissions were ~1-3% valid on generative families. The same padded
surface feeds calibration pools, which also explains part of the seed-29
validation→test coverage shift (0.359 → 0.1875). Fix is bug-class: one shared encode
primitive for harvest, calibration, and audit.

**RC-3. Difficulty was never a parameter.** Anchor generators are single-operation
templates with hard-coded operand ranges (`behavior_anchor_rows`: no difficulty
argument; the episode "difficulty" field is the constant string "deterministic").
Bimodality is therefore expected, not accidental: no knob exists between "one
multiply" and text-to-SQL.

**Literature convergence (three independent sweeps agree):**
- The dead read-out is textbook posterior collapse under a powerful decoder
  (Variational Lossy Autoencoder 2017; Wang/Blei/Cunningham 2021 non-identifiability).
  "Encourager" fixes (KL annealing, free bits, MI bonuses) do not apply to a frozen
  decoder; the only FORCERS are information asymmetry (the latent carries information
  the decoder input lacks) and dense latent supervision.
- Every published success of gated insertion into a frozen LM had the gated signal
  carrying information ABSENT from the decoder's other inputs (Flamingo: images;
  gisting/AutoCompressors: the compressed text is removed; Coconut: the CoT text the
  latent replaces is deleted by curriculum).
- Latent-reasoning gains are real at OUR scale — GPT-2 124M — but only with dense
  latent supervision or shortcut removal (CODI self-distillation EMNLP 2025; SIM-CoT
  ICLR 2026; ICoT-SI stepwise internalization 2024; CoLaR 1-1.5B NeurIPS 2025).
  Pause/filler tokens without such signal do NOT help at small scale.
- Learned abstention beats logprob only on long answers or domain mixtures /
  distribution shift (Kamath 2020 selective QA; ASPIRE 2024; logprob is strongest
  exactly on short verifiable answers — v8.0's regime). Verifiers must be trained
  on-policy on the generator's own samples (Cobbe 2021 → V-STaR 2024 → LiLaVe 2025);
  teacher-forced fitting inverts under shift, which is precisely Failure 5.

## 1. Design

One sentence: **give the workspace information the answer channel does not have, so
ignoring it is no longer a loss minimum; supervise the latent densely; fix the
harvest bug; band the difficulty; realign the abstention metric.**

Honest scoping, stated up front (a reviewer will say it, so we say it first): the
masked-row setting is CONSTRUCTED — we withhold input from one channel to make the
latent necessary. A pass demonstrates the mechanism (the latent channel can carry
task-critical information through the frozen LM — exactly what Studies 8-9 falsified
in the redundant setting). It is NOT a capability claim; capability still requires
the full-context plain-LoRA control, which stays gated.

### M1. Information-asymmetric channels (Claim L core)

- Collator emits per-row `input_ids` (full context) AND `answer_input_ids` (masked
  context: the answer-determining premise span replaced by a fixed `[…]` token).
  Premise spans are emitted TOKEN-LEVEL BY THE GENERATOR at data-build time (the
  anchor generators are templates — they know exactly which tokens carry the
  operands), not by char-span regex mapping. This kills the truncation-fragility
  fatal from red-team.
- Workspace builds from full context (modeling_hlwm.py:2281 unchanged); teacher
  forcing and decode for the workspace channel condition on masked context + prefix;
  the masked-causal arm (masked context, no prefix) is the killer baseline and must
  crater; full-context causal is the skyline.
- Applied to a masked stratum: 50% of anchor-family joint-phase workspace batches.
  Unmasked batches continue as v8.0 (so full-context behavior is preserved and the
  causal channel stays honest).
- **Leak integrity, four layers (fatal-flaw repair):** (1) build-time dataset assert:
  withheld token ids absent from `answer_input_ids`; (2) unit tests on the builder;
  (3) runtime batch assert in the collator; (4) audit-time leak-probe GATE:
  token-level overlap of withheld premise vs masked prompt must be 0 on every masked
  row. Leak gate failure voids the masked claims (ladder rung 0) — it does not kill
  the rest of the audit.
- **No gold-seeded canvases on masked rows (fatal-flaw repair):** prefixes consumed
  by masked-row answer CE are built with the full reverse pass from pure noise (the
  audit path), not the q_sample-noised supervised path whose t=T keep-prob leaves
  ~39% verbatim target tokens in the canvas.
- **KL anchor removed on masked rows** (its reference cannot answer there; it would
  push away from answering). Optional 0.02 KL on unmasked rows only.
- Lane demotion executed: 1 lane. Prefix = 16 windows + 1 summary + 8 synthesis =
  **25 positions** (all layout/VRAM math uses 25; fatal-flaw repair).
- Gate alpha: kept, logged, REFRAMED as diagnostic-only telemetry (RC-1); no gate on
  alpha. `readout_ablation` fixed to slice the FULL prefix; a memory-only slice is
  kept as a descriptive second arm.

### M2. Dense latent supervision (small-scale-proven signal)

SIM-CoT/CODI-class auxiliary loss: a probe head reads ONLY the prefix positions and
is trained to decode the WITHHELD PREMISE TOKENS (never the gold answer —
answer-injection fatal repaired: supervising the answer would let the sidecar solve
the task and the LM transcribe it, making Claim L unfalsifiable). Cost control: score
only over the premise token ids (no 151,936-vocab softmax), applied on masked rows
only, weight 0.1. Declared canary: if aux premise-decoding accuracy is high while
Claim L fails, the finding is "channel carries content the frozen LM cannot use" —
reported as such (that outcome retires the mechanism at rung 2 with better
attribution than v8.0 ever had).

### M3. Attribution control (fatal-flaw repair — killer baseline for the canvas)

A GIST-ENCODER control arm in the audit: same 25-token prefix budget produced by a
trivial mean-pool projection of the full context (no canvas, no diffusion, no
refinement), teacher-forced/decoded identically on masked rows. If gist ties the
workspace canvas, the canvas is retired and the claim re-scopes to "latent memory
channel" (rung 3). This is the control that separates "the architecture computes"
from "any compression carries bits."

### M4. Harvest + verifier repair (Claim V)

- ONE encode primitive (unpadded, preserve-ends) shared by harvest, calibration, and
  audit (RC-2 fix). Expected effect: per-candidate validity on anchors rises toward
  audit-observed causal validity (0.82 on easy/medium), making the 16/family floor
  arithmetically reachable at 512 attempts.
- The floor becomes a GATE (`validity_floor_met`), closing the v8.0 prereg-vs-code
  gap (the plan said floor, the code shipped `trained`≥8/8).
- Verifier trained on-policy only: features/labels exclusively from sampled
  emissions on the unpadded surface, pairwise within-anchor objective
  (V-STaR/DPO-style) on correct/incorrect pairs from the same prompt. Gate:
  verifier margin on GENERATED audit emissions > 0 on both seeds (kills the
  teacher-forced inversion mode).
- Disclosed fallback if a family still starves: STaR-style hint-conditioned
  positives, labeled as off-policy in the artifact.

### M5. Difficulty banding (RC-3 fix; powers everything)

- Parameterize anchor generators: chain length 1-3 operations, operand digit count,
  distractor count (the hard-coded literals become arguments).
- Calibrate ONCE against the FROZEN v8.0 seed-17 checkpoint as difficulty estimator
  (role-separated: the estimator is not the subject of any gate); target medium band
  pass@8 ∈ [0.25, 0.625]; freeze the strata into the bundle at build time.
- Audit composition: 224 banded anchors (target ≥120 medium) + 64 SQL (risk-only
  domain; NOT in any selection statistic — the shipped `answer_signature` has no
  vote-equivalence for near-unique strings; fatal-flaw repair) . Code rows from
  data/combined are NOT admitted this run (equivalence-by-execution is real work;
  deferred, disclosed).
- Band drift disclosed: trained model's pass@8 on the frozen strata is reported;
  a `headroom_present` void-check (≥10% of medium rows with 1-7 of 8 valid) guards
  any selection statistic.
- Claim S stays retired. A single report-only weighted-vs-plurality comparison runs
  on banded anchors (it is the preregistered "harder rows" continuation), gated to
  VOID unless ≥25 discordant pairs exist (power floor from McNemar arithmetic:
  25 pairs at 70/30 split gives mid-p<0.05 ~80% of the time). No headline rests on it.

### M6. Abstention realignment (Claim A2)

- Conformal coverage machinery kept exactly (it worked): target 0.40 (inflated from
  0.35 — the v8.0 seed-29 miss was 1.25 pts, and calibration now runs on the fixed,
  unpadded surface which removes the known score shift), band [0.20, 0.55].
- THREE-WAY split for validity (fatal-flaw repair): combiner fit on A (96), any
  score-form selection on B1 (64), conformal threshold on B2 (96), disjoint by
  episode-id hash. No adopt-if-wins on the threshold split.
- Score granularity repair: publish score kept CONTINUOUS for ranking (the sigmoid
  output ranks; the threshold only decides publication) + two new features with
  literature support at small scale: within-pool agreement count (semantic-consistency
  proxy) and length-normalized logprob. Leave-one-family-out fitting (Kamath-style
  domain-mixture regime — the regime where learned scores actually beat logprob).
- Preregistered statistics, arithmetic-checked (fatal-flaw repair):
  - `coverage_in_band` per seed.
  - `risk_ucb`: Clopper-Pearson 95% UCB ≤ 0.15 at the operating point, with the
    feasibility precondition n_published ≥ 20 (UCB at n=20, 0 errors is 0.139 —
    winnable; the v9 audit's expected coverage 0.2-0.4 of 288 rows gives n≈58-115).
  - Headline `abstention_beats_logprob_partial_augrc`: paired bootstrap on AUGRC
    restricted to coverage ∈ [0.05, 0.50] (the deployable region; full-curve AUGRC
    on a 59%-easy pool measured logprob's saturation, not abstention quality),
    heads win in ≥80% of 1,000 resamples per seed (0.90 was set with no power
    analysis; 0.80 at n=288 with the mixture design is the pre-computed achievable
    bar). Anchors-only parity is an acceptable declared secondary outcome (short
    verifiable answers are logprob's strongest regime — literature).

## 2. Budget (from MEASURED Session F numbers; fatal-flaw repair)

Measured: 5.20 s/step train; audit 18-19 s/row (anchors ~13s, SQL ~33s at 96-token
family budgets); harvest+calibration ~1.2 h combined.

| Phase | v9.0 | Cost |
|---|---|---|
| Train: 128 overfit + 768 local + 2304 joint = 3200 steps ×5.2s ×1.15 (aux+masked-row full-reverse overhead) | | 5.3 h |
| Harvest 512 attempts (unpadded, ~60% shorter prompts than padded surface) | | 0.5 h |
| Calibration 256 anchors × 8-pool | | 0.7 h |
| Audit 288 rows (224 anchors ~13s + 64 SQL ~33s = 1.4 h) + masked arms (masked-TF/decode ×3 arms on 112 masked rows ≈ +0.6 h) + gist control on masked rows (+0.3 h) | | 2.3 h |
| **Total** | | **8.8 h vs 9.0 ceiling** |

Guards (preregistered degradations, wall-clock-triggered, in order): (1) pool 8→6
temperatures in calibration; (2) audit row-priority list drops SQL rows first, never
masked anchors; (3) per-phase checkpoints so a truncated seed resumes instead of
voiding. Decode runaway guard: per-family max-new-tokens + newline stop + per-row
wall-clock cap with count of capped rows reported (the v6-style register pathology
would otherwise 6x the harvest cost — red-team arithmetic).

Step-count note: 3200 < 4224 is a disclosed change; the overfit/local/joint ratio is
preserved and Session F showed loss plateaus by ~2/3 of joint.

## 3. Gate battery (22)

Training/infra (6): training_complete · zero_skipped_updates ·
truncation_integrity (build+runtime asserts passed) · validity_floor_met (16/family,
now a real gate) · calibration_fitted (3-way split, τ interior, tie mass ≤20%) ·
canary_clean (marker rate 0 at every boundary from step 1536).

Claim L (6): masked_leak_zero (GATE, voids L if failed) · masked_causal_floor
(masked prefix-None accuracy ≤ 0.10 — confirms the withholding bit) ·
**latent_channel_live: masked workspace accuracy − masked prefix-None ≥ 0.30** ·
full_ablation_consistent (full-prefix ablation on masked rows ≥ 0.30 too — same
fact via the fixed instrument) · unmasked_parity (unmasked workspace within 0.05 of
causal — the v8.0 result must not regress) · aux_premise_decodable (probe accuracy
≥ 0.50 on masked rows; report-only diagnostic if L fails).

Attribution (2): gist_control_run · canvas_beats_gist (workspace − gist ≥ 0.10 on
masked rows; failure → rung 3 re-scope, not a program failure).

Claim V (3): verifier_margin_positive_on_generated (both seeds) ·
clean_commit_ranked_above_corrupt · publish_score_continuous (≥100 distinct values
across audit — kills the granularity failure silently).

Claim A2 (4): coverage_in_band [0.20,0.55] · risk_ucb_015 (with n≥20 precondition;
n<20 → gate VOID not failed, disclosed) · abstention_beats_logprob_partial_augrc
(≥80% of paired bootstraps, coverage-restricted) · safe_abstention_probes.

General (1): complete_answer_rate ≥ 0.50.

Every gate checked for arithmetic winnability in this plan (the two v8.0-class
unwinnable gates found by red-team — UCB ≤0.05 needing n≥59, and a harvest gate
contradicting its own band — are the reason this section exists).

## 4. Pre-mortem (detectors and responses fixed before the run)

P1 truncation leak → detector: leak gates; response: L voided, session reported as
infrastructure failure, no mechanism verdict. P2 masked rows unlearnable (workspace
AND causal-skyline both floor) → banding failed; L void with re-band, not mechanism
death. P3 aux dominates, prefix = premise copy, L still fails → "channel without
use" terminal finding (better attribution than v8.0). P4 harvest still starves a
family → STaR hint fallback, disclosed. P5 band drift re-saturates medium →
headroom void-check fires; L unaffected (L is not headroom-dependent). P6 decode
runaway → wall-clock caps, capped-row count reported. P7 coverage band missed again
→ Mondrian NOT introduced (n too small per red-team); pooled band widened as above;
miss on one seed = A2 fails that seed, per-seed rule unchanged. P8 gist ties canvas
→ rung 3 re-scope (a REAL result: compression suffices). P9 masked-row training
destabilizes unmasked parity → unmasked_parity gate catches; response: mask ratio
0.5→0.25 is the single preregistered retry knob (one retry, disclosed). P10 the
notebook crashes post-science again → every cell that consumes audit JSON is
wrapped in row.get() with a smoke test on a synthetic record in the unit suite.

## 5. Binding ladder

0. Leak/integrity gates fail → infrastructure failure; no mechanism verdict; fix
   and rerun authorized (this rung does not consume the mechanism's last chance).
1. `latent_channel_live` FAILS on both seeds (leak clean, floor confirmed) → the
   latent channel cannot carry even structurally-necessary information through the
   frozen LM at 0.6B. TERMINAL for the workspace generative mechanism — no v10 of
   this mechanism at this scale. The paper reports it as the completed falsification.
2. L passes one seed only → supported-not-replicated; one preregistered rerun of
   the failing seed authorized (seed variance is the program's documented pathology).
3. L passes both but `canvas_beats_gist` fails → canvas retired; claim re-scopes to
   "latent memory channel at matched budget"; paper says compression suffices.
4. A2 headline fails both seeds with V passing → abstention re-scopes permanently to
   coverage-control-without-superiority (guarantees, not wins).
5. L core + A2 pass both seeds → plain-LoRA control IMMEDIATELY (full-context rows
   only — masked rows would rig the comparison, stated in §1), then benchmarks.

## 6. Build checklist (fix hooks from code diagnosis, verified line refs)

1. data.py:43-100 + 489-501: generator-emitted premise token spans; collator emits
   `answer_input_ids`/`answer_attention_mask`; build+runtime asserts.
2. train_kaggle.py:526-543 (hlwm_loss) + modeling_hlwm.py:2188-2211
   (forward_hlwm signature): thread the masked pair; masked rows use pure-noise
   full-reverse canvases; KL skipped on masked rows.
3. modeling_hlwm.py:2670-2671: full-prefix ablation (keep memory-only slice as
   second arm); single-lane config; 25-token prefix everywhere.
4. NEW aux premise probe head (small, prefix-positions-only, premise-token scoring).
5. ONE shared encode primitive: harvest (train_kaggle.py:1390-1406), calibration,
   audit all call it (RC-2).
6. Anchor generator difficulty parameters + one-time banding script against the
   frozen v8.0 seed-17 checkpoint; strata frozen into the bundle.
7. evaluate_checkpoint.py: masked arms, gist control, partial-AUGRC, 3-way split,
   n-precondition gates, row-priority + wall-clock guards; fix the gates-cell debug
   print (KeyError root cause of the Session F crash).
8. Unit tests: leak battery, masked/unmasked channel equivalence when span empty,
   full-vs-sliced ablation, encode-primitive equality across call sites, gate
   arithmetic sanity (every gate winnable on a synthetic passing run).
9. Notebook: dry-run timing cell before the dual launch; manifest sha verification
   kept; smoke-test the post-audit cells on a synthetic record.

## 7. What this plan does NOT claim

No production readiness. No model superiority. The masked-row setting is a
constructed mechanism demonstration, disclosed as such. Claim S stays retired.
Benchmarks stay gated behind the plain-LoRA control, which stays gated behind rung 5.
If rung 1 fires, the honest sentence in the paper is: "with redundancy removed, dense
latent supervision, a repaired harvest, and difficulty-banded evidence, the latent
workspace still could not carry necessary information through the frozen substrate;
the mechanism is falsified at this scale," and the program's generative line ends.

## 8. Build-time amendments (frozen with the bundle, 2026-09-02)

Declared before the run; each is a concretization or repair of the sections
above, none loosens a gate.

**A1. Banding executes in-session, not at build time.** §M5 said "freeze the
strata into the bundle at build time"; the frozen v8.0 seed-17 checkpoint is
multi-GB Kaggle-side state that cannot be exercised locally. The banding pass
therefore runs as the notebook cell immediately BEFORE any v9.0 training step,
using a frozen procedure (8 causal decodes per row: 1 greedy + 7 at T=0.8,
pass@8 bins with the audit's exact thresholds, selection rule: medium up to
168 then hard then easy fill to 224). Estimator preference: the attached v8.0
seed-17 checkpoint; fallback: the pinned base model, disclosed in the strata
file. The selected ids are frozen to disk and the audit consumes them
verbatim, so the strata cannot drift with the audited model.

**A2. The gist control is trained, not audit-only.** An untrained gist arm
would make `canvas_beats_gist` a strawman comparison. Masked joint-phase
batches alternate the answer channel's prefix between the canvas and the gist
projection (step parity), so both arms receive equal training signal.

**A3. Score-form selection concretized.** §M6's "two new features" reduce to
one: the "length-normalized logprob" feature is arithmetically identical to
the existing exp(mean logprob) feature (the mean IS length-normalized), so the
added feature is within-pool agreement only. Split B1 selects between the
4-feature and 5-feature forms by partial AUGRC over the rule's own selected
candidates; ties prefer the simpler form. This is the Kamath-style selection
step, executed on its own split.

**A4. Masking shapes per family.** numeric: every operand replaced by "[?]"
(operators stay visible); unit: the quantity replaced (units visible);
ordering: every list value replaced; abstention rows are never masked (there
is no premise to withhold). The masked flag covers ~50% of maskable rows
(hash-derived): train 405/768, validation 132/288, test 215/432.

**A5. Counts and budget.** Anchors: train 1,024 / validation 384 / test 576
(banding pool). Audit 288 rows = 224 banded anchors + 64 SQL. Training 3,200
microsteps. Guards: train 6.5h cap, harvest 1.0h cap, audit 3.0h cap with
anchor-first row priority. Planned session ~9.5h against the 12h quota.

**A6. Governing document.** The notebook shipped inside the sha-verified
bundle is the executable preregistration; where prose and notebook differ,
the notebook governs (the Session F lesson, §4 of its report).

**A7. Session F instrument corrections carried in.** The full-prefix ablation
is shipped (`disable_workspace_prefix`; on unmasked rows it is read off the
causal arm, which is the identical computation); the memory-only slice remains
as a descriptive second arm; `prefix_gate_tanh` is telemetry only and no gate
reads it; the KL anchor is removed on masked rows and reduced to 0.02 on
unmasked rows.

## 9. Session G rung-0 event and hotfix (appended 2026-09-03, session light-of-the-embel-f3)

**Event.** The first v9.0 launch (Session G, Kaggle 2xT4, seeds 17/29)
completed all 3,200 training microsteps on both seeds (~4.5h), completed
policy collection (512 attempts each), then **crashed identically on both
seeds** inside `calibrate_commitment_policy`:
`ValueError: combiner expects [N, 3|4] features and N labels`
(train_kaggle.py:909, raised from the `for feature_count in (4, 5)` loop at
line 1222-1223). The final canary before the crash was healthy on the
integrity axes: prompt_leak_rate 0.0, format_marker_rate 0.0 on both seeds.

**Classification.** Ladder rung 0 (infrastructure failure before any gate
was evaluated): fix and rerun authorized; does not consume the mechanism's
last chance. No science number from Session G is interpreted.

**Root cause.** §A3's two-form selection fits 4- and 5-feature combiners, and
`publish_score`/config already accept five weights, but `fit_publish_combiner`
retained the Version 8.0 input guard `shape[-1] not in (3, 4)`. The unit suite
never exercised a 5-column fit, so 69/69 passed around the defect.

**Hotfix (2026-09-03).** Three changes, nothing else:
1. `train_kaggle.py:906/909` — guard widened to `(3, 4, 5)`; docstring names
   the fifth (within-pool agreement) feature.
2. `test_modeling_hlwm.py` — the combiner test now fits 3-, 4- AND 5-column
   forms (the calibrator's exact call), and asserts 6 columns are rejected.
3. Notebook training cell — the resume glob now falls back to raw
   `hlwm-v9.0-seed-{seed}/checkpoint-*.pt` (highest step) from an attached
   crashed-run output; previously it only matched the `-resumable.pt` name
   that the (never-reached) packaging cell creates.

Rebuilt via `scripts/build_hlwm_v90_bundle.py`; suite 69/69. Inside the zip
only `train_kaggle.py`, `test_modeling_hlwm.py` and `manifest.json` changed;
all five data entries are CRC-identical to the launched bundle (the
preregistered data surface is untouched).
- bundle zip sha256: `a107c493014ca4fc1cdc6430352a5a696b32e34f4465ec01ad11be2fa59890b8` (was 084b3c25)
- notebook sha256: `e96923064aded0b38e718d2491727b17f4ec1c2592b1538f8b878b60fdb4c2e3` (was 21aeba43)
- train_kaggle.py sha256: `cdace14b...` (was c05cae62), test_modeling_hlwm.py `3d8bb435...` (was 7591be98)

**Salvage protocol for the relaunch.** Both seeds saved a step-3200 resumable
checkpoint (`checkpoint-step-003200.pt`, with optimizer, scaler and
python/torch/cuda RNG state) BEFORE the crash. Attach the crashed Session G
version's output as an additional input dataset, keep every other attachment
identical, replace the bundle dataset with the a107c493 zip and the kernel
with the e96923064 notebook, and rerun: the training cell resumes at
completion (`resumed_at_completion`), skips the training loop and the overfit
gate, and proceeds directly to policy collection -> calibration -> audit
(~4.5-5.5h instead of ~9.5h). A fresh full run with the fixed bundle is the
fallback if the checkpoint attachment is impractical.

**Non-blocking observations from the crashed logs** (recorded for comparison,
not interpreted): policy-collection validity floor unmet on both seeds
(41/512 and 43/512 valid; numeric and unit families far below the 16/family
floor at the banded difficulty), abstention-family grader accuracy ~0.94,
commit-head separation after fit healthy on both seeds.

**Operational hazard discovered while rebuilding.** This repo lives under
`~/Documents`, which iCloud had partially evicted ("Optimize Mac Storage"):
13,118/13,156 venv `.py` files were dataless, which made `pytest` hang for
20+ minutes at import (blocked in per-file cloud restores). After forced
rehydration the same suite runs in 2.9s. Before any future local build, check
`ls -lO` for `dataless` under `.venv` and bulk-warm first.
