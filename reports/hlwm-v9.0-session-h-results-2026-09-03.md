# HLWM v9.0 "Necessary Channel" — Session H results (2026-09-03)

**Verdict: BOTH SEEDS FAILED. Preregistered binding rung 1 — TERMINAL for the
workspace generative mechanism.** The notebook's own verdict cell:

> "rung 1: latent_channel_live failed on both seeds - the latent channel
> cannot carry structurally necessary information at 0.6B; TERMINAL for the
> workspace generative mechanism"

This is the first v9.0 run to complete the entire pipeline (train → collect →
calibrate → audit → verdict → packaging). The instrument was clean; the
reading was zero.

## 1. Provenance

- Kernel: `karthikyaddanapudi/notebooka2efd2fa03` (2×T4, seeds 17/29).
  Bundle `a107c493…` (Session G hotfix), notebook `e9692306…`; in-session
  suite 69/69; `bundle verified: all 9 code shas match`.
- **Salvage resume**: both seeds resumed from Session G's step-3200
  checkpoints (`resumed_at_completion`), skipping the 4.5h training phase.
  RNG restoration was exact: policy-collection family-valid counts are
  digit-identical to Session G (s17 41 = 16/6/16/3, s29 43 = 16/9/12/6).
  The salvage is bit-equivalent to an uninterrupted run, not an approximation.
- Session G training metrics (3,021,250 / 2,972,385 bytes — byte-matching the
  crash-time heartbeats) archived at `artifacts/kaggle/hlwm-v9.0/session-g/`;
  Session H audit artifacts at `artifacts/kaggle/hlwm-v9.0/session-h/`.
- Banding: `estimator: "pinned-base"` (v8.0 checkpoint output was not
  attached). Pool 576 → easy 31 / medium 102 / hard 443; selected 224;
  `medium_floor_met: false` (disclosed; the audit pool skews hard).

## 2. Gate battery (22 + verdict): 15 passed / 7 real failures / 1 artifact

**Failed (7 real, identical on both seeds):**

| Gate | s17 | s29 | Bar |
|---|---|---|---|
| latent_channel_live | 0.000 − 0.000 = 0.000 | same | masked workspace − floor ≥ 0.30 |
| aux_premise_decodable | probe 0.297 | 0.334 | ≥ 0.50 |
| full_ablation_consistent | 0.000 | 0.000 | workspace − full-ablation ≥ 0.30 |
| canvas_beats_gist | 0.000 − 0.000 | same | ≥ 0.10 |
| validity_floor_met | 41/512 valid | 43/512 | ≥16/family (numeric 6/9, unit 3/6) |
| verifier_margin_positive_on_generated | −0.277 | −0.227 | > 0 |
| abstention_beats_logprob_partial_augrc | bootstrap 0.746 | 0.792 | ≥ 0.80 |

**canary_clean "failed" is a resume artifact, not a leak.** The audit reads
canaries from the checkpoint-directory `metrics.jsonl`; the resumed run's file
holds only the `resumed_at_completion` marker (46 bytes), so `canary_seen==0`
→ `None` → recorded False (`canary_boundaries_seen: false` in gate_notes).
Certified out-of-band on Session G's training metrics with the audit's exact
rule (step ≥ 1536, `format_marker_rate > 0`): **7 canary records per seed,
0 dirty, 0 prompt leaks → canary_clean TRUE on both seeds.** Effective
failure count is therefore 7, and no leak-class failure occurred anywhere:
audit `prompt_leak_rate` 0.0, `no_scaffold_leak_rate` 0.0, masked
`leak_rows` 0/99 on both seeds.

**Passed (notable):** masked_leak_zero, masked_causal_floor (floor exactly
0.0 — necessity by construction held), unmasked_parity (gap −0.037 / 0.000;
s17's workspace arm was *better* than causal, 0.508 vs 0.471),
calibration_fitted, coverage_in_band (0.295 / 0.267 — the conformal
order-statistic fix killed the coverage strangle), risk_ucb_015 (0.072 /
0.038), safe_abstention_probes (1.00/1.00 commit and accuracy),
publish_score_continuous (329 / 372 distinct scores — the degenerate
threshold stays dead), clean_commit_ranked_above_corrupt (+0.354 / +0.418),
complete_answer_rate 0.997, truncation_integrity, training_complete,
zero_skipped_updates.

## 3. The headline result (Claim L / Claim C)

The one question v9.0 was built to answer: with the premise physically
withheld from the answer channel, can the latent workspace carry it?

- **Masked workspace accuracy: 0.000 on 99/99 rows, both seeds.**
- Floor (no prefix): 0.000 — the masking is airtight; the question was
  answerable only through the channel.
- Trained gist control: 0.000 — no attribution confound; nothing generative
  moved on masked rows.
- **Premise probe: 0.297 / 0.334 (bar 0.50).** The latent state contains
  *partial, weakly decodable* premise information, but the generative path
  extracts none of it. This is the same shape as v6.0's diagnosis (selector
  near-oracle, channel corrupt) and v8.0's (parity by irrelevance), now
  measured on a leak-proof, difficulty-banded, unpadded instrument with the
  ablation removing the entire prefix: **the information asymmetry design
  did not induce the channel to carry the payload.**

Verifier margin on generated candidates is negative on both seeds (−0.277 /
−0.227): the verifier head ranks its own generated-valid candidates *below*
teacher-forced references, consistent with the generative channel emitting
low-grade candidates at banded difficulty (validity 8%).

## 4. What survived: the calibrated-abstention machinery (again)

Operationally flawless for the third consecutive study, and for the first
time directionally better than the logprob baseline on both seeds:

| Metric | s17 | s29 |
|---|---|---|
| Published rows | 85/288 (cov 0.295) | 77/288 (cov 0.267) |
| Selective accuracy | 0.976 | **1.000 (77/77 correct)** |
| Risk UCB95 | 0.072 | 0.038 |
| Partial AUGRC heads vs logprob | 0.0128 vs 0.0156 ✓ | 0.0073 vs 0.0114 ✓ |
| Bootstrap partial win | 0.746 | 0.792 (bar 0.80) |
| Full AUGRC heads vs logprob | 0.127 vs 0.141 (win 0.994) | 0.139 vs 0.126 (win 0.039) |

The preregistered gate (partial-band bootstrap ≥ 0.80) failed on both seeds,
so the abstention claim is NOT confirmed — but the effect direction agreed
across seeds in the preregistered band for the first time in program history
(s29 missed by 0.008). Selective prediction remains the program's only
consistently positive thread.

## 5. Ladder application

Rung 1 is defined in the plan and notebook as **terminal for the generative
mechanism**: no further seed-runs, re-tunes, or design variants of the
workspace-as-generative-channel claim at this scale are authorized by this
preregistration. The claim family that dies here: "a diffusion-canvas latent
workspace can synthesize/carry answer content that the causal path cannot."
What the program can still assert from v9.0's clean instrument:

1. **Negative (now with structural necessity):** at 0.6B with matched budget,
   the latent prefix carries no generatively usable information even when it
   is the only path (0.000 over 198 masked row-evaluations), while containing
   probe-detectable premise traces (~0.30) — a posterior-collapse-shaped
   outcome robust to every instrument correction accumulated since v5.5.
2. **Positive (unclaimed, sub-threshold):** calibrated selective prediction
   with head-derived scores matches or beats logprob abstention in the
   preregistered coverage band on both seeds (0.746/0.792 bootstrap, bar
   0.80), with risk UCBs ≤ 0.072 and zero unsafe publications on s29.
3. Parity holds: the workspace path does no harm on unmasked rows.

## 6. Session accounting

Session H: banding ~40 min + resumed pipeline ~76 min + audit ~? (both seeds
returncode 0 at 118 min; full session including audit and packaging within
quota). Salvage saved ~4.5h vs a fresh run. Combined G+H cost ≈ 4.6h + ~5h.
Plain-LoRA control and external benchmarks remain UNAUTHORIZED (10 studies,
0 full passes).

## 7. Open decisions (for the user)

- Write Study 10 (v9.0) results into `paper/hlwm-paper.tex` as the terminal
  negative for the mechanism + the abstention near-miss (the paper currently
  ends at Study 9 results; session -31 that owned the last paper edit is
  gone).
- Whether to pursue the abstention thread as its own claim (it needs its own
  preregistration; the 0.80 bar was missed by 0.054/0.008).
- Any continuation at larger scale (e.g. 1.7B with cross-session resume) is
  OUTSIDE this preregistration and would be a new program, honestly labeled.
