# HLWM Version 8.0 - repaired channel, decoupled claims

Version 8.0 is the response to the Version 6.0 two-seed replication (Session E, both
seeds failed). That run produced one line-level defect and one design defect, and
Version 8.0 fixes both rather than re-running the same battery.

**The defect.** `Qwen3-0.6B-Base` has no BOS token, so `bos_token_id` resolves to
`eos` (`<|endoftext|>`). Every workspace-channel teacher-forcing and decode call
seeded the answer with that id, injecting a mid-sequence document boundary. The
model did the reasonable thing and started a new turn, emitting `Human:` /
`Assistant:` scaffold. The damage was confined to the workspace channel, which is
exactly where the study's claims live: workspace answers lost roughly 30 points of
quality against the causal path on the same rows. Version 8.0 removes the seeded id
entirely (`_channel_prefill` / `_channel_teacher_force` never insert BOS) and moves
the response cue out of the prompt string into `config.response_cue_ids`, appended
internally as embeddings.

**The design defect.** The Version 6.0 headline gate asked four candidates at
p≈0.8 to beat self-consistency. The headroom for *any* selector there is
N·p·(1−p)^(N−1) ≈ 2-5 points, one or two rows out of 36 — the gate was unwinnable
regardless of the selector's quality. Version 8.0 decouples the claims and powers
each one separately.

## The three claims

- **Claim G (generation parity).** The workspace channel must reach the causal
  channel's quality on identical rows: no scaffold leak, accuracy within 5 points,
  token-F1 within 10%, a live read-out memory (ablation delta ≥ 0.05), and an open
  prefix gate. This is the repair, verified.
- **Claim S (selection).** Verifier-weighted self-consistency beats plurality
  self-consistency on the *medium*-difficulty stratum, where the headroom actually
  exists, tested with McNemar mid-p on discordant pairs and a SQL-free
  anchors-only sensitivity check. `heads_not_logprob` guards against the heads
  merely re-deriving mean logprob (|Spearman| < 0.95).
- **Claim A (abstention).** Split-conformal publication at
  `c_nominal = 0.35` with the coverage floor `k = ⌊(n+1)(1−c)⌋` on a held-out
  calibration half, beating mean-logprob abstention on AUGRC in ≥ 90% of paired
  bootstrap resamples, with a Clopper-Pearson upper bound on selective risk.
  The conformal order statistic replaces the margin-midpoint threshold, so
  coverage can no longer be strangled the way it was on seed 29 (0.083).

## Channel and training repairs

1. **R1 no BOS anywhere**; **R2** response cue as internal embeddings;
   **R3** `prefix_gate = tanh(α)` with `α` initialized at 0.05 and the memory
   projections at `std 0.02`, so diversity and orthogonality gradients are live
   from step 0 instead of provably inert under zero-init;
   **R4** a KL-to-base fluency anchor on the synthesis channel
   (`--synthesis-kl-weight 0.05`) against a detached causal reference;
   **R5** structurally asymmetric lane briefs (`torch.tensor_split` over context
   positions) rather than a penalty on symmetric inputs;
   **R6** a four-feature publish rule that must consume the candidate's mean
   logprob, fitted on calibration half A with the threshold set on half B.
2. **Candidate pool** N=8 (1 greedy + 7 at T=0.8) decoded through the *causal*
   channel with a KV cache, which is what affords the 320-row audit inside the
   session budget.
3. **Audit** covers every light-gradable test row (anchors first, n=320) with a
   difficulty-binned breakdown and a base-model control on a format-matched prompt.

24-gate battery, 58 unit tests. Plan, pre-mortem (P1-P15) and the five-rung
fallback ladder: `reports/hlwm-v8.0-plan-2026-09-02.md`. Session E results:
`reports/hlwm-v6.0-session-e-results-2026-09-02.md`.

The Version 5.5/6.0 background below is retained for module-level context.

---

# HLWM Version 5.5

Version 5.5 is the next production-track research iteration on the pinned
`Qwen/Qwen3-0.6B-Base` substrate. It targets a single A100 MIG `1g.5gb`
partition (about 4.75 GB of device memory) inside a 3.5-hour session, while
remaining runnable on a single T4. It keeps the strict boundary between a
research gate and a production-ready model.

The public path is unchanged from Version 5.4:

1. isolated categorical-diffusion lanes create private workspace states;
2. structured summaries cross a synchronization barrier;
3. those summaries condition Qwen through a learned prefix;
4. Qwen generates one frozen autoregressive candidate;
5. candidate-level commitment, risk, and verification heads apply thresholds
   calibrated on generated validation emissions;
6. the system publishes the exact candidate or returns a fixed safe fallback.

Private canvas tokens are never exposed.

## Why Version 5.5 exists

Version 5.4 (both seeds, 4,224 microsteps, zero skipped updates, ~4.77 GB peak
per T4) fixed the private-denoising publication error and lifted the selected
seed to 0.813 semantic probe accuracy with zero prompt leakage. It still
failed the complete capability gate for two preregistered reasons:

- **Policy-score overlap.** Generated-emission calibration accepted 0.839 of
  positives but rejected only 0.583 of negatives (gate: 0.70). The fitted
  operating point was degenerate (commitment threshold 0), showing the three
  candidate heads never learned to separate on-policy emissions; they had only
  ever been trained on teacher-forced references and mechanical corruptions.
- **Routing collapse.** Seed 17 sent 98.4% of audited decisions through one
  expert and seed 29 materially used only two, so the earlier
  "two routes above 1%" gate was too weak to mean anything.

Version 5.5 responds directly:

- an **on-policy policy-head phase** generates train-anchor emissions with the
  frozen decision path, grades them with the deterministic semantic graders,
  and fits only the three candidate-level heads on those emissions plus two
  hard-negative families (the anchor's fluent wrong-value answer and a
  mechanically corrupted snapshot); thresholds are then fitted, as before, on
  generated validation emissions only, and the test split remains untouched;
- a **router marginal-entropy regularizer** (KL of mean routing probabilities
  to uniform) joins the Switch-style balance term during both local and joint
  training;
- evaluation reports **preregistered routing quantities**: normalized route
  entropy, second-route load (gate: at least 0.10), and a causal intervention
  that pins routing to the least-used expert and measures policy-score
  movement on the identical frozen candidate;
- **BF16 autocast** on devices that support it (A100/MIG), which removes
  loss-scale skip risk; FP16 with the existing GradScaler remains the T4 path;
- the real-Qwen preflight now also runs a **joint-stage forward/backward** and
  enforces a **device memory headroom limit**, so a 5 GB partition fails in
  minutes instead of mid-session;
- reduced token budgets (context 160, canvas 96, brief 96, causal 320) keep
  the FP32 full-vocabulary loss path inside the MIG memory ceiling.

## Training method

Unchanged in structure: DiffusionBlocks-style one-pass training for the tied
fast private transition, short joint unrolling for slow/global state, routing,
verification, synthesis, and commitment, then the new on-policy head phase,
then threshold calibration. Inference executes the full `T -> 1` reverse
schedule before Qwen decoding.

The single-device session runs one seed with:

- 128 fixed-example overfit microsteps and a 5% improvement gate;
- 768 one-pass local-denoising microsteps;
- 1,664 short joint-tuning microsteps;
- 40% causal-language batches and a 75% verified-anchor sampling ratio;
- frozen Qwen base weights in the autocast dtype, rank-16 FP32 LoRA sidecars
  in the final eight blocks, FP32 full-vocabulary loss/softmax;
- a wall-clock budget that checkpoints resumable state before the session
  limit; a second session resumes with `--resume` and identical example order.

A failed or truncated run is valid research evidence, never production-ready.

## Data and claim boundary

The Reasoning9000 `direct_fast_no_judge` release remains architecture and
language material only; its policy labels stay masked. Only narrow
programmatically verified behavior anchors supervise policy heads, and they
test arithmetic, unit conversion, ordering, and missing-evidence behavior.
They do not establish broad reasoning quality or production safety.
Production readiness still requires independently reviewed correction and
abstention data, external benchmarks, adversarial testing, stronger
calibration with confidence bounds, multi-seed replication, and serving
reliability tests.

## Dual-T4 multi-seed replication

`embel-hlwm-v5.5-kaggle-2xt4.ipynb` runs the identical Version 5.5
architecture on two Kaggle T4s, one independent seed per GPU (session 1:
seeds 17 and 29; session 2: `HLWM_SEEDS=41,73`), at the Version 5.4-proven
T4 scale of 4,224 microsteps and context 192 / canvas 128 / brief 96 /
causal 384, FP16 with the GradScaler and the zero-skip requirement. Each
seed produces its own deliverable, gate file, and named resumable
checkpoint; the session writes a cross-seed replication verdict. Plan:
`reports/hlwm-v5.5-t4-replication-plan-2026-08-30.md`.

## Version 5.6 — routing redesign on verified multi-domain data

`embel-hlwm-v5.6-kaggle-2xt4.ipynb` (bundle
`hlwm-v5.6-candidate-bundle.zip`, built by
`scripts/build_hlwm_v56_bundle.py`) changes exactly the mechanisms Study 4
falsified and the data the review found missing:

- **Routing objective**: the marginal-entropy (KL-to-uniform) regularizer is
  removed — Study 4 showed it drives soft marginals uniform while hard
  decisions stay concentrated, homogenizing the experts. The Switch-style
  balance loss on hard assignments is strengthened (`--router-aux-weight
  0.05`) and a pairwise expert-output diversity penalty is added
  (`--expert-diversity-weight 0.05`). Declared fallback if the routing gates
  fail again: `--num-experts 2`.
- **Policy heads**: `--policy-epochs 120` and `--policy-label-smoothing
  0.05` de-saturate the calibrated score scale (v5.5 fitted rails at ~5e-5 /
  0.9999), which also unconfounds the routing causal-liveness measurement.
- **Data** (`scripts/build_hlwm_v56_dataset.py` → `data/hlwm-v5.6/`):
  Reasoning9000 episodes (policy-masked) + 11 independently adjudicated
  builder episodes + execution-verified benchmark train-split episodes
  (MBPP / APPS / TACO-verified / CodeContests — the official reference must
  pass its own executed checks at build time or the row is dropped) +
  Spider exact-match rows + audited OpenThoughts/DeepCoder packets, with
  per-source dev slices, cross-split collision drops, and cross-source
  dedup. SWE-bench is deferred to the 8B candidate.
- **Budgets**: context 256 / canvas 128 / brief 96 / causal 1024 so the code
  data fits (v5.5's causal 384 fit ~25% of it). Audits are domain-stratified
  with 288 generation tokens and execution/SQL graders; per-domain metrics
  are report-only.

Gate thresholds are unchanged from the Version 5.5 preregistration. Plan:
`reports/hlwm-v5.6-plan-2026-08-30.md`.

## Persistence

Run the notebook (or `run_v55_a100_5gb.sh`) to completion. The final cell must
expose:

- `hlwm-v5.5-deliverable.zip` and its checksum;
- `hlwm-v5.5-resumable.pt` and its checksum;
- the numbered checkpoint parts and `parts.json` manifest.

Download the deliverable ZIP first. For resumable training, download either
the full checkpoint or every numbered part plus the parts manifest.
