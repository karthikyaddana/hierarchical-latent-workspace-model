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

## Persistence

Run the notebook (or `run_v55_a100_5gb.sh`) to completion. The final cell must
expose:

- `hlwm-v5.5-deliverable.zip` and its checksum;
- `hlwm-v5.5-resumable.pt` and its checksum;
- the numbered checkpoint parts and `parts.json` manifest.

Download the deliverable ZIP first. For resumable training, download either
the full checkpoint or every numbered part plus the parts manifest.
