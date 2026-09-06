# HLWM Kaggle Version 5.3

Version 5.3 is a production-oriented research prototype built on the pinned
`Qwen/Qwen3-0.6B-Base` language substrate. It executes the complete HLWM theory
on two Kaggle T4 GPUs, while keeping a strict boundary between a passed research
gate and a production-ready model.

The public path is:

1. isolated categorical-diffusion lanes create private workspace states;
2. verified summaries cross a synchronization barrier;
3. those summaries condition Qwen through a learned prefix;
4. Qwen generates a normal autoregressive answer;
5. validation-calibrated commitment, risk, and verifier thresholds decide
   whether to publish the candidate or return a safe public fallback.

Private canvas tokens are never exposed.

## What changed after the Version 5.2 run

Version 5.2 proved that both T4 seeds could complete 896 stable steps with the
real 151,936-token Qwen vocabulary and about 4.73 GB peak allocated memory per
GPU. It also exposed three measurable shortcomings: exact-phrase graders
rejected correct answers, safe missing-evidence responses were treated as
non-public decisions, and the two private lanes were nearly collapsed.

Version 5.3 addresses those findings by:

- grading arithmetic, units, ordering, and abstention with structured semantic
  specifications rather than exact wording;
- treating a supported “cannot determine from the supplied evidence” response
  as a committed public answer;
- expanding programmatically verified, split-isolated anchors to 512 train,
  128 validation, and 128 test records, exactly balanced across four task
  classes;
- fitting all publication thresholds on verified validation clean/corrupt
  pairs only, then embedding the fitted thresholds in checkpoints and
  adapters;
- applying squared-cosine lane-diversity loss during both one-pass local and
  short-unrolled joint training, with a higher weight;
- increasing each seed to 1,024 microsteps and the fresh held-out audit to 32
  outputs per seed;
- persisting a compact best-adapter ZIP and a separate resumable checkpoint,
  each with a SHA-256 sidecar and an explicit notebook download link.

## Training method

DiffusionBlocks-style one-pass training is used only for the tied fast private
transition. A local batch samples one categorical corruption time and trains a
single transition, avoiding backpropagation through the complete reverse chain.
Short joint unrolling remains necessary for slow/global state, routing,
verification, synthesis, and commitment. Inference always executes the full
`T -> 1` reverse schedule before Qwen decoding.

Each T4 independently trains seed 17 or 29 with:

- 64 fixed-example overfit microsteps and a 5% improvement gate;
- 384 one-pass local-denoising microsteps;
- 576 short joint-tuning microsteps;
- rank-8 FP32 LoRA sidecars in the final four Qwen blocks;
- frozen FP16 Qwen base weights and FP32 full-vocabulary loss/softmax;
- deterministic ordering, automatic mixed-precision checks, resumable
  checkpoints, and validation-only policy calibration.

## Data and claim boundary

The current Reasoning9000 `direct_fast_no_judge` release remains useful for
architecture learning but is not accepted as factual or policy ground truth.
Its publish, halt, risk, and verifier labels are masked. Only independently
adjudicated records or the narrow programmatic anchors supervise policy heads.

The programmatic anchors are strong tests of specific behaviors, not broad
reasoning evidence. Production readiness still requires independently reviewed
correction/refusal data, external benchmarks, adversarial and safety testing,
multi-seed replication at longer scale, and deployment reliability tests.

## Kaggle persistence

Run the supplied notebook with **Save Version → Save & Run All**. Do not rely on
a Quick Version or paths printed by an interactive session. The final cell must
finish successfully and show links for:

- `hlwm-v5.3-best-deliverable.zip` and its checksum;
- `hlwm-v5.3-best-resumable.pt` and its checksum.

Download those files from the saved version’s Output tab. A run that fails a
capability gate remains useful evidence, but must not be labeled production
ready.
