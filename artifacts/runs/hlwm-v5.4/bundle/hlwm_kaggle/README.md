# HLWM Kaggle Version 5.4

Version 5.4 is a production-oriented research prototype built on the pinned
`Qwen/Qwen3-0.6B-Base` language substrate. It tests the executable HLWM path on
two Kaggle T4 GPUs while keeping a strict boundary between a research gate and
a production-ready model.

The public path is:

1. isolated categorical-diffusion lanes create private workspace states;
2. structured summaries cross a synchronization barrier;
3. those summaries condition Qwen through a learned prefix;
4. Qwen generates one frozen autoregressive candidate;
5. candidate-level commitment, risk, and verification heads apply thresholds
   calibrated on generated validation emissions;
6. the system publishes the exact candidate or returns a fixed safe fallback.

Private canvas tokens are never exposed.

## Why Version 5.4 exists

Both Version 5.3 seeds completed 1,024 microsteps with zero skipped optimizer
updates and roughly 4.73 GB peak allocated memory per T4. The held-out audit
also improved lane separation substantially. It nevertheless failed the two
capability gates:

- semantic probe accuracy was 0.625 for seed 17 and 0.5625 for seed 29;
- probe commitment accuracy was 0.625 and 0.3125;
- correct missing-evidence answers were generated inconsistently across seeds;
- validation calibration scored teacher-forced references, while the test gate
  scored free generated candidates;
- private denoising-token error was used as a hard gate on final answers and
  rejected semantically correct generated responses.

Version 5.4 responds directly by:

- adding a third candidate-level verifier logit trained on clean/negative
  candidate pairs;
- fitting all three publication thresholds on semantically graded generated
  validation candidates, never on the test split;
- expanding to 1,024 train, 256 validation, and 256 test anchors with multiple
  prompt templates and matched operand distributions across isolated splits;
- increasing anchor-rich training to 4,224 microsteps per seed;
- using rank-16 FP32 LoRA sidecars across the final eight Qwen blocks;
- auditing 64 held-out outputs per seed and requiring correct safe abstentions
  to be published, not merely generated;
- splitting the resumable checkpoint into download-friendly parts in addition
  to retaining the full checkpoint and SHA-256 manifests.

## Training method

DiffusionBlocks-style one-pass training is used only for the tied fast private
transition. A local batch samples one categorical corruption time and trains a
single transition. Short joint unrolling remains necessary for slow/global
state, routing, verification, synthesis, and commitment. Inference executes the
full `T -> 1` reverse schedule before Qwen decoding.

Each T4 independently trains seed 17 or 29 with:

- 128 fixed-example overfit microsteps and a 5% improvement gate;
- 1,024 one-pass local-denoising microsteps;
- 3,072 short joint-tuning microsteps;
- 40% causal-language batches and a 75% verified-anchor sampling ratio;
- frozen FP16 Qwen base weights, rank-16 FP32 LoRA sidecars in eight blocks,
  and FP32 full-vocabulary loss/softmax;
- deterministic ordering, numerical preflight, resumable checkpoints, and
  generated-emission validation calibration.

## Data and claim boundary

The Reasoning9000 `direct_fast_no_judge` release remains useful for architecture
and language learning but is not factual or policy ground truth. Its publish,
halt, risk, and verifier labels remain masked. Only independently adjudicated
records or narrow programmatically verified anchors supervise policy heads.

The anchors test arithmetic, unit conversion, ordering, and missing-evidence
behavior. They do not establish broad reasoning quality or production safety.
Production readiness still requires independently reviewed correction and
abstention data, external benchmarks, adversarial testing, stronger calibration
with confidence bounds, multi-seed replication, and serving reliability tests.

## Kaggle persistence

Run the supplied notebook with **Save Version -> Save & Run All**. The last cell
must finish and expose:

- `hlwm-v5.4-best-deliverable.zip` and its checksum;
- `hlwm-v5.4-best-resumable.pt` and its checksum;
- the numbered checkpoint parts and `parts.json` manifest.

Download the deliverable ZIP first. For resumable training, download either the
full checkpoint or every numbered part plus the parts manifest. A failed gate
is valid research evidence but must not be labeled production-ready.
