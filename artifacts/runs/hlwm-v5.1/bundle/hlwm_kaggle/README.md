# HLWM Kaggle Version 5.1

Version 5.1 is a correctness-first production candidate built around the pinned
`Qwen/Qwen3-0.6B-Base` language substrate. It tests the complete execution
theory on Kaggle T4 hardware without claiming that the present synthetic data
already makes the resulting checkpoint production-ready.

The public output path is:

1. isolated categorical-diffusion lanes create private workspace states;
2. verified lane summaries cross a synchronization barrier;
3. those summaries become a learned prefix for Qwen;
4. Qwen generates the answer autoregressively;
5. publication requires high commitment, low predicted risk and low verifier
   error.

Private canvas `argmax` tokens are never returned as the public answer.

## Version 5.1 corrections

The Version 5 pilot proved the execution path but produced incomplete prompt
paraphrases. Version 5.1 therefore:

- uses one bounded public prompt for training and evaluation, preserving both
  the instruction and response cue;
- removes the prototype-warning text from model input;
- adds 96 split-isolated, programmatically verified arithmetic, conversion,
  ordering and missing-evidence behavior anchors;
- oversamples those anchors without presenting them as human adjudication;
- gives lanes explicit constructor and independent-critic roles;
- tunes one final Qwen decoder block while leaving the rest frozen;
- evaluates 16 generated outputs per seed;
- runs seeds 17 and 29 concurrently, one on each Kaggle T4.

## Training method

DiffusionBlocks-style training is used only for the tied fast private
transition. A local batch samples one categorical corruption timestep and
trains one transition, so it does not backpropagate through a complete reverse
chain. Joint tuning uses a short explicit schedule for slow/global state,
routing, verification, workspace synthesis and commitment. Inference always
executes the complete `T -> 1` reverse schedule before Qwen decoding.

Each of the two default runs is progressive:

- 64 microsteps on a fixed 32-example overfit set;
- an automatic fixed-corruption loss gate;
- 320 one-pass local-denoising microsteps;
- 512 short joint-tuning microsteps;
- checksummed checkpoint and safetensors export;
- fresh-process generation and corruption-rejection evaluation on eight unseen
  anchors and eight held-out Reasoning9000 records per seed.

The trainer stops instead of spending the remaining Kaggle budget if the tiny
overfit gate fails. It also saves a resumable checkpoint at the configured
runtime limit.

The exported `hlwm-adapter-*.safetensors` is self-describing: its metadata pins
the Qwen revision, HLWM configuration and number of tuned Qwen tail blocks.
`inference_hlwm.py` reloads that adapter and always runs the complete reverse
schedule before applying the publication gate.

## Implemented invariants

- exact multinomial forward corruption and x0 reverse posterior;
- equal-corruption-mass timestep sampling;
- normalized recurrent diffusion embeddings;
- tied Qwen-initialized transition weights;
- state-isolated private lanes and protected attention;
- connected root-to-leaf sparse adapter paths;
- route refresh only at bounded window boundaries;
- one-pass local denoising and short joint unrolling;
- full reverse diffusion at inference;
- summary-only synchronization barrier;
- Qwen autoregressive workspace-conditioned answer generation;
- paired clean/corrupt commitment training only on policy-eligible records;
- conservative three-condition publication gating;
- deterministic resume ordering, mixed-precision stability checks, SHA-256
  sidecars and adapter export.

## Data boundary

Reasoning9000 is retained for architecture learning, but its current
`direct_fast_no_judge` release is not treated as calibrated policy truth.
Publish, halt, risk and verifier labels are masked for the unreviewed release;
only independently adjudicated records and narrow programmatically checked
anchors may supervise those heads. Deterministic corrupted snapshots provide a
mechanical rejection contrast; they are not a substitute for a human-reviewed
abstention and correction set.

The behavior anchors are intentionally narrow. They detect prompt copying,
arithmetic errors and unsupported guessing, but cannot validate broad factual
or technical correctness.

Production readiness still requires independently adjudicated clean, incorrect
and abstain examples, held-out external benchmarks, calibration, multiple
seeds, long-run reliability and deployment load tests. Version 5.1 is designed to
make those gates measurable rather than to hide that they have not yet passed.
