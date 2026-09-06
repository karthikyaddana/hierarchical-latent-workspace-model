# HLWM Kaggle Version 5

Version 5 is a correctness-first production candidate built around the pinned
`Qwen/Qwen3-0.6B-Base` language substrate. It tests the complete execution
theory on Kaggle T4 hardware without claiming that the present synthetic data
already makes the resulting checkpoint production-ready.

The important Version 5 correction is the output path:

1. isolated categorical-diffusion lanes create private workspace states;
2. verified lane summaries cross a synchronization barrier;
3. those summaries become a learned prefix for Qwen;
4. Qwen generates the answer autoregressively;
5. publication requires high commitment, low predicted risk and low verifier
   error.

Private canvas `argmax` tokens are never returned as the public answer.

## Training method

DiffusionBlocks-style training is used only for the tied fast private
transition. A local batch samples one categorical corruption timestep and
trains one transition, so it does not backpropagate through a complete reverse
chain. Joint tuning uses a short explicit schedule for slow/global state,
routing, verification, workspace synthesis and commitment. Inference always
executes the complete `T -> 1` reverse schedule before Qwen decoding.

The default run is progressive:

- 48 microsteps on a fixed 24-example overfit set;
- an automatic fixed-corruption loss gate;
- 120 one-pass local-denoising microsteps;
- 80 short joint-tuning microsteps;
- checksummed checkpoint and safetensors export;
- fresh-process generation and corruption-rejection evaluation.

The trainer stops instead of spending the remaining Kaggle budget if the tiny
overfit gate fails. It also saves a resumable checkpoint at the configured
runtime limit.

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
- paired clean/corrupt commitment training;
- conservative three-condition publication gating;
- deterministic resume ordering, mixed-precision stability checks, SHA-256
  sidecars and adapter export.

## Data boundary

Reasoning9000 is retained for architecture learning, but its current
`direct_fast_no_judge` release is not treated as calibrated policy truth.
Publish, halt, risk and verifier labels are masked unless an episode explicitly
records independent adjudication. Deterministic corrupted snapshots provide a
mechanical rejection contrast; they are not a substitute for a human-reviewed
abstention and correction set.

Production readiness still requires independently adjudicated clean, incorrect
and abstain examples, held-out external benchmarks, calibration, multiple
seeds, long-run reliability and deployment load tests. Version 5 is designed to
make those gates measurable rather than to hide that they have not yet passed.
