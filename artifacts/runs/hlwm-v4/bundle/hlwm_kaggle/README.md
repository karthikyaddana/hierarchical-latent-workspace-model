# HLWM Kaggle Version 4

This package turns the pinned `Qwen/Qwen3-0.6B-Base` checkpoint into a
prototype with a normal causal path plus private categorical-diffusion lanes.
The Qwen language weights initialize the shared transition; they are frozen for
the first T4 experiment while the root, routing, recurrent state, verifier,
synthesis, halting and commitment modules train.

Implemented in the first executable prototype:

- exact categorical forward corruption and an x0-parameterized reverse posterior;
- causal committed-context attention and bidirectional private-canvas attention;
- tied recurrent private refinement with fast and private slow state;
- an always-executed root adapter and sparse routed specialist adapters;
- state-isolated lanes folded into the batch dimension;
- a synchronization barrier that exports summaries rather than hidden lane state;
- verifier, global update, synthesis, private halt and atomic commitment heads;
- a preserved autoregressive Qwen path and causal-anchor training batches;
- adapter-only checkpoints that can resume within Kaggle's weekly quota.

Research boundary: this is the minimum language-model prototype, not proof of
the complete paper. The first run uses a flat specialist bank and a single
macrocycle. Variable-depth connected expert-graph traversal, verifier recurrence,
route-window redispatch, calibrated EVI and multiple macrocycles are the next
progressive-training phase. Reasoning9000 is a synthetic development release;
its pilot audit reviewed 50 episodes and accepted 32, so it can drive training
and debugging but cannot by itself establish factual quality or superiority.

Version 4 turns the successful wiring smoke test into a reproducible research
baseline:

- all 4,300 Reasoning9000 master episodes are packaged (3,310 train, 585
  validation and 405 test), rather than the former 256-episode smoke subset;
- deterministic global-step sampling makes resumed example order reproducible;
- frozen Qwen weights stay FP16 while trainable HLWM parameters retain FP32
  master weights;
- optimizer warmup advances on applied optimizer updates, not microsteps;
- dynamic loss scaling starts conservatively and records every skipped update;
- checkpoints include RNG and trainer state plus a SHA-256 sidecar;
- trainable weights are also exported as a standalone safetensors adapter;
- training logs route load, lane similarity, corruption, halt, risk and
  commitment diagnostics without non-standard JSON NaN values;
- `evaluate_checkpoint.py` reloads the saved model and compares pinned-base
  causal, HLWM causal and private-diffusion outputs on fixed test episodes.

The clean notebook runs a fresh 250-microstep baseline. It deliberately fails
its gate if any optimizer update is skipped. Only after checkpoint reload,
generation and diagnostic assertions pass should the run be extended.
