# HLWM Kaggle T4 prototype

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
progressive-training phase. The current Reasoning9000 subset is unreviewed
synthetic data and may only support architecture testing.

The notebook intentionally starts with a short run. Once it passes, increase
`--steps` while keeping the same output directory and use `--resume` with the
latest checkpoint.

