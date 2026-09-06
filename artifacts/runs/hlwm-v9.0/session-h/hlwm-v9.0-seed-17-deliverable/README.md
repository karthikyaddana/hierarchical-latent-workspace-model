# HLWM Version 9.0 - necessity by construction

Version 9.0 is the response to the Version 8.0 run (Session F): every repair
worked and replicated, and every mechanism measured null or lost to its
training-free baseline. The post-session code audit reduced that outcome to
three root causes, and this bundle fixes all three.

**RC-1 (the core finding).** The latent read-out was redundant by
construction: both channels conditioned on the same full context, so the model
could minimize its loss while ignoring the workspace, and it did (textbook
posterior collapse under a strong decoder). Two v8.0 instruments were also
broken: the "ablation" only removed the 34 memory tokens (the 8 gated
synthesis tokens stayed in both arms), and the prefix gate's trajectory was
structurally flat (RMSNorm is scale-invariant, so the attention-visible prefix
content is independent of the gate for any positive value). The v8.0 KL anchor
referenced the same model without the prefix, so its only stable point was a
prefix that changes nothing.

**RC-2.** The head-training harvest generated from the training collator's
fixed-length right-padded prompt surface while the audit encodes unpadded;
candidate validity on generative families was 1-3% against 0.82 causal
validity on the audit surface, so the 16-per-family validity floor starved.

**RC-3.** Anchor difficulty was never a parameter (single-operation templates,
hard-coded operand ranges), so the audit distribution was bimodal and every
selection/abstention comparison ran without headroom.

## The Version 9.0 mechanisms

- **M1 information-asymmetric channels (Claim L).** Maskable anchor rows carry
  a withheld-premise prompt variant (`user_request_masked`); on masked rows the
  answer channel conditions on it while the workspace reads the full context,
  so the gated latent prefix is the only path from the operands to the answer.
  Ignoring the latent now costs loss. Leak integrity has four layers:
  generator construction, a normalization-time substring assert, a collator
  token-subsequence assert, and an audit leak gate that voids Claim L.
- **M2 dense latent supervision.** A latent-only probe decodes the withheld
  premise token ids from the prefix positions (restricted candidate scoring,
  never the gold answer), weight 0.1, masked rows only.
- **M3 gist attribution control.** A mean-pool projection prefix at the same
  25-token budget, trained on alternating masked batches, audited as the
  canvas's killer baseline (`canvas_beats_gist`).
- **M4 harvest/verifier repair.** One shared unpadded encode primitive
  (`data.encode_preserving_ends`) for harvest, calibration and audit; the
  validity floor is now a real gate; a wall-clock guard caps the harvest.
- **M5 difficulty banding.** Generators take chain length (1-3), operand
  digits (2-6) and distractor count (0-2); the audited 224 anchors are chosen
  by an in-session banding pass against a frozen estimator before training.
- **M6 abstention realignment (Claim A2).** Conformal target coverage 0.40 in
  band [0.20, 0.55]; three-way calibration split (weights on A, score form on
  B1, threshold on B2); a five-feature publish rule nests plurality agreement;
  the headline is coverage-restricted partial AUGRC in [0.05, 0.50] at >= 80%
  of paired bootstraps; risk UCB <= 0.15 with an n >= 20 precondition.

Single lane (the preregistered unconditional lane exit executed after
failures eight and nine). Prefix layout: 8 synthesis + 16 windows + 1 summary
= 25 positions, scaled by a telemetry-only gate that gates nothing.

## Provenance and boundaries

Preregistration: `reports/hlwm-v9.0-plan-2026-09-02.md` (amendments in its
final section). The masked-row setting is a constructed mechanism
demonstration, not a capability claim; the matched plain-LoRA control stays
gated behind a full pass (binding ladder rung 5), and external benchmarks stay
gated behind the control. Base model `Qwen/Qwen3-0.6B-Base` at revision
`da87bfb6...`, frozen; trainable sidecar ~76.6M parameters.
