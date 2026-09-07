# Preregistered: v11 pilot — vocabulary-grounded thoughts against the necessity instrument

Frozen 2026-09-07, before any v11 training step. Extends the closed v10.0 record;
motivated by the J-space verdict (`hlwm-v10.0-jspace-results-2026-09-06.md`) and
its external corroboration (J-CoT, arXiv:2607.21981: vocabulary-indexed thought
interfaces work where dense-vector passing struggles).

## Hypothesis

**H-v11:** the v10.0 channel failed because its thoughts were written into
decoder-inert directions (measured: premise content probes at 0.075/0.100 inside
the decoder-sensitive subspace vs 0.400/0.725 outside). If each thought is
constructed *inside* the decoder's own vocabulary basis — state → frozen
`lm_head` → temperature softmax → coefficients → embedding = coefficients ×
frozen embedding matrix — then at the SAME budget where the certified run
measured zero transfer, the channel should begin to transfer withheld content.

## Two changes, nothing else

Diff against the certified v10.0 harness (sha256s in the run manifest):

1. **Vocabulary bottleneck** in `produce_latent_thoughts` behind a new config
   flag (`vocab_grounded_thoughts`): the projected thought state is decoded by
   the frozen `lm_head`, softmaxed at temperature `vocab_thought_tau`, and the
   thought embedding is that distribution's average of frozen input embeddings,
   then scale-grounded as before. Top-5 coefficient tokens are logged per
   thought: the channel becomes human-readable in the token stream's own terms.
2. **In-context aligned decode loss**: the student pass's existing
   `segment_logits` (frozen `lm_head` over the thought positions *inside the
   masked deployment context*) receive the same per-window trace-token CE the
   trained `latent_step_logits` head got in v10 (weight `--incontext-decode-weight`,
   default 0.5). Supervision through the frozen unembedding in context, where
   v10's only premise supervision was a separate trained probe head — the
   readable-but-not-usable design error.

Everything else — data, seeds (17, 29), warm phase (600), main budget (600,
matching the certified run's 1200 total microsteps), gates machinery, audit
battery — is byte-identical to the certified configuration.

## Preregistered gates (per seed unless stated)

- **G-A (primary, content transfer):** in the audit's gold-forced-CE block,
  `channel_masked − causal_masked ≤ −0.10` nats/token on masked rows (certified
  baseline: +0.005/+0.023). This is the sensitive continuous measure of the
  channel carrying answer-relevant bits.
- **G-B (headline):** `gonogo_masked_numeric_em > 0` and the audit's full arm
  strictly beats its shuffled-thoughts arm on the masked core (certified: all
  arms 0.000).
- **G-C (no regression):** unmasked full-pipeline accuracy within ±0.05 of the
  certified 0.517/0.510; warm reconstruction gate still ≥ 0.50; masked leak
  scan still zero.
- **G-D (mechanism moved):** the J-space audit (`scripts/jspace_projection_audit.py`,
  A1 secondary) on the v11 checkpoint reads the in-subspace probe at
  ≥ chance + 0.20 (certified: 0.075/0.100 ≈ chance).

Readings: all four pass on both seeds → the misaligned-write account is
confirmed *constructively* and the channel works at pilot scale. G-A passes
without G-B → content transfers but not enough for exact-match; report as
partial, extend budget once (main 600 → 2400) under the same gates. G-A fails
with G-D passing → alignment achieved but content still unused: H-v11 wrong in
an informative way; stop and report. Nothing passes → stop and report; no
third configuration inside this preregistration.

## Budget and environment

Kaggle `NvidiaTeslaT4`, seeds sequential in one kernel; 1,200 microsteps/seed
(≈75 optimizer updates — deliberately the certified budget, so the certified
run is the matched control); full audit per seed. The capability question at
published-success budgets (≥2×10⁵ presentations) remains out of scope: this
pilot decides whether alignment was the binding constraint at the measured
failure point, not whether HLWM reaches parity.

## Provenance

v11 code ships as a Kaggle dataset (exact bytes, sha256s in its manifest);
the kernel mounts it beside the certified run's output and copies the
certified data unchanged. All deviations recorded in the results file.
