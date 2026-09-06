# Preregistered analysis plan: is the Exp-10 channel content outside the decoder-sensitive subspace?

Date frozen: 2026-09-06 (before any checkpoint forward pass on this machine).
Analyst: local CPU/MPS session, Apple M2, torch 2.13.0. No training. No new data.

## Motivation

The certified v10.0 run (Kaggle kernel `sirishayaddanapudi/hlwm-v10-0-certified-run`,
status COMPLETE, step 1200, 75 optimizer updates, zero skipped) recorded, per seed:

| seed | probe acc (chance 0.129) | preregistered bar | masked EM | latent_channel_live |
|------|--------------------------|-------------------|-----------|---------------------|
| 17   | 0.6212 (n=263 held-out)  | >= 0.50: PASS     | 0.000     | false               |
| 29   | 0.7576 (n=263 held-out)  | >= 0.50: PASS     | 0.000     | false               |

This is the first run in the program's history where the decodability bar passed
while generation through the channel stayed at exactly zero. Anthropic's workspace
result (Transformer Circuits, 2026-07) reports that verbalizable content occupies a
privileged subspace distinct from what probes can decode. Hypothesis under test:

**H1 (misaligned write):** the premise content carried by the six latent thoughts
lies predominantly outside the decoder's output-sensitive subspace at the injection
site, so an external probe reads it while the frozen decoder cannot.

## Fixed inputs

- Checkpoints: `hlwm-v10.0-seed-{17,29}-resumable.pt` from the certified kernel
  (format `hlwm-trainable-checkpoint-v5`, step 1200), sha256 recorded at ingest.
- Base: `Qwen/Qwen3-0.6B-Base` revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (pinned).
- Rows: the audit's masked core, first 160 masked rows in ascending
  `str(episode_id)` order (the preregistered ordering rule), restricted to the
  first **64** rows per seed for the Jacobian stage (compute budget, fixed here).
- Probe rows: the identical 263 held-out masked rows the audit used.
- Code: `artifacts/kaggle/hlwm-v10.0/bundle/hlwm_kaggle` (sha256 in the run manifest),
  imported unmodified. Analysis script: `scripts/jspace_projection_audit.py`.

## Procedure

1. Rebuild the model (frozen base + checkpoint trainables), fp32 on CPU / bf16 on MPS.
2. **G0 instrument gate.** Greedy-decode the first 20 masked-core rows and 20 unmasked
   rows. Required: masked exact-match = 0.000 (as recorded) and unmasked accuracy
   within +/- 0.10 of the recorded 0.517/0.510 band. Fail -> STOP, report
   environment mismatch; no further claims.
3. **G1 probe replication.** Recompute thought-state means on the 263 held-out rows,
   retrain the audit probe (same code path, seed+4). Required: accuracy within
   +/- 0.07 of the recorded value per seed. Also train the same probe protocol on the
   **thought embeds** (what the decoder actually reads, post projection + grounding).
   - If embeds probe < chance + 0.10 while states probe replicates: outcome
     **O-PROJ** (the projection itself destroys the content before the decoder
     sees it). Decisive; skip to reporting.
4. **Sensitivity and subspace (64 masked-core rows/seed).** For each row, teacher-force
   `[masked prompt][6 thoughts][cue]` and take the position of the first answer token.
   Compute the Jacobian of the 10 digit-token logits at that position w.r.t. the
   6x1024 thought-embed block (10 backward passes), and the same w.r.t. the last 6
   valid prompt-token embeddings (baseline).
   - **M1 deadness ratio** = median_row ||J_thoughts||_F / ||J_prompt||_F.
   - **M2 subspace split**: per row, SVD of J_thoughts (10 x 6144), S = top-8 right
     singular vectors. Project each row's flattened thought embeds onto S and its
     complement; train the audit-protocol probe on each component across rows
     (labels = the probe digit labels of those 64 rows; if fewer than 8 labelled
     rows survive, extend to the next masked-core rows until 64 labelled).

## Preregistered readings (frozen before execution)

- **DEAD CHANNEL:** M1 < 0.10 on both seeds -> the decoder's answer position is
  order-of-magnitude insensitive to the thoughts; the write is causally
  disconnected regardless of content. (Consistent with prefix_gate_median 0.079
  in the training record if the gate, not the content, is the bottleneck.)
- **H1 CONFIRMED (misaligned write):** M1 >= 0.10 and probe(S-component) <
  chance + 0.10 while probe(complement) >= chance + 0.20, both seeds -> the
  decoder listens to the thoughts, but the premise content sits in directions it
  does not listen to.
- **H1 REFUTED:** probe(S-component) >= chance + 0.20 on both seeds -> the
  readable content IS inside the decoder-sensitive subspace and the blockage is
  elsewhere (nonlinear, positional, or decoding-policy). Equally reportable.
- Any mixed/other pattern: **INCONCLUSIVE**, report numbers without a mechanism claim.

## Pre-committed consequences

- CONFIRMED or DEAD CHANNEL -> (a) new analysis subsection in `paper/hlwm-paper.tex`
  Experiment 10; (b) certified-run artifacts enter `artifacts/kaggle/hlwm-v10.0/`
  and the public repo; (c) the declared-routing hybrid prototype is written into
  the HLWM codebase as a documented, untested successor mechanism.
- REFUTED -> (a) and (b) only; no successor mechanism claim from this analysis.
- The papers' published numbers are corrected only where this record shows them
  stale (the certified run supersedes "undecided at 125 updates" phrasing: it is a
  second independent abort at 75 updates with the decodability bar passed).

## Deviations

Any deviation from this plan will be recorded in the results file with a reason.
