# J-space projection audit: results (2026-09-06)

Plan: `hlwm-v10.0-jspace-analysis-plan-2026-09-06.md` (frozen before execution;
Amendment A1 frozen after seed 17's primary, before any secondary run).
Script: `scripts/jspace_projection_audit.py`. Inputs: the certified v10.0
step-1200 checkpoints (sha256s in `artifacts/kaggle/hlwm-v10.0/session-i6-certified/`).

## Verdict

**H1 CONFIRMED on both seeds (misaligned write).** The certified run's latent
channel is not gradient-dead, and the premise content its probe reads lies
almost entirely outside the subspace the decoder's answer logits are sensitive
to. "Readable but not usable" has a measured mechanism in this system: the
training objective stored the withheld premise in decoder-inert directions.

## Environments

Two independent executions, both passing every instrument gate:

- **local**: Apple M2, MPS, bf16 substrate (deviation D3), torch 2.14
- **T4**: Kaggle NvidiaTeslaT4, fp16 substrate — the recorded audit's own
  conditions — kernel `sirishayaddanapudi/hlwm-jspace-projection-audit` v4

Instrument gates: G0 masked EM = 0.000 (both seeds, both environments; unmasked
within band of the recorded 0.517/0.510). G1 probe replication on the T4 was
exact to three decimals (0.621 / 0.758 vs recorded 0.6212 / 0.7576).

## Numbers

Chance = 0.129 throughout. "S" = component of the thought embeds inside the
row's decoder-sensitive subspace (top-8 right singular vectors of the 10-digit
Jacobian at the first gold digit position); "perp" = orthogonal complement.

### Primary (n=64 flattened 6144-d features, preregistered)

| env | seed | M1 deadness | probe S | probe perp | flat ref | outcome |
|-----|------|------------|---------|-----------|----------|---------|
| local | 17 | 0.292 | 0.000 | 0.188 | 0.125 | INCONCLUSIVE (ref at chance: underpowered) |
| local | 29 | 0.363 | 0.062 | 0.375 | 0.375 | **H1 CONFIRMED** |
| T4    | 17 | 0.296 | 0.125 | 0.250 | 0.188 | INCONCLUSIVE |
| T4    | 29 | 0.352 | 0.062 | 0.312 | 0.438 | INCONCLUSIVE (perp 0.312 vs 0.329 bar) |

### Amendment A1 secondary (n=160, mean-projected 1024-d features, T4)

| seed | probe S̄ | probe perp̄ | reference | power ok | outcome |
|------|---------|-----------|-----------|----------|---------|
| 17 | 0.075 | 0.400 | 0.425 | yes | **H1 CONFIRMED** |
| 29 | 0.100 | 0.725 | 0.800 | yes | **H1 CONFIRMED** |

A1's binding power check (reference >= chance + 0.20) passed on both seeds, so
the secondary is the adjudicating reading per the amendment.

## Reading

1. **The channel is causally connected but content-empty where it matters.**
   Median gradient-norm ratio ~0.3 versus real prompt tokens: the decoder
   listens to the thought positions. But the digit content decodable from the
   thoughts probes at chance inside the decoder-sensitive component (0.075 /
   0.100) and at up to 0.725 in the orthogonal complement.
2. This is the constructive counterpart of the verbalizable-workspace result
   (Transformer Circuits, 2026-07): an external probe can read directions the
   model itself cannot use; membership in the decoder-sensitive subspace, not
   probe decodability, is the property that matters. The program's preregistered
   decodability bar (>= 0.50) measured the wrong property — the certified run
   passed it (0.62/0.76) while generation stayed at exactly 0.000.
3. Budget caveat unchanged: 75 optimizer updates. Nothing here says the write
   could not align under more training; it says that at this budget the failure
   is specifically an alignment failure, not a capacity, liveness, or
   readability failure.

## Deviations

D1 (digit position), D2 (G0 power), D3 (bf16 off-CUDA) as recorded in the
result JSONs. A1 as recorded in the plan. No other deviations.

## Consequences executed (per the plan's pre-commitments)

- (a) analysis subsection added to `paper/hlwm-paper.tex` (Experiment 10);
- (b) certified-run artifacts and both environments' result JSONs committed to
  the public repo;
- (c) declared-routing / aligned-write successor prototype documented in the
  repo as untested code.
