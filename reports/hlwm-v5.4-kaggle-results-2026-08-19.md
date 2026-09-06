# HLWM V5.4 Kaggle Results

Date: 2026-08-19

## Artifact verification

- Deliverable: `hlwm-v5.4-best-deliverable.zip`
- Local SHA-256: `a2747e20d5089f7bafa71a1a28366274d0194e2e6b4c4440413be0051df03ea1`
- ZIP CRC: all 12 members passed
- Selected seed: 29
- Resumable checkpoint size reported by Kaggle: 868,763,419 bytes
- Downloaded checkpoint fragment: `part-000`, 199,229,440 bytes
- Fragment SHA-256: `f5b713eb6d4b082d8f26001cd149a07ac15a07d2b4d92ff81a5ad45bdaa793c8`
- The fragment is not independently loadable; reconstruction requires parts 000--004 and the parts manifest.

## Training result

Both seeds completed 4,224 microsteps with zero skipped optimizer updates. Peak allocated memory was approximately 4.77 GB per T4. Fixed-noise overfit improvement was 54.58% for seed 17 and 55.97% for seed 29.

## Held-out audit

| Metric | Seed 17 | Seed 29 |
|---|---:|---:|
| Candidate token F1 | 0.527 | 0.536 |
| Frozen-base token F1 | 0.253 | 0.253 |
| Quality pass rate | 0.813 | 0.891 |
| Prompt leakage | 0 | 0 |
| Probe semantic accuracy | 0.625 | 0.813 |
| Probe commitment accuracy | 0.625 | 0.750 |
| Safe-abstention accuracy | 1.000 | 0.875 |
| Safe-abstention committed rate | 1.000 | 0.875 |
| Mean lane-summary cosine | 0.707 | 0.805 |

Seed 29 passed every held-out capability sub-gate except validation calibration. Seed 17 also failed semantic and probe-commitment gates.

## Calibration failure

For selected seed 29, generated-validation calibration used 64 isolated behavior anchors. Generated-candidate semantic accuracy was 0.875. The fitted three-head gate accepted 0.839 of positive candidates but rejected only 0.583 of negatives, producing balanced accuracy 0.711. The preregistered negative-rejection minimum was 0.70, so calibration and the complete capability gate failed.

The fitted operating point was degenerate: commitment threshold 0, risk threshold 0.0010605, and verifier-error threshold 0.0010123. Threshold tuning cannot repair the underlying overlap between positive and negative policy scores.

## Decision

Version 5.4 is a substantial improvement over Version 5.3 and supports continued HLWM experimentation. It is not production-ready and does not authorize deployment or a superiority claim over Qwen.

The next controlled version should train the policy heads on generated training-anchor emissions and semantically graded hard negatives before fitting thresholds on generated validation emissions. Routing gates must also be strengthened: seed 17 used one route for 98.4% of audited decisions, while seed 29 materially used only two routes.
