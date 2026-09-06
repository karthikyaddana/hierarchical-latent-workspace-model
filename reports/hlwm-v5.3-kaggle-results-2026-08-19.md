# HLWM Version 5.3 Kaggle result record

Date: 2026-08-19

## Artifact verification

- Downloaded deliverable: `hlwm-v5.3-best-deliverable.zip`
- Observed SHA-256: `8ae6fa290dafd920c2e1ce580cdb7942de0fd0a436014971b8dd42d081edd148`
- Sidecar SHA-256: exact match
- ZIP integrity: all 11 members passed CRC validation
- Best seed: 29
- Resumable checkpoint expected SHA-256:
  `919e39d7233baaaa892fc4d4e8b5a1df1fd65d5ad4f7e0ca74f140bd44c75bff`
- Resumable checkpoint status: checksum downloaded, checkpoint file not yet
  downloaded locally

## Training stability

Both seeds completed 1,024 planned microsteps and 256 optimizer updates with
zero skipped optimizer updates. Peak allocated memory was 4.732 GB for seed 17
and 4.733 GB for seed 29. The fixed-noise overfit loss improved by 46.87% and
47.73%, respectively.

## Held-out audit

Each seed was evaluated on 16 behavior anchors and 16 Reasoning9000 records.

| Metric | Seed 17 | Seed 29 |
|---|---:|---:|
| Candidate token F1 | 0.390 | 0.421 |
| Frozen-base token F1 | 0.288 | 0.288 |
| Quality pass rate | 0.750 | 0.781 |
| Complete-answer rate | 0.531 | 0.500 |
| Prompt-leak rate | 0.000 | 0.000 |
| Probe semantic accuracy | 0.625 | 0.563 |
| Safe-abstention accuracy | 0.000 | 1.000 |
| Probe commitment accuracy | 0.625 | 0.313 |
| Commitment coverage | 0.125 | 0.125 |
| Mean lane-summary cosine | 0.496 | 0.331 |

Both capability gates failed. The run demonstrates stable execution, improved
lane separation, positive candidate-vs-base F1 on this small diagnostic, and
zero observed prompt leakage. It does not demonstrate production readiness or
architectural superiority.

## Failure diagnosis

The validation calibration reported 0.918 balanced accuracy on teacher-forced
clean references and paired corruptions, but the final gate evaluated free
generated candidates. This train/calibration-to-inference mismatch produced
only 12.5% publication coverage and low probe commitment accuracy. In
addition, the hard verifier threshold used private denoising-token error rather
than a score of the frozen public candidate. Correct missing-evidence answers
therefore received high private error and were withheld.

Version 5.4 replaces that hard gate with a candidate-level verifier, calibrates
on semantically labeled generated validation emissions, uses more varied
split-isolated anchors, increases anchor-rich training, and expands the held-out
audit.
