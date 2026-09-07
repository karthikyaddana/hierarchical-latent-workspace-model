# Results: v11 pilot — vocabulary-grounded thoughts against the necessity instrument

Preregistration: `hlwm-v11-vocab-grounded-plan-2026-09-07.md` (frozen before any
v11 training step). Kernel `sirishayaddanapudi/hlwm-v11-vocab-grounded-pilot`,
Kaggle T4×2, started 2026-09-07 02:44 UTC. Matched control: the certified v10.0
run, same data, seeds, budget and audit.

## Headline

**H-v11 is not supported. The channel still transfers no usable content.**

Masked-row generation scored **0.000 on both seeds**, exactly as the certified
v10.0 run did, while the premise probe read **0.773 against a 0.129 chance
baseline**. Constructing each thought inside the decoder's own vocabulary basis
did not convert readability into usability at this budget.

## Gate verdicts

| Gate | Bar | Seed 17 | Seed 29 | Verdict |
| --- | --- | --- | --- | --- |
| **G-A** (primary) | `channel_masked − causal_masked ≤ −0.10` nats | **−0.0089** (n=54) | not measured | **FAIL** |
| **G-B** (headline) | masked EM > 0, full > shuffled | 0.000 | 0.000 | **FAIL** |
| **G-C** (no regression) | unmasked ±0.05, warm ≥ 0.50, zero leak | pass | partial | **PASS** |
| **G-D** (mechanism) | in-subspace probe ≥ chance + 0.20 | pending | not run | **pending** |

G-A detail, seed 17 masked rows: `channel_masked` 1.100844, `causal_masked`
1.109767. The certified v10.0 baseline was **+0.005 / +0.023**, so the sign did
flip — the channel now costs slightly less than nothing rather than slightly
more — but −0.0089 nats is an eleventh of the preregistered bar. Reporting the
sign flip as progress would be reading noise as signal; the honest statement is
that the measure moved in the predicted direction by an amount the plan
declared, in advance, to be insufficient.

Per-family (seed 17, `channel_masked − causal_masked`): abstention −0.041
(n=19), ordering −0.043 (n=36), unit −0.028 (n=36), numeric **+0.050** (n=37).
The one family whose gold answers are the digits the whole instrument is built
around is the one where the channel still *costs* the decoder.

G-C detail (seed 17): unmasked parity gap −0.0329 (bar ±0.05), warm gate
em_600 0.5930 (bar 0.50), input-leak rows 0, scaffold rows 0. Seed 29 passed
its warm gate (em_600 0.6105) but has no audit, so its unmasked-parity and leak
readings are absent; G-C is recorded as passing on seed 17 and partial overall.

Other measured quantities (seed 17): `expert_liveness_ratio` 5.52,
`router_agreement` 1.000, `cross_routing_cost` 0.215, `failure_auroc_head`
0.996, `masked_core_n` 160, `probe_heldout_n` 263.

## Deviations

- **D-v11-1 (material): seed 29 has no audit.** The kernel was cancelled at
  roughly 8.6 h elapsed, during seed 29's audit; seed 17's audit had already
  completed and was written. Both seeds' training completed to step 1200 and
  both wrote checkpoints and metrics. Whether the cancellation was Kaggle's
  session limit, a GPU-quota exhaustion or a manual stop is **not determinable
  from the artifacts**, and is not asserted here. Consequence: every audit-side
  number above is single-seed. The two-seed claim in this record is limited to
  the training-side gates (warm gate, go/no-go), which agree.
- **D-v11-2:** G-D runs post hoc from the retrieved seed-17 checkpoint rather
  than inside the kernel, because the kernel was cancelled before that stage.
  Same script (`scripts/jspace_projection_audit.py`), same preregistered
  Amendment A1 secondary, run on CPU.

## Reading under the preregistration

The plan fixed the readings in advance. G-A fails, so the branches are:

- **G-A fails with G-D passing** → alignment was achieved and the content is
  still unused: H-v11 is wrong in an informative way. Stop and report.
- **Nothing passes** → stop and report; no third configuration inside this
  preregistration.

Either way the plan's instruction is *stop and report*, and no extension is
authorized here. The budget extension (main 600 → 2400) was conditioned on G-A
passing without G-B, which did not occur.

What this does not license: the capability question at published-success
budgets (≥2×10⁵ presentations) remains out of scope and undecided, exactly as
in the closed v10 record. This pilot tested whether write alignment was the
binding constraint at the measured failure point. On the primary measure, at
this budget, on one audited seed: it was not the only one.

## Provenance

Code: `manifest-v11.json` (sha256 per file) in the kernel's mounted dataset and
in `successor/v11/`. Checkpoints `checkpoint-step-001200.pt` for both seeds,
sha256 sidecars beside them. Retrieved artifacts: `preflight.json`, per-seed
`metrics.jsonl`, seed 17 `v10-audit.json` and `v10-gates.json`, both training
logs and the orchestration log.
