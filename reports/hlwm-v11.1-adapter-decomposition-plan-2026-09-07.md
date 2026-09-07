# Preregistered: v11.1 arm — adapter decomposition (DoRA / PiSSA) as a cold-start control

Frozen 2026-09-07, before any v11.1 training step. Separate arm from the running
v11 pilot (`hlwm-v11-vocab-grounded-plan-2026-09-07.md`), which is preregistered
as "two changes, nothing else" and must not be amended mid-run.

## What this arm is not

DoRA (arXiv:2402.09353) and PiSSA (arXiv:2404.02948) are **not** fixes for the
diagnosed failure. The J-space result
(`hlwm-v10.0-jspace-results-2026-09-06.md`) localized v10.0's failure to *write
alignment*: premise content probed at 0.075/0.100 inside the decoder-sensitive
subspace against 0.400/0.725 outside it. No adapter parameterization moves where
a channel writes; that is what the v11 vocabulary bottleneck addresses. Claiming
otherwise would be the same category error the post-mortem documents.

## What this arm is

A control for a **confound** in that result, not a competitor to it.

The certified run performed **75 optimizer updates**. Standard LoRA zero-inits
`up`, so the sidecar contributes exactly zero at step 0, and — measured directly
in `verify_adapters.py` — `∂L/∂down` is exactly 0 at initialization for every one
of the 28 adapter tensors, because that gradient is proportional to `upᵀ`. The
down projection only becomes trainable after `up` leaves zero. At 75 updates it
is a live possibility that the language adapters were still in their cold start
when the go/no-go fired, in which case "the channel wrote into decoder-inert
directions" would be partly a statement about an under-opened adapter stack
rather than about the objective's preferred write direction.

PiSSA removes exactly this cold start: both factors are initialized from the
base weight's principal singular subspace, so both carry non-zero gradient from
step 0 (measured: `up=4.5e-3 down=5.8e-3` after one step, against LoRA's
`down=5.3e-6`). DoRA adds a per-output-row magnitude, decoupling update
direction from update norm, which the DoRA paper reports as the larger gain
precisely in the low-update, low-rank regime this program runs in.

**H-v11.1:** if the v10.0 alignment verdict is a property of the objective and
not of adapter cold start, then re-running the certified configuration with
`--adapter pissa` (and, separately, `--adapter dora`) reproduces the same
J-space split at the same budget. If instead the split narrows substantially
under PiSSA, the alignment verdict is budget-conditioned and must be reported
with that caveat.

Note the direction of the prediction: **this arm is preregistered to confirm the
existing negative result.** A null here strengthens the J-space finding; a
positive weakens a conclusion already published. It is run because the confound
is real, not because a win is expected.

## Changes

Diff against the v11 harness (which is itself the certified v10.0 harness plus
the two v11 flags; sha256s in `manifest-v11.json`):

1. `LoRAResidual` gains `mode ∈ {lora, dora, pissa}`.
   - **dora**: `magnitude = nn.Parameter(ones(output_width))`, initialized after
     the substrate transplant to `‖W + s·BA‖_row`; forward output scaled by
     `magnitude / ‖W + s·BA‖_row`. 1-D, so the optimizer's existing no-decay
     group picks it up (the same exemption that protects the prefix gates).
   - **pissa**: `down`/`up` initialized from `svd_lowrank(W)`'s top-`r`
     subspace; frozen buffers `pissa_down0`/`pissa_up0` hold that init and the
     sidecar emits `s·(BAx − B₀A₀x)`, which is identically zero at step 0. The
     base weight is left untouched, so the residual-variant of PiSSA — not the
     weight-surgery variant — is what runs here; this keeps the frozen substrate
     bit-identical to every other arm in the record.
2. `--adapter {lora,dora,pissa}` on `train_kaggle.py`, default `lora`.

`adapter_mode="lora"` is byte-compatible with v10/v11 behavior: the DoRA and
PiSSA branches are unreachable and no new parameter or buffer is allocated.

## Preregistered gates

- **G-1 (init identity, already verified offline):** at construction, `dora` and
  `pissa` reproduce the frozen base model's outputs exactly. Recorded:
  `max|diff| = 0.000e+00` for both.
- **G-2 (liveness, already verified offline):** after one optimizer step, every
  enabled adapter tensor in the model carries non-zero gradient in all three
  modes — 0 dead of 28 sites per mode. DoRA's `magnitude` grad is non-zero and
  perturbing it moves the output (`max|diff| = 1.4e-1`), i.e. the gain is wired
  into the forward path.
- **G-3 (no regression):** the v10 battery passes under each mode
  (148 tests × {lora, dora, pissa}). Recorded 2026-09-07: all pass.
- **G-4 (primary):** on the certified configuration and budget, the J-space
  in-subspace vs complement probe split under `pissa` stays within ±0.10 of the
  certified 0.075/0.100 vs 0.400/0.725. Within → alignment verdict holds and is
  not a cold-start artifact. Outside → the verdict is budget-conditioned, and
  the paper's post-termination addendum and the repository README must be
  amended to say so.
- **G-5 (secondary, capability):** unmasked full-pipeline accuracy under each
  mode against the certified 0.517/0.510. Reported, not gated: this program has
  no matched plain-LoRA control, so no adapter comparison here supports a
  capability claim, and none will be made.

## Sequencing

Runs only **after** the v11 pilot reports, on the same T4×2 budget, seeds 17
and 29. If v11 passes G-A/G-B, this arm runs against the v11 configuration
instead of the certified one; that substitution is declared here rather than
chosen after seeing the numbers. Order is fixed: `pissa` first (it is the mode
that bears on the confound), `dora` only if budget remains.

## Provenance

Implementation verified by `verify_adapters.py` (V1 init identity, V2 liveness
after one step, V3 mode separation, V4 magnitude liveness) and by the v10
battery under `HLWM_TEST_ADAPTER_MODE`. Both ship with the code. All deviations
recorded in the results file.
