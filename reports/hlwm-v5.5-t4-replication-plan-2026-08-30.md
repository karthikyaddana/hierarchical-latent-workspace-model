# HLWM V5.5 Preregistered Plan — Kaggle dual-T4 multi-seed replication

Date: 2026-08-30
Package: `artifacts/kaggle/hlwm-v5.5/hlwm-v5.5-candidate-bundle.zip` (manifest `5.5.0`, unchanged)
Notebook: `artifacts/kaggle/hlwm-v5.5/embel-hlwm-v5.5-kaggle-2xt4.ipynb`
Supersedes nothing: `reports/hlwm-v5.5-plan-2026-08-30.md` remains the plan for the
A100 MIG venue. This plan governs the Kaggle T4 runs.

## Why this venue and this scale

The first A100 MIG `1g.5gb` session crashed at step 896 — the exact first
joint-phase step — with an allocator failure surfacing as an NVML internal
assert: the joint stage is the memory-heaviest phase and another tenant held
~1.18 GB of the 4.75 GB slice. Before the crash the run was healthy: 0.59
s/step, zero skipped updates, BF16 path confirmed, router marginal entropy
near uniform. The architecture is not in question; the venue was.

Kaggle T4 x2 provides 15 GB per GPU (Version 5.4 peaked at ~4.77 GB at these
lengths), ~12-hour sessions, two devices for two independent seeds, and 30
hours of weekly quota. This directly serves roadmap step 2 (multi-seed
replication) while re-attempting step 1 (a passed Version 5.5 gate).

## Deltas from the MIG preregistration (declared, not silent)

| Quantity | MIG plan | This plan | Reason |
|---|---|---|---|
| Seeds per session | 1 (29) | 2 (17, 29; then 41, 73) | two devices |
| Microsteps per seed | 2,560 | 4,224 (128+1,024+3,072) | v5.4-proven T4 count; 2,560 was a 3.5-h cut |
| Token budgets | 160/96/96/320 | 192/128/96/384 | v5.4-proven at ~4.77 GB on 15 GB |
| Precision | BF16, no scaler | see deviation D1 below | `auto` resolved differently than predicted |
| Audit outputs per seed | 48 | 64 | tighter rate estimates; v5.4 precedent |
| Training cap | 2.3 h | 8.5 h | 12-h session |

### Deviation D1 — precision resolved to BF16, not FP16 (observed 2026-08-30, session A)

This plan predicted FP16 + GradScaler on Turing. The actual session A run
resolved `--precision auto` to **BF16** on both T4s. `train_kaggle.py:1527`
selects BF16 whenever `torch.cuda.is_bf16_supported()` is true, and under
torch 2.10 / CUDA 12.8 that returns true on compute capability 7.5 — it
gates on CUDA support for the dtype, not on the presence of BF16 tensor
cores, which Turing lacks. Preflight logged `"precision": "bf16"`,
`"amp_dtype": "torch.bfloat16"`, `passed: true`, peak 3.86 GB of 14.56 GB.

Consequences, recorded before results are known:

- **Numerically safe.** BF16 carries the FP32 exponent range, so the
  overflow/underflow failure mode the GradScaler exists to manage is absent.
  The preflight's finiteness hooks passed across 281 checked stage calls.
- **Slower than predicted.** Without BF16 tensor cores these matmuls do not
  get the acceleration FP16 would have received on this hardware. Observed
  throughput is nonetheless adequate (see session A timing).
- **Weakens one gate's evidentiary value.** `skipped_optimizer_updates == 0`
  is trivially satisfied with no loss scaler in play; it should not be read
  as evidence about numerical stability in this venue. It remains asserted.
- **Improves cross-venue comparability.** The A100 MIG run also used BF16,
  so T4 and MIG results are now numerically comparable rather than
  confounded by an FP16/BF16 difference. This is a genuine, if accidental,
  benefit for the replication claim.
- Precision was **not** re-pinned mid-session. Changing it would have
  invalidated the run in progress. Session B will use the same `auto`
  resolution so all four seeds match.

No preregistered threshold is changed by this deviation.

Architecture, optimizer, LoRA configuration, on-policy head phase (96
emissions, 400 epochs, lr 0.001), calibration (64 validation emissions,
`validation_generated_candidate_joint_threshold_v2`, test split untouched),
router entropy weight 0.02, and every gate threshold are unchanged from the
Version 5.5 preregistration.

## Session plan and quota ledger (30 h available this week)

| Session | Content | Est. charge |
|---|---|---|
| A | seeds 17 + 29 concurrently, full pipeline, per-seed audit, packaging | 6–9 h |
| B | seeds 41 + 73 (`HLWM_SEEDS=41,73`), same pipeline | 6–9 h |
| C | external benchmark harness (roadmap step 3) on the passing checkpoints | 2–3 h |
| reserve | resumes for truncated seeds, one re-run on failure | remainder |

Session B runs only after session A's artifacts are reviewed. A truncated
seed saves `hlwm-v5.5-seed-<seed>-resumable.pt`; the next session resumes it
by attaching that file (discovered by exact name). No gate claim is valid
from a truncated seed.

## Preregistered endpoints

Per-seed gate: all Version 5.5 sub-gates, evaluated once on 64 fresh-process
outputs — leakage 0, quality ≥ 0.75, complete answers ≥ 0.50, semantic
≥ 0.75, safe abstention ≥ 0.75 with commit rate ≥ 0.70, commit accuracy
≥ 0.70, validation calibration ≥ 0.70/≥ 0.70/≥ 0.70 fitted after the
on-policy head phase, positive margins, second-route load ≥ 0.10, normalized
route entropy ≥ 0.25, least-used-expert intervention |Δ commit| ≥ 0.01, lane
cosine < 0.90, candidate F1 ≥ 0.8× base, `training_complete` with zero
skipped updates.

- **Primary endpoint:** both session-A seeds (17, 29) pass the full per-seed
  gate. This is the replication claim.
- **Secondary endpoint:** ≥ 3 of 4 seeds across sessions A and B pass, with
  the three routing gates passing on every completed seed.
- FP16 note: `skipped_optimizer_updates == 0` is asserted per seed. A seed
  with skips fails `training_complete` regardless of other results.

A passing primary endpoint authorizes the external-benchmark step (GSM8K,
ARC, MMLU subsets against Qwen3-0.6B-Instruct and same-size peers), measuring
both standard accuracy and selective/abstention accuracy. It is not a
production-readiness or Qwen-superiority claim.

## Benchmark preview (step 3, harness to be built during session A)

Non-binding sketch, to be preregistered in its own plan before session C:
~250-item subsets of GSM8K and ARC-Challenge plus a fixed MMLU slice; HLWM
committed-answer accuracy, abstention-adjusted selective accuracy, and raw
candidate accuracy vs `Qwen/Qwen3-0.6B-Base` and `Qwen3-0.6B-Instruct` under
identical prompting and token budgets; fixed seeds; scored by the
deterministic graders where applicable and exact-match elsewhere.

## Environment fixes carried into this notebook

Version-pinned installs that survive preinstalled different versions
(metadata version check, `--user` fallback, `sys.path` refresh); pytest
fallback to direct execution; `garbage_collection_threshold:0.8` instead of
`expandable_segments` (MIG incompatibility found 2026-08-30); `HF_HOME`
pinned under `$HOME`; widened bundle glob `hlwm-v5*candidate-bundle.zip`
(upload paths strip dots); dual-GPU assert with per-device memory print.
