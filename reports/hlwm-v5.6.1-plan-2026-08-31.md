# HLWM V5.6.1 Preregistered Plan — two variance fixes, everything else frozen

Date: 2026-08-31
Package: `artifacts/kaggle/hlwm-v5.6.1/hlwm-v5.6.1-candidate-bundle.zip`
(manifest `5.6.1`; sha256 in `build-report.json`)
Notebook: `artifacts/kaggle/hlwm-v5.6.1/embel-hlwm-v5.6.1-kaggle-2xt4.ipynb`
Governing results: `reports/hlwm-v5.6-session-a-results-2026-08-31.md`
Inherits everything not listed below from `reports/hlwm-v5.6-plan-2026-08-30.md`.

## The only two changes

| Change | Mechanism | Traces to |
|---|---|---|
| `--expert-init-scale 0.01` (expert up-projections initialized at std 0.01 instead of zero) | Session A telemetry showed the expert-diversity penalty inert all run (0.0002–0.0018): zero-initialized output projections keep all expert deltas at zero, so there is nothing to decorrelate exactly when collapse pressure is strongest. Nonzero init gives each expert an independent function from step 1, making the penalty live and breaking the symmetric start. | Seed 17's 100% one-expert collapse |
| `--policy-records 128` with per-family round-robin (`balanced_anchor_order`) | 11 of seed 29's 12 wrong commit decisions were confident false rejections of valid answers scoring exactly at the smoothed negative rail — a family (ordering weakest) that random head-phase sampling under-represented as positives. | Seed 29's 0.625 commit accuracy (gate 0.70) |

Data, budgets, routing weights, label smoothing, steps (4,224), gate
thresholds, and audit design are unchanged. Dataset is byte-identical to
v5.6 (same `data/hlwm-v5.6/`); only trainer code and the two flags differ.

## Session and endpoints

Session B: seeds 17 + 29 again (deliberately the same seeds — the question
is whether the fixes rescue seed 17's exact failure mode and lift seed 29's
one remaining behavioral gate). Est. 6–7 h of ~19 h quota. Fresh training
(do not attach v5.6 resumables; names changed to `hlwm-v5.6.1-*` to prevent
accidental resume).

- **Primary endpoint (unchanged):** both seeds pass the complete gate.
- **Diagnostic sub-questions, declared in advance:** (a) does
  `expert_diversity` telemetry now move materially above its inert 0.0002–
  0.0018 band; (b) does seed 17 avoid the workspace-prefix degeneration
  (prompt-leak 0, lane cosine < 0.90); (c) does seed 29's probe commit
  accuracy clear 0.70 with the balanced head phase; (d) does the
  intervention delta reach 0.01 now that diversity pressure is live.
- Fallback ladder unchanged: if both seeds again fail the routing gates,
  `--num-experts 2`; if joint-phase outcomes remain a seed lottery, simplify
  the workspace per the paper's reduction rule.
- The plain-LoRA control (code complete, 36/36 tests) runs after a stable
  pair exists — postponed by explicit user decision 2026-08-31.

## Paper state

`paper/hlwm-paper.tex` updated 2026-08-31 with Study 5 (v5.6 session A):
abstract, provenance table, §9.6 with full results table, failure-mode rows
(routing collapse annotation + new joint-phase seed variance row), Known
Limitations item 6, conclusion. 35 pp compiled clean; installed at root.
