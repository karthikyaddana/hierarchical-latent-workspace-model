# HLWM V5.7 Preregistered Plan — workspace-only

Date: 2026-09-01
Governing results: `reports/hlwm-v5.6.2-session-c-results-2026-09-01.md`
Inherits everything not listed below from `reports/hlwm-v5.6-plan-2026-08-30.md`
(data byte-identical; token budgets, 4,224 steps, LR, label smoothing, audit
design, and all behavioral gate thresholds frozen).
Artifacts: `artifacts/kaggle/hlwm-v5.7/` (bundle sha 5a76c709…, 42/42 tests).

## Why workspace-only

Study 7 met rung 1 of the ladder ("if either seed still collapses at two
experts"): seed 17 collapsed a third time under a third configuration, and
causal liveness is 0-for-10 lifetime. The routed expert graph exits the
small-scale claim; the paper's routing sections stand as proposal plus
negative evidence. v5.7 measures what the workspace itself — lanes,
synthesis, verification, calibrated commitment — delivers without routing,
and ships the two repairs Study 7 made mandatory.

## The four changes

| Change | Mechanism | Traces to |
|---|---|---|
| `--num-experts 1` | A single always-active workspace adapter; no routing decision exists. Trainable budget stays matched to Study 6 within 0.9% (74,508,423 vs 75,174,348; LoRA rank unchanged), so the declared content comparison is clean. Router-aux and expert-diversity weights set to 0 (no siblings); `output_diversity` guarded to return 0 below two experts. | Rung 1; Study 7 content regression |
| De-correlated policy-head labels | Commit trains on semantic validity; risk on the mechanical-corruption event **only** (corrupt → [0,1,0]); verifier-error on the fluent-but-wrong event **only** (invalid emission / wrong-value reference → [0,0,1]); ranking losses applied per matching kind. The three thresholds no longer describe one valid/invalid axis. Method tag `on_policy_train_anchor_head_fit_v2_decorrelated`, asserted by the notebook. | Six degenerate-threshold recurrences; Known Limitations item 3 |
| Domain-stratified canary | Canary rows round-robin over dataset domains (not anchor families); semantic validity graded only by loop-safe graders (code-executing graders stay in the audit); per-run domain list logged. | Study 7: anchor-only canary provably blind to routing and non-anchor pathologies |
| Gate battery 18 → 15 | The three routing gates (second-route load, normalized entropy, causal liveness) are removed with the mechanism they measured; routing metrics remain report-only in the audit. Lane distinctness stays (lanes still exist). | Rung 1 |

Kept: validity-gated head phase (≥16 valid positives per family, cap 512 —
it worked), expert-init-scale 0.01, all v5.6 data/budgets/steps.

## Session and endpoints

Session D of the week: seeds 17 + 29, fresh training (older resumables are
shape-incompatible), est. 5–6 h of the ~9 h remaining quota.

- **Primary endpoint:** both seeds pass the complete 15-gate battery.
- **Declared content question (the reduction caveat test):** does probe
  content accuracy recover to Study 6 levels (0.75–0.78)? If yes, the
  six-expert content contribution was fungible capacity; if not, the paper's
  reduction claim carries the caveat that the expert bank was load-bearing
  for content.
- **Declared diagnostics:**
  (a) commitment threshold non-degenerate (∈[0.02, 0.98]) for the first time
  — the de-correlation fix's direct test;
  (b) validity floor met, and commit accuracy ≥0.70 on both seeds now that
  labels are de-correlated and (if content recovers) the generator is back to
  Study 6 strength;
  (c) canary domain coverage: leak/marker/validity trajectories per domain
  bucket, and whether any failure seen in the audit was visible to the canary
  by step 2,048;
  (d) lane distinctness at 15-gate scope: does removing routing pressure
  change lane cosine (Study 7 seed 29 passed it for the first time).
- **Fallback ladder:**
  1. Both seeds pass all 15 gates → the plain-LoRA control runs
     **immediately** as the next session on the same seeds (binding order,
     unchanged), then external benchmarks per the draft prereg.
  2. Content does not recover (probe accuracy < 0.70 on either seed) →
     restore capacity explicitly: one always-active adapter widened to match
     the removed expert parameters (bottleneck 64 → ~384) OR LoRA rank 24,
     declared as a capacity study, not a routing revival.
  3. Threshold still degenerate despite de-correlated labels → the joint
     three-threshold publication rule itself is the defect; v5.8 replaces it
     with a single calibrated commit score and scalar threshold.

## Claim boundary

v5.7 makes no routing claim of any kind. A full pass authorizes only the
matched control; nothing here is a production-readiness claim.
