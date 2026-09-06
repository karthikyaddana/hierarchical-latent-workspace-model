# Post-hardening calibration 007 — failed final acceptance

- Stable range: offset 260, count 10 (indices 260–269)
- Blueprint planner/critic-rewriter: `gpt-5.6-luna`
- Independent blueprint quality critic: `Kimi-K2.5`
- Episode constructor: `DeepSeek-V4-Flash`
- Primary final judge: `gpt-5.4-mini`
- Independent secondary final judge: `Kimi-K2.6`
- Approved blueprints: 10/10
- Locally valid constructed episodes: 8/10
- Construction failures: 2/10 (one local contract failure; one bounded Azure malformed-JSON/timeout failure)
- Primary first-pass accepts: 1/8
- Dual-judge first-pass accepts: 1/8 constructed, 1/10 scheduled
- Repairs: 0
- Required calibration gate: at least 5/10 dual accepts
- Automated result: failed
- Manual result: failed

## Primary rejection pattern

All eight constructed records passed static and deterministic adversarial checks before model review. The primary judge rejected seven:

- five episodes scored below the 0.80 expertise-uplift floor, usually because the constructor rendered an accepted blueprint as a compact pattern application, checklist, or source-near adaptation;
- three episodes repeated the same central reconstruction across multiple claim-verification rows instead of producing claim-specific independent verifier outcomes;
- one distributed-systems episode contained a substantive recovery/state-transition inconsistency;
- several episodes included unnecessary `proposed` or `not_run` tool records that looked more evidentiary than their status allowed.

The categories overlap because one episode may contain more than one defect.

## Manual audit of the sole dual accept

The sole dual accept, `research-and-decision-making-ff78b14c419b5712`, is not eligible for promotion. Its selected source chunk is an acknowledgments page from *The Psychology of Money*. The episode accurately quotes the page, but the quoted acknowledgments contribute only decorative words about support and feedback; the actual statistical task is supplied by invented scenario premises. This violates the requirement that source material contribute a substantive transferable mechanism rather than topic words. The accepted set therefore contains one manual false accept.

## Required hardening before another calibration

1. Deterministically exclude obvious acknowledgments, copyright/colophon, table-of-contents, author-biography, index, and bare-reference chunks from generation job selection.
2. Require every claim's verification row to use claim-specific visible inputs, operation, reconstructed result, mutation, recomputed outcome, rejection rule, and verdict. Reusing one central transcript across claims is forbidden.
3. Preserve expert depth during construction: instantiate full state traces, calculations, artifact fields, edge cases, and discriminator logic rather than collapsing the blueprint into a checklist.
4. Emit an empty `tool_runs` list when the pipeline supplied no execution evidence and no tool proposal is necessary.
5. Keep all thresholds, all five deployment roles, dual fail-closed final review, no-repair calibration policy, and the physical scale lock unchanged.
6. Permanently retire offsets 260–269; do not repair, promote, or reuse them.

Pilot 5 and 9,000-scale generation remain blocked.
