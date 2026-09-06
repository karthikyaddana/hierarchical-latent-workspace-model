# Post-hardening calibration 005 — failed before episode construction

- Stable range: offset 240, count 10 (indices 240–249)
- Blueprint planner/critic-rewriter: `gpt-5.6-luna`
- Episode constructor: `DeepSeek-V4-Flash`
- Blueprint quality gate and planned primary episode judge: `gpt-5.4-mini`
- Planned independent secondary episode judge: `Kimi-K2.6`
- Blueprint attempts recorded: 40
- Independent blueprint reviews recorded: 14
- Approved blueprints: 0/10
- Generated episodes: 0
- Judged episodes: 0
- Repairs: 0
- Result: failed the required minimum of 5/10 first-pass dual-judge accepts

## Exact attempt-level rejection breakdown

Each of the 40 attempts was classified by its first decisive rejection family:

- 14 independent blueprint-quality rejections;
- 10 pre-barrier sibling-artifact dependency violations;
- 8 misplaced or missing top-level `constraint_trace` structures;
- 4 malformed lane or artifact structures;
- 3 execution-claim false positives caused by the validator treating the noun `build` as a claimed successful build outcome;
- 1 other static contract failure.

The quality gate also produced several false or overbroad objections: it rejected fully declared hypothetical scenario premises merely because they were invented, and it sometimes required completed episode artifacts during blueprint review. Those objections conflict with the blueprint contract, which permits explicit scenario premises and evaluates planned artifacts before episode construction. Other objections—source-near tasks, weak lane independence, underspecified discriminators, and ornamental falsification—were substantive.

## Required hardening before another calibration

1. Preserve all thresholds and the four distinct model roles.
2. Mechanically normalize only unambiguous JSON-shape errors such as a misplaced `constraint_trace`, a one-key lane wrapper, a span ID used where its parent chunk ID is required, and `logical_counterexample` used as a claim type rather than an evidence mode.
3. Correct the execution-outcome regex so nouns such as `build ordering` are not treated as fabricated successful execution.
4. Tell the independent blueprint judge to treat declared scenario premises as authoritative hypothetical inputs, while still requiring source material to contribute a real transferable mechanism.
5. Tell the judge not to demand completed artifacts at blueprint stage; it must assess whether the planned artifact contract is sufficient for later deterministic review.
6. Preserve the failed range permanently. Do not repair, promote, or reuse any calibration-005 record.

The physical `data/reasoning9000/STOP` scale lock remains present. Pilot 5 and 9,000-scale generation remain blocked.
