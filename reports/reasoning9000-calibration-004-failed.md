# Post-hardening calibration 004 — failed before episode construction

- Stable range: offset 230, count 10 (indices 230–239)
- Blueprint attempts recorded: 32
- Independent blueprint reviews recorded: 13
- Jobs permanently failed before stop: 7/10
- Approved blueprints: 0
- Generated episodes: 0
- Result: mathematically unable to reach 5/10; stopped before further spend, with no judging, repair, promotion, or Pilot 5

## Why the run was stopped

Once six of ten jobs had permanently failed blueprint construction, at most four records could remain, so the required 5/10 dual-judge acceptance was impossible. The remaining calls were stopped and the range was retired.

## Rejection audit

The failures were substantively valid rather than false gate rejections. They included:

- unsupported exact GUI, webhook-payload, runtime, API, and empirical behavior;
- source-near textbook tasks with expertise uplift far below 0.80;
- solver lanes that consumed or reviewed sibling artifacts before the barrier;
- duplicated artifact ownership and claim IDs used as premise IDs;
- support-span IDs placed where canonical chunk or premise evidence was required;
- infeasible verifier simulations and ornamental counterexamples;
- underdetermined recommendations presented as source-backed conclusions.

The first independent gate used `gpt-5.4-mini` only as a critic. DeepSeek repeatedly failed to rewrite rejected blueprints into contract-compliant candidates within four attempts. Running another calibration with the same role assignment would repeat the failure mode.

## Required model-role separation

The next calibration must use:

- `gpt-5.6-luna` as the blueprint planner and critic-rewriter;
- `DeepSeek-V4-Flash` only after blueprint approval, for episode construction;
- `gpt-5.4-mini` as the primary final episode judge;
- a distinct independent final judge for dual fail-closed acceptance.

The planner must not judge its own final episode. Thresholds remain unchanged. The physical `data/reasoning9000/STOP` scale lock remains present, and offsets 230–239 are permanently excluded from Pilot 5.
