# Post-hardening calibration 009 — stopped after mathematical failure

- Stable range: offset 280, count 10 (indices 280–289)
- Blueprint planner: `gpt-5.6-luna`
- Blueprint critic: `Kimi-K2.5`
- Episode constructor: `DeepSeek-V4-Flash`
- Construction critic: `DeepSeek-V3.2-Speciale`
- Planned final judges: `gpt-5.4-mini` and `Kimi-K2.6`
- Approved blueprints: 9/10
- Construction-critic reviews recorded: 16
- Episodes accepted by the construction critic and local gates: 1/10
- Permanently failed jobs before stop: 8/10
- Remaining in-flight job when stopped: 1/10
- Final judging: not run
- Repairs: 0
- Result: stopped because reaching 5/10 was mathematically impossible

## Diagnosis

The construction critic caught substantive defects that the prior local gate missed: a confounded factorial design, invalid sensitivity arithmetic, a public-interface migration gap, a word-count violation, an arithmetic contradiction, and incomplete counterfactual controls. This validates the need for a construction-quality stage.

However, the constructor's correction rate was too low because retries received issue text but not the rejected candidate they needed to revise. One critic rejection was also invalid: it claimed that the schema-required post-barrier `verification` array was self-judging and should be removed. In this architecture, episode verification is an internal artifact that final judges independently audit; it is mandatory.

The source-quality heuristic also missed an `Other Books You May Enjoy` promotional bibliography packet. The blueprint critic rejected it correctly, but it should be removed before spending a planner call.

## Required hardening

1. On construction retries, send DeepSeek the full rejected episode plus critic/local errors and require a targeted rewrite rather than a fresh reconstruction from issue text alone.
2. Explicitly tell the construction critic that the episode's post-barrier `verification` array is mandatory and distinct from final model judging.
3. Extend deterministic source filtering to promotional book lists such as `Other Books You May Enjoy`.
4. Preserve the original four construction attempts, all thresholds, both independent final judges, no-repair calibration policy, disjoint offsets, and the physical scale lock.

Offsets 280–289 are retired permanently. Pilot 5 and 9,000-scale generation remain blocked.
