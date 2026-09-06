# Post-hardening calibration 006 — failed before episode construction

- Stable range: offset 250, count 10 (indices 250–259)
- Blueprint planner/critic-rewriter: `gpt-5.6-luna`
- Episode constructor: `DeepSeek-V4-Flash`
- Blueprint quality gate and planned primary episode judge: `gpt-5.4-mini`
- Planned independent secondary episode judge: `Kimi-K2.6`
- Blueprint attempts recorded: 37
- Independent blueprint reviews recorded: 21
- Approved blueprints: 0/10
- Generated or judged episodes: 0
- Repairs: 0

## Exact attempt-level rejection breakdown

- 21 independent blueprint-quality rejections;
- 5 sibling-lane dependency violations;
- 3 malformed blueprint shapes;
- 2 source-support span ID transcription errors;
- 6 other static contract failures;
- 1 job ended after its first malformed response exposed a `None`-handling defect in the new shape normalizer.

The clarified GPT-5.4-mini critic still repeatedly rejected declared hypothetical scenario premises merely because they were scenario-invented, despite the system contract explicitly requiring those premises to be treated as authoritative task inputs. It also continued to request completed artifacts at blueprint stage. Substantive rejections—internal arithmetic mismatches, underspecified state transitions, weak discriminators, source-near tasks, and ambiguous lane ownership—remain valid and must stay fail-closed.

## Required hardening before another calibration

1. Correct missing-list handling in the mechanical normalizer.
2. Resolve a mistyped support-span ID only when the candidate's exact support quote uniquely matches a pipeline-issued span in the cited chunk.
3. Move blueprint-quality review to a fifth model deployment, separate from the planner, constructor, and two final judges. The final judge pair remains `gpt-5.4-mini` plus `Kimi-K2.6`.
4. Preserve all thresholds, deterministic checks, dual final review, no-repair calibration policy, and the physical scale lock.
5. Permanently retire offsets 250–259; do not repair, promote, or reuse them.

Pilot 5 and 9,000-scale generation remain blocked.
