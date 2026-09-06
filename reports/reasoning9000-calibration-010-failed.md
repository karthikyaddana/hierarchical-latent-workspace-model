# Post-hardening calibration 010 — stopped after mathematical failure

- Stable range: offset 290, count 10 (indices 290–299)
- Approved blueprints: 9/10
- Construction-critic reviews recorded: 7
- Episodes accepted through construction: 0/10
- Permanently failed jobs before stop: 6/10
- Remaining in-flight jobs when stopped: 4/10
- Final judging: not run
- Repairs: 0
- Result: stopped because reaching 5/10 was mathematically impossible

## Conclusive bottleneck

Sending DeepSeek the full rejected episode improved edit specificity but did not improve the pass rate enough. The construction critic continued to find real defects—incorrect acronym boundary code, missing selected constraint artifacts, absent ordered central reconstruction, contradictory arithmetic, and invalid schema-scale blueprint output—and DeepSeek did not reliably correct them within the four-attempt budget.

DeepSeek remains useful as the first-draft parallel reasoner. It should no longer own all critic-driven full rewrites.

## Required role change

- Attempt 1: `DeepSeek-V4-Flash` constructs the first episode draft.
- Attempts 2–4: `gpt-5.6-luna`, acting as a non-final episode editor/reconstructor, rewrites the rejected draft from the approved blueprint and bounded structured critic feedback.
- The construction critic remains `DeepSeek-V3.2-Speciale`.
- Critic feedback is capped at the three highest-priority blocking defects and must use a compact category/location/defect/correction form.
- `gpt-5.4-mini` and `Kimi-K2.6` remain independent final judges and receive no role in drafting or editing.

The total four-attempt budget, thresholds, no-repair calibration policy, disjoint offsets, and physical scale lock remain unchanged. Offsets 290–299 are retired permanently.
