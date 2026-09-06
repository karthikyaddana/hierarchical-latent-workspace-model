# Post-editor calibration 011 — stopped after mathematical failure

- Stable range: offset 300, count 10 (indices 300–309)
- Approved blueprints: 10/10
- DeepSeek first-draft attempts: used for every episode
- Luna editor/reconstructor attempts recorded: 22
- Construction-critic reviews recorded: 16
- Episodes accepted through construction: 0/10
- Permanently failed jobs before stop: 9/10
- Remaining in-flight job when stopped: 1/10
- Final judging: not run
- Repairs: 0
- Result: stopped because reaching 5/10 was mathematically impossible

## Result of the requested role change

The role separation worked operationally: DeepSeek produced first drafts, Luna received the full rejected draft plus no more than three prioritized critic defects for attempts 2–4, and neither final judge participated in drafting, editing, or construction criticism. Provenance records distinguish `first_draft_constructor` from `episode_editor_reconstructor`.

Luna produced more targeted and structurally coherent rewrites than DeepSeek, but no candidate cleared the construction critic. Several critic findings were substantive: dtype-policy mismatch, incomplete pipeline handoff contract, invalid arithmetic, leakage ambiguity, incorrect potential-outcome construction, and unresolved serialization or state-boundary behavior.

The construction critic also overreached on some candidates. One rejection explicitly described an extra parsing checkpoint as “not harmful” but still treated it as blocking. This means the critic is not yet calibrated to its own fail-closed contract: it sometimes converts non-blocking design differences into rejection even though the prompt limits it to blocking defects.

## Conclusion

The constructor rewrite bottleneck is improved but not solved. The next change should not lower the 0.84 or 0.80 thresholds. It should calibrate the construction critic against labelled blocking/non-blocking examples or replace it with a critic whose decisions agree with the independent final acceptance standard. No additional calibration should run until that critic precision is demonstrated offline.

Offsets 300–309 are retired permanently. Pilot 5 and 9,000-scale generation remain blocked by the physical STOP file.
