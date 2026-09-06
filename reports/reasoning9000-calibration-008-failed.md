# Post-hardening calibration 008 — failed final acceptance

- Stable range: offset 270, count 10 (indices 270–279)
- Eligible source chunks after deterministic filtering: 47,743
- Non-substantive source chunks removed: 960
- Approved blueprints: 9/10
- Locally valid constructed episodes: 7/10
- Construction failures: 3/10 (one low-uplift blueprint, one repeated arithmetic failure, one bounded Azure timeout failure)
- Primary first-pass accepts: 1/7
- Dual-judge first-pass accepts: 1/7 constructed, 1/10 scheduled
- Repairs: 0
- Required gate: at least 5/10 dual accepts
- Result: failed

## What improved

- No front-matter, acknowledgment, copyright, table-of-contents, biography, index, or bare-reference false accept was observed.
- The duplicated-verification reconstruction defect from calibration 007 did not recur.
- The sole dual accept passed manual inspection with no blocking defect.

## Primary rejection pattern

- Two accepted blueprints were still below the final 0.80 expertise-uplift floor.
- Four episodes contained substantive implementation defects: inconsistent retry recovery, incorrect TensorFlow mask composition, an ambiguous serialization boundary mutation, or an incompletely exercised operating-control counterexample.
- Several verifier records contained all required labels but still restated lane artifacts instead of independently applying a changed boundary to a frozen artifact.

## Required hardening before another calibration

Add a pre-final construction-quality critic that reviews each locally valid episode during the bounded construction loop and returns defects to the constructor before the episode is saved. This critic must be distinct from the planner, constructor, blueprint critic, and both final judges. It is not a repair stage: rejected candidates remain inside the original construction attempt budget, and the final dual judges still see the resulting episode for the first time.

All thresholds, no-repair calibration policy, dual fail-closed final review, disjoint offsets, and the physical scale lock remain unchanged. Offsets 270–279 are retired permanently.
