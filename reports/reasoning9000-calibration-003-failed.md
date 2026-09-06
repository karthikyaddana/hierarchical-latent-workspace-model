# Post-hardening calibration 003 — failed

- Stable range: offset 220, count 10 (indices 220–229)
- Approved blueprints: 10/10
- Locally valid episodes generated: 9/10
- First-pass primary-judge accepts: 2/10
- First-pass dual-judge accepts: 0/10
- Promotion requirement: at least 5/10 dual-judge accepts with zero deterministic false accepts
- Result: failed; no repair, no promotion, and no Pilot 5

## Generation result

Nine episodes passed local construction. `general-reasoning-and-learning-849eb96e7afbc5ac` exhausted four construction attempts: two drafts cited packet-valid support-span IDs that the episode canonicalizer did not yet translate, one draft omitted the required second counterfactual, and the final draft remained invalid.

## Deterministic-gate correction

The first judge invocation incorrectly stopped eight episodes before model review. These were deterministic false rejections caused by three parser defects:

- ordered reconstruction labels separated by periods were not recognized;
- `$820 / $4,100 = 0.20` was parsed as the false fragment `4,100 = 0.20`;
- a proposed future survey or A/B test was misclassified as claimed execution.

The eight wrappers were preserved under `data/reasoning9000/history/calibration-003-prehardening-gate-offset220`. The parser was corrected, regression tests were added, and all nine unchanged episodes then passed deterministic replay. The original primary review for the ninth episode was preserved; only the eight false rejections were resubmitted.

## Model-review result

Six episodes failed the primary judge. Two more passed primary review but failed the independent secondary judge. The recurring blocking defects were:

- shallow, source-near tasks with expertise uplift below 0.80;
- a second lane that reviewed a sibling artifact instead of independently solving a necessary subproblem;
- hypothetical falsification language rather than an applied mutation, recomputed outcome, and rejection rule;
- internally inconsistent arithmetic, probability events, counts, or artifact behavior;
- scenario facts required by the task but absent from the approved premises;
- unsupported external outcomes such as exact runtime errors, compilation validity, empirical effects, or invented ROI inputs;
- incomplete code/design artifacts that did not preserve the named behavior or instantiate the claimed extension;
- claimed token totals that did not match the frozen artifacts.

## Required hardening before calibration 004

- Add a separate quality review of every structurally valid blueprint before episode construction. It must reject weak expertise uplift, source paraphrase, incomplete scenario premises, infeasible evidence, duplicated lanes, and ornamental falsification.
- Enforce lane isolation locally: a pre-barrier claim may not depend on an artifact owned by another lane.
- Run deterministic acceptance checks during episode construction so the teacher can correct failures before a record is written.
- Require central falsification to contain a concrete mutation, recomputed outcome, and explicit rejection rule.
- Canonicalize any packet-valid `chunk-id::span-NNN` reference to its known chunk ID.
- Recompute declared whitespace-token totals directly from their cited artifacts.
- Normalize escaped newline labels before review and tell judges that the ordered verifier record belongs in `verification[].result`, not necessarily in the user-facing answer.

The physical `data/reasoning9000/STOP` scale lock remains present. Calibration 003 records are permanently excluded from Pilot 5.
