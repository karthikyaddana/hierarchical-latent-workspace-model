# Reasoning 9,000 — Pilot 4 failure report

## Run identity

- Pilot: `pilot-004-failed`
- Stable job range: offset `150`, count `50`
- Generator and repair model: `DeepSeek-V4-Flash`
- Primary judge: `gpt-5.4-mini`
- Secondary judge: `gpt-5.6-luna`
- Acceptance thresholds: unchanged (`>=50%` first-pass, `>=75%` final, `>=0.85` accepted mean expertise, `>=0.80` per-episode expertise, `>=0.84` overall score, `>=12` accepted domains)
- Scale status: **locked**

## Stage results

1. Generation produced 50 unique, locally valid task blueprints and 50 locally valid episodes.
2. Corrected first pass accepted 0/50:
   - 7 deterministic adversarial-gate rejects;
   - 38 primary-judge rejects;
   - 5 primary accepts rejected by the secondary judge.
3. First repair run saved 26 repairs and failed 24:
   - 21 Azure 429 rate-limit failures;
   - 1 Azure timeout;
   - 1 fail-closed local repair-contract failure;
   - 1 malformed JSON response after transport retries.
4. Low-concurrency retry saved 21 more repairs and failed 3:
   - `edge-ai-91dc2916af4fc46a`: failed the blueprint constraint/artifact/validation trace contract after both repair-validation attempts;
   - `software-architecture-02462c24e1b66736`: failed the blueprint constraint/artifact/validation trace contract after both repair-validation attempts;
   - `product-strategy-ea7e2dc88b009df9`: the second repair-validation response remained malformed JSON after all six transport attempts.
5. In total, 47/50 episodes were repaired once. The other three retained their rejected first-pass episodes.
6. Final dual judging accepted 0/50:
   - 5 deterministic adversarial-gate rejects;
   - 37 primary-judge rejects;
   - 1 primary accept rejected locally for nonempty unsupported-claim IDs before secondary judging (`machine-learning-systems-5a6543e4bd254313`);
   - 7 primary accepts rejected by the secondary judge;
   - 0 accepted episodes.

## Exact deterministic failures

First pass:

- exact 400-word constraint observed 195 words;
- 140–160-word constraint observed 131 words;
- 500–1,000-word constraint observed 335 words;
- an unattested claim that a G*Power calculation had been performed;
- an unattested claim that code tests confirmed four behaviors;
- one claim relied on an unexecuted/inaccessible non-numeric-input verification;
- one claim relied on an unexecuted/inaccessible momentum-overshoot verification.

Final pass:

- exact 400-word constraint observed 195 words;
- 140–160-word constraint observed 131 words;
- 500–1,000-word constraint observed 335 words;
- one published answer made an unattested execution claim while simultaneously saying the comparison had not run;
- one claim relied on an unexecuted/inaccessible momentum-overshoot verification.

## Overlapping judge failure signals

Primary-judge records:

- 37 rejected and scored below 0.84;
- 31 scored expertise uplift below 0.80;
- 27 contained unsupported claim IDs;
- 37 required substantive fixes.

Secondary-judge records:

- 7 rejected and scored below 0.84;
- 5 scored expertise uplift below 0.80;
- all 7 contained unsupported claim IDs;
- all 7 required substantive fixes.

The recurring substantive defects were ornamental rather than independent verification, unsupported external or scenario facts, proposed tests presented as outcomes, incomplete published deliverables, arbitrary weights or thresholds presented as objective, weak empirical experiment design, source chunks that did not support the cited proposition, and insufficient expert difficulty.

## Formal gate and manual audit

- Reviewed: 50
- First-pass acceptance: 0%
- Final acceptance: 0%
- Static failure rate in reviewed wrappers: 0%
- Language failures: 0
- Private-reasoning failures: 0
- Accepted review inconsistencies: 0
- Accepted adversarial failures: 0
- Accepted dual-judge failures: 0
- Exact accepted duplicates: 0
- Accepted domains: 0
- Accepted-corpus hash: `4f53cda18c2baa0c0354bb5f9a3ecbe5`
- Manual accepted-set inspection: not applicable because the accepted set is empty; the manual gate remains false rather than reporting a vacuous pass
- Scale allowed: **false**

The scale gate fails acceptance rate, first-pass acceptance rate, expertise uplift, accepted expertise floor, accepted score floor, and domain coverage. The 9,000-episode runner remains locked.

## Regression result

The complete local suite passed immediately after the run: **61 passed**.

## Post-run hardening implemented

Pilot 4 is not being repaired again. Its measured defects were used to harden the next clean pilot:

- every source-backed blueprint premise now requires a verbatim support quote that is deterministically found in its cited chunk;
- every claim declares a closed evidence mode, and unavailable supplied-execution evidence is rejected before episode generation;
- blueprints require at least two independent solver lanes, a marginal-necessity statement for each lane, a concrete root-only baseline failure, an interacting-constraint explanation, and a central post-barrier reconstruction;
- premise dependencies are automatically carried into episode claim evidence and checked against the blueprint;
- a publish commitment now fails local validation when any claim is refuted, insufficient, or open at the barrier;
- proposed/not-run tools cannot report passing, successful, executed, or otherwise fabricated outcomes;
- supported claims cannot rely on external execution methods when no trusted execution attestation exists;
- the execution-claim detector now recognizes explicit negation such as "no actual comparison has been executed";
- empty accepted sets no longer receive a vacuous manual-audit pass;
- generation, repair, and judge instructions now require full deliverables, visible reconstruction, source entailment, conditional treatment of unknowns, and fail-closed integration.
- the episode schema now requires the complete blueprint constraint trace, repair receives the full output contract, and trace IDs/artifact IDs are reconstructed deterministically from the approved blueprint;
- judges now classify defects as episode-repairable or blueprint-fatal; blueprint-fatal records are quarantined for clean regeneration instead of wasting an episode-only repair call;
- verifier-authored replacement evidence is explicitly forbidden, and judge acceptance is incompatible with any unsupported claim or required fix;
- execution negation is checked clause by clause, so a negated benchmark statement cannot hide a positive claim such as "tests passed."
- affirmative execution claims are scanned in selected artifacts as well as the published answer, and each claim or external verification must cite its own pipeline-trusted run rather than borrowing trust from an unrelated run ID.

The complete suite after these changes: **76 passed**.
