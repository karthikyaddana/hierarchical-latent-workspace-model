# Post-hardening calibration 002 — failed

- Stable range: offset 210, count 10 (indices 210–219)
- Locally valid episodes generated: 6/10
- First-pass primary-judge accepts: 2/10
- First-pass dual-judge accepts: 0/10
- Promotion requirement: at least 5/10 dual-judge accepts with zero deterministic false accepts
- Result: failed; no repair, no promotion, and no Pilot 5

## Generation failures

- `research-and-decision-making-c1ba28cebef96905`: its blueprint emitted six success criteria, exceeding the schema maximum of five.
- `edge-ai-4bafa18b83d959fb`: its blueprint used source IDs where exact chunk IDs were required.
- `sales-and-copywriting-d56ac7eda5568cb0`: the generated episode cited valid pipeline-issued support-span IDs, but episode evidence validation did not recognize those IDs.
- `cloud-and-devops-1ff25b58907504c0`: the generated episode hit the same valid-support-span evidence-validation mismatch.

## First-pass review results

All six locally valid episodes passed static and adversarial checks. Four were rejected by the primary judge. The primary judge accepted two, but the independent secondary judge rejected both, so judge consensus accepted zero.

- `communication-and-writing-0e6750fda907dbad`: unsupported fictional identity and affiliation, incomplete central reconstruction, and expertise uplift below threshold.
- `llm-engineering-ac20bd164127ee02`: self-attested rather than reconstructed verification, artifact/claim naming mismatch, incomplete failure ordering, and expertise uplift below threshold.
- `programming-and-web-f3a5bd17f3d81a7f`: shallow source-near transformation, non-necessary lane split, and incomplete reconstruction.
- `psychology-and-behavior-19eb4999995b4d04`: the proposed design did not isolate the confound, the measurement schedule was incomplete, and the causal inference was invalid.
- `marketing-and-brand-8f26604114d08b48`: secondary review found contradictory word counts, a missing required tension statement, and missing element-level source attribution.
- `software-architecture-a390ca388f6b15c7`: secondary review found that the proposed synchronous extension was behaviorally broken and that the verifier never exercised the extension or a mutation rejection.

## Required hardening before calibration 003

- Normalize blueprint lists before schema validation: cap success criteria at five without dropping distinct constraints, and translate source-ID references to their selected exact chunk IDs.
- Treat a valid `chunk-id::span-NNN` as available claim evidence whenever the authoritative premise carries that span and the chunk prefix is in the job packet.
- Require the central verification artifact to contain an explicit inputs, operation, reconstructed result, and verdict chain, plus at least one genuinely discriminating falsification.
- Reject unverifiable real-person or institution claims unless they are source-backed or explicitly hypothetical.
- Detect conflicting numeric reconstructions, especially word-count claims, across artifacts, lane summaries, verifier output, and the published answer.
- Strengthen blueprint novelty and lane-necessity checks so source-near restatements and builder-plus-reviewer decompositions do not pass.
- Require proposed code/design artifacts to verify the exact extension or counterexample named by the central claim rather than self-certifying it.

The physical `data/reasoning9000/STOP` scale lock remains present. Calibration 002 records are permanently excluded from Pilot 5.
