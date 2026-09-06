# Foundation dataset pilot report

Date: 2026-08-17

## Outcome

The authorized 11-book corpus was ingested and used for a 20-episode pilot. `DeepSeek-V4-Flash` generated the episodes and performed one feedback-driven repair pass. The independent `gpt-5.4-mini` judge accepted 14 episodes and rejected 6 at the unchanged 0.84 quality threshold.

The accepted set is valid and usable as a small architecture/data-format pilot. It is not large enough to train a useful production model, and the observed workflow should not yet be scaled directly to 4,000 episodes.

## Source ingestion

- Authorized files: 11
- Extracted documents/pages: 1,898
- Accepted chunks: 3,035
- Curriculum domains: 8
- Empty chunks: 0
- Exact duplicate groups: 0
- Corrupted replacement characters: 0

## Generation and adjudication

- Scheduled: 20
- Generated successfully: 20
- Generation failures: 0
- Repaired from judge feedback: 19
- Final accepted: 14
- Final rejected: 6
- Acceptance rate: 70%
- Mean score among accepted episodes: 0.897
- Duplicate accepted episodes removed: 0

Two generation requests and four repair requests timed out once and recovered automatically through retry.

## Accepted episode distribution

| Domain | Generated | Accepted |
|---|---:|---:|
| Deep-learning fundamentals | 2 | 1 |
| Distributed data systems | 3 | 2 |
| Edge AI | 2 | 1 |
| Framework engineering | 2 | 2 |
| LLM engineering | 3 | 2 |
| Machine-learning systems | 3 | 3 |
| Mathematics and optimization | 2 | 2 |
| Software architecture | 3 | 1 |

## Rejection causes

The six rejected records remain outside every final training export:

1. An edge-AI episode retained an unsupported accuracy-recovery target.
2. A deep-learning episode retained an unverified gradient-norm bound.
3. An LLM-engineering episode triggered the private-reasoning marker guard.
4. A distributed-systems episode contradicted its own exactly-once requirement.
5. A software-architecture episode used incorrect quantity arithmetic.
6. A software-architecture episode failed to adapt the source example to the requested domain.

## Materialized outputs

| View | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Master episodes | 9 | 2 | 3 | 14 |
| Native HLWM records | 117 | 24 | 39 | 180 |
| Chat-SFT records | 113 | 23 | 37 | 173 |

Native stages include 29 isolated private-lane records, 14 barriers, 14 verification records, 14 synthesis records, 15 continuation records, 38 counterfactual records, and 14 root-retention records.

Final validation passed with no errors or warnings. All five pipeline unit tests passed.

## Azure token use

| Operation | Prompt tokens | Completion tokens | Total |
|---|---:|---:|---:|
| DeepSeek generation | 69,635 | 88,749 | 158,384 |
| DeepSeek repair | 207,297 | 93,241 | 300,538 |
| Independent judging | 405,910 | 22,671 | 428,581 |
| Connection doctor | 25 | 12 | 37 |
| **Total** | **682,867** | **204,673** | **887,540** |

A naive 4,000-episode run using this exact multi-pass workflow would scale to roughly 177.5 million tokens before additional failures, re-repairs, human review, or training. Actual Azure cost depends on the deployment pricing. No 4,000-episode run was launched.

## Scale decision

Do not scale this exact run directly to 4,000 episodes. The final 70% acceptance is promising, but 19 of 20 records required repair and the repair/adjudication passes consumed most tokens. The next defensible gate is a fresh 50-episode pilot using the corrected generator and judge prompts, with manual review of all accepted records. Scale only if first-pass acceptance improves materially and the held-out evaluation confirms that the native records improve routing, parallel-lane utility, verification, and root capability retention.
