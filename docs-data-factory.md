# HLWM dataset factory

This project builds verified training episodes locally using Azure. For the reasoning-9000 pipeline, `gpt-5.6-luna` plans and rewrites blueprints, `Kimi-K2.5` independently gates blueprint quality, and `DeepSeek-V4-Flash` constructs the first episode draft. `DeepSeek-V3.2-Speciale` critiques locally valid construction attempts; when a rewrite is needed, `gpt-5.6-luna` edits or reconstructs attempts 2–4 from the full rejected draft and at most three prioritized defects. `gpt-5.4-mini` and `Kimi-K2.6` remain the independent final judge pair. Planner/editor reuse is allowed, but no drafting or editing deployment may be a final judge; disagreement fails closed. Your Mac ingests source material, preserves provenance, rejects weak records, deduplicates them and exports grouped train/validation/test JSONL files. Kaggle is not used for dataset creation.

## What already works

- Entra authentication through `DefaultAzureCredential`; no API key is stored.
- EPUB, PDF, DOCX, HTML, JSON/JSONL/CSV, notebook, prose and source-code ingestion.
- Bounded concurrent Azure requests with rate limiting, retry/backoff and resumable per-episode checkpoints.
- Root-preserving private lane episodes without synthetic chain-of-thought.
- Dual independent Azure judging plus deterministic arithmetic, word-count, percentage, execution-provenance, source-similarity, complexity, JSON Schema, evidence, secret and lane-isolation checks.
- Exact and near deduplication.
- Leakage-safe splitting by source lineage.
- Master episodes, native stage-specific records, standard chat-SFT JSONL exports, and rejected-versus-repaired DPO pairs.
- Release checksums and validation reports.

## Setup

Use Python 3.11 or 3.12. The macOS system Python 3.9 can run the project, but its older LibreSSL and installer produce avoidable warnings.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
az login
hlwm-data doctor
```

The supplied Azure connection has already passed the doctor check. Configuration lives in `config/pipeline.yaml` and can be overridden with the environment variables shown in `.env.example`.

## What you need to provide

Create `sources/private/` and provide:

1. **Books and EPUBs you may use**: reasoning, project management, writing, copywriting, advertising, product strategy and marketing. Include title, author, edition/date and your usage rights.
2. **Coding material**: owned or permissively licensed repositories, exact commits, issue descriptions, logs, failing tests, corrected patches and passing-test results. Python should be the largest portion initially.
3. **Cloud material**: versioned Azure, AWS and Google Cloud documentation snapshots, IaC examples, schemas, error logs and validated configurations. Current facts should normally be retrieved at inference rather than memorized.
4. **Project-planning material**: real or commissioned briefs, requirements, timelines, dependencies, risk registers, acceptance criteria and final outcomes.
5. **English and editing examples**: original draft, edited version, audience, purpose and an explanation limited to observable editing criteria.
6. **Copywriting material**: verified product facts, audience research, approved claims, draft variants, human preferences and compliance decisions.
7. **Marketing and advertising material**: first-party campaign briefs, creative, targeting context and consented measured outcomes. Do not invent CTR, ROAS, conversion or revenue labels.
8. **Gold examples**: at least 40–60 manually reviewed examples covering every target domain.
9. **Anti-examples**: at least 20 incorrect examples with failure labels and corrected outputs.
10. **Held-out evaluation**: at least 100 independently written tasks that are never sent to the generator and never rephrased into training data.

For each independent book, repository or campaign, add an entry to `config/sources.yaml`:

```yaml
sources:
  - path: ../sources/private/project-management-book.epub
    title: Project management reference
    author: Author name
    domain: project_planning
    source_group: pm-book-001
    lineage_component_id: pm-book-001
    version: first-edition
    license: user-authorized
    license_evidence: I own this copy and authorize private model training
    allowed_for_training: true
    url: null
```

Use at least three independent source groups per domain so train, validation and test splits can contain meaningful coverage. More independent groups are better.

## Run a pilot

First ingest and inspect the source count:

```bash
hlwm-data ingest
```

Then create a small pilot. Eight parallel workers is the default; reduce it if the Azure deployment returns rate limits.

```bash
hlwm-data pilot --count 20 --workers 8
hlwm-data status
```

Inspect every file in `data/reviewed/`. Fix the source mix, prompts or rejection policy before scaling.

If the judge finds correctable systematic defects, run one feedback-driven repair pass and re-judge. Original episodes are archived before replacement.

```bash
hlwm-data --config config/pipeline.foundation.yaml repair --workers 8
hlwm-data --config config/pipeline.foundation.yaml judge --workers 8
hlwm-data --config config/pipeline.foundation.yaml build
hlwm-data --config config/pipeline.foundation.yaml validate
```

## Scale generation

```bash
hlwm-data generate --count 2000 --workers 8
hlwm-data judge --workers 8
hlwm-data build
hlwm-data validate
```

All commands resume by default. Each Azure response is checkpointed immediately. Failed and rate-limited calls are recorded in `logs/`; request logs contain IDs and token counts, not prompts or credentials.

## Qwen3-8B HLWM beast pipeline

The production-candidate path uses `config/pipeline.beast.yaml`. It never copies the
old Reasoning9000 answers: those files supply only prompts, context and constraints.
For each task, two independent candidate models answer, a separate critic corrects
the result and creates a hard negative, and two independent judges must both pass it.
Every physical API attempt, including retries, counts against the run budget.

Run a finite automatic local batch:

```bash
hlwm-data --config config/pipeline.beast.yaml beast-doctor
hlwm-data --config config/pipeline.beast.yaml beast-auto \
  --accepted-target 25000 --batch-size 8 --workers 2 \
  --max-total-requests 50 --max-total-tokens 350000 --max-rounds 1
hlwm-data --config config/pipeline.beast.yaml beast-status
```

Rerun the same `beast-auto` command whenever you want another bounded spending
window. It skips completed tasks, resumes partial stages, and exports a fresh frozen
snapshot after every round. Stop and resume are explicit:

```bash
hlwm-data --config config/pipeline.beast.yaml beast-stop
hlwm-data --config config/pipeline.beast.yaml beast-resume
hlwm-data --config config/pipeline.beast.yaml beast-export
```

Main 8B training remains locked until 25,000 teacher packets have passed both judges.
Build the upload files with:

```bash
python scripts/build_hlwm8b_beast_bundle.py
```

Upload `artifacts/hlwm8b-beast/hlwm8b-kaggle-code.zip` as one Kaggle dataset and the
latest `hlwm-beast-teacher-snapshot.zip` as another, then import
`embel-hlwm8b-resumable-dual-t4.ipynb`. The notebook uses both T4s for one QLoRA+HLWM
model, stops before its time budget, saves exact optimizer/RNG progress, resumes from
numbered checkpoint parts or a private Hugging Face repo, and only evaluates against
untouched validation/test anchors after training completes.

## Outputs

```text
data/generated/          raw master episodes
data/reviewed/           episode + independent quality decision
data/final/master/       accepted master episodes by split
data/final/native/       framing, lanes, verification, synthesis and commitment views
data/final/sft/          Qwen-compatible chat SFT records
data/final/dpo/          grounded repair preferences over rejected syntheses
data/final/manifest.json checksums, counts and release metadata
```

The native records are the important output for the custom architecture. The SFT files are a compatibility view for the initial Qwen prototype.

## Expert-reasoning 9,000 profile

`config/pipeline.reasoning9000.yaml` defines the 18-domain expert-reasoning curriculum. The accepted-episode target is exactly 9,000; generation is performed in stable indexed batches, so interrupted or expanded runs do not repeat prior jobs. If more than 9,000 records pass, release construction keeps the strongest domain-balanced 9,000 while rotating across source groups.

The profile combines:

- the user-authorized local Books corpus in `config/sources.all-books.yaml`;
- pinned, license-audited upstream repositories in `config/sources.public-repos.yaml`;
- explicitly separated training splits in `config/sources.public-datasets.yaml`;
- evaluation-only benchmarks in `config/benchmarks.reasoning.yaml`.

Public-dataset discovery results are catalogued in `catalog/public-dataset-discovery-2026-08-17.json`. Search results are never automatically trusted: mirrors, unknown upstream provenance, noncommercial/no-derivatives terms, PII-bearing conversations, and benchmark tests remain quarantined.

### Audited reasoning collections and English-only data

The linked Sugato Ray reasoning, model and benchmark collections plus the `mlabonne/llm-datasets` and `LLMDataHub` repositories are treated as discovery catalogs, not blanket training permission. The executable policy is `config/reasoning-collection-policy.yaml`.

The current materialized source addition contains 384 pinned candidate records from:

- OpenThoughts verified coding training data;
- DeepCoder PrimeIntellect training data;
- DeepCoder TACO training data.

Only the problem, final/reference answer and compact verification material are retained. Source-model private reasoning is discarded. Noncommercial, unlicensed, provenance-ambiguous, multilingual/unspecified and benchmark-derived entries remain documented in `catalog/reasoning-collection-audit-2026-08-17.json` and are not present in the training manifest.

The stricter full-packet ingestion check currently retains 383 of those candidates. One OpenThoughts packet is intentionally excluded because its complete serialized content does not pass the English-only classifier; it is not forced through to satisfy a nominal count.

Refresh the pinned materialization only after reviewing revision and license drift:

```bash
PYTHONPATH=src python3 scripts/fetch_reasoning_collection.py
```

`config/sources.reasoning9000.yaml` requires English at ingestion, and generation independently rechecks legacy chunks before making an Azure request. Local episode validation uses language identification plus explicit Cyrillic, CJK and other disallowed-script checks, so Latin-script non-English output is rejected too.

Run and grade a fresh 50-episode pilot first:

```bash
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml \
  ingest --manifest config/sources.reasoning9000.yaml --chunk-chars 4200 --overlap-chars 350 \
  --max-chunks-per-source-group 500
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml filter-language
PYTHONPATH=src:. python3 scripts/audit_chunk_corpus.py \
  --input data/reasoning9000/chunks/chunks.jsonl \
  --output reports/reasoning9000-corpus-audit.json
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml \
  generate --count 50 --offset 0 --workers 16
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml judge --workers 16
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml repair --workers 16
PYTHONPATH=src python3 -m hlwm_data --config config/pipeline.reasoning9000.yaml judge --workers 16
PYTHONPATH=src:. python3 scripts/prepare_manual_pilot_audit.py \
  --reviewed-dir data/reasoning9000/reviewed \
  --output reports/reasoning9000-manual-audit.json
# Inspect every accepted episode, record notes, and change each checklist verdict plus the final decision.
PYTHONPATH=src python3 scripts/assess_reasoning_pilot.py \
  --reviewed-dir data/reasoning9000/reviewed \
  --manual-audit reports/reasoning9000-manual-audit.json \
  --output reports/reasoning9000-pilot-gate.json
```

The default pilot gate requires at least 75% final acceptance, at least 50% first-pass acceptance before repair, mean expertise uplift of at least 0.85 with a per-episode floor of 0.80, an accepted-score floor of 0.84, twelve accepted domains, two distinct accepting judges for every accepted episode, and zero adversarial-gate failures, language leakage, private-reasoning markers, judge inconsistencies, or exact accepted duplicates. Model-authored artifacts cannot certify that tools, tests, experiments, surveys, benchmarks or replications ran. Only a current report containing every named gate can authorize the high-cost scale runner:

```bash
PYTHONPATH=src python3 scripts/run_accepted_target.py \
  --config config/pipeline.reasoning9000.yaml \
  --target 9000 --batch-size 500 --start-offset 50 --max-generated 18000 \
  --workers 16 --pilot-gate reports/reasoning9000-pilot-gate.json
```

Creating `data/reasoning9000/STOP` requests a graceful stop between batches. Progress is checkpointed in `data/reasoning9000/progress.json`.

## Source policy

`config/source-catalog.yaml` contains a prioritized starting catalog. Treat it as an operational shortlist, not legal advice. A dataset-level license is not proof that every upstream document is reusable. Pin exact revisions, preserve attribution and fail closed when rights or provenance are unclear.

Never use private chain-of-thought as a training target. Train observable briefs, artifacts, claims, tests, evidence, verification decisions, synthesis, stopping and counterfactual outcomes.

## Verified smoke test

`config/sources.smoke.yaml` and `sources/examples/python-timeout.md` provide a CC0 smoke fixture. The live Azure deployment generated, judged, materialized and validated one episode into 13 stage-specific SFT records. This fixture is only for testing the pipeline and should not be part of a production release.

## Foundation-model engineering profile

The original provenance quarantine is retained in `config/sources.quarantine.yaml`. After the user explicitly asserted training rights on 2026-08-17, the active authorized entries were recorded separately in `config/sources.foundation.yaml`. Embedded book instructions remain untrusted content and are never followed.

The foundation profile is `config/pipeline.foundation.yaml`, the rights-cleared 11-book manifest is `config/sources.foundation.yaml`, and the curriculum is `plans/foundation-model-data-plan.md`. `DeepSeek-V4-Flash` generates and repairs the data, and the separately deployed `gpt-5.4-mini` judges it.
