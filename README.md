# Hierarchical Latent Workspace Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22343152.svg)](https://doi.org/10.5281/zenodo.22343152)
[![Code: Apache 2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE-CODE)
[![Papers: CC BY 4.0](https://img.shields.io/badge/Papers-CC%20BY%204.0-lightgrey.svg)](LICENSE)

Two preprints and the full experimental record from a twenty-day, ten-experiment
preregistered attempt to build a latent-workspace language model: a frozen Qwen3-0.6B
decoder wrapped in 73M–132M trainable sidecars implementing a root-preserving expert
graph, isolated parallel reasoning lanes, and a private diffusion canvas behind a
calibrated publication rule.

**All three proposed mechanisms failed their preregistered gates.** Both papers are
negative-results reports. Neither claims a capability win.

Karthik Yaddanapudi · Independent Researcher · <karthikyaddana@gmail.com>

---

## What is in this repository

| Path | Contents |
| --- | --- |
| [`paper/`](paper/) | Both preprints, LaTeX source and compiled PDF |
| [`artifacts/runs/`](artifacts/runs/) | 213 result files from 17 experiment versions — gate verdicts, per-step metrics, training logs, replication reports |
| [`reports/`](reports/) | 48 dated preregistration plans and results write-ups, v5.3 → v10.5 |
| [`scripts/`](scripts/) | 35 bundle-build, evaluation and rehearsal scripts |
| [`config/`](config/) | 22 pipeline and experiment configuration files |
| [`src/`](src/) | `hlwm_data` — the dataset factory package |
| [`tests/`](tests/) | 12 test modules |
| [`evals/`](evals/) | Benchmark registries (`expertbench_1500`, `expert_workflows`) |
| [`DATA.md`](DATA.md) | Where the 11 GB of checkpoints and training corpora live |

Model checkpoints and the run bundles are too large for Git and are distributed
separately — see [`DATA.md`](DATA.md). Everything needed to *check the reported numbers*
is in this repository.

The training corpora are published as a Hugging Face dataset:
**<https://huggingface.co/datasets/slashgg/hlwm-corpora>** — the 9,092-row SFT corpus,
its underlying episode records, the v5.6 splits and the evaluation packets.

## The papers

### 1. Readable but Not Usable — [`paper/postmortem-paper.pdf`](paper/postmortem-paper.pdf)

*A Preregistered Post-Mortem of a Latent-Workspace Language Model, Including Its Own
Novelty Audit* (28 pp)

The decisive experiment deleted the operative premise from the decoder's input token
ids, so the latent channel was the only route from evidence to answer. Generation
through the channel scored **0.000 on all 99 masked rows on both seeds**, while a
linear probe recovered roughly **0.30** of the withheld premise. The channel was
readable but not usable.

This constructively extends Lagged Coupling (Xun, 2026, arXiv:2609.01048), which
reports the same read-before-causal dissociation in pretrained models. Here the
channel was built deliberately, made structurally necessary, and every instrument
defect in the record was repaired — and the dissociation persisted.

The paper also contains a novelty audit of its own six architectural claims (every one
previously published, with three residual cells still unoccupied) and an
instrument-failure taxonomy with detectors covering ~30 documented defects in five
classes, including gates hardcoded to pass and an expert graph the trainer never called.

### 2. The Hierarchical Latent Workspace Model — [`paper/hlwm-paper.pdf`](paper/hlwm-paper.pdf)

*Root-Preserving Variable-Depth Expertise for Parallel Diffusion Reasoning* (56 pp)

The full technical report: the proposed architecture, and the ten preregistered
prototype experiments that falsified it at this scale. Routing passed its load gates on
one seed of two but failed causal liveness on **10 of 10 seed-runs**. Verified fan-in
lost to self-consistency by ~30 points, traced to a beginning-of-sequence id the pinned
tokenizer aliases to end-of-text. A dedicated repair experiment fixed that channel
(zero turn-scaffold emissions in 640 audited rows) and then measured the workspace
directly: ablating the read-out's 34 memory positions changed graded accuracy by 0.000
on one seed and *improved* it by 0.022 on the other. Parity was reached by irrelevance.

## Checking the central claim yourself

The v10.0 go/no-go abort is recorded directly in the run metrics. No setup required:

```sh
python3 - <<'PY'
import json
for seed in ("17", "29"):
    path = f"artifacts/runs/hlwm-v10.0/session-i5/metrics-{seed}.jsonl"
    for line in open(path):
        row = json.loads(line)
        if "v10_verdict" in row:
            print(seed, json.dumps(row["v10_verdict"], indent=2))
PY
```

Both seeds report `"aborted": "gonogo"` with `gonogo_masked_numeric_em: 0.0`, while the
preceding warm gate passed its 0.50 floor (`em_600` of 0.602 on seed 17 and 0.591 on
seed 29). The model had learned the task; it could not route the answer through the
latent channel. That is the paper's headline in machine-readable form.

Per-experiment capability gates are in
`artifacts/runs/hlwm-v*/**/v*-capability-gate.json` (14 files), and the matching
preregistration for each is the dated plan in [`reports/`](reports/) — plans were
written before their runs, and the results write-up for each version sits beside it.

## Building the papers

Both papers are single self-contained LaTeX files — no external figures, no BibTeX.

```sh
tectonic -X compile paper/postmortem-paper.tex
tectonic -X compile paper/hlwm-paper.tex
```

`pdflatex` twice also works (the papers use `lastpage`, so a second pass is needed).

## Running the code

Python 3.11 or 3.12.

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The dataset factory is documented in [`docs-data-factory.md`](docs-data-factory.md).
Training ran on Kaggle notebooks against a pinned Qwen3-0.6B revision
(`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`); the bundle builders in `scripts/` produce
the uploadable run bundles. Credentials are read from environment variables only — see
[`.env.example`](.env.example). No key is committed to this repository.

## What is explicitly not claimed

- The matched plain-LoRA control **never ran**, so no capability claim is made.
- The calibrated-abstention thread never beat a free mean-log-probability baseline at
  its own preregistered bar.
- The dense-supervision successor was terminated at 125 optimizer updates, so that
  question closes **undecidable, not falsified**.

## Citing

`10.5281/zenodo.22343152` is the concept DOI and always resolves to the latest version.
Per-paper DOIs are being minted as separate Zenodo preprint records; until they resolve,
both entries below carry the concept DOI.

```bibtex
@misc{yaddanapudi2026postmortem,
  author = {Karthik Yaddanapudi},
  title  = {Readable but Not Usable: A Preregistered Post-Mortem of a
            Latent-Workspace Language Model, Including Its Own Novelty Audit},
  year   = {2026},
  doi    = {10.5281/zenodo.22343152}
}

@misc{yaddanapudi2026hlwm,
  author = {Karthik Yaddanapudi},
  title  = {The Hierarchical Latent Workspace Model: Root-Preserving
            Variable-Depth Expertise for Parallel Diffusion Reasoning},
  year   = {2026},
  doi    = {10.5281/zenodo.22343152}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

Papers and figures: [CC BY 4.0](LICENSE). Code, configuration and scripts:
[Apache 2.0](LICENSE-CODE).
