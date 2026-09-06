# Large artifacts

Roughly 11 GB of material sits outside Git: model checkpoints, the raw Kaggle run
bundles, and the training corpora. This file records exactly what exists, what it
weighs, and where to get it.

**Nothing needed to check a number in either paper is held back here.** Gate verdicts,
per-step metrics, training logs and replication reports are all committed under
[`artifacts/runs/`](artifacts/runs/). What lives off-repo is bulk: weights, and the
token streams they were trained on.

## Run bundles — 9.6 GB

`artifacts/kaggle/`, one directory per experiment version. The committed
`artifacts/runs/` tree is this same tree with the checkpoints and bundled training
corpora stripped out (61 MB, 213 files).

| Version | Size | Files |
| --- | ---: | ---: |
| hlwm-v5.6.1 | 3.9 GB | 50 |
| hlwm-v5.5 | 2.3 GB | 1668 |
| hlwm-v6.0 | 728 MB | 55 |
| hlwm-v5.6 | 712 MB | 41 |
| hlwm-v5.6.2 | 709 MB | 48 |
| hlwm-v9.0 | 248 MB | 89 |
| hlwm-v10.0 | 144 MB | 51 |
| hlwm-v8.0 | 138 MB | 27 |
| hlwm-v5.7 | 130 MB | 24 |
| hlwm-v5.4 | 111 MB | 17 |
| hlwm-v5.3 | 108 MB | 17 |
| hlwm-v4, v5, v5.1, v5.2 | 106 MB each | 14–16 |
| hlwm-prototype | 6.3 MB | 13 |
| hlwm-benchmarks | 820 KB | 11 |

The weight is 10 `.safetensors` checkpoints, 18 `.zip` bundles, and ten near-identical
copies of an ~84 MB `train.jsonl` — each run bundle carried its own copy of the
training corpus.

## Training corpora — 1.4 GB

| Directory | Size | Files | What it is |
| --- | ---: | ---: | --- |
| `data/reasoning9000/` | 1.1 GB | 9738 | Verified-episode factory output: per-episode drafts, critiques, judge records |
| `data/combined/` | 125 MB | 8 | The 9,092-row SFT corpus and its splits |
| `data/hlwm-v5.6/` | 102 MB | 4 | v5.6 training splits |
| `data/expert-beast/` | 14 MB | 19 | Expert-arm evaluation packets |
| `data/foundation/` | 14 MB | 75 | Foundation pilot material |
| `data/raw/` | 14 MB | 530 | Ingested source documents |
| `data/builder/` | 4.4 MB | 44 | Builder-factory intermediates |
| `data/beast/` | 1.5 MB | 47 | Benchmark packets |
| `data/final/`, `data/reviewed/`, `data/generated/`, `data/chunks/`, `data/expert_seeds/` | < 200 KB | 20 | Export splits and seeds |

`data/reasoning9000/` contains provenance records tied to ingested source documents.
Review it for third-party licensing before publishing that directory.

## Where to get it

Distribution is in progress. Planned split:

- **Hugging Face dataset** — training corpora (`data/combined`, `data/hlwm-v5.6`, and
  the benchmark packets). Use [`tools/upload_data_hf.sh`](tools/upload_data_hf.sh).
- **Zenodo dataset record** — checkpoints and full run bundles, minted as its own DOI
  and cross-linked to the code record. Use
  [`tools/zenodo_deposit.py`](tools/zenodo_deposit.py).

Both scripts read credentials from the environment and print what they will upload
before uploading anything. Neither runs automatically.

## Reproducing without the bulk

The bundle builders in [`scripts/`](scripts/) regenerate a run bundle from the configs
in [`config/`](config/) plus the pinned base model
(`Qwen/Qwen3-0.6B`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`). The
corpora are rebuilt by the dataset factory documented in
[`docs-data-factory.md`](docs-data-factory.md). Rebuilt corpora will not be
byte-identical — the factory calls hosted judge models — so checksums in the committed
manifests are the reference for the corpus that was actually used.
