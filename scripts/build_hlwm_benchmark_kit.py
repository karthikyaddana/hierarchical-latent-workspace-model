#!/usr/bin/env python3
"""Build the session C external-benchmark kit (Kaggle dual-T4).

The kit rides on top of an existing HLWM candidate bundle: the notebook
extracts the v5.6.1 bundle for the core code, overlays this kit's
`evaluate_benchmarks.py` + updated `semantic_grading.py` (multiple-choice and
execution graders) + test file, verifies the pinned eval packets against
their recorded SHA-256, and runs GSM8K on GPU 0 concurrently with
ARC-Challenge + MMLU on GPU 1 against an attached passing-seed checkpoint.

Session C is authorized only after a full preregistered gate passes
(standing rule since the v5.5 plan). Building the kit costs nothing and is
not an authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_hlwm_kaggle_bundle import PIP_CELL_V55, sha256  # noqa: E402

SOURCE = ROOT / "experiments" / "kaggle_hlwm"
SUBSETS = ROOT / "sources" / "public" / "benchmark-eval-subsets"
DEFAULT_OUTPUT = ROOT / "artifacts" / "kaggle" / "hlwm-benchmarks"

OVERLAY_FILES = ("evaluate_benchmarks.py", "semantic_grading.py", "test_modeling_hlwm.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instruct-model", type=str, default="Qwen/Qwen3-0.6B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    kit = output / "kit" / "hlwm_benchmark_kit"
    if output.exists():
        shutil.rmtree(output)
    (kit / "benchmark-eval-subsets").mkdir(parents=True)
    (kit / "overlay").mkdir(parents=True)

    fetch_manifest = json.loads((SUBSETS / "fetch-manifest.json").read_text())
    for suite in fetch_manifest["suites"]:
        shutil.copy2(
            SUBSETS / (suite + ".jsonl"),
            kit / "benchmark-eval-subsets" / (suite + ".jsonl"),
        )
    shutil.copy2(SUBSETS / "fetch-manifest.json", kit / "benchmark-eval-subsets" / "fetch-manifest.json")
    for name in OVERLAY_FILES:
        shutil.copy2(SOURCE / name, kit / "overlay" / name)

    try:
        from huggingface_hub import HfApi

        instruct_revision = HfApi().model_info(args.instruct_model).sha
    except Exception as error:  # offline build still produces a usable kit
        print("warning: could not pin instruct revision:", error)
        instruct_revision = None

    manifest: Dict[str, Any] = {
        "name": "hlwm-benchmark-kit",
        "kit_version": "1.0.0",
        "eval_only": True,
        "requires": [
            "hlwm-verified-multidomain v5.6.1 candidate bundle (code)",
            "one passing-seed hlwm resumable checkpoint",
        ],
        "suites": fetch_manifest["suites"],
        "sample_seed": fetch_manifest["sample_seed"],
        "per_suite": fetch_manifest["per_suite"],
        "instruct_model": args.instruct_model,
        "instruct_revision": instruct_revision,
        "overlay_sha256": {name: sha256(kit / "overlay" / name) for name in OVERLAY_FILES},
        "authorization": (
            "run only after a full preregistered capability gate passes; "
            "results without a passing gate are exploratory and unreportable"
        ),
    }
    (kit / "kit-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    notebook = output / "embel-hlwm-benchmarks-2xt4.ipynb"
    notebook.write_text(
        json.dumps(notebook_document_benchmarks(manifest), indent=1) + "\n",
        encoding="utf-8",
    )
    archive = output / "hlwm-benchmark-kit.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted((output / "kit").rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output / "kit"))

    report = {
        "kit": str(archive),
        "kit_sha256": sha256(archive),
        "kit_bytes": archive.stat().st_size,
        "notebook": str(notebook),
        "notebook_sha256": sha256(notebook),
        "instruct_revision": instruct_revision,
        "suites": {k: v["rows"] for k, v in fetch_manifest["suites"].items()},
    }
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def notebook_document_benchmarks(manifest: Dict[str, Any]) -> Dict[str, Any]:
    def markdown(source: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code(source: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    suite_shas = {k: v["sha256"] for k, v in manifest["suites"].items()}
    instruct_model = manifest["instruct_model"]
    instruct_revision = manifest["instruct_revision"] or ""

    cells = [
        markdown(
            "# Embel HLWM external benchmarks (session C) - dual T4\n\n"
            "**Authorization gate:** this session runs only after a full "
            "preregistered HLWM capability gate has passed. Attach (1) the "
            "v5.6.1 candidate bundle, (2) the `hlwm-benchmark-kit.zip`, and "
            "(3) the passing seed's `hlwm-v5.6.1-seed-<seed>-resumable.pt`.\n\n"
            "GSM8K (numeric, 250 items) runs on GPU 0 while ARC-Challenge + "
            "MMLU (multiple choice, 250 each) run on GPU 1. Every system --- "
            "the HLWM decision path, its causal path, the pinned frozen base, "
            "and the pinned instruct variant --- receives identical prompts "
            "and token budgets. Baselines record confidence so selective "
            "accuracy is compared at HLWM's achieved coverage (the post-hoc "
            "threshold favors the baselines).\n"
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')\n"
            "import platform, subprocess, sys, time\n"
            "SESSION_START = time.time()\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
        ),
        code(PIP_CELL_V55),
        code(
            "from pathlib import Path\n"
            "import hashlib, json, shutil, zipfile\n"
            "KAGGLE = Path('/kaggle/input').exists()\n"
            "WORK = Path('/kaggle/working/hlwm-benchmarks') if KAGGLE else Path.cwd() / 'hlwm-benchmarks-work'\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
            "bundles = [p for root in search_roots for p in root.rglob('hlwm-v5.6.1*candidate-bundle.zip')]\n"
            "bundles += [p for root in search_roots for p in root.rglob('hlwm-v561*candidate-bundle.zip')]\n"
            "assert bundles, 'attach the v5.6.1 candidate bundle'\n"
            "with zipfile.ZipFile(bundles[0]) as bundle: bundle.extractall(WORK)\n"
            "PROJECT = WORK / 'hlwm_kaggle'\n"
            "kits = [p for root in search_roots for p in root.rglob('hlwm-benchmark-kit*.zip')]\n"
            "if kits:\n"
            "    with zipfile.ZipFile(kits[0]) as kit: kit.extractall(WORK / 'kit')\n"
            "    KIT = WORK / 'kit' / 'hlwm_benchmark_kit'\n"
            "else:\n"
            "    candidates = [p.parent for root in search_roots for p in root.rglob('kit-manifest.json')]\n"
            "    assert candidates, 'attach hlwm-benchmark-kit.zip'\n"
            "    KIT = Path(shutil.copytree(candidates[0], WORK / 'kit' / 'hlwm_benchmark_kit'))\n"
            "kit_manifest = json.loads((KIT/'kit-manifest.json').read_text())\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "EXPECTED_SHAS = " + json.dumps(suite_shas) + "\n"
            "BENCH = KIT / 'benchmark-eval-subsets'\n"
            "for suite, expected in EXPECTED_SHAS.items():\n"
            "    actual = file_sha256(BENCH / (suite + '.jsonl'))\n"
            "    assert actual == expected, f'{suite} packet hash mismatch'\n"
            "for name in ('evaluate_benchmarks.py','semantic_grading.py','test_modeling_hlwm.py'):\n"
            "    shutil.copy2(KIT/'overlay'/name, PROJECT/name)\n"
            "print('bundle + kit verified; overlay applied')\n"
        ),
        code(
            "import importlib.util, py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.device_count() == 2, 'select GPU T4 x2'\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','evaluate_checkpoint.py','evaluate_benchmarks.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "if importlib.util.find_spec('pytest') is not None:\n"
            "    test_command = [sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')]\n"
            "else:\n"
            "    test_command = [sys.executable, str(PROJECT/'test_modeling_hlwm.py')]\n"
            "result = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)\n"
            "print((result.stdout or '')[-3000:])\n"
            "if result.returncode != 0:\n"
            "    print((result.stderr or '')[-3000:])\n"
            "    raise RuntimeError('tests failed')\n"
            "checkpoints = [p for root in search_roots for p in root.rglob('hlwm-v5*resumable.pt')]\n"
            "assert checkpoints, 'attach the passing seed resumable checkpoint'\n"
            "CHECKPOINT = sorted(checkpoints)[0]\n"
            "print('checkpoint:', CHECKPOINT)\n"
        ),
        code(
            "os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))\n"
            "Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)\n"
            "import torch\n"
            "payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=payload['base_model'], revision=payload['base_revision'])\n"
            "INSTRUCT_MODEL = " + json.dumps(instruct_model) + "\n"
            "INSTRUCT_REVISION = " + json.dumps(instruct_revision) + " or None\n"
            "if INSTRUCT_MODEL:\n"
            "    snapshot_download(repo_id=INSTRUCT_MODEL, revision=INSTRUCT_REVISION)\n"
            "del payload\n"
            "print('model snapshots cached')\n"
        ),
        code(
            "import subprocess, sys, time\n"
            "JOBS = {0: 'gsm8k', 1: 'arc-challenge,mmlu'}\n"
            "processes, handles, outputs = {}, {}, {}\n"
            "for gpu, suites in JOBS.items():\n"
            "    outdir = Path(f'/kaggle/working/benchmarks-gpu{gpu}')\n"
            "    if outdir.exists(): shutil.rmtree(outdir)\n"
            "    outputs[gpu] = outdir\n"
            "    command = [sys.executable, str(PROJECT/'evaluate_benchmarks.py'),\n"
            "        '--checkpoint', str(CHECKPOINT), '--benchmarks-dir', str(BENCH),\n"
            "        '--output-dir', str(outdir), '--suites', suites,\n"
            "        '--context-tokens', '256', '--canvas-tokens', '128', '--max-new-tokens', '320',\n"
            "        '--instruct-model', INSTRUCT_MODEL]\n"
            "    if INSTRUCT_REVISION: command += ['--instruct-revision', INSTRUCT_REVISION]\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = Path(f'/kaggle/working/benchmarks-gpu{gpu}.log').open('w'); handles[gpu] = handle\n"
            "    processes[gpu] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "    print('launched', suites, 'on GPU', gpu)\n"
            "while any(process.poll() is None for process in processes.values()):\n"
            "    time.sleep(60)\n"
            "    print({gpu: {'returncode': process.poll(), 'elapsed_min': round((time.time()-SESSION_START)/60,1)}\n"
            "           for gpu, process in processes.items()})\n"
            "for handle in handles.values(): handle.close()\n"
            "failures = {gpu: process.returncode for gpu, process in processes.items() if process.returncode != 0}\n"
            "if failures:\n"
            "    for gpu in failures: print(Path(f'/kaggle/working/benchmarks-gpu{gpu}.log').read_text()[-8000:])\n"
            "    raise RuntimeError(f'benchmark jobs failed: {failures}')\n"
            "print('both benchmark jobs finished')\n"
        ),
        code(
            "merged = {'suites': {}, 'kit_manifest': kit_manifest, 'checkpoint': str(CHECKPOINT)}\n"
            "for gpu, outdir in outputs.items():\n"
            "    partial = json.loads((outdir/'benchmarks-evaluation.json').read_text())\n"
            "    merged['suites'].update(partial['report']['suites'])\n"
            "    merged.setdefault('reports', {})[str(gpu)] = partial['report']\n"
            "print('SUITE COMPARISON (accuracy | HLWM selective@coverage vs baselines matched-coverage):')\n"
            "for suite, block in merged['suites'].items():\n"
            "    h = block['hlwm']\n"
            "    line = f\"{suite:14s} hlwm cand {h['candidate_accuracy']:.3f} causal {h['causal_accuracy']:.3f} \"\n"
            "    line += f\"| selective {h['selective_accuracy'] if h['selective_accuracy'] is not None else float('nan'):.3f} @ cov {h['coverage']:.2f}\"\n"
            "    for name in ('base','instruct'):\n"
            "        if name in block:\n"
            "            b = block[name]; mc = b['matched_coverage_selective'] or {}\n"
            "            line += f\" | {name} {b['accuracy']:.3f}\"\n"
            "            if mc: line += f\" (sel {mc['selective_accuracy']:.3f})\"\n"
            "    print(line)\n"
            "out = Path('/kaggle/working/hlwm-benchmarks-report.json')\n"
            "out.write_text(json.dumps(merged, indent=2, sort_keys=True)+'\\n')\n"
            "staging = Path('/kaggle/working/hlwm-benchmarks-deliverable')\n"
            "if staging.exists(): shutil.rmtree(staging)\n"
            "staging.mkdir()\n"
            "shutil.copy2(out, staging/out.name)\n"
            "for gpu, outdir in outputs.items():\n"
            "    shutil.copy2(outdir/'benchmarks-evaluation.json', staging/f'benchmarks-evaluation-gpu{gpu}.json')\n"
            "archive = Path(shutil.make_archive('/kaggle/working/hlwm-benchmarks-deliverable', 'zip', staging))\n"
            "archive.with_suffix(archive.suffix+'.sha256').write_text(file_sha256(archive)+'  '+archive.name+'\\n')\n"
            "print('deliverable:', archive)\n"
            "print('total session minutes:', round((time.time()-SESSION_START)/60, 1))\n"
        ),
        markdown(
            "## Reading the result\n\n"
            "The preregistered claim under test is **calibrated selective "
            "prediction**: at HLWM's achieved coverage, its selective accuracy "
            "should beat each baseline's matched-coverage selective accuracy "
            "(a comparison that structurally favors the baselines, whose "
            "abstention thresholds are fit post hoc on the test items). Raw "
            "candidate accuracy vs the instruct model is reported, not "
            "claimed. Coverage must always be reported next to accuracy, per "
            "seed. No production or superiority claim follows from a single "
            "seed or a single session.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    main()
