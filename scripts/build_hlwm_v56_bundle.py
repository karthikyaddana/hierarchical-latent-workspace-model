#!/usr/bin/env python3
"""Build the Version 5.6 verified multi-domain HLWM package (Kaggle dual-T4).

Version 5.6 changes over 5.5, per the Study 4 verdict and the 2026-08-30 data
review (`reports/hlwm-v5.6-data-review-2026-08-30.md`):

* routing objective: the marginal-entropy (KL-to-uniform) regularizer is
  removed (it homogenized experts); the Switch-style hard-assignment balance
  loss is strengthened and a pairwise expert-output diversity penalty is
  added;
* policy heads: fewer on-policy epochs plus label smoothing so calibration
  operates on an unsaturated score scale (also unconfounds the routing
  causal-liveness measurement);
* data: the v5.6 master dataset (`scripts/build_hlwm_v56_dataset.py`) —
  Reasoning9000 + adjudicated builder episodes + execution-verified benchmark
  train-split episodes (MBPP/APPS/TACO/CodeContests) + Spider exact-match +
  audited packets, with per-source dev slices;
* budgets: context 256 / canvas 128 / brief 96 / causal 1024 so the code data
  actually fits (v5.5's causal 384 fit ~25% of it).

Gate thresholds are unchanged from the Version 5.5 preregistration; the new
per-domain audit metrics are report-only for this first run on the new data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_hlwm_kaggle_bundle import (  # noqa: E402
    PIP_CELL_V55,
    behavior_anchor_rows,
    data_quality_report,
    line_count,
    sha256,
    validate_behavior_anchors,
)

SOURCE = ROOT / "experiments" / "kaggle_hlwm"
DATASET = ROOT / "data" / "hlwm-v5.6"
DEFAULT_OUTPUT = ROOT / "artifacts" / "kaggle" / "hlwm-v5.6.1"

ANCHOR_COUNTS = {"train": 1024, "validation": 256, "test": 256}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_split(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_prompt(record: Dict[str, Any]) -> str:
    return " ".join(str((record.get("input") or {}).get("user_request", "")).split())


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    project = output / "bundle" / "hlwm_kaggle"
    if output.exists():
        shutil.rmtree(output)
    project.mkdir(parents=True)

    for name in (
        "__init__.py",
        "modeling_hlwm.py",
        "data.py",
        "semantic_grading.py",
        "train_kaggle.py",
        "evaluate_checkpoint.py",
        "inference_hlwm.py",
        "test_modeling_hlwm.py",
        "README.md",
    ):
        shutil.copy2(SOURCE / name, project / name)

    # ---- dataset with cross-split prompt-collision fail-closed drop ----
    splits = {
        split: load_split(DATASET / "master" / (split + ".jsonl"))
        for split in ("train", "validation", "test")
    }
    held_out_prompts = {
        episode_prompt(record)
        for split in ("validation", "test")
        for record in splits[split]
        if episode_prompt(record)
    }
    seen_train: set[str] = set()
    kept_train: List[Dict[str, Any]] = []
    dropped_collisions = 0
    dropped_duplicates = 0
    for record in splits["train"]:
        prompt = episode_prompt(record)
        if prompt and prompt in held_out_prompts:
            dropped_collisions += 1
            continue
        if prompt and prompt in seen_train:
            dropped_duplicates += 1
            continue
        seen_train.add(prompt)
        kept_train.append(record)
    splits["train"] = kept_train

    dataset_counts: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    anchor_requests: Dict[str, set] = {}
    for split, records in splits.items():
        anchors = behavior_anchor_rows(split, ANCHOR_COUNTS[split], args.seed)
        validate_behavior_anchors(anchors, split)
        anchor_requests[split] = {
            str((row.get("input") or {}).get("user_request", "")) for row in anchors
        }
        destination = project / "data" / "master" / (split + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in anchors:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        dataset_counts[split] = len(records)
        counts[split] = len(records) + ANCHOR_COUNTS[split]

    split_names = sorted(anchor_requests)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if anchor_requests[left] & anchor_requests[right]:
                raise ValueError(
                    "behavior anchor prompt leakage between %s and %s" % (left, right)
                )

    quality = data_quality_report(project / "data" / "master")
    quality["generation_method"] = "mixed_v5.6_verified_multidomain"
    quality["programmatically_verified_behavior_anchors"] = sum(ANCHOR_COUNTS.values())
    quality["behavior_anchor_counts"] = ANCHOR_COUNTS
    quality["behavior_anchor_task_types"] = ["numeric", "unit", "ordering", "abstention"]
    quality["safe_abstention_is_public_target"] = True
    quality["policy_calibration_split"] = "generated_validation_emissions_only"
    quality["cross_split_prompt_collisions_dropped"] = dropped_collisions
    quality["duplicate_train_prompts_dropped"] = dropped_duplicates
    dataset_manifest = json.loads((DATASET / "manifest.json").read_text())
    quality["v56_dataset_manifest"] = dataset_manifest
    quality["policy_supervision_sources"] = (
        "programmatic behavior anchors; independently adjudicated builder episodes "
        "(executed pytest); benchmark episodes whose official reference passed its "
        "own executed checks at build time (marked programmatically_verified). "
        "Spider sql_exact and audited packets remain causal/workspace-only."
    )
    (project / "data-quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "name": "hlwm-verified-multidomain-v5-6-1-candidate",
        "package_version": "5.6.1",
        "purpose": (
            "routing-objective redesign (hard-assignment balance + expert diversity, "
            "marginal-KL removed) with execution-verified multi-domain data, sized for "
            "a Kaggle dual-T4 two-seed session"
        ),
        "review_status": (
            "Reasoning9000 rows remain unreviewed and policy-masked; builder episodes "
            "are the first independently adjudicated rows; benchmark episodes carry "
            "build-time execution-verified labels. Not authorized as production or "
            "factual-quality evidence until the preregistered gate passes."
        ),
        "counts": counts,
        "dataset_counts": dataset_counts,
        "behavior_anchor_counts": ANCHOR_COUNTS,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "base_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "seed": args.seed,
        "training_defaults": {
            "overfit_steps": 128,
            "local_steps": 1024,
            "joint_steps": 3072,
            "gradient_accumulation": 4,
            "learning_rate": 0.00008,
            "initial_loss_scale": 1024.0,
            "warmup_updates": 32,
            "causal_ratio": 0.40,
            "anchor_ratio": 0.50,
            "unfreeze_tail_layers": 0,
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "lora_dropout": 0.05,
            "lora_tail_layers": 8,
            "precision": "auto",
            "router_entropy_weight": 0.0,
            "router_aux_weight": 0.05,
            "expert_diversity_weight": 0.05,
            "context_tokens": 256,
            "canvas_tokens": 128,
            "brief_tokens": 96,
            "causal_tokens": 1024,
            "policy_records": 128,
            "policy_new_tokens": 96,
            "policy_epochs": 120,
            "expert_init_scale": 0.01,
            "policy_label_smoothing": 0.05,
            "policy_learning_rate": 0.001,
            "calibration_records": 64,
            "calibration_new_tokens": 96,
            "max_runtime_hours": 8.5,
        },
        "session_plan_hours": 9.0,
        "target_hardware": "Kaggle T4 x2, one independent seed per GPU",
        "v5_6_changes": [
            "marginal-entropy router regularizer removed (Study 4: it homogenized experts)",
            "Switch-style hard-assignment balance loss strengthened (weight 0.01 -> 0.05)",
            "pairwise expert-output diversity penalty on shared probe rows (weight 0.05)",
            "policy-head de-saturation: 400 -> 120 epochs plus 0.05 label smoothing",
            "execution-verified multi-domain data: MBPP/APPS/TACO/CodeContests episodes "
            "whose references passed their own executed checks, Spider exact-match, "
            "audited packets, adjudicated builder episodes, per-source dev slices",
            "budgets context 256 / canvas 128 / brief 96 / causal 1024 for the code data",
            "anchor ratio 0.75 -> 0.50 so external verified rows receive real exposure",
            "audit: domain-stratified sampling, execution/sql graders, per-domain metrics "
            "(report-only), generation budget 288 tokens",
            "v5.6.1: expert up-projections initialized at std 0.01 so the "
            "diversity penalty is live from step 1 (Session A telemetry showed "
            "it inert under zero init while seed 17 collapsed to one expert)",
            "v5.6.1: on-policy head phase enlarged 96 -> 128 records with "
            "per-family round-robin (Session A seed 29 confidently rejected "
            "valid answers from an under-represented anchor family)",
        ],
    }
    code_hashes = {path.name: sha256(path) for path in sorted(project.glob("*.py"))}
    manifest["code_sha256"] = code_hashes
    (project / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    notebook_t4 = output / "embel-hlwm-v5.6.1-kaggle-2xt4.ipynb"
    notebook_t4.write_text(
        json.dumps(
            notebook_document_v56_t4(
                counts,
                dataset_counts,
                int(quality["independently_adjudicated_episodes"]),
            ),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    archive = output / "hlwm-v5.6.1-candidate-bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted((output / "bundle").rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output / "bundle"))

    report = {
        "notebook_t4": str(notebook_t4),
        "notebook_t4_sha256": sha256(notebook_t4),
        "bundle": str(archive),
        "bundle_sha256": sha256(archive),
        "bundle_bytes": archive.stat().st_size,
        "counts": counts,
        "dataset_counts": dataset_counts,
        "cross_split_prompt_collisions_dropped": dropped_collisions,
        "duplicate_train_prompts_dropped": dropped_duplicates,
        "code_sha256": code_hashes,
    }
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def notebook_document_v56_t4(
    counts: Dict[str, int],
    dataset_counts: Dict[str, int],
    adjudicated_episodes: int,
) -> Dict[str, Any]:
    """Return the Version 5.6 dual-T4 two-seed notebook (Kaggle)."""

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

    counts_literal = json.dumps(counts, sort_keys=True)
    dataset_counts_literal = json.dumps(dataset_counts, sort_keys=True)

    cells = [
        markdown(
            "# Embel HLWM Version 5.6.1 - variance fixes on the Session A configuration\n\n"
            "Two independent seeds, one per T4. Version 5.6 removes the marginal-entropy "
            "router regularizer (Study 4 showed it homogenizes experts), strengthens the "
            "hard-assignment balance loss, adds an expert-output diversity penalty, "
            "de-saturates the policy heads (120 epochs + 0.05 label smoothing), and trains "
            "on the execution-verified multi-domain dataset (MBPP / APPS / TACO / "
            "CodeContests episodes whose official references passed their own executed "
            "checks, Spider exact-match, audited packets, adjudicated builder episodes) at "
            "context 256 / canvas 128 / brief 96 / causal 1024. Gate thresholds are "
            "unchanged from the Version 5.5 preregistration; per-domain audit metrics are "
            "report-only. Plan: `reports/hlwm-v5.6-plan-2026-08-30.md`.\n\n"
            "**Important:** start with **Save Version -> Save & Run All**. Session 1 runs "
            "seeds 17 and 29; session 2 sets `HLWM_SEEDS=41,73`. Attached "
            "`hlwm-v5.6.1-seed-<seed>-resumable.pt` checkpoints are picked up by name.\n"
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')\n"
            "import platform, subprocess, sys, time\n"
            "SESSION_START = time.time()\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(PIP_CELL_V55),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "KAGGLE = Path('/kaggle/input').exists()\n"
            "WORK = Path('/kaggle/working/hlwm-v5.6.1') if KAGGLE else Path.cwd() / 'hlwm-v5.6.1-work'\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
            "archives = [p for root in search_roots for p in root.rglob('hlwm-v5.6.1*candidate-bundle.zip')]\n"
            "archives += [p for root in search_roots for p in root.rglob('hlwm-v561*candidate-bundle.zip')]\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for root in search_roots:\n"
            "        for path in root.rglob('manifest.json'):\n"
            "            try:\n"
            "                if json.loads(path.read_text()).get('name') == 'hlwm-verified-multidomain-v5-6-1-candidate': manifests.append(path)\n"
            "            except (OSError, json.JSONDecodeError): pass\n"
            "    if not manifests: raise FileNotFoundError('Attach the HLWM Version 5.6 candidate bundle first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "quality = json.loads((PROJECT/'data-quality.json').read_text())\n"
            "assert manifest['package_version'] == '5.6.1'\n"
            "assert manifest['counts'] == " + counts_literal + "\n"
            "assert manifest['dataset_counts'] == " + dataset_counts_literal + "\n"
            "assert manifest['behavior_anchor_counts'] == {'train': 1024, 'validation': 256, 'test': 256}\n"
            "assert quality['independently_adjudicated_episodes'] == "
            + str(int(adjudicated_episodes))
            + "\n"
            "print(json.dumps({'project': str(PROJECT), 'counts': manifest['counts']}, indent=2))\n"
        ),
        code(
            "import importlib.util, py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.device_count() == 2, f'expected two T4 GPUs, found {torch.cuda.device_count()} (select GPU T4 x2)'\n"
            "for index in range(2):\n"
            "    properties = torch.cuda.get_device_properties(index)\n"
            "    print(index, properties.name, round(properties.total_memory/2**30, 2), 'GB')\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "if importlib.util.find_spec('pytest') is not None:\n"
            "    test_command = [sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')]\n"
            "else:\n"
            "    print('pytest unavailable; falling back to direct execution')\n"
            "    test_command = [sys.executable, str(PROJECT/'test_modeling_hlwm.py')]\n"
            "result = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)\n"
            "print((result.stdout or '')[-4000:])\n"
            "if result.returncode != 0:\n"
            "    print((result.stderr or '')[-4000:])\n"
            "    raise RuntimeError('Version 5.6 tests failed')\n"
            "print('Version 5.6 architecture, grader, routing and data tests passed')\n"
        ),
        code(
            "os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))\n"
            "Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])\n"
            "print('Pinned Qwen snapshot cached before the dual launch')\n"
        ),
        markdown(
            "## Real-Qwen numerical and memory preflight (GPU 0)\n\n"
            "Local **and joint** forward/backward passes at the Version 5.6 production "
            "lengths (context 256 / canvas 128 / causal 1024) with the full 151,936-token "
            "vocabulary, plus the 92% device-memory headroom limit. Both T4s are "
            "identical, so one preflight covers the dual launch.\n"
        ),
        code(
            "import os, subprocess, sys\n"
            "PREFLIGHT_OUTPUT = WORK / 'preflight'\n"
            "if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)\n"
            "preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '17', '--preflight-only',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',\n"
            "    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "    '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--precision', 'auto', '--router-entropy-weight', '0.0',\n"
            "    '--router-aux-weight', '0.05', '--expert-diversity-weight', '0.05',\n"
            "    '--expert-init-scale', '0.01',\n"
            "    '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',\n"
            "    '--max-validation', '1', '--num-workers', '0']\n"
            "preflight_environment = os.environ.copy(); preflight_environment['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "subprocess.run(preflight_command, cwd=PROJECT, env=preflight_environment, check=True)\n"
            "print('Local+joint real-Qwen preflight and memory headroom check passed on GPU 0')\n"
        ),
        markdown(
            "## Concurrent two-seed Version 5.6 training\n\n"
            "Each seed runs 128 overfit + 1,024 local + 3,072 joint microsteps (5% "
            "fixed-noise gate, zero skipped updates required) on the verified "
            "multi-domain data, then generates 96 train-anchor emissions, grades them, "
            "fits the three candidate heads for 120 epochs with 0.05 label smoothing, "
            "and calibrates thresholds on 64 generated validation emissions. The "
            "8.5-hour cap checkpoints exact resumable state; a truncated seed resumes "
            "next session via its attached `hlwm-v5.6.1-seed-<seed>-resumable.pt`.\n"
        ),
        code(
            "import os, subprocess, sys, time\n"
            "SEEDS = [int(part) for part in os.environ.get('HLWM_SEEDS', '17,29').split(',')]\n"
            "assert len(SEEDS) == 2, 'this notebook schedules exactly two seeds, one per T4'\n"
            "OUTPUTS = {seed: Path(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}') for seed in SEEDS}\n"
            "processes, handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    if output.exists(): shutil.rmtree(output)\n"
            "    resume = [p for root in search_roots for p in root.rglob(f'hlwm-v5.6.1-seed-{seed}-resumable.pt')]\n"
            "    command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),\n"
            "        '--overfit-steps', '128', '--local-steps', '1024', '--joint-steps', '3072',\n"
            "        '--overfit-examples', '64', '--overfit-min-improvement', '0.05',\n"
            "        '--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32',\n"
            "        '--causal-ratio', '0.40', '--anchor-ratio', '0.50', '--unfreeze-tail-layers', '0',\n"
            "        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "        '--precision', 'auto', '--router-entropy-weight', '0.0',\n"
            "        '--router-aux-weight', '0.05', '--expert-diversity-weight', '0.05',\n"
            "        '--expert-init-scale', '0.01',\n"
            "        '--skip-real-qwen-preflight',\n"
            "        '--initial-loss-scale', '1024', '--max-skipped-updates', '2',\n"
            "        '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "        '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',\n"
            "        '--max-validation', '256', '--calibration-records', '64', '--calibration-new-tokens', '96',\n"
            "        '--policy-records', '128', '--policy-new-tokens', '96', '--policy-epochs', '120',\n"
            "        '--policy-label-smoothing', '0.05', '--policy-learning-rate', '0.001',\n"
            "        '--num-workers', '2', '--eval-every', '512', '--save-every', '1024',\n"
            "        '--max-runtime-hours', '8.5']\n"
            "    if resume:\n"
            "        command += ['--resume', str(resume[0])]\n"
            "        print('seed', seed, 'resuming from', resume[0])\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = Path(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}-training.log').open('w'); handles[seed] = handle\n"
            "    processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "    print('launched seed', seed, 'on physical GPU', gpu)\n"
            "while any(process.poll() is None for process in processes.values()):\n"
            "    time.sleep(30)\n"
            "    print({seed: {'returncode': process.poll(),\n"
            "                  'elapsed_min': round((time.time()-SESSION_START)/60, 1),\n"
            "                  'metrics_bytes': (OUTPUTS[seed]/'metrics.jsonl').stat().st_size if (OUTPUTS[seed]/'metrics.jsonl').exists() else 0}\n"
            "           for seed, process in processes.items()})\n"
            "for handle in handles.values(): handle.close()\n"
            "failures = {seed: process.returncode for seed, process in processes.items() if process.returncode != 0}\n"
            "if failures:\n"
            "    for seed in failures: print(Path(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}-training.log').read_text()[-8000:])\n"
            "    raise RuntimeError(f'dual training failed: {failures}')\n"
            "print('Both seeds finished')\n"
        ),
        code(
            "summaries = {seed: json.loads((output/'summary.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "RUN_COMPLETE = {}\n"
            "for seed, summary in summaries.items():\n"
            "    assert summary['status'] in ('stable_prototype_training_complete', 'time_budget_checkpoint_saved'), summary['status']\n"
            "    RUN_COMPLETE[seed] = summary['status'] == 'stable_prototype_training_complete'\n"
            "    assert summary['planned_steps'] == 4224, summary['planned_steps']\n"
            "    assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "    assert summary['overfit_gate']['passed'], summary['overfit_gate']\n"
            "    assert summary['on_policy_policy_head'] is not None and summary['on_policy_policy_head']['trained'], summary['on_policy_policy_head']\n"
            "    assert summary['on_policy_policy_head']['label_smoothing'] == 0.05\n"
            "    calibration = summary['commitment_calibration']\n"
            "    assert calibration['method'] == 'validation_generated_candidate_joint_threshold_v2'\n"
            "    assert calibration['test_split_used'] is False\n"
            "    assert calibration['policy_heads_trained_on_policy'] is True\n"
            "    assert Path(summary['checkpoint']).exists() and Path(summary['adapter']['path']).exists()\n"
            "    print(seed, json.dumps({'run_complete': RUN_COMPLETE[seed], 'steps': summary['steps'],\n"
            "        'precision': summary['precision'], 'peak_gb': summary['peak_gpu_allocated_gb'],\n"
            "        'policy_separation_after': summary['on_policy_policy_head'].get('separation_after'),\n"
            "        'calibration': {k: calibration[k] for k in ('positive_accept_rate','negative_reject_rate','balanced_accuracy','passed_research_gate')}}, indent=2))\n"
        ),
        markdown(
            "## Fresh reload and per-seed 64-output audit with routing interventions\n\n"
            "Each seed reloads its checksummed checkpoint in a fresh process on its own "
            "GPU. The audit is domain-stratified (anchors plus code, sql, software "
            "reasoning, builder and Reasoning9000 domains), executes generated code "
            "against official checks, and measures normalized route entropy, second-route "
            "load, and the policy-score movement when routing is pinned to the least-used "
            "expert. Generation budget is 288 new tokens so code answers can complete.\n"
        ),
        code(
            "evaluation_processes, evaluation_handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "        '--checkpoint', summaries[seed]['checkpoint'], '--data-dir', str(DATA), '--output-dir', str(output),\n"
            "        '--samples', '64', '--context-tokens', '256', '--canvas-tokens', '128',\n"
            "        '--max-new-tokens', '288', '--seed', str(seed + 1000)]\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = (output/'evaluation.log').open('w'); evaluation_handles[seed] = handle\n"
            "    evaluation_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "for seed, process in evaluation_processes.items():\n"
            "    if process.wait() != 0:\n"
            "        for handle in evaluation_handles.values(): handle.close()\n"
            "        print((OUTPUTS[seed]/'evaluation.log').read_text()[-5000:])\n"
            "        raise RuntimeError(f'evaluation failed for seed {seed}')\n"
            "for handle in evaluation_handles.values(): handle.close()\n"
            "print('Both fresh-process evaluations finished')\n"
        ),
        code(
            "evaluations = {seed: json.loads((output/'checkpoint-evaluation.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "gates = {}\n"
            "for seed, evaluation in evaluations.items():\n"
            "    aggregate, records = evaluation['aggregate'], evaluation['records']\n"
            "    assert len(records) == 64 and all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in records)\n"
            "    gate = {\n"
            "        'training_complete': RUN_COMPLETE[seed],\n"
            "        'no_prompt_leak': aggregate['prompt_leak_rate'] == 0.0,\n"
            "        'quality_pass_rate_at_least_75pct': aggregate['quality_pass_rate'] >= 0.75,\n"
            "        'complete_answer_rate_at_least_50pct': aggregate['complete_answer_rate'] >= 0.50,\n"
            "        'semantic_probe_accuracy_at_least_75pct': aggregate['probe_content_accuracy'] >= 0.75,\n"
            "        'safe_abstention_accuracy_at_least_75pct': aggregate['safe_abstention_accuracy'] >= 0.75,\n"
            "        'safe_abstention_commit_rate_at_least_70pct': aggregate['safe_abstention_commit_rate'] >= 0.70,\n"
            "        'probe_commit_accuracy_at_least_70pct': aggregate['probe_commit_accuracy'] >= 0.70,\n"
            "        'validation_calibration_passed': bool(evaluation['validation_calibration'] and evaluation['validation_calibration']['passed_research_gate']),\n"
            "        'policy_heads_trained_on_policy': bool(evaluation['validation_calibration'] and evaluation['validation_calibration'].get('policy_heads_trained_on_policy')),\n"
            "        'clean_commit_ranked_above_corrupt': aggregate['mean_commit_margin'] > 0.0,\n"
            "        'corrupt_risk_ranked_above_clean': aggregate['mean_risk_margin'] > 0.0,\n"
            "        'corrupt_candidate_verifier_ranked_above_clean': aggregate['mean_verifier_error_margin'] > 0.0,\n"
            "        'second_route_load_at_least_10pct': aggregate['second_route_load'] >= 0.10,\n"
            "        'route_entropy_normalized_at_least_0p25': aggregate['route_entropy_normalized'] >= 0.25,\n"
            "        'routing_causally_live': aggregate['routing_intervention']['mean_abs_commit_delta'] >= 0.01,\n"
            "        'lanes_materially_distinct': abs(aggregate['mean_lane_summary_cosine']) < 0.90,\n"
            "        'candidate_f1_not_worse_than_80pct_base': aggregate['mean_hlwm_candidate_token_f1'] >= 0.8 * aggregate['mean_base_token_f1'],\n"
            "    }\n"
            "    gate['passed'] = all(gate.values()); gates[seed] = gate\n"
            "    (OUTPUTS[seed]/'v5.6.1-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')\n"
            "    print('SEED', seed); print(json.dumps(aggregate, indent=2)); print(json.dumps(gate, indent=2))\n"
            "    print('PER-DOMAIN (report-only):'); print(json.dumps(aggregate['domain_metrics'], indent=2))\n"
            "    for row in records[:2]:\n"
            "        print('\\n', row['episode_id'], 'expected_action=', row['expected_action'], 'committed=', row['committed'])\n"
            "        print('OUTPUT:', row['hlwm_candidate'][:500])\n"
            "replication = {'seeds': SEEDS,\n"
            "    'per_seed_passed': {str(seed): gates[seed]['passed'] for seed in SEEDS},\n"
            "    'replicated': all(gates[seed]['passed'] for seed in SEEDS),\n"
            "    'planned_steps': 4224,\n"
            "    'key_rates': {str(seed): {key: evaluations[seed]['aggregate'][key] for key in (\n"
            "        'probe_content_accuracy','safe_abstention_accuracy','probe_commit_accuracy',\n"
            "        'second_route_load','route_entropy_normalized')} for seed in SEEDS},\n"
            "    'domain_metrics': {str(seed): evaluations[seed]['aggregate']['domain_metrics'] for seed in SEEDS}}\n"
            "Path('/kaggle/working/hlwm-v5.6.1-replication-verdict.json').write_text(json.dumps(replication, indent=2)+'\\n')\n"
            "print(json.dumps(replication, indent=2))\n"
        ),
        code(
            "import hashlib\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "ARTIFACTS = [Path('/kaggle/working/hlwm-v5.6.1-replication-verdict.json')]\n"
            "for seed in SEEDS:\n"
            "    output = OUTPUTS[seed]; summary = summaries[seed]\n"
            "    staging = Path(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}-deliverable')\n"
            "    if staging.exists(): shutil.rmtree(staging)\n"
            "    staging.mkdir()\n"
            "    for source in [Path(summary['adapter']['path']), output/'summary.json', output/'commitment-calibration.json',\n"
            "                   output/'policy-head-training.json', output/'checkpoint-evaluation.json',\n"
            "                   output/'v5.6.1-capability-gate.json', output/'metrics.jsonl',\n"
            "                   PROJECT/'manifest.json', PROJECT/'README.md', PROJECT/'modeling_hlwm.py',\n"
            "                   PROJECT/'semantic_grading.py', PROJECT/'inference_hlwm.py']:\n"
            "        shutil.copy2(source, staging/source.name)\n"
            "    archive = Path(shutil.make_archive(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}-deliverable', 'zip', staging))\n"
            "    resumable = Path(f'/kaggle/working/hlwm-v5.6.1-seed-{seed}-resumable.pt')\n"
            "    if resumable.exists(): resumable.unlink()\n"
            "    source_checkpoint = Path(summary['checkpoint'])\n"
            "    shutil.move(str(source_checkpoint), resumable)\n"
            "    sidecar = source_checkpoint.with_suffix(source_checkpoint.suffix + '.sha256')\n"
            "    if sidecar.exists(): sidecar.unlink()\n"
            "    for stale in output.glob('checkpoint-*.pt*'):\n"
            "        if stale.is_file(): stale.unlink()\n"
            "    for artifact in (archive, resumable):\n"
            "        assert artifact.exists() and artifact.stat().st_size > 1_000_000, artifact\n"
            "        artifact.with_suffix(artifact.suffix+'.sha256').write_text(file_sha256(artifact)+'  '+artifact.name+'\\n')\n"
            "    parts = []\n"
            "    with resumable.open('rb') as source:\n"
            "        part_index = 0\n"
            "        while True:\n"
            "            payload = source.read(190 * 1024 * 1024)\n"
            "            if not payload: break\n"
            "            part = Path(str(resumable) + f'.part-{part_index:03d}')\n"
            "            part.write_bytes(payload)\n"
            "            part.with_suffix(part.suffix+'.sha256').write_text(file_sha256(part)+'  '+part.name+'\\n')\n"
            "            parts.append(part); part_index += 1\n"
            "    parts_manifest = Path(str(resumable) + '.parts.json')\n"
            "    parts_manifest.write_text(json.dumps({'checkpoint': resumable.name, 'checkpoint_bytes': resumable.stat().st_size,\n"
            "        'checkpoint_sha256': file_sha256(resumable),\n"
            "        'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': file_sha256(part)} for part in parts]}, indent=2)+'\\n')\n"
            "    ARTIFACTS += [archive, archive.with_suffix(archive.suffix+'.sha256'), resumable,\n"
            "                  resumable.with_suffix(resumable.suffix+'.sha256'), parts_manifest, *parts]\n"
            "Path('/kaggle/working/HLWM-V5.6.1-DOWNLOADS.txt').write_text('Save & Run All output. Download every line below.\\n'+'\\n'.join(str(path) for path in ARTIFACTS)+'\\n')\n"
            "print('replicated:', replication['replicated'])\n"
            "for seed in SEEDS: print(seed, 'run_complete', RUN_COMPLETE[seed], 'gate_passed', gates[seed]['passed'])\n"
            "try:\n"
            "    from IPython.display import FileLink, display\n"
            "    for path in ARTIFACTS: display(FileLink(str(path)))\n"
            "except Exception as error:\n"
            "    print('links unavailable outside a notebook UI:', error)\n"
            "import time\n"
            "print('total session minutes:', round((time.time()-SESSION_START)/60, 1))\n"
        ),
        markdown(
            "## Decision boundary\n\n"
            "Both seeds passing replicates the Version 5.6 research gate; it authorizes "
            "the external-benchmark step (eval-only registry suites against same-size "
            "instruct models), not deployment. Per-domain dev-slice metrics are "
            "report-only for this first run on the new data. A time-truncated seed must "
            "be resumed (attach its `hlwm-v5.6.1-seed-<seed>-resumable.pt`) before any gate "
            "claim for that seed. If both seeds fail the routing gates again, the "
            "declared fallback is `--num-experts 2` with the same objective. The matched "
            "plain-LoRA control (same data, steps, and trainable-parameter budget, no "
            "HLWM modules) is session B's first job either way.\n"
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
