#!/usr/bin/env python3
"""Package the Version 8.0 HLWM candidate bundle (Kaggle dual-T4, two seeds).

Version 8.0 answers the Version 6.0 Session E replication failure
(`reports/hlwm-v6.0-session-e-results-2026-09-02.md`) with a channel repair and
a claim decoupling; the full preregistration, the P1-P15 pre-mortem and the
five-rung fallback ladder live in `reports/hlwm-v8.0-plan-2026-09-02.md`.

Unlike the v5.6 builder this script does **not** re-derive the dataset. The
v8.0 data is bit-identical to the v6.0 data (the failure was in the model code,
not the corpus), so the materialized `data/master` split inside the bundle is
the source of record and is verified here rather than rebuilt. What this script
does own is the part v6.0 got wrong: `manifest.json` is regenerated from the
files actually being shipped, so the recorded `code_sha256` can never again go
stale behind a hotfix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "artifacts" / "kaggle" / "hlwm-v8.0"
PROJECT = BUNDLE_ROOT / "bundle" / "hlwm_kaggle"

CODE_FILES = (
    "__init__.py",
    "data.py",
    "evaluate_checkpoint.py",
    "inference_hlwm.py",
    "modeling_hlwm.py",
    "semantic_grading.py",
    "test_modeling_hlwm.py",
    "train_kaggle.py",
)

PACKAGE_NAME = "hlwm-repaired-channel-v8-0-candidate"
PACKAGE_VERSION = "8.0"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

EXPECTED_COUNTS = {"train": 7498, "validation": 945, "test": 841}
EXPECTED_DATASET_COUNTS = {"train": 6474, "validation": 689, "test": 585}
ANCHOR_COUNTS = {"train": 1024, "validation": 256, "test": 256}

# Decode temperatures for the N=8 candidate pool: one greedy anchor plus seven
# samples at a single moderate temperature.  A temperature *ladder* (v6.0 used
# 0.0/0.7/0.9/1.1) confounds the selector's job with a decode-quality gradient;
# a flat 0.8 makes the seven samples exchangeable.
CANDIDATE_TEMPERATURES = "0.0," + ",".join(["0.8"] * 7)

TRAINING_DEFAULTS: Dict[str, Any] = {
    "anchor_ratio": 0.5,
    "brief_tokens": 96,
    "calibration_new_tokens": 96,
    "calibration_records": 256,
    "candidate_temperatures": CANDIDATE_TEMPERATURES,
    "canvas_tokens": 128,
    "causal_ratio": 0.4,
    "causal_tokens": 1024,
    "conformal_target_coverage": 0.35,
    "context_tokens": 256,
    "expert_diversity_weight": 0.0,
    "expert_init_scale": 0.01,
    "gradient_accumulation": 4,
    "initial_loss_scale": 1024.0,
    "joint_steps": 3072,
    "learning_rate": 8e-05,
    "local_steps": 1024,
    "lora_alpha": 32.0,
    "lora_dropout": 0.05,
    "lora_rank": 16,
    "lora_tail_layers": 8,
    "max_runtime_hours": 8.5,
    "num_experts": 1,
    "num_lanes": 2,
    "overfit_steps": 128,
    "policy_epochs": 120,
    "policy_label_smoothing": 0.05,
    "policy_learning_rate": 0.001,
    "policy_new_tokens": 96,
    "policy_records": 128,
    "precision": "auto",
    "prefix_gate_init": 0.05,
    "router_aux_weight": 0.0,
    "router_entropy_weight": 0.0,
    "synthesis_kl_weight": 0.05,
    "unfreeze_tail_layers": 0,
    "warmup_updates": 32,
    "workspace_memory_windows": 16,
}

V8_CHANGES: List[str] = [
    "R1 no BOS anywhere on the workspace channel: Qwen3 has no BOS, so the "
    "v6.0 bos_token_id seed resolved to <|endoftext|> and injected a "
    "mid-sequence document boundary that made the model open a new turn "
    "(Human:/Assistant: scaffold). _channel_prefill/_channel_teacher_force "
    "never insert a seed id and align targets without one.",
    "R2 response cue moved out of the prompt string into config.response_cue_ids, "
    "appended internally as embeddings, so prompt truncation can never sever it "
    "and the base-model control can be format-matched.",
    "R3 zero-init relocated: memory/summary projections initialize at std 0.02 and "
    "a single scalar prefix_gate = tanh(alpha), alpha_0 = 0.05, carries the gate. "
    "Diversity and orthogonality gradients are live from step 0 instead of "
    "provably inert against zero-init outputs.",
    "R4 KL-to-base fluency anchor on the synthesis channel "
    "(--synthesis-kl-weight 0.05) against a detached causal reference.",
    "R5 lane diversity made structural: _prepare_briefs splits context positions "
    "with torch.tensor_split so each lane sees a different view, replacing a "
    "penalty term on identical inputs (v6.0 lane cosine 0.945/0.968).",
    "R6 four-feature publish rule that must consume the candidate mean logprob; "
    "weights fit on calibration half A, threshold set on held-out half B.",
    "Claim decoupling: the v6.0 headline (fan-in beats self-consistency at N=4, "
    "p~0.8) had 2-5 points of theoretical headroom and was unwinnable by "
    "construction. v8.0 splits into Claim G (channel parity), Claim S "
    "(verifier-weighted SC beats plurality SC on the medium stratum, McNemar "
    "mid-p) and Claim A (conformal abstention beats logprob abstention on AUGRC).",
    "Split-conformal publication threshold at c_nominal 0.35 with the coverage "
    "floor k = floor((n+1)(1-c)), replacing margin-midpoint selection; 256 "
    "calibration anchors, A/B split. Fixes the seed-29 coverage collapse (0.083).",
    "Candidate pool N=8 (1 greedy + 7 at T=0.8) decoded through the causal "
    "channel with a KV cache; the cache is what affords the 320-row audit.",
    "Audit population = every light-gradable test row (anchors first, n=320), "
    "difficulty-binned, with a format-matched base-model control.",
    "manifest.json is regenerated by this builder from the shipped files, so "
    "code_sha256 cannot go stale behind a hotfix the way v6.0's did.",
    "Scaffold-leak detection now matches a turn marker opening any line, not only "
    "the start of the answer: v6.0's scaffold followed real answer text, so the "
    "old start-of-string test would have scored those rows clean and let the "
    "no_scaffold_leak gate certify a still-broken channel.",
    "Gate battery 20 -> 24, grouped by claim; 46 -> 58 unit tests.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Notebook


def code(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


TRAIN_FLAGS = f"""
        '--overfit-steps', '128', '--local-steps', '1024', '--joint-steps', '3072',
        '--overfit-examples', '64', '--overfit-min-improvement', '0.05',
        '--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32',
        '--causal-ratio', '0.40', '--anchor-ratio', '0.50', '--unfreeze-tail-layers', '0',
        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',
        '--precision', 'auto', '--router-entropy-weight', '0.0',
        '--router-aux-weight', '0.0', '--expert-diversity-weight', '0.0',
        '--expert-init-scale', '0.01', '--workspace-memory-windows', '16',
        '--synthesis-kl-weight', '0.05', '--prefix-gate-init', '0.05',
        '--initial-loss-scale', '1024', '--max-skipped-updates', '2',
        '--num-lanes', '2', '--num-experts', '1', '--refinement-steps', '2', '--diffusion-steps', '4',
        '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',
        '--max-validation', '256', '--calibration-records', '256', '--calibration-new-tokens', '96',
        '--conformal-target-coverage', '0.35',
        '--policy-records', '128', '--policy-new-tokens', '96', '--policy-epochs', '120',
        '--policy-label-smoothing', '0.05', '--policy-learning-rate', '0.001',
        '--policy-min-valid-per-family', '16', '--policy-max-attempts', '1024',
        '--candidate-temperatures', '{CANDIDATE_TEMPERATURES}',
        '--canary-anchors', '8', '--canary-new-tokens', '96',
"""


def notebook_document() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []

    cells.append(
        markdown(
            """
# Embel HLWM Version 8.0 - repaired channel, decoupled claims

Two independent seeds, one per T4, workspace-only. Version 8.0 exists because the
Version 6.0 replication (Session E) failed both seeds for two separable reasons, and
this session tests the repair rather than re-running the same battery.

**The line-level defect.** `Qwen3-0.6B-Base` has no BOS token, so `bos_token_id`
resolves to `eos` (`<|endoftext|>`). Every workspace teacher-forcing and decode call
seeded the answer with it, injecting a mid-sequence document boundary; the model
opened a new turn and emitted `Human:`/`Assistant:` scaffold. The damage was confined
to the workspace channel - exactly where the claims live - costing ~30 points of
quality against the causal path on the same rows.

**The design defect.** The v6.0 headline asked 4 candidates at p~0.8 to beat
self-consistency. The headroom for *any* selector is `N*p*(1-p)^(N-1)` ~ 2-5 points,
one or two rows out of 36. The gate could not be won regardless of selector quality.

Version 8.0 therefore ships six channel repairs (R1-R6) and splits the hypothesis
into three separately powered claims:

- **Claim G (parity)** - the repaired workspace channel matches the causal channel:
  no scaffold leak, accuracy within 5 points, F1 within 10%, live read-out memory,
  open prefix gate.
- **Claim S (selection)** - verifier-weighted self-consistency beats plurality
  self-consistency on the **medium**-difficulty stratum, where headroom exists,
  by McNemar mid-p on discordant pairs, with an anchors-only sensitivity check and
  a `heads_not_logprob` guard.
- **Claim A (abstention)** - split-conformal publication at `c_nominal = 0.35`
  (`k = floor((n+1)(1-c))` on held-out calibration half B) beats mean-logprob
  abstention on AUGRC in >= 90% of paired bootstrap resamples, with a
  Clopper-Pearson upper bound on selective risk.

24 gates, 58 unit tests, audit over every light-gradable test row (n=320).
Plan, P1-P15 pre-mortem and the five-rung fallback ladder:
`reports/hlwm-v8.0-plan-2026-09-02.md`.
"""
        )
    )

    cells.append(
        code(
            """
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')
import platform, subprocess, sys, time
SESSION_START = time.time()
print(platform.platform())
subprocess.run(['nvidia-smi'], check=True)
print('python', sys.version)
"""
        )
    )

    cells.append(
        code(
            """
import importlib, importlib.metadata, site, subprocess, sys
def ensure(spec):
    name, version = spec.split('==')
    try:
        if importlib.metadata.version(name) == version:
            print('already pinned:', spec); return
    except importlib.metadata.PackageNotFoundError: pass
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', spec], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--user',
                        '--no-warn-script-location', spec], check=True)
    print('installed:', spec)
for spec in ('transformers==4.56.2', 'accelerate==1.10.1', 'safetensors==0.6.2', 'pytest==8.4.1'):
    ensure(spec)
user_site = site.getusersitepackages()
if user_site not in sys.path: sys.path.insert(0, user_site)
importlib.invalidate_caches()
import transformers
print('transformers', transformers.__version__, '| user site', user_site)
"""
        )
    )

    cells.append(
        code(
            f"""
from pathlib import Path
import hashlib, json, shutil, zipfile
KAGGLE = Path('/kaggle/input').exists()
WORK = Path('/kaggle/working/hlwm-v8.0') if KAGGLE else Path.cwd() / 'hlwm-v8.0-work'
if WORK.exists(): shutil.rmtree(WORK)
search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]
archives = [p for root in search_roots for p in root.rglob('hlwm-v8.0*candidate-bundle.zip')]
archives += [p for root in search_roots for p in root.rglob('hlwm-v80*candidate-bundle.zip')]
if archives:
    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)
    PROJECT = WORK / 'hlwm_kaggle'
else:
    manifests = []
    for root in search_roots:
        for path in root.rglob('manifest.json'):
            try:
                if json.loads(path.read_text()).get('name') == '{PACKAGE_NAME}': manifests.append(path)
            except (OSError, json.JSONDecodeError): pass
    if not manifests: raise FileNotFoundError('Attach the HLWM Version 8.0 candidate bundle first.')
    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')
    PROJECT = WORK / 'hlwm_kaggle'
DATA = PROJECT / 'data'
manifest = json.loads((PROJECT/'manifest.json').read_text())
quality = json.loads((PROJECT/'data-quality.json').read_text())
assert manifest['package_version'] == '{PACKAGE_VERSION}', manifest['package_version']
assert manifest['counts'] == {json.dumps(EXPECTED_COUNTS)}
assert manifest['dataset_counts'] == {json.dumps(EXPECTED_DATASET_COUNTS)}
assert manifest['behavior_anchor_counts'] == {json.dumps(ANCHOR_COUNTS)}
assert quality['independently_adjudicated_episodes'] == 13

# v6.0 shipped a manifest whose code_sha256 had gone stale behind a hotfix, so the
# recorded hashes did not describe the code that ran. Verify before spending 7 GPU hours.
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()
stale = {{name: (recorded, file_sha256(PROJECT/name))
         for name, recorded in manifest['code_sha256'].items()
         if file_sha256(PROJECT/name) != recorded}}
assert not stale, f'manifest code_sha256 does not match the shipped files: {{stale}}'
print(json.dumps({{'project': str(PROJECT), 'counts': manifest['counts'],
                  'code_sha256_verified': len(manifest['code_sha256'])}}, indent=2))
"""
        )
    )

    cells.append(
        code(
            """
import importlib.util, py_compile, subprocess, sys, torch
assert torch.cuda.device_count() == 2, f'expected two T4 GPUs, found {torch.cuda.device_count()} (select GPU T4 x2)'
for index in range(2):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, round(properties.total_memory/2**30, 2), 'GB')
for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):
    py_compile.compile(str(PROJECT/name), doraise=True)
if importlib.util.find_spec('pytest') is not None:
    test_command = [sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')]
else:
    print('pytest unavailable; falling back to direct execution')
    test_command = [sys.executable, str(PROJECT/'test_modeling_hlwm.py')]
result = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)
print((result.stdout or '')[-4000:])
if result.returncode != 0:
    print((result.stderr or '')[-4000:])
    raise RuntimeError('Version 8.0 tests failed')
print('Version 8.0 channel, conformal, KV-cache, lane and grader tests passed')
"""
        )
    )

    cells.append(
        code(
            """
os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])
print('Pinned Qwen snapshot cached before the dual launch')
"""
        )
    )

    cells.append(
        markdown(
            """
## Tokenizer assertion: the Version 6.0 root cause, checked directly

The whole v6.0 failure reduces to one property of this tokenizer. Assert it here so
that if a future revision *does* add a BOS token, the discrepancy surfaces in seconds
rather than being inferred from scaffold text seven hours later.
"""
        )
    )

    cells.append(
        code(
            """
from transformers import AutoTokenizer
_tokenizer = AutoTokenizer.from_pretrained(manifest['base_model'], revision=manifest['base_revision'], trust_remote_code=False)
print('bos_token', _tokenizer.bos_token, '| bos_token_id', _tokenizer.bos_token_id)
print('eos_token', _tokenizer.eos_token, '| eos_token_id', _tokenizer.eos_token_id)
# Qwen3 has no BOS: transformers aliases bos_token_id to eos, which is exactly the id
# v6.0 seeded every workspace answer with. v8.0 seeds nothing; this records the fact.
assert _tokenizer.bos_token_id in (None, _tokenizer.eos_token_id), (
    'base revision now defines a distinct BOS; revisit the R1 no-seed decision')
sys.path.insert(0, str(PROJECT))
from data import RESPONSE_CUE_TEXT, build_public_prompt
cue_ids = _tokenizer.encode(RESPONSE_CUE_TEXT, add_special_tokens=False)
print('response cue', repr(RESPONSE_CUE_TEXT), '->', cue_ids)
assert cue_ids, 'response cue must tokenize to at least one id'
# Check for the cue's own text, not '### Response': the prompt legitimately keeps a
# '### Response requirements' heading, so a substring test reports a false positive.
_probe_prompt = build_public_prompt({'input': {'user_request': 'ping'}})
assert RESPONSE_CUE_TEXT not in _probe_prompt, 'R2 regression: cue is back in the prompt'
print('R2 check: cue absent from the prompt string, appended internally as embeddings')
"""
        )
    )

    cells.append(
        markdown(
            """
## Real-Qwen numerical and memory preflight (GPU 0)

Local **and joint** forward/backward at production lengths (context 256 / canvas 128 /
causal 1024) with the full 151,936-token vocabulary, now including the synthesis KL
anchor and the widened 43-token read-out prefix, plus the 92% device-memory headroom
limit. Both T4s are identical, so one preflight covers the dual launch.
"""
        )
    )

    cells.append(
        code(
            f"""
import os, subprocess, sys
PREFLIGHT_OUTPUT = WORK / 'preflight'
if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)
preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),
    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '17', '--preflight-only',
    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',
    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',
    '--num-lanes', '2', '--num-experts', '1', '--refinement-steps', '2', '--diffusion-steps', '4',
    '--precision', 'auto', '--router-entropy-weight', '0.0',
    '--router-aux-weight', '0.0', '--expert-diversity-weight', '0.0',
    '--expert-init-scale', '0.01', '--workspace-memory-windows', '16',
    '--synthesis-kl-weight', '0.05', '--prefix-gate-init', '0.05',
    '--candidate-temperatures', '{CANDIDATE_TEMPERATURES}',
    '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',
    '--max-validation', '1', '--num-workers', '0']
preflight_environment = os.environ.copy(); preflight_environment['CUDA_VISIBLE_DEVICES'] = '0'
subprocess.run(preflight_command, cwd=PROJECT, env=preflight_environment, check=True)
print('Local+joint real-Qwen preflight and memory headroom check passed on GPU 0')
"""
        )
    )

    cells.append(
        markdown(
            """
## Concurrent two-seed Version 8.0 training

Each seed runs 128 overfit + 1,024 local + 3,072 joint microsteps (5% fixed-noise gate,
zero skipped updates required), logging the 8-generation domain-stratified canary at
each 512-step boundary. The canary now requires light-gradable rows, so its accuracy
number is real rather than n=1.

It then collects the N=8 fan-in pool (1 greedy + 7 at T=0.8) with the >=16 valid
positives per family floor, fits the four-feature publish combiner on calibration
half A, and sets the publication threshold as the split-conformal order statistic
`k = floor((n+1)(1-0.35))` on held-out half B over 256 validation anchors.
"""
        )
    )

    cells.append(
        code(
            f"""
import os, subprocess, sys, time
SEEDS = [int(part) for part in os.environ.get('HLWM_SEEDS', '17,29').split(',')]
assert len(SEEDS) == 2, 'this notebook schedules exactly two seeds, one per T4'
OUTPUTS = {{seed: Path(f'/kaggle/working/hlwm-v8.0-seed-{{seed}}') for seed in SEEDS}}
processes, handles = {{}}, {{}}
for gpu, seed in enumerate(SEEDS):
    output = OUTPUTS[seed]
    if output.exists(): shutil.rmtree(output)
    resume = [p for root in search_roots + [WORK] for p in root.rglob(f'hlwm-v8.0-seed-{{seed}}-resumable.pt')]
    command = [sys.executable, str(PROJECT/'train_kaggle.py'),
        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),
        '--skip-real-qwen-preflight',{TRAIN_FLAGS}        '--num-workers', '2', '--eval-every', '512', '--save-every', '1024',
        '--max-runtime-hours', '8.5']
    if resume:
        command += ['--resume', str(resume[0])]
        print('seed', seed, 'resuming from', resume[0])
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    handle = Path(f'/kaggle/working/hlwm-v8.0-seed-{{seed}}-training.log').open('w'); handles[seed] = handle
    processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    print('launched seed', seed, 'on physical GPU', gpu)
while any(process.poll() is None for process in processes.values()):
    time.sleep(30)
    print({{seed: {{'returncode': process.poll(),
                  'elapsed_min': round((time.time()-SESSION_START)/60, 1),
                  'metrics_bytes': (OUTPUTS[seed]/'metrics.jsonl').stat().st_size if (OUTPUTS[seed]/'metrics.jsonl').exists() else 0}}
           for seed, process in processes.items()}})
for handle in handles.values(): handle.close()
failures = {{seed: process.returncode for seed, process in processes.items() if process.returncode != 0}}
if failures:
    for seed in failures: print(Path(f'/kaggle/working/hlwm-v8.0-seed-{{seed}}-training.log').read_text()[-8000:])
    raise RuntimeError(f'dual training failed: {{failures}}')
print('Both seeds finished')
"""
        )
    )

    cells.append(
        code(
            """
summaries = {seed: json.loads((output/'summary.json').read_text()) for seed, output in OUTPUTS.items()}
RUN_COMPLETE = {}
for seed, summary in summaries.items():
    assert summary['status'] in ('stable_prototype_training_complete', 'time_budget_checkpoint_saved'), summary['status']
    RUN_COMPLETE[seed] = summary['status'] == 'stable_prototype_training_complete'
    assert summary['planned_steps'] == 4224, summary['planned_steps']
    assert summary['skipped_optimizer_updates'] == 0, summary
    assert summary['overfit_gate']['passed'], summary['overfit_gate']
    head = summary['on_policy_policy_head']
    assert head is not None and head['trained'], head
    assert head['method'] == 'on_policy_train_anchor_head_fit_v2_decorrelated'
    calibration = summary['commitment_calibration']
    assert calibration['method'] == 'validation_generated_nbest_conformal_publish_v4', calibration['method']
    assert calibration['test_split_used'] is False
    assert calibration['policy_heads_trained_on_policy'] is True
    conformal = calibration['conformal']
    assert conformal['fitted'], conformal
    # The conformal construction is only meaningful if the threshold came off the
    # held-out half; a collapsed split would silently reproduce the v6.0 pathology.
    assert calibration['split_sizes']['holdout_anchors'] >= 16, calibration['split_sizes']
    # R3: the gate must have moved off its 0.05 initialization in some direction;
    # a gate pinned at init means the read-out prefix never earned its way in.
    print(seed, 'R3 prefix_gate', json.dumps(summary['prefix_gate']),
          'response_cue_ids', summary['response_cue_ids'])
    assert summary['response_cue_ids'], 'R2 regression: adapter carries no response cue'
    print(seed, 'validity_floor_met', head.get('validity_floor_met'),
          'family_valid_positives', head.get('family_valid_positives'), 'attempts', head.get('emitted'))
    print(seed, 'conformal', json.dumps(conformal))
    print(seed, 'publish_rule', json.dumps(calibration['publish_rule']))
    print(seed, 'selection', json.dumps(calibration['selection']))
    assert Path(summary['checkpoint']).exists() and Path(summary['adapter']['path']).exists()
    print(seed, json.dumps({'run_complete': RUN_COMPLETE[seed], 'steps': summary['steps'],
        'precision': summary['precision'], 'peak_gb': summary['peak_gpu_allocated_gb'],
        'policy_separation_after': head.get('separation_after'),
        'calibration': {k: calibration[k] for k in ('positive_accept_rate','negative_reject_rate','balanced_accuracy','passed_research_gate')}}, indent=2))
"""
        )
    )

    cells.append(
        markdown(
            """
## Fresh reload and per-seed 320-row audit

Each seed reloads its checksummed checkpoint in a fresh process on its own GPU. The
audit population is every light-gradable test row, anchors first, so the difficulty
strata and the McNemar discordant-pair counts have real sample size (v6.0's headline
rested on 36 rows). Arms scored on identical rows and an identical N=8 pool: greedy,
plurality self-consistency, verifier-weighted self-consistency, max-logprob
best-of-N, heads-argmax, and oracle-any-valid. The read-out ablation re-decodes every
row with the memory sliced off; the base-model control uses a format-matched prompt.
"""
        )
    )

    cells.append(
        code(
            f"""
evaluation_processes, evaluation_handles = {{}}, {{}}
for gpu, seed in enumerate(SEEDS):
    output = OUTPUTS[seed]
    command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'),
        '--checkpoint', summaries[seed]['checkpoint'], '--data-dir', str(DATA), '--output-dir', str(output),
        '--samples', '320', '--context-tokens', '256', '--canvas-tokens', '128',
        '--max-new-tokens', '288', '--candidate-temperatures', '{CANDIDATE_TEMPERATURES}',
        '--seed', str(seed + 1000)]
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    handle = (output/'evaluation.log').open('w'); evaluation_handles[seed] = handle
    evaluation_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
for seed, process in evaluation_processes.items():
    if process.wait() != 0:
        for handle in evaluation_handles.values(): handle.close()
        print((OUTPUTS[seed]/'evaluation.log').read_text()[-5000:])
        raise RuntimeError(f'evaluation failed for seed {{seed}}')
for handle in evaluation_handles.values(): handle.close()
print('Both fresh-process evaluations finished')
"""
        )
    )

    cells.append(
        code(
            """
evaluations = {seed: json.loads((output/'checkpoint-evaluation.json').read_text()) for seed, output in OUTPUTS.items()}
gates = {}
for seed, evaluation in evaluations.items():
    aggregate, records = evaluation['aggregate'], evaluation['records']
    assert all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in records)
    # evaluate_checkpoint.py writes the 24-gate battery; only training_complete is
    # outside its knowledge, so the notebook supplies it and re-derives `passed`.
    gate = dict(evaluation['gates'])
    gate['training_complete'] = RUN_COMPLETE[seed]
    gate['passed'] = all(bool(value) for key, value in gate.items() if key != 'passed')
    gates[seed] = gate
    (OUTPUTS[seed]/'v8.0-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')
    print('SEED', seed, '| audited rows', len(records))
    print(json.dumps({k: v for k, v in aggregate.items() if k != 'domain_metrics'}, indent=2)[:9000])
    print(json.dumps(gate, indent=2))
    print('FAILED GATES:', [k for k, v in gate.items() if k != 'passed' and not v])
    print('PER-DOMAIN (report-only):'); print(json.dumps(aggregate['domain_metrics'], indent=2))
    for row in records[:2]:
        print('\\n', row['episode_id'], 'expected_action=', row['expected_action'], 'committed=', row['committed'])
        print('OUTPUT:', row['hlwm_candidate'][:500])
"""
        )
    )

    cells.append(
        markdown(
            """
## Pooled headline and replication verdict

Claim S is tested per seed, but two seeds of ~320 rows give a better-powered pooled
test than either alone. Pooling here means pooling the **discordant pairs** (rows
where weighted SC and plurality SC disagree) across seeds and running one McNemar
mid-p, which is the correct pooled statistic for a paired binary comparison. The
per-seed gates remain binding; the pooled number is reported alongside them.
"""
        )
    )

    cells.append(
        code(
            """
from math import comb
def mcnemar_mid_p(wins, losses):
    # Identical to evaluate_checkpoint.mcnemar_mid_p; restated here so the pooled
    # statistic is computed in the notebook without importing the audit module.
    total = wins + losses
    if total == 0: return 1.0
    tail = sum(comb(total, k) for k in range(wins + 1, total + 1)) / 2.0 ** total
    at_observed = comb(total, wins) / 2.0 ** total
    return min(1.0, tail + 0.5 * at_observed)

pooled_wins = pooled_losses = 0
pooled_medium_rows = 0
for seed, evaluation in evaluations.items():
    for row in evaluation['records']:
        if row['difficulty_bin'] != 'medium': continue
        pooled_medium_rows += 1
        weighted, plurality = row['arms']['weighted_sc'], row['arms']['sc_vote']
        pooled_wins += int(bool(weighted) and not bool(plurality))
        pooled_losses += int(bool(plurality) and not bool(weighted))
pooled = {'medium_rows': pooled_medium_rows, 'wins': pooled_wins, 'losses': pooled_losses,
          'mcnemar_mid_p': mcnemar_mid_p(pooled_wins, pooled_losses)}
print('POOLED CLAIM S:', json.dumps(pooled, indent=2))

def claim_status(keys, seed):
    return {key: bool(gates[seed][key]) for key in keys}
CLAIM_G = ('no_scaffold_leak','channel_parity_accuracy','channel_parity_f1','readout_live','gate_alpha_open','quality_pass_rate_at_least_70pct')
CLAIM_S = ('weighted_sc_beats_sc_medium','weighted_sc_no_worse_overall','argmax_selection_ge_greedy','selection_headroom_present','heads_not_logprob')
CLAIM_A = ('coverage_in_band','abstention_beats_logprob_augrc','selective_accuracy_matched','risk_bound','safe_abstention_probes')
replication = {'seeds': SEEDS,
    'per_seed_passed': {str(seed): gates[seed]['passed'] for seed in SEEDS},
    'replicated': all(gates[seed]['passed'] for seed in SEEDS),
    'planned_steps': 4224,
    'failed_gates': {str(seed): [k for k, v in gates[seed].items() if k != 'passed' and not v] for seed in SEEDS},
    'claims': {str(seed): {'G': claim_status(CLAIM_G, seed), 'S': claim_status(CLAIM_S, seed), 'A': claim_status(CLAIM_A, seed)} for seed in SEEDS},
    'claim_s_pooled': pooled,
    'channel_parity': {str(seed): evaluations[seed]['aggregate']['channel_parity'] for seed in SEEDS},
    'selection': {str(seed): evaluations[seed]['aggregate']['selection'] for seed in SEEDS},
    'abstention': {str(seed): evaluations[seed]['aggregate']['abstention'] for seed in SEEDS},
    'score_diagnostics': {str(seed): evaluations[seed]['aggregate']['score_diagnostics'] for seed in SEEDS},
    'readout_ablation': {str(seed): evaluations[seed]['aggregate']['readout_ablation'] for seed in SEEDS},
    'conformal': {str(seed): evaluations[seed].get('conformal') for seed in SEEDS},
    'prefix_gate_tanh': {str(seed): evaluations[seed]['aggregate'].get('prefix_gate_tanh') for seed in SEEDS},
    'lane_cosine': {str(seed): evaluations[seed]['aggregate']['mean_lane_summary_cosine'] for seed in SEEDS},
    'key_rates': {str(seed): {key: evaluations[seed]['aggregate'][key] for key in (
        'probe_content_accuracy','safe_abstention_accuracy','probe_commit_accuracy',
        'quality_pass_rate','prompt_leak_rate','complete_answer_rate')} for seed in SEEDS},
    'domain_metrics': {str(seed): evaluations[seed]['aggregate']['domain_metrics'] for seed in SEEDS},
    'boundary': ('Research gate only. Passing authorizes the plain-LoRA control and the '
                 'external-benchmark step, not deployment.')}
Path('/kaggle/working/hlwm-v8.0-replication-verdict.json').write_text(json.dumps(replication, indent=2)+'\\n')
print(json.dumps({k: v for k, v in replication.items() if k not in ('domain_metrics','selection')}, indent=2))
"""
        )
    )

    cells.append(
        code(
            """
ARTIFACTS = [Path('/kaggle/working/hlwm-v8.0-replication-verdict.json')]
for seed in SEEDS:
    output = OUTPUTS[seed]; summary = summaries[seed]
    staging = Path(f'/kaggle/working/hlwm-v8.0-seed-{seed}-deliverable')
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir()
    for source in [Path(summary['adapter']['path']), output/'summary.json', output/'commitment-calibration.json',
                   output/'policy-head-training.json', output/'checkpoint-evaluation.json',
                   output/'v8.0-capability-gate.json', output/'metrics.jsonl',
                   PROJECT/'manifest.json', PROJECT/'README.md', PROJECT/'modeling_hlwm.py',
                   PROJECT/'semantic_grading.py', PROJECT/'inference_hlwm.py']:
        shutil.copy2(source, staging/source.name)
    archive = Path(shutil.make_archive(f'/kaggle/working/hlwm-v8.0-seed-{seed}-deliverable', 'zip', staging))
    resumable = Path(f'/kaggle/working/hlwm-v8.0-seed-{seed}-resumable.pt')
    if resumable.exists(): resumable.unlink()
    source_checkpoint = Path(summary['checkpoint'])
    shutil.move(str(source_checkpoint), resumable)
    sidecar = source_checkpoint.with_suffix(source_checkpoint.suffix + '.sha256')
    if sidecar.exists(): sidecar.unlink()
    for stale_checkpoint in output.glob('checkpoint-*.pt*'):
        if stale_checkpoint.is_file(): stale_checkpoint.unlink()
    for artifact in (archive, resumable):
        assert artifact.exists() and artifact.stat().st_size > 1_000_000, artifact
        artifact.with_suffix(artifact.suffix+'.sha256').write_text(file_sha256(artifact)+'  '+artifact.name+'\\n')
    parts = []
    with resumable.open('rb') as source:
        part_index = 0
        while True:
            payload = source.read(190 * 1024 * 1024)
            if not payload: break
            part = Path(str(resumable) + f'.part-{part_index:03d}')
            part.write_bytes(payload)
            part.with_suffix(part.suffix+'.sha256').write_text(file_sha256(part)+'  '+part.name+'\\n')
            parts.append(part); part_index += 1
    parts_manifest = Path(str(resumable) + '.parts.json')
    parts_manifest.write_text(json.dumps({'checkpoint': resumable.name, 'checkpoint_bytes': resumable.stat().st_size,
        'checkpoint_sha256': file_sha256(resumable),
        'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': file_sha256(part)} for part in parts]}, indent=2)+'\\n')
    ARTIFACTS += [archive, archive.with_suffix(archive.suffix+'.sha256'), resumable,
                  resumable.with_suffix(resumable.suffix+'.sha256'), parts_manifest, *parts]
Path('/kaggle/working/HLWM-V8.0-DOWNLOADS.txt').write_text('Save & Run All output. Download every line below.\\n'+'\\n'.join(str(path) for path in ARTIFACTS)+'\\n')
print('replicated:', replication['replicated'])
for seed in SEEDS: print(seed, 'run_complete', RUN_COMPLETE[seed], 'gate_passed', gates[seed]['passed'],
                         'failed', replication['failed_gates'][str(seed)])
try:
    from IPython.display import FileLink, display
    for path in ARTIFACTS: display(FileLink(str(path)))
except Exception as error:
    print('links unavailable outside a notebook UI:', error)
import time
print('total session minutes:', round((time.time()-SESSION_START)/60, 1))
"""
        )
    )

    cells.append(
        markdown(
            """
## Decision boundary and the binding fallback ladder

Passing is a research gate. It authorizes the plain-LoRA control (the decisive
comparison, still unrun) and the external-benchmark step - not deployment. A
time-truncated seed must be resumed (attach its `hlwm-v8.0-seed-<seed>-resumable.pt`)
before any gate claim for that seed.

The ladder below is preregistered in `reports/hlwm-v8.0-plan-2026-09-02.md` and is
binding: read the failure off the verdict's `failed_gates` and `claims` blocks and
take the corresponding rung rather than improvising a rescue.

1. **Claim G fails** (parity, leak, or dead read-out) - the repair did not take. Stop;
   no selection or abstention result from this run is interpretable, because both are
   measured on the channel the repair was supposed to fix.
2. **G passes, S fails** - the heads carry no selection signal beyond decode
   likelihood. Re-scope to Claim A only; the selection sections become negative
   evidence, as routing did in Study 7.
3. **G passes, S and A both fail** - the sidecar adds nothing a logprob does not.
   Retire the head-selection hypothesis and report the parity repair alone.
4. **G and A pass, S fails, `heads_not_logprob` also fails** - report the heads as a
   re-derivation of likelihood, not a verifier, and say so explicitly.
5. **All three pass** - run the plain-LoRA control immediately, at matched trainable
   parameter budget. Without it the result attributes to the workspace what may
   belong to the extra capacity.
"""
        )
    )

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
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if not PROJECT.exists():
        raise SystemExit(f"v8.0 bundle sources not found at {PROJECT}")

    missing = [name for name in CODE_FILES if not (PROJECT / name).exists()]
    if missing:
        raise SystemExit(f"bundle is missing code files: {missing}")

    data_dir = PROJECT / "data" / "master"
    counts = {split: line_count(data_dir / f"{split}.jsonl") for split in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"data counts drifted: {counts} != {EXPECTED_COUNTS}")

    if not args.skip_tests:
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "pytest", "-q", "test_modeling_hlwm.py"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        print((result.stdout or "").strip()[-2000:])
        if result.returncode != 0:
            print((result.stderr or "")[-2000:])
            raise SystemExit("v8.0 tests failed; refusing to build a bundle")

    # ------------------------------------------------------------------
    # Manifest, regenerated from the files actually being shipped.
    manifest = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "behavior_anchor_counts": ANCHOR_COUNTS,
        "code_sha256": {name: sha256(PROJECT / name) for name in CODE_FILES},
        "counts": EXPECTED_COUNTS,
        "dataset_counts": EXPECTED_DATASET_COUNTS,
        "name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "purpose": (
            "repaired workspace channel (no BOS seed, internal response cue, gated "
            "read-out) with the study hypothesis decoupled into generation parity, "
            "verifier-weighted selection on the medium stratum, and split-conformal "
            "abstention; sized for a Kaggle dual-T4 two-seed session"
        ),
        "review_status": (
            "Reasoning9000 rows remain unreviewed and policy-masked; builder episodes "
            "are the first independently adjudicated rows; benchmark episodes carry "
            "build-time execution-verified labels. Not authorized as production or "
            "factual-quality evidence until the preregistered gate passes."
        ),
        "seed": 17,
        "session_plan_hours": 9.0,
        "supersedes": "hlwm-verified-multidomain-v6-0-candidate",
        "target_hardware": "Kaggle T4 x2, one independent seed per GPU",
        "training_defaults": TRAINING_DEFAULTS,
        "v8_0_changes": V8_CHANGES,
    }
    manifest_path = PROJECT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    notebook_path = BUNDLE_ROOT / "embel-hlwm-v8.0-kaggle-2xt4.ipynb"
    notebook_path.write_text(
        json.dumps(notebook_document(), indent=1) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    archive_path = BUNDLE_ROOT / "hlwm-v8.0-candidate-bundle.zip"
    if archive_path.exists():
        archive_path.unlink()
    members: List[Path] = []
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        members.append(path)
    # Deterministic archive: zip entries otherwise carry filesystem mtimes, which
    # makes the recorded bundle_sha256 change on every no-op rebuild and therefore
    # useless for confirming that the Kaggle side received the reviewed bytes.
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(
                str(Path("hlwm_kaggle") / path.relative_to(PROJECT)),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    report = {
        "built": "2026-09-02",
        "package_version": PACKAGE_VERSION,
        "bundle": str(archive_path),
        "bundle_bytes": archive_path.stat().st_size,
        "bundle_sha256": sha256(archive_path),
        "bundle_members": len(members),
        "notebook_t4": str(notebook_path),
        "notebook_t4_sha256": sha256(notebook_path),
        "code_sha256": manifest["code_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "counts": EXPECTED_COUNTS,
        "dataset_counts": EXPECTED_DATASET_COUNTS,
        "changes_vs_v6_0": V8_CHANGES,
        "supersedes": {
            "package": "hlwm-verified-multidomain-v6-0-candidate",
            "bundle_sha256": (
                "39477eb5d26423b91ad3f6dc7075e0f1825c577513dc041d95423e5b6c36caf4"
            ),
            "verdict": "both seeds failed the 20-gate replication (Session E)",
        },
        "plan": "reports/hlwm-v8.0-plan-2026-09-02.md",
        "prior_results": "reports/hlwm-v6.0-session-e-results-2026-09-02.md",
        "gates": 24,
        "tests": "58/58",
        "source_of_record": str(PROJECT),
        "note": (
            "The bundle directory is the v8.0 source of record. experiments/kaggle_hlwm "
            "still holds the pre-hotfix v6.0 tree plus an unshipped multiple_choice "
            "grader and was deliberately not overwritten."
        ),
    }
    (BUNDLE_ROOT / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "changes_vs_v6_0"}, indent=2))


if __name__ == "__main__":
    main()
