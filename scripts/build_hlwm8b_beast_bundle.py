from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "kaggle_hlwm8b"
OUTPUT = ROOT / "artifacts" / "hlwm8b-beast"
PINNED_MODEL = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
PINNED_REVISION = "54641c1611fcff44fa4865626462445e0a153fc7"
FILES = (
    "checkpointing_hlwm8b.py",
    "data_hlwm8b.py",
    "energy_hlwm8b.py",
    "evaluate_hlwm8b.py",
    "inference_hlwm8b.py",
    "modeling_hlwm8b.py",
    "package_hlwm8b.py",
    "semantic_hlwm8b.py",
    "test_hlwm8b.py",
    "train_hlwm8b.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def notebook_document() -> Dict[str, Any]:
    cells = [
        markdown(
            """# EMBEL HLWM 8B — resumable dual-T4 training

Use **Save Version → Save & Run All** with **GPU T4 x2** and Internet enabled.
Attach the `hlwm8b-kaggle-code` and `hlwm-beast-teacher-snapshot` datasets. The
run stops early enough to checkpoint and finish cleanly. On the next version it
resumes from a checkpoint attached as a dataset, or automatically from a private
Hugging Face repo when Kaggle secrets `HF_TOKEN` and `HF_REPO` are configured.

Main training stays locked until the snapshot contains the configured minimum of
licence-audited direct pairs plus locally gated one-teacher records. A small snapshot
can still run the real-Nemotron preflight.
"""
        ),
        code(
            """import os, platform, subprocess, sys, torch
print(platform.platform())
subprocess.run(['nvidia-smi'], check=True)
assert torch.cuda.device_count() == 2, f'Choose GPU T4 x2; found {torch.cuda.device_count()} GPU(s)'
print('python', sys.version)
"""
        ),
        code(
            """!python -m pip install -q 'transformers==4.56.2' 'accelerate==1.10.1' 'peft==0.17.1' 'bitsandbytes>=0.46,<0.49' 'safetensors==0.6.2' 'sentencepiece==0.2.1' 'pytest==8.4.1' 'huggingface_hub>=0.34,<1'
"""
        ),
        code(
            """from pathlib import Path
import json, shutil, zipfile

WORK = Path('/kaggle/working/hlwm8b-beast')
WORK.mkdir(parents=True, exist_ok=True)
code_archives = list(Path('/kaggle/input').rglob('hlwm8b-kaggle-code.zip'))
if len(code_archives) != 1:
    raise FileNotFoundError('Attach exactly one hlwm8b-kaggle-code dataset.')
CODE_ROOT = WORK / 'code'
if CODE_ROOT.exists(): shutil.rmtree(CODE_ROOT)
with zipfile.ZipFile(code_archives[0]) as archive: archive.extractall(CODE_ROOT)
PROJECT = CODE_ROOT / 'hlwm8b_kaggle'

data_archives = list(Path('/kaggle/input').rglob('hlwm-beast-teacher-snapshot.zip'))
if len(data_archives) != 1:
    raise FileNotFoundError('Attach exactly one hlwm-beast-teacher-snapshot dataset.')
DATA_ROOT = WORK / 'data-snapshot'
if DATA_ROOT.exists(): shutil.rmtree(DATA_ROOT)
with zipfile.ZipFile(data_archives[0]) as archive: archive.extractall(DATA_ROOT)
manifest_paths = list(DATA_ROOT.rglob('manifest.json'))
if len(manifest_paths) != 1: raise RuntimeError('Teacher snapshot manifest is missing or ambiguous.')
DATA = manifest_paths[0].parent
MANIFEST = json.loads(manifest_paths[0].read_text())
assert MANIFEST['name'] == 'hlwm-beast-teacher-snapshot'
print(json.dumps({'project': str(PROJECT), 'data': str(DATA), 'manifest': MANIFEST}, indent=2))
"""
        ),
        code(
            """# Optional zero-touch persistence through a private Hugging Face model repository.
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for name in ('HF_TOKEN', 'HF_REPO'):
        try:
            value = secrets.get_secret(name)
            if value: os.environ[name] = value
        except Exception:
            pass
except Exception:
    pass
print('automatic remote resume:', bool(os.getenv('HF_TOKEN') and os.getenv('HF_REPO')))
"""
        ),
        code(
            """# Reconstruct a numbered resume archive when it was uploaded as Kaggle data.
OUTPUT = Path('/kaggle/working/hlwm8b-output')
OUTPUT.mkdir(parents=True, exist_ok=True)
parts_manifests = list(Path('/kaggle/input').rglob('hlwm8b-checkpoint-*.zip.parts.json'))
for parts_manifest in parts_manifests:
    spec = json.loads(parts_manifest.read_text())
    archive_path = OUTPUT / spec['archive']
    with archive_path.open('wb') as destination:
        for part in spec['parts']:
            matches = list(parts_manifest.parent.rglob(part['name']))
            if len(matches) != 1: raise FileNotFoundError(part['name'])
            with matches[0].open('rb') as source: shutil.copyfileobj(source, destination)
    import hashlib
    digest_object = hashlib.sha256()
    with archive_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest_object.update(chunk)
    digest = digest_object.hexdigest()
    if digest != spec['archive_sha256']: raise ValueError('Reconstructed resume archive checksum mismatch')
    checkpoint_dir = OUTPUT / Path(spec['archive']).stem
    if checkpoint_dir.exists(): shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive: archive.extractall(checkpoint_dir)
    archive_path.unlink()
    print('reconstructed', checkpoint_dir)
for archive_path in Path('/kaggle/input').rglob('hlwm8b-checkpoint-*.zip'):
    checkpoint_dir = OUTPUT / archive_path.stem
    if checkpoint_dir.exists(): continue
    checkpoint_dir.mkdir()
    with zipfile.ZipFile(archive_path) as archive: archive.extractall(checkpoint_dir)
    print('imported full checkpoint archive', checkpoint_dir)
"""
        ),
        code(
            """import py_compile, subprocess
for path in PROJECT.glob('*.py'):
    py_compile.compile(str(path), doraise=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_hlwm8b.py')], cwd=PROJECT, check=True)
print('Static checks and HLWM unit tests passed')
"""
        ),
        code(
            f"""from huggingface_hub import snapshot_download
snapshot_download(repo_id='{PINNED_MODEL}', revision='{PINNED_REVISION}')
print('Pinned Nemotron Nano 8B snapshot cached')
"""
        ),
        markdown(
            """## Exact real-Nemotron preflight

This performs a real two-process 4-bit Nemotron 8B forward/backward pass before training.
It is allowed on a small snapshot, but it never bypasses the main data-readiness gate.
"""
        ),
        code(
            """PREFLIGHT = Path('/kaggle/working/hlwm8b-preflight')
if PREFLIGHT.exists(): shutil.rmtree(PREFLIGHT)
preflight = ['accelerate', 'launch', '--multi_gpu', '--num_processes', '2', '--mixed_precision', 'fp16', str(PROJECT/'train_hlwm8b.py'),
    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT), '--preflight-only', '--allow-small-data',
    '--resume', 'off', '--batch-size', '1', '--gradient-accumulation', '1', '--num-workers', '0',
    '--joint-updates', '1', '--on-policy-updates', '0', '--max-prompt-tokens', '512', '--max-answer-tokens', '512']
subprocess.run(preflight, cwd=PROJECT, check=True)
print(json.dumps(json.loads((PREFLIGHT/'preflight.json').read_text()), indent=2))
"""
        ),
        markdown(
            """## Pausable two-T4 conversion

Both T4s train one distributed 8B model. The run saves every 50 updates, reacts to
`/kaggle/working/HLWM_STOP`, and stops 20 minutes before its 8.5-hour budget. Re-run
this notebook with the checkpoint output attached, or configure HF secrets for
automatic remote resume.
"""
        ),
        code(
            """if not MANIFEST.get('ready_for_main_training'):
    raise RuntimeError(f"Data snapshot has {MANIFEST.get('verified_source_records', 0)} verified source records; main training requires {MANIFEST.get('minimum_verified_source_records', 1500)}. Export a newer snapshot and replace the Kaggle dataset.")

command = ['accelerate', 'launch', '--multi_gpu', '--num_processes', '2', '--mixed_precision', 'fp16',
    str(PROJECT/'train_hlwm8b.py'), '--data-dir', str(DATA), '--output-dir', str(OUTPUT),
    '--input-root', '/kaggle/input', '--resume', 'auto', '--joint-updates', '800',
    '--on-policy-updates', '100', '--batch-size', '1', '--gradient-accumulation', '8',
    '--learning-rate', '0.00008', '--sidecar-learning-rate', '0.00016',
    '--save-every-updates', '50', '--eval-every-updates', '100', '--keep-checkpoints', '1',
    '--max-runtime-hours', '8.5', '--runtime-save-buffer-minutes', '20',
    '--max-prompt-tokens', '512', '--max-answer-tokens', '512', '--num-workers', '2',
    '--num-lanes', '3', '--num-experts', '6', '--prefix-tokens', '16']
if os.getenv('HF_REPO'): command.extend(['--hf-repo', os.environ['HF_REPO']])
subprocess.run(command, cwd=PROJECT, check=True)
RUN_STATUS = json.loads((OUTPUT/'run-status.json').read_text())
TRAINING_COMPLETE = RUN_STATUS['status'] == 'training_complete'
print(json.dumps({'status': RUN_STATUS['status'], 'global_update': RUN_STATUS['global_update'], 'pause_reason': RUN_STATUS.get('pause_reason')}, indent=2))
"""
        ),
        code(
            """sys.path.insert(0, str(PROJECT)) if str(PROJECT) not in sys.path else None
from checkpointing_hlwm8b import find_resume_checkpoint
CHECKPOINT = find_resume_checkpoint(OUTPUT)
if CHECKPOINT is None: raise RuntimeError('No verified checkpoint was produced')
EVALUATION = OUTPUT / 'evaluation'
if TRAINING_COMPLETE:
    if EVALUATION.exists(): shutil.rmtree(EVALUATION)
    evaluation = [sys.executable, str(PROJECT/'evaluate_hlwm8b.py'), '--checkpoint', str(CHECKPOINT),
        '--data-dir', str(DATA), '--output-dir', str(EVALUATION), '--validation-samples', '32',
        '--test-samples', '32', '--max-new-tokens', '384', '--max-prompt-tokens', '512',
        '--max-answer-tokens', '512']
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = '0'
    subprocess.run(evaluation, cwd=PROJECT, env=environment, check=True)
else:
    print('Training paused safely. Evaluation will run automatically after a later resumed session completes.')
"""
        ),
        code(
            """DOWNLOADS = Path('/kaggle/working/hlwm8b-downloads')
if DOWNLOADS.exists(): shutil.rmtree(DOWNLOADS)
package = [sys.executable, str(PROJECT/'package_hlwm8b.py'), '--checkpoint', str(CHECKPOINT),
    '--output-dir', str(DOWNLOADS), '--code-dir', str(PROJECT), '--data-manifest', str(DATA/'manifest.json')]
if TRAINING_COMPLETE: package.extend(['--evaluation-dir', str(EVALUATION)])
subprocess.run(package, cwd=PROJECT, check=True)
download_spec = json.loads((DOWNLOADS/'downloads.json').read_text())
print(json.dumps(download_spec, indent=2))
for path in OUTPUT.glob('hlwm8b-checkpoint-*'):
    if path.is_dir(): shutil.rmtree(path)
print('Removed packaged checkpoint directories to preserve Kaggle disk space.')
"""
        ),
        code(
            """from IPython.display import FileLink, display
files = [DOWNLOADS/'downloads.json']
for manifest in DOWNLOADS.glob('*.parts.json'):
    files.append(manifest)
    spec = json.loads(manifest.read_text())
    files.extend(DOWNLOADS/part['name'] for part in spec['parts'])
for path in files:
    if path.exists(): display(FileLink(str(path)))
print('If status was paused, attach every checkpoint part plus its .parts.json to the next Kaggle run unless HF resume is configured.')
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True, "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    project = OUTPUT / "hlwm8b_kaggle"
    project.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(SOURCE / name, project / name)
    requirements = "\n".join(
        (
            "transformers==4.56.2",
            "accelerate==1.10.1",
            "peft==0.17.1",
            "bitsandbytes>=0.46,<0.49",
            "safetensors==0.6.2",
            "sentencepiece==0.2.1",
            "pytest==8.4.1",
            "huggingface_hub>=0.34,<1",
            "",
        )
    )
    (project / "requirements.txt").write_text(requirements, encoding="utf-8")
    readme = """# HLWM 8B Kaggle code

Attach this code bundle and a frozen `hlwm-beast-teacher-snapshot.zip` to the generated
notebook. Use T4 x2 and Save & Run All. The notebook preflights the pinned Nemotron model,
resumes exact optimizer/RNG state, stops before Kaggle's session boundary, evaluates
against untouched anchors, and emits split checkpoint downloads.
"""
    (project / "README.md").write_text(readme, encoding="utf-8")
    bundle_manifest = {
        "name": "hlwm8b-kaggle-code",
        "version": "1.0.0",
        "base_model": PINNED_MODEL,
        "base_revision": PINNED_REVISION,
        "files": {
            str(path.relative_to(project)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(project.rglob("*"))
            if path.is_file()
        },
    }
    (project / "bundle-manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8"
    )
    archive = OUTPUT / "hlwm8b-kaggle-code.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(project.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(OUTPUT))
    notebook = OUTPUT / "embel-hlwm8b-resumable-dual-t4.ipynb"
    notebook.write_text(json.dumps(notebook_document(), indent=1) + "\n", encoding="utf-8")
    data_archive = ROOT / "data" / "expert-beast" / "hlwm-beast-teacher-snapshot.zip"
    if data_archive.exists():
        shutil.copy2(data_archive, OUTPUT / data_archive.name)
    for artifact in [archive, notebook, *( [OUTPUT / data_archive.name] if data_archive.exists() else [])]:
        artifact.with_suffix(artifact.suffix + ".sha256").write_text(
            sha256(artifact) + "  " + artifact.name + "\n", encoding="utf-8"
        )
    result = {
        "code_bundle": str(archive),
        "code_bundle_sha256": sha256(archive),
        "notebook": str(notebook),
        "notebook_sha256": sha256(notebook),
        "data_snapshot": str(OUTPUT / data_archive.name) if data_archive.exists() else None,
        "data_snapshot_sha256": sha256(OUTPUT / data_archive.name) if data_archive.exists() else None,
    }
    snapshot_manifest_path = ROOT / "data" / "expert-beast" / "snapshot" / "manifest.json"
    snapshot_manifest = (
        json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        if snapshot_manifest_path.exists()
        else {}
    )
    upload_readme = f"""HLWM 8B KAGGLE UPLOADS

1. Create a Kaggle dataset named embel-hlwm8b-kaggle-code and upload:
   hlwm8b-kaggle-code.zip
2. Create a Kaggle dataset named embel-hlwm8b-teacher-snapshot and upload:
   hlwm-beast-teacher-snapshot.zip
3. Import embel-hlwm8b-resumable-dual-t4.ipynb as a new notebook.
4. Attach both datasets, choose GPU T4 x2, enable Internet, then Save Version -> Save & Run All.
5. Optional zero-touch resume: create a private Hugging Face model repo and add Kaggle secrets
   HF_TOKEN and HF_REPO. Without them, attach every numbered checkpoint part and its parts manifest
   to the next notebook version.

CURRENT SNAPSHOT
Verified source records: {snapshot_manifest.get('verified_source_records', 0)}
One-teacher records: {snapshot_manifest.get('accepted_teacher_records', 0)}
Direct locally verified records: {snapshot_manifest.get('accepted_direct_pair_records', 0)}
Required for this T4 stage: {snapshot_manifest.get('minimum_verified_source_records', 1500)}
Ready for main training: {snapshot_manifest.get('ready_for_main_training', False)}

The current snapshot is safe for the real-Nemotron preflight. Replace it with a later beast-export
snapshot before main training when Ready for main training is false.
"""
    (OUTPUT / "UPLOAD-README.txt").write_text(upload_readme, encoding="utf-8")
    (OUTPUT / "build.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
