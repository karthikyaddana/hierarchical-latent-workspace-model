"""HLWM Version 10.0 dual-T4 Kaggle notebook (executable preregistration).

Cell inventory and the session lessons each one carries:
  0  title/ladder pointer (markdown)
  1  environment dump
  2  pinned pip installs
  3  bundle locate by PINNED code digest + sha verify + build_mode guard
     (G: verify before GPU commitment; the smoke bundle physically cannot
     launch; I-5: a stale tree in an attached kernel output can no longer
     win the selection, because selection is by content and fails closed)
  4  GPU inventory
  5  in-session test suite (blocking)
  6  data sanity spot checks under the real tokenizer
  7  BLOCKING on-device preflight + gamma calibration + throughput ledger
     (B/D/G: no training until the CUDA-class checks pass on this device)
  8  dual-seed launch with dual-pattern resume glob (G/H: raw
     checkpoint-*.pt fallback) and heartbeat monitor
  9  per-seed audit as one subprocess per GPU (I-4/I-5: a crash in one
     seed's audit must not destroy the other's; also halves audit
     wall-clock against the 12h session cap)
  10 pooled verdict + binding rung (F: every field access defensive; no
     display-only crash may ever eat a finished run again)
  11 packaging + resumable renames
  12 binding ladder (markdown)
"""

from __future__ import annotations

from typing import Any, Dict, List


def _markdown(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def notebook_document(
    expected_counts: Dict[str, int], code_digest: str = ""
) -> Dict[str, Any]:
    """Build the notebook document.

    ``code_digest`` is the sha256 of the just-written manifest's
    ``code_sha256`` map: the notebook pins it so cell 3 can select the code
    tree it was built with by CONTENT, and refuse to run any other.
    """

    cells: List[Dict[str, Any]] = []

    cells.append(_markdown(
        "# HLWM Version 10.0 — Dense Supervision (dual T4)\n\n"
        "Executable preregistration for the v10 run: CODI-style latent "
        "self-distillation with structural necessity, family-routed experts, "
        "and the e-process abstention claim. Plan of record: "
        "`reports/hlwm-v10-plan-2026-09-03.md` including the binding §10 "
        "red-team amendments. Where prose and this notebook differ, THIS "
        "NOTEBOOK GOVERNS (the Study 9 lesson).\n\n"
        "Run order is strict: every cell is a gate for the next. The "
        "preflight cell aborts the session BEFORE any GPU-hour commitment."
    ))

    cells.append(_code(
        "import platform, subprocess, sys, time\n"
        "SESSION_START = time.time()\n"
        "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n"
        "print(platform.platform())\n"
        "print('python', sys.version.split()[0])"
    ))

    cells.append(_code(
        "import subprocess, sys\n"
        "for spec in ('transformers==4.56.2', 'accelerate==1.10.1', 'safetensors==0.6.2', 'pytest==8.4.1'):\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '-q', spec], check=True)\n"
        "    print('installed:', spec)\n"
        "import site; site.main()\n"
        "import transformers\n"
        "print('transformers', transformers.__version__)"
    ))

    cells.append(_code(
        "from pathlib import Path\n"
        "import hashlib, json, shutil, zipfile\n"
        "# The code tree to run is PINNED to the digest of the bundle this\n"
        "# notebook was built with. A resume attaches a previous kernel output,\n"
        "# and that output contains a COMPLETE OLD COPY of this tree whose\n"
        "# manifest name, version and counts are identical -- picking\n"
        "# candidates[0] was a coin flip that could silently run stale code for\n"
        "# a full session. Selection is now by content, and unmatched means\n"
        "# stop, never guess.\n"
        f"EXPECTED_CODE_DIGEST = {code_digest!r}\n"
        "WANTED = 'hlwm-dense-supervision-v10-0-candidate'\n"
        "KAGGLE = Path('/kaggle/input').exists()\n"
        "WORK = Path('/kaggle/working/hlwm-v10.0') if KAGGLE else Path.cwd() / 'hlwm-v10.0-work'\n"
        "if WORK.exists(): shutil.rmtree(WORK)\n"
        "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
        "def code_digest(mapping):\n"
        "    return hashlib.sha256(json.dumps(mapping or {}, sort_keys=True).encode()).hexdigest()\n"
        "candidates = []\n"
        "for root in search_roots:\n"
        "    for path in sorted(root.rglob('*candidate-bundle*.zip')):\n"
        "        if 'smoke' in path.name: continue\n"
        "        try:\n"
        "            with zipfile.ZipFile(path) as bundle:\n"
        "                found = json.loads(bundle.read('hlwm_kaggle/manifest.json'))\n"
        "        except Exception: continue\n"
        "        if found.get('name') == WANTED:\n"
        "            candidates.append(('zip', path, code_digest(found.get('code_sha256'))))\n"
        "    for path in sorted(root.rglob('manifest.json')):\n"
        "        try: found = json.loads(path.read_text())\n"
        "        except Exception: continue\n"
        "        if found.get('name') == WANTED:\n"
        "            candidates.append(('tree', path.parent, code_digest(found.get('code_sha256'))))\n"
        "print('expected code digest:', EXPECTED_CODE_DIGEST[:16])\n"
        "print('v10.0 code trees attached:', len(candidates))\n"
        "for kind, path, digest in candidates:\n"
        "    print(' ', digest[:16], 'MATCH ' if digest == EXPECTED_CODE_DIGEST else 'stale ', kind, path)\n"
        "matching = [item for item in candidates if item[2] == EXPECTED_CODE_DIGEST]\n"
        "if not matching:\n"
        "    for root in search_roots:\n"
        "        print('ATTACHED under', root, ':')\n"
        "        for path in sorted(root.glob('*'))[:24]: print('  ', path)\n"
        "    raise FileNotFoundError(\n"
        "        'No attached bundle matches this notebook. Upload the zip built alongside '\n"
        "        'it (code digest %s) -- a stale tree from a previous kernel output must '\n"
        "        'never be run.' % EXPECTED_CODE_DIGEST[:16])\n"
        "kind, source, digest = matching[0]\n"
        "if kind == 'zip':\n"
        "    with zipfile.ZipFile(source) as bundle: bundle.extractall(WORK)\n"
        "else:\n"
        "    shutil.copytree(source, WORK / 'hlwm_kaggle')\n"
        "PROJECT = WORK / 'hlwm_kaggle'\n"
        "print('bundle source (%s):' % kind, source)\n"
        "DATA = PROJECT / 'data'\n"
        "manifest = json.loads((PROJECT / 'manifest.json').read_text())\n"
        "assert manifest['package_version'] == '10.0', manifest['package_version']\n"
        "assert manifest['name'] == 'hlwm-dense-supervision-v10-0-candidate', manifest['name']\n"
        "assert manifest.get('build_mode', 'full') == 'full', (\n"
        "    'SMOKE bundle attached: rerun scripts/build_hlwm_v100_bundle.py without --smoke '\n"
        "    'and re-upload; a smoke build must never spend GPU hours')\n"
        f"assert manifest['counts'] == {expected_counts!r}, manifest['counts']\n"
        "def file_sha256(path):\n"
        "    digest = hashlib.sha256()\n"
        "    with open(path, 'rb') as stream:\n"
        "        for chunk in iter(lambda: stream.read(1 << 20), b''): digest.update(chunk)\n"
        "    return digest.hexdigest()\n"
        "# Verify shipped code BEFORE committing GPU hours (v6.0 shipped stale shas).\n"
        "for name, expected in manifest['code_sha256'].items():\n"
        "    actual = file_sha256(PROJECT / name)\n"
        "    assert actual == expected, f'code sha mismatch for {name}'\n"
        "print('bundle verified: all', len(manifest['code_sha256']), 'code shas match at', PROJECT)"
    ))

    cells.append(_code(
        "import torch\n"
        "assert torch.cuda.is_available(), 'no CUDA device'\n"
        "for index in range(torch.cuda.device_count()):\n"
        "    properties = torch.cuda.get_device_properties(index)\n"
        "    print(index, properties.name, round(properties.total_memory / 2**30, 2), 'GB')\n"
        "assert torch.cuda.device_count() >= 2, 'this notebook schedules one seed per GPU'"
    ))

    cells.append(_code(
        "import subprocess, sys\n"
        "suite = ['test_modeling_hlwm.py', 'test_selective_stats.py', 'test_evaluate_v10.py', 'test_data_v10.py']\n"
        "result = subprocess.run([sys.executable, '-m', 'pytest', *suite, '-q'], cwd=PROJECT,\n"
        "                        capture_output=True, text=True)\n"
        "print(result.stdout[-4000:])\n"
        "if result.returncode != 0:\n"
        "    print(result.stderr[-4000:])\n"
        "    raise RuntimeError('Version 10.0 in-session test suite failed')\n"
        "print('Version 10.0 channel, stats, audit and data tests passed')"
    ))

    cells.append(_code(
        "import json, sys\n"
        "sys.path.insert(0, str(PROJECT))\n"
        "from transformers import AutoTokenizer\n"
        "MODEL = 'Qwen/Qwen3-0.6B-Base'\n"
        "REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)\n"
        "if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token\n"
        "print('bos_token', tokenizer.bos_token, '| bos_token_id', tokenizer.bos_token_id)\n"
        "from data import RESPONSE_CUE_TEXT, encode_trace_segments\n"
        "cue_ids = tokenizer.encode(RESPONSE_CUE_TEXT, add_special_tokens=False)\n"
        "print('response cue', repr(RESPONSE_CUE_TEXT), '->', cue_ids)\n"
        "assert tokenizer.eos_token_id not in cue_ids\n"
        "checked = masked_ok = 0\n"
        "with open(DATA / 'master' / 'test.jsonl', 'r', encoding='utf-8') as stream:\n"
        "    for line in stream:\n"
        "        row = json.loads(line)\n"
        "        payload = row.get('v10')\n"
        "        if not payload: continue\n"
        "        ids, _ = encode_trace_segments(tokenizer, payload['steps'], payload.get('teacher_excluded_steps', 1))\n"
        "        assert len(ids) >= 6, row['episode_id']\n"
        "        assert tokenizer.eos_token_id not in ids, row['episode_id']\n"
        "        masked_request = row.get('input', {}).get('user_request_masked')\n"
        "        if masked_request:\n"
        "            for literal in row.get('evaluation', {}).get('withheld_literals', []):\n"
        "                assert literal not in masked_request, row['episode_id']\n"
        "            masked_ok += 1\n"
        "        checked += 1\n"
        "        if checked >= 200: break\n"
        "print('trace/window/leak spot check passed on', checked, 'rows (', masked_ok, 'masked )')"
    ))

    cells.append(_code(
        "# BLOCKING PREFLIGHT: the CUDA-class checks (autocast, the two-SDPA\n"
        "# gate path, memory peak, decode parity) run on the REAL model on\n"
        "# THIS device before any training commitment, and calibrate gamma.\n"
        "import json, os, subprocess, sys\n"
        "PREFLIGHT_DIR = Path('/kaggle/working/hlwm-v10.0-preflight')\n"
        "command = [sys.executable, str(PROJECT / 'train_kaggle.py'),\n"
        "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_DIR), '--seed', '17',\n"
        "    '--v10-preflight-only', '--skip-architecture-smoke', '--skip-real-qwen-preflight',\n"
        "    '--latent-thoughts', '6', '--kv-prefix-slots', '16', '--kv-prefix-rank', '64',\n"
        "    '--prefix-attn-gate-init', '0.08', '--mlp-expert-count', '2', '--mlp-expert-rank', '16',\n"
        "    '--steps', '2000', '--warm-steps', '600', '--wo-l1-branch-steps', '300',\n"
        "    '--batch-size', '1', '--gradient-accumulation', '16',\n"
        "    '--learning-rate', '0.0008', '--weight-decay', '0.1', '--warmup-updates', '4',\n"
        "    '--lora-rank', '64', '--lora-alpha', '128', '--lora-tail-layers', '28',\n"
        "    '--unfreeze-tail-layers', '0', '--num-lanes', '1', '--num-experts', '1',\n"
        "    '--refinement-steps', '2', '--diffusion-steps', '4',\n"
        "    '--context-tokens', '256', '--canvas-tokens', '64', '--brief-tokens', '32',\n"
        "    '--causal-tokens', '512', '--trace-tokens', '96', '--precision', 'auto',\n"
        "    '--max-validation', '256', '--num-workers', '0']\n"
        "environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = '0'\n"
        "result = subprocess.run(command, cwd=PROJECT, env=environment, capture_output=True, text=True)\n"
        "print(result.stdout[-6000:])\n"
        "if result.returncode != 0:\n"
        "    print(result.stderr[-6000:])\n"
        "    raise RuntimeError('BLOCKING preflight failed (exit %d): do not train' % result.returncode)\n"
        "preflight = json.loads((PREFLIGHT_DIR / 'preflight.json').read_text())\n"
        "assert preflight['passed'], preflight['failed_checks']\n"
        "CALIBRATED_GAMMA = preflight['calibrated_gamma']\n"
        "SECONDS_PER_STEP = preflight['checks']['seconds_per_step']\n"
        "print('calibrated gamma', CALIBRATED_GAMMA)\n"
        "print('measured s/step', round(SECONDS_PER_STEP, 2),\n"
        "      '| projected training hours', round(preflight['checks']['projected_training_hours'], 2))\n"
        "print('peak preflight memory GB', round(preflight['checks'].get('peak_memory_gb', 0.0), 2))\n"
        "print('teacher ceiling by family', preflight['checks']['teacher_ceiling_by_family'])"
    ))

    cells.append(_code(
        "import os, re, subprocess, sys, time, shutil\n"
        "SEEDS = [int(part) for part in os.environ.get('HLWM_SEEDS', '17,29').split(',')]\n"
        "assert len(SEEDS) == 2, 'this notebook schedules exactly two seeds, one per T4'\n"
        "OUTPUTS = {seed: Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}') for seed in SEEDS}\n"
        "processes, handles = {}, {}\n"
        "for gpu, seed in enumerate(SEEDS):\n"
        "    output = OUTPUTS[seed]\n"
        "    if output.exists(): shutil.rmtree(output)\n"
        "    resume = [p for root in search_roots + [WORK] for p in root.rglob(f'hlwm-v10.0-seed-{seed}-resumable.pt')]\n"
        "    if not resume:\n"
        "        # A run that crashed before packaging leaves raw checkpoint\n"
        "        # files; resume from the highest step (the Session G lesson).\n"
        "        raw = [p for root in search_roots for p in root.rglob(f'hlwm-v10.0-seed-{seed}/checkpoint-*.pt')]\n"
        "        resume = sorted(raw, key=lambda p: int((re.findall(r'(\\d+)', p.stem) or ['0'])[-1]))[-1:]\n"
        "    command = [sys.executable, str(PROJECT / 'train_kaggle.py'),\n"
        "        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),\n"
        "        '--skip-architecture-smoke', '--skip-real-qwen-preflight',\n"
        "        '--latent-thoughts', '6', '--kv-prefix-slots', '16', '--kv-prefix-rank', '64',\n"
        "        '--prefix-attn-gate-init', '0.08', '--mlp-expert-count', '2', '--mlp-expert-rank', '16',\n"
        "        '--steps', '2000', '--warm-steps', '600', '--wo-l1-branch-steps', '300',\n"
        "        '--batch-size', '1', '--gradient-accumulation', '16',\n"
        "        '--learning-rate', '0.0008', '--weight-decay', '0.1', '--warmup-updates', '4',\n"
        "        '--lora-rank', '64', '--lora-alpha', '128', '--lora-tail-layers', '28',\n"
        "        '--unfreeze-tail-layers', '0', '--num-lanes', '1', '--num-experts', '1',\n"
        "        '--refinement-steps', '2', '--diffusion-steps', '4',\n"
        "        '--context-tokens', '256', '--canvas-tokens', '64', '--brief-tokens', '32',\n"
        "        '--causal-tokens', '512', '--trace-tokens', '96', '--precision', 'auto',\n"
        "        '--max-validation', '256', '--num-workers', '2',\n"
        "        '--distill-gamma', str(CALIBRATED_GAMMA), '--telemetry-every', '100',\n"
        "        '--masked-fraction-floor', '0.5', '--warm-em-floor', '0.50',\n"
        "        '--gonogo-step', '1200', '--gonogo-masked-numeric-em', '0.05',\n"
        "        '--save-every', '500', '--max-runtime-hours', '5.5']\n"
        "    if resume:\n"
        "        command += ['--resume', str(resume[0])]\n"
        "        print('seed', seed, 'resuming from', resume[0])\n"
        "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
        "    handle = Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-training.log').open('w'); handles[seed] = handle\n"
        "    processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment,\n"
        "                                       stdout=handle, stderr=subprocess.STDOUT)\n"
        "    print('launched seed', seed, 'on physical GPU', gpu)\n"
        "while any(process.poll() is None for process in processes.values()):\n"
        "    time.sleep(30)\n"
        "    print({seed: {'returncode': process.poll(),\n"
        "                  'elapsed_min': round((time.time() - SESSION_START) / 60, 1),\n"
        "                  'metrics_bytes': (OUTPUTS[seed] / 'metrics.jsonl').stat().st_size\n"
        "                  if (OUTPUTS[seed] / 'metrics.jsonl').exists() else 0}\n"
        "           for seed, process in processes.items()})\n"
        "for handle in handles.values(): handle.close()\n"
        "failures = {seed: process.returncode for seed, process in processes.items() if process.returncode != 0}\n"
        "if failures:\n"
        "    for seed in failures:\n"
        "        print(Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-training.log').read_text()[-8000:])\n"
        "    raise RuntimeError(f'dual training failed: {failures}')\n"
        "print('Both seeds finished')\n"
        "verdicts = {}\n"
        "for seed in SEEDS:\n"
        "    for line in (OUTPUTS[seed] / 'metrics.jsonl').read_text().splitlines():\n"
        "        try: record = json.loads(line)\n"
        "        except json.JSONDecodeError: continue\n"
        "        if 'v10_verdict' in record: verdicts[seed] = record['v10_verdict']\n"
        "    print(seed, json.dumps(verdicts.get(seed, {}), sort_keys=True)[:800])\n"
        "ABORTED = {seed: verdicts.get(seed, {}).get('aborted') for seed in SEEDS}\n"
        "print('aborted:', ABORTED)"
    ))

    cells.append(_code(
        "# Per-seed audit, ONE SEED PER GPU in its own process. Two reasons\n"
        "# this is not the in-process loop it used to be: the sequential audit\n"
        "# left GPU1 idle and put the session against the 12h cap, and in both\n"
        "# sessions I-4 and I-5 an exception while auditing the first seed\n"
        "# destroyed the second seed's audit too. A crash now costs one seed.\n"
        "import json, os, subprocess, sys, time\n"
        "audit_processes, audit_handles = {}, {}\n"
        "for gpu, seed in enumerate(SEEDS):\n"
        "    verdict = verdicts.get(seed, {})\n"
        "    if verdict.get('aborted'):\n"
        "        print(seed, 'training aborted at', verdict['aborted'], '- audit runs on the abort checkpoint for the failure branch')\n"
        "    command = [sys.executable, str(PROJECT / 'evaluate_v10.py'),\n"
        "               '--output-dir', str(OUTPUTS[seed]), '--data-dir', str(DATA), '--seed', str(seed)]\n"
        "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
        "    handle = Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-audit.log').open('w')\n"
        "    audit_handles[seed] = handle\n"
        "    audit_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment,\n"
        "                                             stdout=handle, stderr=subprocess.STDOUT)\n"
        "    print('audit launched: seed', seed, 'on physical GPU', gpu)\n"
        "while any(process.poll() is None for process in audit_processes.values()):\n"
        "    time.sleep(60)\n"
        "    print({seed: {'returncode': process.poll(),\n"
        "                  'elapsed_min': round((time.time() - SESSION_START) / 60, 1),\n"
        "                  'gates_written': (OUTPUTS[seed] / 'v10-gates.json').exists()}\n"
        "           for seed, process in audit_processes.items()})\n"
        "for handle in audit_handles.values(): handle.close()\n"
        "gates_by_seed = {}\n"
        "for seed in SEEDS:\n"
        "    code = audit_processes[seed].returncode\n"
        "    gates_path = OUTPUTS[seed] / 'v10-gates.json'\n"
        "    if code != 0 or not gates_path.exists():\n"
        "        print('SEED', seed, 'AUDIT FAILED (exit', code, ') - log tail:')\n"
        "        print(Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-audit.log').read_text()[-6000:])\n"
        "        continue\n"
        "    gates_by_seed[seed] = json.loads(gates_path.read_text())\n"
        "    print(seed, 'gates:', {name: value for name, value in gates_by_seed[seed].items()\n"
        "                           if isinstance(value, bool)})\n"
        "if not gates_by_seed:\n"
        "    raise RuntimeError('both seed audits failed; see the log tails above')"
    ))

    cells.append(_code(
        "import json\n"
        "import evaluate_v10 as ev10\n"
        "rung = ev10.binding_rung(gates_by_seed)\n"
        "verdict_document = {\n"
        "    'package_version': '10.0',\n"
        "    'binding_rung': rung,\n"
        "    'seeds': {str(seed): gates_by_seed.get(seed, {}) for seed in SEEDS},\n"
        "    'training_verdicts': {str(seed): verdicts.get(seed, {}) for seed in SEEDS},\n"
        "}\n"
        "Path('/kaggle/working/hlwm-v10.0-replication-verdict.json').write_text(\n"
        "    json.dumps(verdict_document, indent=1, sort_keys=True, default=str))\n"
        "print('BINDING RUNG:', rung)\n"
        "for seed in SEEDS:\n"
        "    gates = gates_by_seed.get(seed, {})\n"
        "    failed = sorted(name for name, value in gates.items()\n"
        "                    if isinstance(value, bool) and not value and name != 'passed')\n"
        "    print(seed, 'passed' if gates.get('passed') else 'FAILED', '| failed gates:', failed)"
    ))

    cells.append(_code(
        "import re, shutil\n"
        "ARTIFACTS = [Path('/kaggle/working/hlwm-v10.0-replication-verdict.json'),\n"
        "             PREFLIGHT_DIR / 'preflight.json']\n"
        "for seed in SEEDS:\n"
        "    output = OUTPUTS[seed]\n"
        "    staging = Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-deliverable')\n"
        "    if staging.exists(): shutil.rmtree(staging)\n"
        "    staging.mkdir()\n"
        "    for source in [output / 'v10-audit.json', output / 'v10-gates.json',\n"
        "                   output / 'metrics.jsonl', PROJECT / 'manifest.json', PROJECT / 'README.md']:\n"
        "        if source.exists(): shutil.copy2(source, staging / source.name)\n"
        "    archive = Path(shutil.make_archive(f'/kaggle/working/hlwm-v10.0-seed-{seed}-deliverable', 'zip', staging))\n"
        "    resumable = Path(f'/kaggle/working/hlwm-v10.0-seed-{seed}-resumable.pt')\n"
        "    if resumable.exists(): resumable.unlink()\n"
        "    source_checkpoint = output / 'checkpoint-v10-full.pt'\n"
        "    if not source_checkpoint.exists():\n"
        "        # Aborted at a tripwire: there is no full checkpoint, and the\n"
        "        # old unconditional sweep deleted every checkpoint-step file,\n"
        "        # so the salvage deliverable shipped ZERO weights and the run\n"
        "        # could not be resumed or re-audited. Promote the newest step.\n"
        "        steps = sorted(output.glob('checkpoint-step-*.pt'),\n"
        "                       key=lambda p: int((re.findall(r'(\\d+)', p.stem) or ['0'])[-1]))\n"
        "        source_checkpoint = steps[-1] if steps else None\n"
        "    if source_checkpoint is not None and source_checkpoint.exists():\n"
        "        shutil.copy2(source_checkpoint, resumable)\n"
        "        print('seed', seed, 'resumable from', source_checkpoint.name)\n"
        "    keep = {source_checkpoint.name} if source_checkpoint is not None else set()\n"
        "    for stale in output.glob('checkpoint-step-*.pt*'):\n"
        "        if stale.is_file() and stale.name not in keep: stale.unlink()\n"
        "    ARTIFACTS += [archive, resumable, output / 'checkpoint-v10-wo-l1-branch.pt']\n"
        "print('deliverables:')\n"
        "for path in ARTIFACTS:\n"
        "    if Path(path).exists(): print(' ', path)"
    ))

    cells.append(_markdown(
        "## Binding ladder (preregistered; read the rung off the verdict, do not improvise)\n\n"
        "0. **Infrastructure failure** (crash, sha mismatch, preflight abort): "
        "fix and rerun authorized; consumes no scientific chance.\n"
        "1. **Warm gate fails on both seeds**: the dense-supervised channel "
        "cannot learn reconstruction at this exposure; terminal for this "
        "design class at 0.6B; the abort branch (fully-powered abstention "
        "audit + plain-LoRA control) is the session deliverable.\n"
        "2. **Warm passes, masked transfer null on both seeds** "
        "(latent_channel_live paired-LCB fails): the channel claim dies; the "
        "w/o-L1, pause, and probe read-outs localize why.\n"
        "3. **One seed passes**: no replication claim; single-seed report; "
        "one replication session authorized.\n"
        "4. **Both seeds pass**: Claim B established at margin tier; the "
        "matched plain-LoRA control session is REQUIRED before any external "
        "claim.\n\n"
        "The abstention claim resolves independently via the anytime-valid "
        "e-process; the in-run DeLong look is futility-only. There is no "
        "rung on which success may be declared by narrative."
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
