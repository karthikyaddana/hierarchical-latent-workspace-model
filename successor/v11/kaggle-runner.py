"""HLWM v11 pilot: vocabulary-grounded thoughts against the necessity instrument.

Preregistered: reports/hlwm-v11-vocab-grounded-plan-2026-09-07.md (in the v11
code dataset and the public repo). Protocol, budget, data, seeds and audit are
the certified v10.0 run's own (kernel sirishayaddanapudi/hlwm-v10-0-certified-run,
mounted as input); the only deltas are --vocab-grounded-thoughts and
--incontext-decode-weight 0.5. The certified run is therefore the matched control.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SEEDS = [17, 29]

# ---- locate mounts -----------------------------------------------------------
v11_manifest = glob.glob("/kaggle/input/**/manifest-v11.json", recursive=True)
if not v11_manifest:
    raise SystemExit("v11 code dataset not mounted")
V11_SRC = Path(v11_manifest[0]).parent
cert = glob.glob("/kaggle/input/**/hlwm-v10.0-seed-17", recursive=True)
if not cert:
    raise SystemExit("certified-run kernel output not mounted")
CERT = Path(cert[0]).parent
DATA = CERT / "hlwm-v10.0" / "hlwm_kaggle" / "data"
assert (DATA / "master" / "test.jsonl").exists(), DATA

PROJECT = WORK / "hlwm_v11"
if PROJECT.exists():
    shutil.rmtree(PROJECT)
shutil.copytree(V11_SRC, PROJECT)
print("v11 code:", PROJECT, "| data:", DATA, flush=True)

COMMON = [
    "--data-dir", str(DATA),
    "--skip-architecture-smoke", "--skip-real-qwen-preflight",
    "--latent-thoughts", "6", "--kv-prefix-slots", "16", "--kv-prefix-rank", "64",
    "--prefix-attn-gate-init", "0.08", "--mlp-expert-count", "2", "--mlp-expert-rank", "16",
    "--steps", "2000", "--warm-steps", "600", "--wo-l1-branch-steps", "300",
    "--batch-size", "1", "--gradient-accumulation", "16",
    "--learning-rate", "0.0008", "--weight-decay", "0.1", "--warmup-updates", "4",
    "--lora-rank", "64", "--lora-alpha", "128", "--lora-tail-layers", "28",
    "--unfreeze-tail-layers", "0", "--num-lanes", "1", "--num-experts", "1",
    "--refinement-steps", "2", "--diffusion-steps", "4",
    "--context-tokens", "256", "--canvas-tokens", "64", "--brief-tokens", "32",
    "--causal-tokens", "512", "--trace-tokens", "96", "--precision", "auto",
    "--max-validation", "256",
    # ---- the two v11 deltas ----
    "--vocab-grounded-thoughts", "--incontext-decode-weight", "0.5",
]

# ---- blocking preflight on GPU0 (calibrates gamma, measures s/step) ----------
PRE = WORK / "hlwm-v11-preflight"
cmd = [sys.executable, str(PROJECT / "train_kaggle.py"),
       "--output-dir", str(PRE), "--seed", "17",
       "--v10-preflight-only", "--num-workers", "0"] + COMMON
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
res = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True)
print(res.stdout[-4000:], flush=True)
if res.returncode != 0:
    print(res.stderr[-4000:], flush=True)
    raise SystemExit("BLOCKING preflight failed (%d)" % res.returncode)
preflight = json.loads((PRE / "preflight.json").read_text())
assert preflight["passed"], preflight.get("failed_checks")
GAMMA = preflight["calibrated_gamma"]
print("calibrated gamma", GAMMA,
      "| s/step", round(preflight["checks"]["seconds_per_step"], 2), flush=True)

# ---- two seeds in parallel, one per T4 ---------------------------------------
OUT = {seed: WORK / f"hlwm-v11-seed-{seed}" for seed in SEEDS}
procs, logs = {}, {}
for gpu, seed in enumerate(SEEDS):
    out = OUT[seed]
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, str(PROJECT / "train_kaggle.py"),
           "--output-dir", str(out), "--seed", str(seed),
           "--num-workers", "2",
           "--distill-gamma", str(GAMMA), "--telemetry-every", "100",
           "--masked-fraction-floor", "0.5", "--warm-em-floor", "0.50",
           "--gonogo-step", "1200", "--gonogo-masked-numeric-em", "0.05",
           "--save-every", "500", "--max-runtime-hours", "5.5"] + COMMON
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(WORK / f"hlwm-v11-seed-{seed}-training.log", "w")
    logs[seed] = log
    procs[seed] = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"seed {seed} -> GPU {gpu} (pid {procs[seed].pid})", flush=True)

codes = {}
while procs:
    for seed, proc in list(procs.items()):
        code = proc.poll()
        if code is None:
            continue
        codes[seed] = code
        logs[seed].close()
        print(f"seed {seed} training exited {code}", flush=True)
        del procs[seed]
    time.sleep(30)
print("training exit codes:", codes, flush=True)

# ---- audits (sequential, GPU0) ------------------------------------------------
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch  # noqa: E402
import evaluate_v10 as ev10  # noqa: E402

gates_by_seed, verdicts = {}, {}
for seed in SEEDS:
    try:
        import train_kaggle as tk
        verdicts[seed] = tk.training_verdict_from_metrics(OUT[seed] / "metrics.jsonl") \
            if hasattr(tk, "training_verdict_from_metrics") \
            else ev10.training_verdict_from_metrics(OUT[seed] / "metrics.jsonl")
    except Exception as error:
        verdicts[seed] = {"error": str(error)}
    try:
        gates = ev10.run_seed_audit(OUT[seed], data_dir=DATA, seed=seed,
                                    device=torch.device("cuda:0"))
        gates_by_seed[seed] = gates
    except Exception as error:
        gates_by_seed[seed] = {"error": str(error), "passed": False}
        print(f"seed {seed} audit error: {error}", flush=True)
    torch.cuda.empty_cache()

rung = ev10.binding_rung(gates_by_seed) if all(
    "error" not in g for g in gates_by_seed.values()) else "audit-error"
verdict = {
    "package_version": "11.0-pilot",
    "preregistration": "hlwm-v11-vocab-grounded-plan-2026-09-07.md",
    "control": "sirishayaddanapudi/hlwm-v10-0-certified-run (matched protocol)",
    "binding_rung": rung,
    "seeds": {str(s): gates_by_seed.get(s, {}) for s in SEEDS},
    "training_verdicts": {str(s): verdicts.get(s, {}) for s in SEEDS},
    "training_exit_codes": {str(s): codes.get(s) for s in SEEDS},
}
(WORK / "hlwm-v11-replication-verdict.json").write_text(
    json.dumps(verdict, indent=1, sort_keys=True, default=str))
print("BINDING RUNG:", rung, flush=True)

# ---- deliverables --------------------------------------------------------------
import re  # noqa: E402
for seed in SEEDS:
    out = OUT[seed]
    staging = WORK / f"hlwm-v11-seed-{seed}-deliverable"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in ("v10-audit.json", "v10-gates.json", "metrics.jsonl"):
        if (out / name).exists():
            shutil.copy2(out / name, staging / name)
    shutil.copy2(PROJECT / "manifest-v11.json", staging / "manifest-v11.json")
    shutil.make_archive(str(staging), "zip", staging)
    # keep exactly one checkpoint per seed
    full = out / "checkpoint-v10-full.pt"
    if not full.exists():
        steps = sorted(out.glob("checkpoint-step-*.pt"),
                       key=lambda p: int((re.findall(r"(\d+)", p.stem) or ["0"])[-1]))
        full = steps[-1] if steps else None
    keep = {full.name} if full else set()
    for stale in out.glob("checkpoint-step-*.pt*"):
        if stale.is_file() and stale.name not in keep:
            stale.unlink()
print("done", flush=True)
