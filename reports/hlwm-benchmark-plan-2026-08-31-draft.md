# HLWM External-Benchmark Plan (Session C) — DRAFT, unauthorized until a gate passes

Date: 2026-08-31 (draft; finalize and re-date before session C launches)
Standing rule (v5.5/v5.6 preregistrations): the external-benchmark step is
authorized **only after both seeds of a revision pass the complete
capability gate**. This draft freezes the design now so the session can
launch immediately when that happens; no numbers may be reported from any
run performed before authorization.

## Materials (all built and checksummed 2026-08-31)

- Kit: `artifacts/kaggle/hlwm-benchmarks/hlwm-benchmark-kit.zip`
  (sha256 `878402bf…`, 131 KB) + `embel-hlwm-benchmarks-2xt4.ipynb`.
- Suites (evaluation-only artifact,
  `sources/public/benchmark-eval-subsets/`, sampled seed 17 from official
  test splits at pinned revisions):
  - GSM8K test, 250 items, numeric grader (`openai/gsm8k` @ `740312a…`, MIT);
  - ARC-Challenge test, 250 items, multiple-choice grader
    (`allenai/ai2_arc` @ pinned, CC-BY-SA-4.0);
  - MMLU test, 250 items across subjects, multiple-choice grader
    (`cais/mmlu` @ `c30699e…`, MIT).
  Gold answers self-grade 750/750 through the deterministic graders.
- Systems, identical prompt text (`### Instruction … ### Response`) and
  identical 320-token generation budget:
  1. HLWM decision path (workspace candidate + calibrated commit/abstain);
  2. HLWM causal path (same checkpoint, plain generation);
  3. `Qwen/Qwen3-0.6B-Base` @ the training pin (frozen base);
  4. `Qwen/Qwen3-0.6B` (instruct) @ `c1899de289a04d12100db370d81485cdf75e47ca`.

## Preregistered endpoints (frozen at draft time)

- **Primary (the paper's claim):** on each suite, HLWM selective accuracy at
  its achieved coverage exceeds every baseline's matched-coverage selective
  accuracy, where the baseline abstains by mean-logprob confidence with a
  threshold fit post hoc on the same test items (a comparison that
  structurally favors the baselines). Coverage is always reported next to
  accuracy, per seed.
- **Secondary (report, not claim):** HLWM candidate accuracy vs baseline raw
  accuracy; HLWM causal-path accuracy vs frozen base (finetuning effect);
  abstention counts.
- **Non-goals declared:** no production-readiness claim; no claim vs the
  instruct model's raw accuracy; a single-seed result is labeled single-seed.

## Session mechanics

Attach: v5.6.1 bundle (code), the kit, and the passing seed's
`hlwm-v5.6.1-seed-<seed>-resumable.pt`. GSM8K runs on GPU 0 concurrently
with ARC+MMLU on GPU 1 (~2.5–3 h estimate: ~750 HLWM decision-path
generations dominate). If both session-B seeds pass, run the better-covered
seed first; the second seed's run is the replication and shares the session
if time allows.

## Amendments log

(Empty. Any change between this draft and launch is recorded here before
results are read.)
