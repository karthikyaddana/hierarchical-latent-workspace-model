# HLWM V5.6 Session A Results — routing breakthrough on one seed, variance on the other

Date: 2026-08-31
Plan: `reports/hlwm-v5.6-plan-2026-08-30.md`
Artifacts: `artifacts/kaggle/hlwm-v5.6/hlwm-v5-3/` (= seed 17),
`artifacts/kaggle/hlwm-v5.6/hlwm-v5-4/` (= seed 29), replication verdict JSON.
Training: both seeds 4,224/4,224 steps, zero skipped updates, BF16, peak
4.82 GB, 3.98 h / 4.22 h — the causal-1024 budget cost ~35% more time than
v5.5, well under the 8.5 h cap.

## Verdict

**The preregistered primary endpoint failed — but for the first time in four
studies the two seeds did not fail the same way, and one of them passed the
routing-load gates that three consecutive versions had failed.**

| Gate cluster | Seed 17 | Seed 29 |
|---|---|---|
| Training complete / 0 skips | PASS | PASS |
| Validation calibration (bal. acc.) | PASS 0.909 | PASS 0.904 |
| Head de-saturation (score rails) | 0.051 / 0.949 | 0.050 / 0.945 |
| Prompt leak | **FAIL 0.266** | PASS 0.0 |
| Quality / semantic probes | **FAIL 0.578 / 0.281** | PASS 0.844 / 0.813 |
| Lane distinctness (< 0.90) | **FAIL 0.942** | PASS 0.890 |
| Second-route load (≥ 0.10) | **FAIL 0.000** | **PASS 0.418** |
| Normalized route entropy (≥ 0.25) | **FAIL 0.000** | **PASS 0.570** |
| Routing causally live (≥ 0.01) | FAIL 0.0035 | FAIL 0.0073 |
| Probe commit accuracy (≥ 0.70) | FAIL 0.656 | FAIL 0.625 |
| Gates failed (of 18) | 8 | 2 |

## Finding 1 — the v5.6 routing objective can produce real route diversity

Seed 29 routed through **three experts** (loads 0.16 / 0.42 / 0.42), passing
second-route load at 4.2× its gate and normalized entropy at 2.3× its gate.
Studies 2–4 never exceeded 0.094 / 0.174. The causal-liveness intervention
also improved — 0.0073 against the 0.01 gate, versus 0.0008–0.0035 in Study 4
— but still fell short: swapping to the least-used (zero-traffic) expert
moves commitment probability by less than a point. Direction strongly right;
one gate short.

Notably, the **expert-output diversity penalty did not do this**: telemetry
shows it never engaged (0.0002–0.0018 throughout, because the expert
adapters' zero-initialized up-projections keep deltas tiny, so there is
nothing to decorrelate at the start and the term never wakes up). The
diversity seed 29 found came from the strengthened hard-assignment balance
loss plus favorable optimization luck. That is exactly what Finding 2 says
must be fixed.

## Finding 2 — same configuration, catastrophically different outcomes

Seed 17 collapsed to **100% one-expert routing** (worse than any prior
study), lane cosine 0.942, and a degenerate candidate-generation path:
26.6% prompt-leak flags, anchor content accuracy 0.281 (v5.5 seed 17:
0.875). Inspection shows the failure texture: candidates are frequently
*numerically correct but format-broken* — e.g. reference "The result is
84325." vs candidate "Human: 80329 + 3996 = 84325." — plus some empty
candidates. Meanwhile seed 17's plain **causal path is completely healthy**
(token F1 0.588, identical to seed 29's 0.589), and only 3 of 9,284 training
rows contain such chat markers, so this is not data contamination: the
degenerate workspace prefix stopped steering generation and the base model's
dialog prior leaked through. Everything upstream looked normal during
training (calibration 0.818/1.000, non-degenerate thresholds 0.91/0.11/0.10,
smooth losses except late denoise spikes) — the collapse is only visible at
generation time. Conclusion: the joint workspace phase has **high training
variance** — routing collapse, lane collapse, and prefix degeneration
co-occur, and nothing in the current training-time telemetry gates on it.

## Finding 3 — head de-saturation worked; the remaining commit misses are a
data-balance problem

Both seeds now hold calibrated scores at ~0.05/0.95 instead of v5.5's
5×10⁻⁵/0.9999 rails, with ranking accuracy 1.0 and balanced accuracy ~0.90
on held-out generated emissions. Seed 29's failing commit accuracy (0.625)
decomposes cleanly: 11 of its 12 wrong decisions are **confident false
rejections of valid answers** (p_commit ≈ 0.06, p_risk ≈ 0.94 — exactly the
smoothed negative target), i.e. a cluster of valid candidates lands in
feature space where the head phase only ever saw negatives. Ordering anchors
(0.625 accuracy) are the weakest family. The commitment threshold also
degenerated to 0.0 again on seed 29 (risk/verifier carry rejection), the
third recurrence of the correlated-label symptom. Seed 17's thresholds, for
once, were non-degenerate (0.912/0.113/0.097).

Per-domain report-only metrics behaved as expected for a 0.6B first pass:
code/SQL graded accuracy ~0 on tiny dev samples (1–2 per domain per audit);
seed 29's anchor F1 0.94. These set the v5.6.1 baselines.

## Decisions (per preregistration)

1. **Benchmarks not authorized** (full gate required).
2. **The num-experts-2 fallback is NOT triggered** — it required both seeds
   to fail the routing gates; seed 29 passed two of three and near-missed
   the third.
3. **Next revision (v5.6.1) targets the two identified mechanisms, nothing
   else:**
   - *Wake the diversity penalty*: initialize expert up-projections with a
     small nonzero scale so `expert_diversity` exerts pressure from step 1;
     this is the specific lever aimed at seed 17's collapse mode, and it was
     provably inert this run.
   - *Balance the head phase*: increase policy records 96→128 with
     per-family balancing so no anchor family is learned as
     "always-negative"; keeps smoothing 0.05.
   - Everything else — data, budgets, balance weight, gates — frozen, so the
     seed-17 variance question gets a clean second measurement.
4. Session B = v5.6.1 seeds 17+29 (same seeds, to see whether the fix
   rescues 17's failure mode). The plain-LoRA control (code ready except one
   test) runs after a stable pair exists — comparing a control against a
   high-variance architecture wastes its 3.5 h.

## Claim boundary

This session establishes: the hard-assignment balance objective *can* yield
multi-expert routing that clears the preregistered load gates (first time in
the program); the head de-saturation fix works and replicates; and the joint
workspace phase has unacceptable seed variance that current telemetry cannot
detect during training. No production, benchmark, or capability claim is
made or implied. Quota after session A: ~19 h.
