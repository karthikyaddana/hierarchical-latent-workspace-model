# HLWM V5.6.1 Session B Results — one fix landed, one was nullified, the lottery stands

Date: 2026-08-31
Plan: `reports/hlwm-v5.6.1-plan-2026-08-31.md`
Artifacts: `artifacts/kaggle/hlwm-v5.6.1/hlwm-v5-3/` (= seed 17),
`artifacts/kaggle/hlwm-v5.6.1/hlwm-v5-4/` (= seed 29).
Config verified from summaries: `expert_init_scale 0.01`, `policy_records 128`,
everything else byte-identical to v5.6 (revision `da87bfb6`, 4,224/4,224 steps,
BF16, num_experts 6).

## Verdict

**Primary endpoint failed on both seeds. But the two preregistered fixes had
cleanly separable fates: nonzero expert init rescued seed 17's entire
generation-degeneration cluster while leaving its routing collapse untouched;
the family-balanced head phase never actually balanced anything because
two-thirds of its emissions were invalid.**

| Gate cluster | Seed 17 (A → B) | Seed 29 (A → B) |
|---|---|---|
| Prompt leak | 0.266 → **PASS 0.0** | 0.0 → 0.0 |
| Quality / semantic probes | 0.578/0.281 → **PASS 0.781** | PASS 0.844/0.813 → PASS 0.75 |
| Probe commit accuracy (≥0.70) | 0.656 → **PASS 0.781** | 0.625 → **FAIL 0.625 (unchanged)** |
| Safe abstention (acc / commit rate) | 0.75 → 1.0 / 1.0 | 1.0 → 0.875 / 0.875 |
| Second-route load (≥0.10) | 0.000 → **FAIL 0.000** | 0.418 → PASS 0.113 (−73%) |
| Normalized route entropy (≥0.25) | 0.000 → **FAIL 0.000** | 0.570 → PASS 0.306 (−46%) |
| Routing causally live (≥0.01) | 0.0035 → FAIL 0.0012 | 0.0073 → FAIL 0.0031 |
| Lane distinctness (<0.90) | 0.942 → FAIL 0.9016 | 0.890 → **FAIL 0.9190** |
| Commitment threshold | 0.912 → **degenerate 0.0** | 0.0 → 0.090 |
| Gates failed (of 18) | 8 → 4 | 2 → 4 |

## Finding 1 — nonzero expert init rescues the degeneration cluster, not the collapse

Seed 17's Session A pathology (26.6% prompt leak, 0.281 anchor accuracy,
dialogue-prior leakage) is **gone**: leak 0.0, probe content 0.781, commit
accuracy 0.781 passing its gate for the first time on this seed. Yet its
routing still collapsed to 100% one expert (second load 0.000, entropy −0.0,
intervention delta 0.0012). Conclusion: prefix degeneration and routing
collapse were **not one failure mode** — the init fix decoupled them. Routing
collapse on seed 17 is now 2-for-2 under two different initializations, and
diagnostic (a) from the plan (does `expert_diversity` telemetry move) is
unanswerable from the uploaded artifacts: `metrics.jsonl` contains only
`resumed_at_completion` markers, so the runs resumed past the telemetry
window. Treat the diversity-penalty liveness question as **unmeasured, not
answered**.

Diagnostics (b) and (c) from the plan: (b) **yes** — seed 17 avoided prefix
degeneration (leak 0, but lane cosine 0.9016 misses the 0.90 gate by 0.002);
(c) **no** — seed 29's commit accuracy did not clear 0.70; (d) **no** —
intervention deltas fell on both seeds (0.0012 / 0.0031 vs 0.01 gate).

## Finding 2 — the head-phase balance fix was nullified before it could act

Seed 29's commit accuracy is bit-for-bit unchanged (0.625, coverage 0.5625,
ordering family still weakest). The mechanism is visible in
`policy-head-training.json`: of 128 requested emissions, **87 (seed 17) / 85
(seed 29) were invalid**, leaving 41/43 valid positives against 343/341
negatives — a 1:8 imbalance essentially identical to the 96-record phase the
fix was supposed to repair. Per-family round-robin balanced the *requests*,
not the *valid records*; the numeric and unit graders scored the emissions at
0.06–0.16 and 0.0 respectively during collection. The fix's premise (more,
balanced records) was correct but its implementation point was wrong: balance
must be enforced **after validity filtering**, with per-family resampling
until a floor of valid positives is met.

## Finding 3 — routing diversity is not merely seed-dependent, it is fragile

Seed 29 kept multi-expert routing but at sharply reduced strength: second
load 0.113 (1.1× its gate, was 4.2×), entropy 0.306 (1.2× gate, was 2.3×).
Same data, same seeds, one flag changed — and the previously strong routing
weakened toward the gate line while the sibling stayed collapsed. Combined
with the causal-liveness gate now 0-for-8 across four studies, the honest
reading: the current 6-expert joint phase does not have a reliable basin of
multi-expert, causally load-bearing routing at this scale.

## Finding 4 — the degenerate-threshold symptom migrated seeds (recurrence #4)

Seed 17's commitment threshold calibrated to 0.0 (Session A: seed 29's did;
Session A seed 17's was healthy 0.912). Fourth recurrence overall, now
observed on both seeds. This is the correlated-policy-label limitation
(paper Known Limitations item 3) presenting through whichever seed's
score geometry happens to be tighter; it will not fix itself.

## Preregistered decision

1. **Benchmarks remain unauthorized** (full gate required).
2. **The fallback ladder's second rung is triggered.** The v5.6.1 plan's
   ladder: (i) num-experts 2 "if both seeds again fail the routing gates" —
   not strictly met (seed 29 passed the two load gates, barely); (ii)
   "if joint-phase outcomes remain a seed lottery, simplify the workspace per
   the paper's reduction rule" — **met**: the targeted fixes ran, and the
   collapse/diversity split across seeds reproduced exactly. The paper's
   failure-mode table row for joint-phase seed variance names this exact
   condition.
3. Next phase is therefore **v5.6.2, a simplification study**, preregistered
   in `reports/hlwm-v5.6.2-plan-2026-08-31.md`: expert count 6 → 2 (the
   strongest-successful-ablation direction the routing-collapse row also
   points to), validity-gated family balancing in the head phase, a
   generation-time canary during training, and a threshold non-degeneracy
   check. Everything else frozen; same seeds.
4. Plain-LoRA control remains postponed until a stable pair exists
   (explicit user decision 2026-08-31, unchanged).

## Claim boundary

Session B establishes: nonzero expert initialization eliminates the
workspace-prefix degeneration mode (n=1 seed, the seed that exhibited it);
the head-phase imbalance persists because emission validity, not record
count, is the binding constraint; and joint-phase routing outcomes remain
seed-dependent after both targeted fixes — triggering the preregistered
architecture-reduction rule. No production, benchmark, or capability claim
is made or implied.
