# HLWM V5.6.2 Session C Results — the reduction bought routing and paid in content; rung 1 falls

Date: 2026-09-01
Plan: `reports/hlwm-v5.6.2-plan-2026-08-31.md`
Artifacts: `artifacts/kaggle/hlwm-v5.6.2/hlwm-v5-3/` (= seed 17),
`artifacts/kaggle/hlwm-v5.6.2/hlwm-v5-4/` (= seed 29).
Config verified from summaries: num_experts 2, expert_init_scale 0.01,
validity floor 16/family cap 512, canary 8 anchors / 512 steps, everything
else frozen. 4.7 h per seed, peak 4.81 GB, telemetry complete this time.

## Verdict

**Primary endpoint failed on both seeds (17: 6 gates, 29: 5 gates), and the
preregistered fallback ladder's rung 1 is triggered: seed 17 collapsed to one
expert even at num_experts 2** (audit route load [1.000, 0.000], third
consecutive collapse under three different configurations). Per the binding
ladder, **v5.7 runs workspace-only (lanes + commitment, no routed experts)
and the paper's routing section is re-scoped to
proposal-plus-negative-evidence.**

| Gate cluster | Seed 17 (B → C) | Seed 29 (B → C) |
|---|---|---|
| Second-route load (≥0.10) | 0.000 → **FAIL 0.000** | 0.113 → **PASS 0.273** |
| Normalized route entropy (≥0.25) | 0.000 → FAIL 0.000 | 0.306 → **PASS 0.846** |
| Lanes distinct (<0.90) | 0.9016 → FAIL 0.9278 | 0.9190 → **PASS 0.8291 (first ever)** |
| Routing causally live (≥0.01) | 0.0012 → FAIL **0.0082 (closest ever)** | 0.0031 → FAIL 0.0037 |
| Probe content accuracy (≥0.75) | 0.781 → **FAIL 0.563** | 0.750 → **FAIL 0.469** |
| Quality pass rate (≥0.75) | 0.828 → **FAIL 0.563** | 0.797 → **FAIL 0.469** |
| Probe commit accuracy (≥0.70) | 0.781 → **PASS 0.844 (best ever)** | 0.625 → FAIL 0.469 |
| Prompt leak (=0) | 0.0 → 0.0 | 0.0 → **FAIL 0.0625** |
| Commitment threshold | 0.0 → **0.0 (degenerate, #5)** | 0.090 → **0.0 (degenerate, #6)** |

## Finding 1 — the reduction is a routing/content trade, not a fix

Seed 29 at two experts posted the healthiest routing in program history
(load 0.273, entropy 0.846, and the lane-distinctness gate passed for the
first time in seven studies) — while its content collapsed: probe accuracy
0.75 → 0.469, quality 0.797 → 0.469, plus its first-ever prompt leak
(0.0625). Seed 17 lost content too (0.781 → 0.563). Numeric probes fell to
0.25 and unit probes to 0.125 **on both seeds** (Session B: 0.625–0.75).
Cutting 6 experts to 2 removed adapter capacity the content path was
actually using. The reduction rule's premise — that the expert graph was
freeloading — is falsified in the direction nobody wanted: the experts were
carrying content, just not routing-diversity.

## Finding 2 — validity-gated balancing worked mechanically and moved the one head gate

The floor logic did its job: 448–512 attempts produced 77–78 valid positives
(vs 41–43 in Session B) at near-perfect family balance (15–16 of 16 per
family; unit at 13–15, floor honestly reported unmet at 0.05–0.08 grader
accuracy). Seed 17's commit accuracy then passed at **0.844, the best ever
measured** — evidence the head-phase imbalance was a real cause of Session
B's false rejections. Seed 29's 0.469 does not refute this: its committed
set is dominated by content errors (graded accuracy 0.469), so the heads were
scoring a worse generator, not scoring worse. The commitment threshold still
calibrated to 0.0 on **both** seeds (recurrences 5 and 6); the risk and
verifier heads carry all rejection. De-correlating policy labels (Known
Limitations item 3) is now the oldest unpaid debt in the program.

## Finding 3 — the canary shipped, and its first lesson is about itself

Telemetry is complete for the first time (diagnostic (a) finally measured:
`expert_diversity` grew 1e−5 → 0.0021 on seed 17 and → 0.0007 on seed 29 —
the penalty engages under nonzero init but stays two orders below the 0.05
weight's bite point). The canary caught seed 17's early instability (step
512: leak 0.125, dialogue-marker rate 0.75, later self-corrected) and
tracked semantic validity rising 0.0 → 0.375. But on routing it was blind by
construction: **both** seeds' canaries showed single-expert routing all run
(second load 0.0 at every boundary), while seed 29's audit showed 0.273 —
behavior anchors route homogeneously, and the diversity lives in the
non-anchor domains the canary never samples. Diagnostic (e) answered: an
anchor-only canary cannot discriminate a collapsing seed from a healthy one.
A v5.7 canary must be domain-stratified, not anchor-stratified.

## Finding 4 — the intervention gate's near-miss is seed 17's, ironically

Seed 17 — the collapsed seed — posted mean commit delta 0.0082 under
least-used-expert pinning, the closest any run has come to the 0.01 gate
(0-for-10 lifetime). With only 2 experts, pinning to the unused expert is a
genuinely out-of-distribution perturbation, so the delta finally moved. This
is consistent with the experts differing (init 0.01 + diversity growth)
while the router still refuses to use both: the failure is in the routing
decision, not expert homogeneity.

## Preregistered decisions

1. **Benchmarks remain unauthorized.**
2. **Rung 1 triggered** ("if either seed still collapses at num_experts 2"):
   v5.7 removes the routed expert graph from the small-scale claim —
   workspace-only (lanes + synthesis + verification + commitment), and the
   paper's routing section is re-scoped to proposal-plus-negative-evidence
   across Studies 3–7 (three collapses under three configurations; one seed
   passing load gates at 6 experts twice and at 2 experts once, never
   causally live).
3. Carry into v5.7: validity-gated head phase (worked; keep floor 16, cap
   512), nonzero init for whatever adapters remain, domain-stratified canary
   (replace anchor-only sampling), threshold non-degeneracy diagnostic
   (now 6 recurrences — v5.7 must also ship the label de-correlation fix,
   not just the diagnostic).
4. Content regression question for v5.7: whether workspace-only at the same
   trainable budget (LoRA rank compensating for removed experts) recovers
   Session B content levels. If it does, the 6-expert content contribution
   was fungible capacity; if not, the expert bank was load-bearing for
   content and the reduction claim needs that caveat.
5. Plain-LoRA control: unchanged decision — runs when a stable pair exists.
   Note it is unaffected by the expert-graph removal (it never had one).

## Claim boundary

Session C establishes: two-expert routing under identical objectives remains
seed-dependent (one collapse, one strong pass); the expert bank was
contributing content capacity (both seeds regressed ~0.2–0.28 probe
accuracy when it shrank); validity-gated balancing fixes head-phase data
imbalance and, on the seed where content held, the commit-accuracy gate;
anchor-only generation canaries cannot see routing health. No production,
benchmark, or capability claim is made or implied.
