# HLWM v5.6 Training-Data Review — combined bundle audit and scope plan

Date: 2026-08-30
Reviewed: `data/combined/` (manifest 2026-08-30T15:21Z), `data/builder/final/`,
`sources/public/benchmark-train-splits/` (fetch manifest 15:02Z), against the
trainer interface in `experiments/kaggle_hlwm/{train_kaggle.py,data.py}`.
Purpose: decide what v5.6 trains and tests on, on Kaggle dual-T4, on the path
to a production-grade calibrated selective-prediction model.

## 1. What the new data is (verified by inspection, not from memory)

**Combined bundle** (`data/combined/`, SHA-256 manifest): master
4,358/844/664 (train/val/test episodes), SFT 9,092/840/659 (messages rows),
25 DPO pairs. Policy block in the manifest confirms: no benchmark eval items,
no teacher private reasoning, external packets train-only, cross-split prompt
collisions dropped (0 needed dropping), unapproved shortlist excluded.

Per-source SFT-train composition, with measured length profile
(chars ÷ 4 ≈ tokens) and fit against candidate causal budgets:

| Source | rows | ~Mtok | fits 384 | fits 768 | fits 1024 | fits 1536 |
|---|---|---|---|---|---|---|
| v5.5 bundle (anchors + R9000) | 4,319 | 4.22 | 24% | 30% | 45% | 85% |
| APPS train (MIT, pinned rev) | 1,199 | 0.66 | 29% | 85% | 95% | 99% |
| Spider train (CC-BY-SA) | 1,200 | 0.09 | 100% | 100% | 100% | 100% |
| TACO-verified train (Apache-2.0) | 716 | 0.40 | 30% | 81% | 94% | 98% |
| MBPP train (CC-BY-4.0) | 464 | 0.07 | 100% | 100% | 100% | 100% |
| SWE-bench train | 398 | 0.62 | 6% | 36% | 46% | 63% |
| CodeContests train | 393 | 0.27 | 22% | 72% | 86% | 96% |
| DeepCoder audited packets | 256 | 0.15 | 27% | 73% | 92% | 100% |
| OpenThoughts audited packets | 127 | 0.13 | 0% | 34% | 65% | 87% |
| Builder factory episodes | 20 | 0.05 | 0% | 0% | 0% | 30% |

**Verification material is present and executable** for every benchmark
source: MBPP carries assert test lists on 464/464 rows; APPS, TACO, and
CodeContests carry input/output pairs; Spider carries `db_id` + gold SQL;
SWE-bench carries the gold patch as reference answer (no in-packet executable
check — a repo harness would be required).

**Builder episodes** are trainer-native (`normalize_episode`-compatible
master schema). 24 master-train episodes across 7 domains; **11 are
`independently_adjudicated: true`**, including 10 `coding_debugging` episodes
whose verification records are `method: executed_pytest` with real run
results (e.g. 34/34 passed, returncode 0). The judge-only rows (copy, deck,
planning, invest, devops) are correctly flagged `false` — fail-closed. These
are the first adjudicated episodes in the program, directly addressing the
paper's "zero independently adjudicated episodes" limitation.

## 2. Leakage and provenance checks — all pass

- MBPP: curation records `protected_ids_excluded: task_id 1-510` (the
  official test/val ID block).
- SWE-bench: curation lists all 12 SWE-bench test repos as excluded
  (astropy, django, matplotlib, seaborn, flask, requests, xarray, pylint,
  pytest, scikit-learn, sphinx, sympy); sampled train rows are from other
  repos.
- Every source pinned to an exact HF revision with license recorded
  (MIT / Apache-2.0 / CC-BY-4.0 / CC-BY-SA-4.0); 88 duplicate prompts
  dropped at bundling; eval-overlap drops: 0.
- The benchmark registry (`config/benchmarks.reasoning.yaml`, 143 entries)
  remains eval-only with `allowed_for_training: false`;
  HumanEval/LiveCodeBench/GPQA-class stay training-forbidden.

## 3. Gaps found (these decide the v5.6 work items)

**G1 — Format gap (blocking).** `train_kaggle.py` consumes only
`master/*.jsonl` episodes. All 5,136 new external rows (benchmark packets +
audited packets) exist only in the SFT split. **Run v5.6 on the bundle as-is
and it trains on zero new external rows.** Required: a packet→episode
converter at bundle-build time (problem → `input`/`frame`, reference answer →
`integration.published_answer`, verification material → `evaluation.answer_spec`),
or a causal-corpus channel in the trainer. The converter is the better move:
it also unlocks graded audits on this data.

**G2 — Token budgets (blocking at v5.5 settings).** v5.5 trained at
context 192 / canvas 128 / brief 96 / causal 384. At causal 384 only ~25% of
the code data fits. At **causal 1024** the usable fractions are 86–100% for
APPS/TACO/CodeContests/DeepCoder/MBPP/Spider. T4 headroom exists: v5.5
peaked 4.77 GB of 14.56 GB; a ~2.7× sequence increase lands well under
budget (verify in preflight). Cost is throughput, not memory.

**G3 — SWE-bench and builder rows don't fit the 0.6B prototype.** SWE-bench
is 46% usable even at 1024 and its patches aren't independently checkable
without a repo harness; builder episodes are median ~9.5K chars. Recommend:
defer SWE-bench and full builder deliverables to the 8B candidate; include
builder's 11 adjudicated episodes anyway for their *policy labels* (heads
read pooled state; note truncation as a caveat), and the rest as
causal-only rows.

**G4 — Policy supervision stays narrow unless we execute checks.**
Eligibility in `data.py` is `independently_adjudicated OR
programmatically_verified` — currently 1,024 anchors + 11 builder episodes.
The benchmark packets carry checkable material but are (correctly) not
flagged. The production-relevant upgrade: at bundle build, **execute MBPP
test lists against reference answers** (cheap, deterministic, sandboxed) and
mark only passing rows `programmatically_verified`; optionally do the same
for a subset of APPS/TACO/CodeContests IO pairs with time limits, and
Spider gold SQL against the Spider DBs. That grows verified
commitment/calibration supervision from 4 toy families to real code/SQL —
exactly the sharpened claim ("knows when not to answer") on real tasks.
Hard negatives come free: wrong-solution variants and cross-problem answer
swaps, execution-labeled.

**G5 — No held-out slice for the new domains.** Bundle policy puts all
external packets in train (correct w.r.t. official splits), but that leaves
nothing to measure in-domain generalization. Recommend carving a per-source
dev slice (e.g. 5%, keyed by problem/lineage ID) out of the *train* packets
at bundle build. This never touches official eval items; the 143-benchmark
registry remains the post-gate external yardstick.

**G6 — Class balance.** Every new row is an "answerable" row; missing-evidence
abstention still comes only from the 1,024 anchors. Keep the trainer's
anchor-fraction guarantee so calibration classes aren't swamped, and keep
abstention anchors in every audit.

**G7 — Small stuff.** DPO pairs: 25 — too few for a DPO phase; use as extra
hard-negative families in the on-policy head phase. Duplicates already
dropped. The pasted NVIDIA API key from the factory session should be
rotated before any public artifact ships.

## 4. Why this data fits the v5.6 routing fix

v5.5's routing failure (expert 0 at 91–94%, experts 2–5 at zero, homogenized
experts) happened on near-homogeneous data: 4 anchor families + R9000's
one-path episodes gave the router almost nothing to specialize on. The new
bundle adds genuinely distinct computation families — competitive
programming, text-to-SQL, python functions, repository debugging, plus 8
builder domains — so the v5.6 mechanism change (Switch-style load balancing
on hard assignments + expert-output diversity penalty, fallback 6→2 experts)
gets tested with data that actually rewards specialization. Domain labels
stay diagnostic priors, not routing supervision, per the paper's invariants.

## 5. Recommended v5.6 scope (to be preregistered before the run)

- **Data**: combined master episodes + converted packets (G1), causal 1024 /
  context 256 / canvas 128 / brief 96 (G2), SWE-bench + oversized builder
  deliverables deferred (G3), MBPP execution-verified policy rows + optional
  APPS/TACO/CC subset (G4), 5% per-source dev slices (G5), anchor-fraction
  guard (G6).
- **Architecture deltas** (already decided from Study 4): hard-assignment
  load-balance aux loss replacing marginal-KL; expert-output diversity
  penalty; declared fallback num-experts 6→2; policy-head de-saturation
  (fewer epochs or label smoothing).
- **Gates**: all v5.5 gates unchanged, plus per-source dev-slice metrics
  reported (MBPP pass-by-execution, Spider exact/execution match), coverage
  paired with risk per seed.
- **Sessions** (~26.5 h quota): A = v5.6 seeds 17+29 dual-T4 (~6–7 h at the
  larger budgets; trim steps if preflight timing says otherwise).
  B = **matched plain-LoRA control** (same data/steps/trainable params, no
  HLWM modules) — the paper's decisive missing control — plus seeds 41/73 if
  A passes. C = external benchmarks (registry, eval-only) only if the full
  gate passes, per standing preregistration rules.

## 6. Verdict

The bundle is well-built: pinned, licensed, deduplicated, leakage-checked,
fail-closed on adjudication, and it finally contains executed-check
supervision. It is **not consumable by the v5.6 trainer as-is** (G1) and
**not usable at v5.5 token budgets** (G2). With the converter, the budget
bump, execution-verified policy marking, and dev slices, v5.6 trains on
~8,600 usable rows across 10+ real domains with ~1,900+ execution-verifiable
policy rows — a substantial step from prototype toward production, while the
official eval registry stays untouched for honest post-gate benchmarks.
