# HLWM v10.5 Build-Ready Specification (workflow wf_6b46ec82-bf1, 2026-09-04)

All line anchors verified against the shipped bundle. Final specification follows.

# HLWM v10.5 BUILD-READY SPECIFICATION (final, 2026-09-04)

Base = merge decision (latent-first, Design A). All three lenses returned survives-with-fixes; every fix below is folded in. **Dropped for fatal-without-fix: nothing** — both fatals (K1 schedule aliasing; K-budget ledger) shipped fixes, adopted as items 1 and 12. Decode-time DA, dual-use student enforcement, and B's adjudication remain deferred per the merge.

## 1. Implementation items (dependency order; blocking tests named inline)

**1. Schedule decorrelation [FATAL fix; before all other trainer work].** Family rotation has period 4 (`train_kaggle.py:3585`, batch 1 ⇒ micro-index==step), so every power-of-2 parity aliases to a fixed family subset — shipped `recon_step/colar_step` (`:3321-3322`) already train recon only on {abstention,ordering} and CoLaR only on {numeric,unit}. Fix: recon/CoLaR alternate on `(step//4)%2` (each 4-step block covers all families); M3 eligibility and M5 enforcement assigned by seeded `crc32(episode_id)`, stratified within family, independent of the masked alternation. Disclose the inherited v10.0 aliasing (§12.13). Tests: **P-SCHED-1** (simulate full 2,000×16 schedule; each family's share of {recon, colar, reflection, DA-enforcement} within ±0.10 of target), **P-SCHED-2** (per-family enforced fraction 0.5±0.05; enforcement⟂masking).

**2. M4 instrument + floors.** "Prefix attention mass" from gate values is a data-independent tanh bijection (separate softmaxes, `modeling_hlwm.py:557-566`) — passes at init forever under a frozen gate. Define **r_l = median over 32 diag rows of ‖g·prefix_attended‖ / (‖attended_word‖+‖g·prefix_attended‖)** per layer; floor = **median over layers 8-28** (plan §A3 wording; per-layer floors would no-go a correct band prediction). Go/no-go: step-600 median r_l ≥ 0.02 AND per-layer gate grad-norm EMA ≥ 1e-7 (grads at `:3988`); step-1200 same floor AND median |tanh g − tanh g_init| ≥ 0.005 (layers 8-28) AND gate-zeroed student-CE delta on the same diag rows reported (content check). Touch: `train_kaggle.py:3491` region + diag pass. Tests: **P-M4-1** (init r_l ≥ 2×floor under M2 init; forced gate-zero → r_l < 0.005), **P-M4-2** (simulated frozen gate fails movement criterion; drifting gate passes).

**3. M2 band init, resume-safe.** Per-layer init list in `HLWMConfig`, consumed inside `__init__` (`modeling_hlwm.py:437-442`) so state_dict load always wins; no post-construction application anywhere. In-band 0.08, out-of-band 0.05. Tests: **P-BAND-1** (init values per band), **P-RESUME-1** (build → perturb gates → save → rebuild+load → bit-equality).

**4. M3 reflection CE (w=0.25), isolation-preserving.** Reflection forward runs **recon-style — no prompt** (`modeling_hlwm.py:2583` pattern), cue-embedding override on `student_channel_teacher_force` (~`:2472`); substitutes **every 4th recon event, masked rows only, main phase only** (recon retains 6/8 slots = 25% reduction of recon pressure, not 50%; warm ships byte-identical to v10.0). Never elicited at eval; task-state serialization content only. Tests: **P-REFL-1** (variant build/tokenization), **P-REFL-2** (reflection forward's visible ids contain zero premise tokens), **P-REFL-3** (substitution = 25% of recon events, masked rows only, under the item-1 schedule), **P-REFL-4** (zero reflection variants for step < warm_steps).

**5. M5 teacher-side DA, hardened.** Builder derives `da_modes` (build-error on unmatched); new `da_masks.py`; enforcement on hash-selected 50% of Group-A rows (item 1). Guards: (a) **P-DA-2** runs the exact runtime collation over the full corpus and asserts every query row has ≥1 allowed key under DA∧causal∧key_valid AND chunk-mask token ids round-trip to derivation text (checksum) — kills the SDPA-NaN and padded-geometry (v9) classes; (b) runtime finite-check on teacher pre-answer states **before** `_update_distill_ema` (`:3400`): skip row, count, disable on repeat; (c) **EMA update skipped on enforced rows**; (d) disable decision moved **600→900** (warm has no Group-A CE, so n=0 at 600): fires if paired within-family enforced-vs-plain CE EMA gap (steps 600-900, ~2,400/arm, ~7σ at 0.2 nats) ≥ 0.2 nats OR normalized pre-answer-state L2 divergence OR enforced-step wall-clock tax > 0.5 s/step OR repeated non-finite rows. Tests: **P-DA-1** (exhaustive mode derivation), **P-DA-2** (above), **P-DA-3** (injected all-False row → caught, skipped, counted), **P-DA-4** (EMA bit-identical with enforced rows present).

**6. Audit deltas (`evaluate_v10.py:115/:211/:1250+`).** (a) Gold-injection arm on the **numeric+unit masked-core stratum only (n=80)** — abstention rows would be lost by a healthy reader; **diagnostic: gate_notes, never in `passed = all(...)` (:1614) or CHANNEL_GATE_KEYS**; preflight **P-GOLD-1** (oracle `torch.equal` injection identity). (b) Donor-key flip regrade = gate_notes diagnostic columns (donor_key_match_rate, regraded acc); **latent_channel_live stays original-key-graded, unchanged**; promotion requires a new preregistered MDE. (c) Expert arms 128→**32 rows/family** (384 decodes). (d) Unmasked-parity capped at a **fixed 160-row unmasked stratum** mirroring the masked-core ordering rule. (e) **P-GATE-WIN**: blocking check asserting shipped n's match the §12 MDE table for every new/changed gate key.

**7. Smoke + ledger machinery (`run_v10_training`, notebook).** Smoke 101 steps must: remeasure s/step and per-arm s/row (blocking, recomputes ledger + audit fit); assert ≥1 enforced Group-A micro-batch executed (**P-SMOKE-1**) and log enforced vs plain s/step separately (T4 SDPA math-path tax). Train-side cut ladder (item 12) implemented as code, decided pre-warm. J-occupancy diag every **512** steps, skipped when projected main < floor+40.

**8. Phase-0 prep artifact (separate session; item 11).** `scripts/build_j_dictionary.py` → `j_lens.npz` (V[l][t] fp16, Q_l, band, corpus sha, **screen verdict + thresholds + null draws**). Preregistered screen (fail-closed): PASS iff (i) band kurtosis AND top-k agreement > 95th percentile of R≥100 token-shuffled + layer-permuted null draws on ≥2 of 3 concept sets {digits, operators, units}; (ii) split-half top-k band agreement ≥ 0.7 (else INCONCLUSIVE → off); (iii) **digit-token median workspace loading ≥ 25th percentile of the all-token distribution** (P2 §3.4 numbers-outside-workspace hazard vs our numeric go/no-go). n ≥ 20k corpus positions. Verdict flips `j_align_enabled` at build time; manifest covers the npz; notebook verifies pre-GPU. Test: **P-J5** (planted band → PASS; shuffled → FAIL; unstable halves → INCONCLUSIVE).

**9. Conditional M1 (screen PASS only).** `KVPrefixProjector.slot_residual_writes()`; **hinge form L_jalign = max(0, 0.3 − ratio)**, values only, w=0.05 — gradient dies once minimally aligned, ending the directed-erasure (v8 KL-anchor) pressure on out-of-span numeric content; ortho-component-norm telemetry wired to M5-style auto-disable. J2 logistic/argmax legs + J-perp arm (64 rows, diagnostic) same condition. Tests: **P-J1..P-J4** (shape/liveness-at-init/no-op-off), **P-J6** (grad exactly 0 at ratio ≥ 0.3).

**10. Closure.** §12 appended to `reports/hlwm-v10-plan-2026-09-03.md` (no silent edits); all mechanisms flag-gated; **P-IDENT-1** (all flags off → bit-exact v10.0 step outputs); deterministic zip; full suite = 140 + ~35 new, all green.

**Compression rule:** if build overruns 1 day, ship items 1-4, 6(a,b,d), 7 (zero prep dependency); M1/J-arms/M5 degrade to the preregistered v10.1 contingency, disclosed.

## 2. Prep-session procedure (item 8; before bundle freeze)

Second Kaggle account (~30h free), single T4, **outside** the 12h session: 1.5-2h to accumulate ≥20k corpus positions of per-layer J estimates on frozen Qwen3-0.6B + run the screen + nulls; write `j_lens.npz`. Local alternative requires venv rehydration first (iCloud eviction, known). Rebuild bundle consuming the npz; `code_sha256` verification precedes any GPU commitment.

## 3. Amended gate list (margins; all other v10.0 gates byte-identical)

| Gate | Criterion / margin | Status |
|---|---|---|
| warm kill (rung 1) | EM ≥ 0.50 @600, full recon pressure | UNCHANGED (warm byte-identical) |
| step-1200 go/no-go | masked numeric EM ≥ 0.05 | UNCHANGED |
| latent_channel_live | paired LCB(full−shuffled) > 0, original-key grading | UNCHANGED; regrade never feeds it |
| M4 liveness go/no-go | @600: median r_l(8-28) ≥ 0.02 AND gate-grad EMA ≥ 1e-7; @1200: + median Δtanh ≥ 0.005; CE-delta reported | NEW (replaces vacuous mass floor) |
| gold_injection_beats_floor | paired one-sided 90% LCB > 0, numeric+unit core n=80, MDE ≈ 0.10 | NEW, **diagnostic only** (gate_notes) |
| M5 auto-disable | @900: gap ≥ 0.2 nats / state-L2 / 0.5 s/step tax / non-finite repeat | NEW (safety, not verdict) |
| P-J1 band prediction | median in-band Δtanh(g) − out-of-band Δtanh(g) > 0 (+ r_l form) | RESTATED delta-from-init (M2 voids absolute form) |
| J2 triple / J-perp | conditional on screen; never pass-blocking, never in CHANNEL_GATE_KEYS | per merge |

Rule: any new/changed gate ships winnability arithmetic (P-GATE-WIN blocks the build without it).

## 4. Revised session ledger (per seed; honest totals)

| Phase | Cost |
|---|---|
| smoke 101 (blocking remeasure) | 0.45-0.55h |
| warm 600 | 2.17-2.83h |
| main (wall-clock-bounded; target 1,400; **binding floor 1,200**; 2,000 total = target, not floor) | remainder |
| w/o-L1 branch | 1.0h (first cut) |
| harvest | 1.0h |
| audit envelope | 3.0h (→2.5h at cut 2) |
| verdict+packaging | 0.4h |

Full stack at 2,000 steps = **13.1-15.4h > 12h** — disclosed; main is wall-clock-bounded by construction. **Train-side cut ladder (smoke-decided, pre-warm):** s/step > 12.5 → drop w/o-L1 branch (+212-277 steps); > 13.5 → audit 3.0→2.5h via named forfeits; > 15.5 → floor unpayable, abort to §6 branch (fully-powered abstention audit + plain-LoRA control) before warm commits. Line items: M1 +≤0.3 s/step (~7 min); J-occupancy 4×32 rows (~2-3 min); M5 tax bounded by its own disable.
Audit inventory (smoke s/row is the arbiter; planning band 4.5-18.5 s/row): pools 576×9.8s = 1.57h; core arms 6×160; gold-injection 80 (~25 min at 18.5); J-perp 64 (conditional, ~20 min); expert 3×128; unmasked-parity 2×160; solvent at ≤~4.7 s/row. **Audit cut order over real arm keys only: J-perp → gold-injection 80→48 → slot_ablation 160→96; never pause/shuffled/floor/full.**

## 5. §12 disclosure list

1. M2 out-of-band init 0.05 (merge deviation from A's 0.03). 2. L_jalign trains linear span, **hinged at τ=0.3**; nonneg-cone = telemetry. 3. Frozen-base J-dictionary; LoRA drift; optional post-train refresh. 4. Screen thresholds/nulls prebound in npz; fail-closed + INCONCLUSIVE→off; non-contiguous band flagged. 5. M5 disable moved 600→900 (600 undecidable, n=0) + hash-stratified assignment + EMA skip on enforced rows. 6. M5 distill-target heterogeneity, bounded by 900 disable. 7. M3 substitutes 2/16 slots = **25% of recon events (6/8 retained)**, masked rows, main phase only. 8. Decode-time DA deferred (P1 scale cliff); `da_masks.py` dormant; **DA never stacks on masking** (D.4 cat-1). 9. P2 digit-weakness headwind + the new digit-loading screen conjunct. 10. J2 diagnostic-only run 1 (routing may bypass broadcast). 11. M4 reframed as §A3 prereg-vs-code fix; single-scalar shipped state + tanh-bijection of any gate-value "mass" noted; instrument = contribution ratio + movement. 12. v10.1 triggers preregistered. 13. **Inherited defect disclosed: shipped v10.0 recon/CoLaR step-parity aliased with the period-4 family rotation; fixed by block-parity.** 14. Honest ledger totals + both cut ladders + phantom-arm correction (no gist/memory-slice arms exist). 15. gold-injection stratum restriction + diagnostic placement; donor-key regrade non-feeding rule. 16. Expert arms 32/family; unmasked-parity fixed 160-row stratum. 17. P-J1 absolute-level reading void under M2 (delta form binding).

Amendment target: `/Users/karthik/Documents/ChatGPT/Light of the Embel/reports/hlwm-v10-plan-2026-09-03.md` §12. Touch-points: `artifacts/kaggle/hlwm-v10.0/bundle/hlwm_kaggle/modeling_hlwm.py` (:194/:285/:437-442/:557-566/:1479/:2472/:2583), `train_kaggle.py` (:3255/:3321-3322/:3400/:3491/:3585/:3776+/:3988), `evaluate_v10.py` (:103/:115/:211-217/:1250-1360/:1614), new `da_masks.py`, `j_lens.py`, `scripts/build_j_dictionary.py`.

---

## Merge decision (for provenance)

# HLWM v10.5 INTEGRATION DECISION (merge lead, 2026-09-04)

**Verification notes before ruling.** (1) The skeptic's claim that per-layer gate telemetry "already exists" at train_kaggle.py:3486 is **wrong**: the code computes per-layer medians then collapses them to one scalar `prefix_gate_median`; the :4017 region is the preflight gate-closed check, not training snapshots. Design A's M4 is therefore a genuine prereg-vs-code fix (plan §A3 requires per-layer liveness floors). (2) Plan §A4 ledger v2 confirmed: warm 600 + wall-clock-bounded main (floor 1,200 steps), w/o-L1 branch 1.0h funded by 1,600→1,400, 101-step smoke with blocking s/step + s/row gates, 3h audit envelope, fixed 160-row masked stratum. (3) Plan ends at §11 → amendment is §12.

## 1. Resolution: LATENT-FIRST. Design A is the base; Design B's decode-time DA and dual-channel adjudication are deferred to HLWM-2-DA.

- **B's unique risk is disqualifying for a single-session program.** B's R2 is self-diagnosed: the DA mask override is the only proposed mechanism that can silently corrupt the *other* channel's verdict (the v9 padded-harvest class, now with adjudication stakes). After rungs terminating 10 studies, we do not ship a cross-channel corruption vector in the same session that decides the latent rung.
- **B's DA side is unwinnable by construction at 0.6B.** P1's measured cliff (29% relative accuracy, 58% parse adherence at ~4B, monotonic) sits ~7× above our scale; preregistering `da_channel_live` at n=160 repeats the v6.0 unwinnable-headline error, and B's own 4-cell table caps a DA win at "promote HLWM-2-DA" — a conclusion reachable without spending 23 build-hours and ~30 audit-min/seed.
- **B halves nothing but risks everything on distillation.** Its dual-use teacher makes ~50% of distill targets enforcement-conditioned — heterogeneity injected into the primary channel's core loss for the benefit of a predicted-dead comparator. A's M5 takes the identical teacher-side mechanism (same 50% enforced rows, same 0.2-nat auto-disable) *without* decode-time DA, marker tokens, or a comparative verdict, so the heterogeneity buys distillation-target structure instead of an adjudication.
- **The user's directive is satisfied latent-first.** Both papers are incorporated as architecture + training pressure (P2: M1/M2/M3 + J instruments; P1: M5 enforced-teacher variant + dormant `da_masks.py` warm-startable fallback), each with blocking CPU preflights — "incorporated and heavily tested," not merely cited. The skeptic's zero-code-change verdict is overridden by the directive, but its tiering is honored: the one mechanism premised on an unverified 0.6B structure (M1) becomes **conditional on a pre-launch existence screen**, and decode-time DA lands exactly where the skeptic put it (HLWM-2).

## 2. Adopt/defer per candidate

**ADOPTED (unconditional):**
- **A.M3 counterfactual-reflection CE (w=0.25, substitutes recon every 4th micro-batch)** — the only candidate that attacks the ten-study signature *causally* (CE through the real generation path conditioned on slots as sole premise source); gradient structurally live (CE on real tokens through trained projector), zero wall-clock, and independent of J-space existence, so the skeptic's scale objection doesn't reach it.
- **A.M4 per-layer gate/prefix-mass telemetry + median-r_l ≥ 0.02 go/no-go floors at 600/1200** — verified prereg-vs-code gap; launch-blocking fix, and it makes M2's falsifiable band prediction and the skeptic's free P2 test actually observable.
- **A.M5 teacher-side DA + dormant `da_masks.py`** — the honest P1 incorporation at a scale where decode-time DA is measured dead: structure enters via distillation targets, FLOPs unchanged, decidable 0.2-nat auto-disable at step 600, fallback warm-startable (G/H salvage lesson).
- **A.M2 band-biased gate init, amended: out-of-band init 0.05 (not 0.03)** — cheap, bias-not-restriction, preregisters a falsifiable P2 prediction; out-of-band raised to half the v10 default (rather than ~⅜) to bound the downside if 0.6B has no band, per the skeptic's "none at all" citation. Disclosed as a merge-lead deviation from Design A.
- **A gold-injection arm + `gold_injection_beats_floor` gate + 3-way writer/reader table** — zero prep dependency, directly separates writer-bottleneck from interface-dead (the v6.0 diagnosis class); winnability arithmetic per B's P-DC6 discipline required before the gate is preregistered.
- **A donor-key flip regrade** — zero decodes, kills the family-ID-relay confound that has haunted shuffled-arm interpretation since v9.
- **J-dictionary prep artifact (`build_j_dictionary.py` + `j_lens.npz`)** — built and screened BEFORE launch; doubles as the skeptic's precondition (b) moved pre-launch, which is exactly what a prep artifact is for.

**ADOPTED CONDITIONAL on the pre-launch J-existence screen** (band-selection nonrandomness per P2 §4.2 kurtosis + top-k-agreement metrics on frozen Qwen3-0.6B; P-J5):
- **A.M1 L_jalign (w=0.05, ratio form, values only)** — live only if the screen passes; otherwise flag off, disclosed. This resolves skeptic finding #1/#2 (don't build a loss on a structure the source paper won't assert at our scale) without abandoning incorporation: the screen is the test.
- **A Gate J2 full triple + J-perp arm + J-occupancy telemetry** — same condition; if screen fails, J2 degrades to free-probe-only diagnostic and J-perp is dropped. J2 stays non-pass-blocking run 1 either way; nothing enters `CHANNEL_GATE_KEYS`.

**DEFERRED:**
- **B decode-time DA (grammar markers in traces, constrained sampling, collator override at decode, arm_da/arm_da_blind/arm_da_nm, DA gate set, `channel_verdict`)** → HLWM-2-DA; unwinnable at 0.6B per P1's own curve, and its mask is a cross-channel corruption vector.
- **B dual-use student-side enforcement (8/16 micro-batches)** → superseded by A.M5's teacher-side-only variant.
- **Skeptic's post-hoc-only stance** → overridden by directive, but its v10.1 triggers (signature reproduction + J-lens screen) are preregistered in §12 as the contingency ladder for anything the screen or the ledger forces out.
- **P2 §7-style ethics/honesty reflection content** → never for this program (skeptic finding #12); M3 uses the *technique* (loss on reflection turn only, never elicited at eval) on task-state serialization.

**INHERITED FROM B into the adopted set:** P-DC6-style winnability arithmetic as a blocking check for every new gate; build-time full-corpus mask sweep (A's P-DA2 ≡ B's P-DC2); the explicit D.4-category-1 statement (DA never stacks on masking) in §12.

## 3. Phased spec

**Phase 0 — prep artifacts (Mac 1–2h or T4 ~15 min, before bundle freeze):** `scripts/build_j_dictionary.py` → `j_lens.npz` (V[l][t] fp16, QR bases Q_l, token ids, band, corpus sha; ~18 MB) with the existence screen verdict written into the artifact; verdict flips `j_align_enabled` in HLWMConfig at build time. Manifest `code_sha256` covers the npz; notebook verifies pre-GPU (v6.0 lesson).

**Phase 1 — core code deltas, dependency order:**
1. M4 telemetry + floors (`train_kaggle.py` :3491 region, go/no-go) — foundation everything else logs through.
2. M2 band init (HLWMConfig + post-construction application; `modeling_hlwm.py` :194/:285/:437) + P-BAND.
3. M3: trace-lib `build_reflection_variant` → collator → `reflect_cue_embedding` (:1479) → `cue_embeds` override on `student_channel_teacher_force` (:2472) → micro-batch substitution in `v10_training_step` → P-REFL-1..4.
4. M5: builder `da_modes` derivation (exhaustive, build-error on unmatched) → `da_masks.py` → teacher-forward `da_allowed_override` on even Group-A rows → enforced/plain CE telemetry + step-600 auto-disable → P-DA1/P-DA2.
5. Audit: gold-injection arm (+oracle `torch.equal` preflight), donor-key flip regrade, free-probe leg of J2; winnability arithmetic for the new gate.

**Phase 2 — conditional on Phase-0 screen:** `KVPrefixProjector.slot_residual_writes()` + L_jalign in `v10_training_step` (P-J1..P-J4), J-logistic/J-argmax legs of J2, J-perp arm (first line in A4 cut order), J-occupancy telemetry every 256 steps.

**Phase 3 — closure:** prereg §12 appended (no silent edits); deterministic zip rebuild + manifest; full suite (140 green + ~30 new: A's ~24, B-inherited winnability/sweep checks, M2-deviation test); every new mechanism flag-gated with v10.0-default bit-exactness certified by identity preflights.

**Compression rule (if build overruns one day):** ship M2/M3/M4 + gold-injection + flip regrade (zero prep dependency); M1/J-arms/M5 degrade to the preregistered v10.1 contingency, disclosed — Phase 1 alone still satisfies "papers incorporated and tested" (M3+M2 are P2; the dormant grammar+masks with P-DA2 passing are P1).

## 4. Revised session ledger (against §A4 v2)

- Smoke 101 steps (blocking s/step + s/row remeasure; ledger recomputed) — unchanged.
- Warm 600 + main wall-clock-bounded, floor 1,200; w/o-L1 branch 1.0h funded by 1,600→1,400 — unchanged. M3 substitutes (no added passes); M5 is a mask on an existing forward; M1 adds <0.3 s/step band matmuls **priced into the smoke remeasure**, and the >13 s/step alternating-schedule trigger already governs.
- Telemetry: +J-occupancy 32-row diag every 256 steps ≈ 4–6 min total; absorbed by the wall-clock bound (main phase stops at budget, so cost is ~15–20 fewer steps, above floor).
- Audit: +gold-injection ~12 min/seed, +flip regrade ~0 (no decodes), +J-perp ~15 min/seed conditional; total delta ≤0.6h/seed inside the 3h envelope; cut order = J-perp → gist → memory-slice (named forfeits stand).
- Net: plan remains ~9.5–10.5h vs 12h quota; no step-floor or row-budget change.

## 5. Disclosure list for §12

1. Merge-lead deviation: M2 out-of-band init 0.05, not Design A's 0.03 (rationale: skeptic's no-band-at-0.6B possibility).
2. L_jalign trains linear span; P2's nonneg-cone reading reserved for telemetry.
3. J-dictionary is frozen-base; effective Jacobian drifts under LoRA (optional post-train refresh as robustness row).
4. All J-mechanisms conditional on the Phase-0 existence screen; screen verdict + metrics disclosed either way; fallback band 9–22 flagged if metric-selected band is non-contiguous.
5. M5 auto-disable event (if fired) + abstention-family rows are all-L.
6. M5 distill-target heterogeneity (50% enforcement-conditioned teacher states), bounded by the step-600 disable.
7. M3 substitutes 4/16 recon micro-batches (recon pressure reduced 25%).
8. Decode-time DA deferred with P1 scale-cliff citation; dormant `da_masks.py` shipped untrained.
9. P2 digit-weakness caveat (§3.4) as headwind on arithmetic anchors.
10. Skeptic's selectivity concern: deterministic family routing may legitimately bypass workspace broadcast — J2 therefore diagnostic, never pass-blocking, run 1.
11. M4 reframed as prereg-vs-code fix (plan §A3), with the verified single-scalar shipped state noted.
12. v10.1 trigger conditions (signature reproduction; screen outcome) preregistered now.

Amendment target: `/Users/karthik/Documents/ChatGPT/Light of the Embel/reports/hlwm-v10-plan-2026-09-03.md` §12. Code touch-points: `artifacts/kaggle/hlwm-v10.0/bundle/hlwm_kaggle/modeling_hlwm.py` (:194/:285/:437/:1276/:1479/:2472), `train_kaggle.py` (:3218/:3491/:3942), `evaluate_v10.py` (:115/:865), new `j_lens.py`, `da_masks.py`, `scripts/build_j_dictionary.py`.