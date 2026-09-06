#!/usr/bin/env python3
"""Preregistered J-space projection audit of the v10.0 certified checkpoints.

Plan (frozen before execution):
    reports/hlwm-v10.0-jspace-analysis-plan-2026-09-06.md

Question: the certified run's probe reads the withheld premise digit from the
latent thoughts (0.62/0.76 vs 0.129 chance) while masked generation is exactly
0.000. Is the probe-readable content orthogonal to the subspace the decoder's
answer logits are sensitive to (H1, misaligned write), or is the channel dead
(gradient ratio), or is the content inside the sensitive subspace (H1 refuted)?

No training. Reuses the audit harness unmodified. Batch 1 throughout (the
audit's own convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

import glob
import os

PROJECT = Path(__file__).resolve().parent.parent


def _find_bundle() -> Path:
    """Locate the harness package: env override, repo copy, or Kaggle mount.

    Defaults prefer the certified kernel's own code+data copy (exact bytes
    that produced the recorded audit; sha256s in its manifest).
    """
    env = os.environ.get("HLWM_BUNDLE")
    if env:
        return Path(env)
    default = PROJECT / "artifacts/kaggle/hlwm-v10.0/bundle/hlwm_kaggle"
    if (default / "evaluate_v10.py").exists():
        return default
    for candidate in glob.glob("/kaggle/input/*/hlwm-v10.0/hlwm_kaggle"):
        return Path(candidate)
    return default


BUNDLE = _find_bundle()
sys.path.insert(0, str(BUNDLE))

import evaluate_checkpoint as ec  # noqa: E402
import evaluate_v10 as ev  # noqa: E402
from data import Reasoning9000Dataset  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digit_token_ids(tokenizer) -> list[int]:
    ids = []
    for digit in "0123456789":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"digit {digit!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def first_digit_position(tokenizer, gold_ids: list[int]) -> int | None:
    """Index of the first gold token whose decoded text contains a digit."""
    for index, token_id in enumerate(gold_ids):
        if any(ch.isdigit() for ch in tokenizer.decode([token_id])):
            return index
    return None


def probe_on(latents: Tensor, labels: list[int], seed: int) -> dict:
    return ev.train_premise_probe(latents, labels, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--audit-json", required=True,
                        help="recorded v10-audit.json for this seed")
    parser.add_argument("--data-dir", default=str(BUNDLE / "data"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--core-rows", type=int, default=64)
    parser.add_argument("--secondary-rows", type=int, default=160,
                        help="Amendment A1: labelled core rows for the "
                             "mean-projected secondary probes (0 disables)")
    parser.add_argument("--g0-rows", type=int, default=20)
    parser.add_argument("--g0-unmasked-rows", type=int, default=60)
    parser.add_argument("--probe-rows", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    torch.manual_seed(0)

    ckpt_path = Path(args.ckpt)
    result: dict = {
        "plan": "reports/hlwm-v10.0-jspace-analysis-plan-2026-09-06.md",
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "device": str(device),
        "dtype": "float32",
        "deviations": [
            "D1: the plan said Jacobian 'at the first answer position'; answers "
            "begin with non-digit boilerplate (e.g. 'The result is ...'), so the "
            "Jacobian is taken at the first GOLD DIGIT position of the "
            "teacher-forced gold answer. Frozen before any Jacobian was computed.",
            "D2: G0's unmasked sample grows from 20 to 60 rows; at 20 rows the "
            "sampling SE (~0.11) exceeds the +/-0.10 band, so the gate would fail "
            "on noise alone. The masked requirement (exact 0.000 on 20 rows) is "
            "unchanged. Recorded before the confirmatory run; the 4-row smoke that "
            "exposed the power problem decoded 4 masked rows (EM 0.000) and 4 "
            "unmasked rows (0.750, n too small to read).",
            "D3: the frozen substrate runs in bfloat16, not the audit's float16, "
            "to avoid fp16 gradient underflow in the Jacobian stage (trainables "
            "stay fp32, matching the harness). The plan's fp32 intent drowned the "
            "9 GB analysis machine in swap; G0/G1 replication gates validate "
            "parity with the recorded fp16 audit empirically.",
        ],
    }

    print(f"[{args.seed}] loading checkpoint (weights_only, mmap) ...", flush=True)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    result["step"] = payload.get("step")
    result["trainer_state"] = payload.get("trainer_state")

    # CUDA: the audit's own float16 (exact parity with the recorded run).
    # MPS/CPU: bfloat16 (D3) to avoid fp16 gradient underflow off-GPU.
    ec.AMP_DTYPE = torch.float16 if device.type == "cuda" else torch.bfloat16
    result["substrate_dtype"] = str(ec.AMP_DTYPE)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        payload["base_model"], revision=payload["base_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ec.load_hlwm(payload, device)
    model.eval()
    base_model_name, base_revision = payload["base_model"], payload["base_revision"]
    del payload
    result["base_model"], result["base_revision"] = base_model_name, base_revision

    rows = Reasoning9000Dataset(Path(args.data_dir) / "master" / "test.jsonl", num_lanes=1).rows
    audit_rows, legacy_excluded = ev.v10_audit_rows(rows)
    ordered = ev.sorted_by_anchor(audit_rows)
    result["n_rows"] = len(ordered)
    result["legacy_rows_excluded"] = legacy_excluded

    recorded = json.loads(Path(args.audit_json).read_text())
    core_ids: list[str] = list(recorded["masked_core"]["anchor_ids"])
    by_id = {ev.anchor_id(row): row for row in ordered}
    core_rows = [by_id[a] for a in core_ids if a in by_id]
    if len(core_rows) != len(core_ids):
        raise RuntimeError("masked-core anchor ids not all present in local data")
    unmasked_rows = [row for row in ordered if not bool(row.get("masked"))]

    # ---------------- G0: instrument gate ----------------
    print(f"[{args.seed}] G0: greedy decode {args.g0_rows} masked core "
          f"+ {args.g0_rows} unmasked rows ...", flush=True)
    g0_core = ev.prepare_rows(model, tokenizer, core_rows[: args.g0_rows], device=device)
    masked_graded = ev.grade_arm(tokenizer, g0_core, ev.arm_full(model, g0_core, seed=args.seed))
    g0_un = ev.prepare_rows(
        model, tokenizer, unmasked_rows[: args.g0_unmasked_rows], device=device
    )
    unmasked_graded = ev.grade_arm(tokenizer, g0_un, ev.arm_full(model, g0_un, seed=args.seed))
    del g0_un
    recorded_unmasked = float(recorded["unmasked"]["full_accuracy"])
    g0 = {
        "masked_accuracy": masked_graded["accuracy"],
        "unmasked_accuracy": unmasked_graded["accuracy"],
        "recorded_unmasked_accuracy": recorded_unmasked,
        "masked_sample_texts": masked_graded["texts"][:3],
        "passed": bool(
            masked_graded["accuracy"] == 0.0
            and abs(unmasked_graded["accuracy"] - recorded_unmasked) <= 0.10
        ),
    }
    result["G0"] = g0
    print(f"[{args.seed}] G0 masked={g0['masked_accuracy']:.3f} "
          f"unmasked={g0['unmasked_accuracy']:.3f} (recorded {recorded_unmasked:.3f}) "
          f"passed={g0['passed']}", flush=True)
    if not g0["passed"]:
        result["outcome"] = "INSTRUMENT_FAILURE"
        Path(args.out).write_text(json.dumps(result, indent=2))
        return

    # ---------------- G1: probe replication (states) + embeds probe ----------------
    heldout_rows = [
        row for row in ordered
        if bool(row.get("masked")) and ev.anchor_id(row) not in set(core_ids)
    ][: args.probe_rows]
    labelled_rows = [row for row in heldout_rows if ev.probe_digit_label(row) is not None]
    print(f"[{args.seed}] G1: preparing {len(labelled_rows)} held-out probe rows ...",
          flush=True)
    state_means, embed_means, labels = [], [], []
    with torch.no_grad():
        for index, row in enumerate(labelled_rows):
            item = ev.prepare_rows(model, tokenizer, [row], device=device)[0]
            state_means.append(item.thought_states.float().mean(dim=1).cpu())
            embed_means.append(item.thought_embeds.float().mean(dim=1).cpu())
            labels.append(ev.probe_digit_label(row))
            if (index + 1) % 50 == 0:
                print(f"[{args.seed}]   prepared {index + 1}/{len(labelled_rows)}",
                      flush=True)
    probe_states = probe_on(torch.cat(state_means), labels, seed=args.seed + 4)
    probe_embeds = probe_on(torch.cat(embed_means), labels, seed=args.seed + 4)
    recorded_probe = float(recorded["probe"]["probe_accuracy"])
    chance = float(recorded["probe"]["chance"])
    g1 = {
        "recorded_probe_accuracy": recorded_probe,
        "states_probe": probe_states,
        "embeds_probe": probe_embeds,
        "chance": chance,
        "replicated": bool(
            probe_states["probe_accuracy"] is not None
            and abs(probe_states["probe_accuracy"] - recorded_probe) <= 0.07
        ),
        "embeds_carry_content": bool(
            probe_embeds["probe_accuracy"] is not None
            and probe_embeds["probe_accuracy"] >= chance + 0.10
        ),
    }
    result["G1"] = g1
    print(f"[{args.seed}] G1 states={probe_states['probe_accuracy']} "
          f"(recorded {recorded_probe:.3f}) embeds={probe_embeds['probe_accuracy']} "
          f"chance={chance:.3f}", flush=True)
    if not g1["replicated"]:
        result["outcome"] = "INSTRUMENT_FAILURE"
        Path(args.out).write_text(json.dumps(result, indent=2))
        return
    if not g1["embeds_carry_content"]:
        result["outcome"] = "O-PROJ"
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[{args.seed}] outcome O-PROJ: the projection into embeds destroys "
              "the content before the decoder sees it.", flush=True)
        return

    # ---------------- Stage 2: sensitivity + subspace on masked-core rows ----------------
    digit_ids = digit_token_ids(tokenizer)
    core_labelled = [row for row in core_rows if ev.probe_digit_label(row) is not None]
    n_stage = max(args.core_rows, args.secondary_rows)
    stage_rows = core_labelled[:n_stage]
    print(f"[{args.seed}] stage 2: {len(stage_rows)} labelled core rows "
          f"(primary on first {args.core_rows}) ...", flush=True)

    deadness_ratios, s_components, perp_components, flat_embeds, stage_labels = (
        [], [], [], [], []
    )
    mean_s, mean_perp, mean_flat = [], [], []  # Amendment A1 features (1024-d)
    slot_grad_norms = []
    skipped = 0
    embed_module = model.backbone.embed_tokens

    for index, row in enumerate(stage_rows):
        item = ev.prepare_rows(model, tokenizer, [row], device=device)[0]
        target = str(row.get("public_target") or "").strip()
        gold_ids = tokenizer.encode(target, add_special_tokens=False)[:48] if target else []
        digit_pos = first_digit_position(tokenizer, gold_ids) if gold_ids else None
        if digit_pos is None:
            skipped += 1
            continue
        gold = torch.tensor([gold_ids], dtype=torch.long, device=device)
        gold_mask = torch.ones_like(gold)
        route = (
            None if item.route_index is None
            else torch.tensor([int(item.route_index)], dtype=torch.long, device=device)
        )

        thoughts_leaf = item.thought_embeds.detach().clone().requires_grad_(True)
        states_leaf = item.thought_states.detach().clone().requires_grad_(True)
        captured: list[Tensor] = []

        def capture_hook(module, inputs, output):
            leaf = output.detach().requires_grad_(True)
            captured.append(leaf)
            return leaf

        handle = embed_module.register_forward_hook(capture_hook)
        try:
            out = model.student_channel_teacher_force(
                item.masked_ids,
                item.masked_mask,
                thoughts_leaf,
                states_leaf,
                gold,
                gold_mask,
                use_prefix_slots=True,
                route_index=route,
            )
        finally:
            handle.remove()
        prompt_leaf = captured[0]  # first embed_tokens call = the masked prompt
        logits = out["logits"][0].float()  # [T_gold, vocab]

        valid = int(item.masked_mask[0].long().sum())
        baseline_slice = slice(max(0, valid - 6), valid)

        jac_thought, jac_prompt = [], []
        slot_norm_sq = 0.0
        for digit_id in digit_ids:
            grads = torch.autograd.grad(
                logits[digit_pos, digit_id],
                [thoughts_leaf, prompt_leaf, states_leaf],
                retain_graph=True,
                allow_unused=True,
            )
            g_thought = grads[0]
            g_prompt = grads[1]
            g_states = grads[2]
            jac_thought.append(
                torch.zeros_like(thoughts_leaf).flatten() if g_thought is None
                else g_thought.detach().flatten().float()
            )
            jac_prompt.append(
                torch.zeros(6 * thoughts_leaf.shape[-1]) if g_prompt is None
                else g_prompt.detach()[0, baseline_slice].flatten().float().cpu()
            )
            if g_states is not None:
                slot_norm_sq += float(g_states.detach().float().pow(2).sum())
        jac_thought_mat = torch.stack([g.cpu() for g in jac_thought])  # [10, 6144]
        jac_prompt_mat = torch.stack(jac_prompt)  # [10, 6144]

        thought_norm = float(jac_thought_mat.norm())
        prompt_norm = float(jac_prompt_mat.norm())
        deadness_ratios.append(thought_norm / max(prompt_norm, 1e-12))
        slot_grad_norms.append(slot_norm_sq ** 0.5)

        embeds_flat = item.thought_embeds.detach().float().flatten().cpu()  # [6144]
        if thought_norm > 0:
            _, _, vh = torch.linalg.svd(jac_thought_mat, full_matrices=False)
            basis = vh[:8]  # [8, 6144]
            coords = basis @ embeds_flat
            e_s = basis.t() @ coords
        else:
            e_s = torch.zeros_like(embeds_flat)
        e_perp = embeds_flat - e_s
        s_components.append(e_s.unsqueeze(0))
        perp_components.append(e_perp.unsqueeze(0))
        flat_embeds.append(embeds_flat.unsqueeze(0))
        stage_labels.append(ev.probe_digit_label(row))
        width = thoughts_leaf.shape[-1]
        mean_s.append(e_s.reshape(-1, width).mean(dim=0, keepdim=True))
        mean_perp.append(e_perp.reshape(-1, width).mean(dim=0, keepdim=True))
        mean_flat.append(embeds_flat.reshape(-1, width).mean(dim=0, keepdim=True))

        del out, logits, thoughts_leaf, states_leaf, captured, item
        if (index + 1) % 8 == 0:
            print(f"[{args.seed}]   jacobian {index + 1}/{len(stage_rows)} "
                  f"(median ratio so far "
                  f"{torch.tensor(deadness_ratios).median():.4f})", flush=True)

    def acc(p):
        return -1.0 if p.get("probe_accuracy") is None else float(p["probe_accuracy"])

    n_primary = min(args.core_rows, len(deadness_ratios))
    ratios = torch.tensor(deadness_ratios[:n_primary])
    m1 = float(ratios.median())
    probe_s = probe_on(
        torch.cat(s_components[:n_primary]), stage_labels[:n_primary], seed=args.seed + 4
    )
    probe_perp = probe_on(
        torch.cat(perp_components[:n_primary]), stage_labels[:n_primary], seed=args.seed + 4
    )
    probe_flat = probe_on(
        torch.cat(flat_embeds[:n_primary]), stage_labels[:n_primary], seed=args.seed + 4
    )

    stage2 = {
        "n_rows": n_primary,
        "n_skipped_no_digit": skipped,
        "M1_deadness_ratio_median": m1,
        "deadness_ratio_quartiles": [
            float(ratios.quantile(0.25)), m1, float(ratios.quantile(0.75))
        ],
        "slot_grad_norm_median": float(torch.tensor(slot_grad_norms).median()),
        "probe_S_component": probe_s,
        "probe_perp_component": probe_perp,
        "probe_flat_reference": probe_flat,
    }
    result["stage2"] = stage2

    # ---------------- preregistered primary reading ----------------
    if m1 < 0.10:
        outcome = "DEAD_CHANNEL"
    elif acc(probe_s) < chance + 0.10 and acc(probe_perp) >= chance + 0.20:
        outcome = "H1_CONFIRMED"
    elif acc(probe_s) >= chance + 0.20:
        outcome = "H1_REFUTED"
    else:
        outcome = "INCONCLUSIVE"
    result["outcome"] = outcome

    # ---------------- Amendment A1: mean-projected secondary ----------------
    if args.secondary_rows > 0 and len(stage_labels) >= 8:
        sec_s = probe_on(torch.cat(mean_s), stage_labels, seed=args.seed + 4)
        sec_perp = probe_on(torch.cat(mean_perp), stage_labels, seed=args.seed + 4)
        sec_ref = probe_on(torch.cat(mean_flat), stage_labels, seed=args.seed + 4)
        power_ok = acc(sec_ref) >= chance + 0.20
        if not power_ok:
            sec_outcome = "INCONCLUSIVE_UNDERPOWERED_FINAL"
        elif acc(sec_s) < chance + 0.10 and acc(sec_perp) >= chance + 0.20:
            sec_outcome = "H1_CONFIRMED"
        elif acc(sec_s) >= chance + 0.20:
            sec_outcome = "H1_REFUTED"
        else:
            sec_outcome = "INCONCLUSIVE_FINAL"
        result["secondary_A1"] = {
            "n_rows": len(stage_labels),
            "probe_S_mean": sec_s,
            "probe_perp_mean": sec_perp,
            "probe_flat_mean_reference": sec_ref,
            "power_ok": power_ok,
            "outcome": sec_outcome,
        }
        print(f"[{args.seed}] A1 secondary (n={len(stage_labels)}): "
              f"S={acc(sec_s):.3f} perp={acc(sec_perp):.3f} ref={acc(sec_ref):.3f} "
              f"power_ok={power_ok} -> {sec_outcome}", flush=True)

    result["runtime_seconds"] = round(time.time() - started, 1)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[{args.seed}] PRIMARY OUTCOME: {outcome}  M1={m1:.4f}  "
          f"S={acc(probe_s):.3f} perp={acc(probe_perp):.3f} flat={acc(probe_flat):.3f} "
          f"({result['runtime_seconds']}s)", flush=True)


def _kaggle_main() -> None:
    """Run both seeds against the certified-run kernel output mounted as input."""
    roots = glob.glob("/kaggle/input/*/hlwm-v10.0-seed-17")
    if not roots:
        raise SystemExit("certified-run kernel output is not mounted as an input")
    root = Path(roots[0]).parent
    for seed in (17, 29):
        sys.argv = [
            "jspace_projection_audit",
            "--seed", str(seed),
            "--ckpt", str(root / f"hlwm-v10.0-seed-{seed}/checkpoint-step-001200.pt"),
            "--audit-json", str(root / f"hlwm-v10.0-seed-{seed}/v10-audit.json"),
            "--device", "cuda",
            "--out", f"/kaggle/working/jspace-{seed}.json",
        ]
        main()


if __name__ == "__main__":
    if Path("/kaggle/input").exists() and len(sys.argv) == 1:
        _kaggle_main()
    else:
        main()
