"""Verification battery for the three adapter modes (lora, dora, pissa).

Checks that are actually load-bearing for the v11 arm:

  V1 init identity   -- at construction, dora and pissa must reproduce the
                        frozen base model's outputs exactly (delta = 0). LoRA
                        gets this from zero-init `up`; DoRA additionally needs
                        magnitude == ||W||_row; PiSSA needs the residual
                        subtraction to cancel its non-zero SVD init.
  V2 grad liveness   -- after ONE optimizer step (which is what opens `up` in
                        any zero-init scheme), every trainable adapter tensor
                        carries non-zero gradient. Testing this AT init is
                        meaningless: standard LoRA also reads exactly 0 on
                        `down` there, by construction.
  V3 mode separation -- after identical steps from an identical seed, dora and
                        pissa outputs differ from lora. Guards against a mode
                        silently degrading to plain LoRA.
  V4 magnitude live  -- perturbing DoRA's magnitude changes the output, i.e.
                        the gain is wired into the forward path and not dead.

Run: python verify_adapters.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration  # noqa: E402

SEED = 11
MODES = ("lora", "dora", "pissa")


def build(mode: str) -> HLWMForConditionalGeneration:
    torch.manual_seed(SEED)
    config = HLWMConfig.tiny(
        latent_thoughts=3, kv_prefix_slots=2, lora_rank=4,
        lora_tail_layers=2, adapter_mode=mode, lora_dropout=0.0,
    )
    model = HLWMForConditionalGeneration(config)
    model.train()
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    return model


def batch(model: HLWMForConditionalGeneration):
    torch.manual_seed(7)
    ids = torch.randint(0, model.config.vocab_size, (2, 6))
    return ids, torch.ones(2, 6, dtype=torch.long)


def hidden(model, ids, mask):
    return model.backbone(
        inputs_embeds=model.backbone.embed_tokens(ids),
        attention_mask=mask, attention_mode="causal",
    )


def adapters(model):
    """Every LoRAResidual in the backbone, keyed by module path."""
    found = {}
    for name, module in model.backbone.named_modules():
        if hasattr(module, "mode") and hasattr(module, "up") and hasattr(module, "down"):
            if getattr(module, "enabled", False):
                found[name] = module
    return found


failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


# ---- V1: init identity ------------------------------------------------------
print("V1 init identity (dora/pissa must equal frozen base at construction)")
reference = build("lora")
ids, mask = batch(reference)
with torch.no_grad():
    base = hidden(reference, ids, mask)
for mode in ("dora", "pissa"):
    model = build(mode)
    with torch.no_grad():
        out = hidden(model, ids, mask)
    delta = (out - base).abs().max().item()
    check(f"{mode} == base at init", delta < 1e-5, f"max|diff| = {delta:.3e}")

# ---- V2: gradient liveness after one step ----------------------------------
print("\nV2 gradient liveness (after one optimizer step, all adapter tensors live)")
trained = {}
for mode in MODES:
    model = build(mode)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1.0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # scale up so tiny-model gradients clear float noise
        (hidden(model, ids, mask).float().pow(2).mean() * 1e4).backward()
        if step == 0:
            optimizer.step()
    layer = model.backbone.layers[-1].self_attn.q_lora
    tensors = {"up": layer.up.weight, "down": layer.down.weight}
    if mode == "dora":
        tensors["magnitude"] = layer.magnitude
    detail = " ".join(
        f"{n}={0.0 if t.grad is None else t.grad.abs().sum().item():.3e}"
        for n, t in tensors.items()
    )
    live = all(t.grad is not None and t.grad.abs().sum().item() > 0 for t in tensors.values())
    check(f"{mode} all adapter grads non-zero", live, detail)
    # every enabled adapter in the model, not just the probe site
    dead = [
        name for name, module in adapters(model).items()
        for tensor in (module.up.weight, module.down.weight)
        if tensor.grad is None or tensor.grad.abs().sum().item() == 0
    ]
    check(f"{mode} no dead adapter site", not dead, f"{len(dead)} dead of {2 * len(adapters(model))}")
    with torch.no_grad():
        trained[mode] = hidden(model, ids, mask).clone()

# ---- V3: mode separation ----------------------------------------------------
print("\nV3 mode separation (dora/pissa must not collapse to plain lora)")
for mode in ("dora", "pissa"):
    delta = (trained[mode] - trained["lora"]).abs().max().item()
    check(f"{mode} differs from lora after training", delta > 1e-6, f"max|diff| = {delta:.3e}")

# ---- V4: DoRA magnitude is wired into the forward path ----------------------
print("\nV4 DoRA magnitude liveness (perturbing magnitude changes the output)")
model = build("dora")
with torch.no_grad():
    before = hidden(model, ids, mask).clone()
    for module in adapters(model).values():
        module.magnitude.mul_(1.05)
    after = hidden(model, ids, mask)
delta = (after - before).abs().max().item()
check("magnitude perturbation moves output", delta > 1e-4, f"max|diff| = {delta:.3e}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
