"""Load an exported HLWM adapter and run the complete public generation path."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from .modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration
except ImportError:
    from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration


SAFE_FALLBACK = "I cannot provide a sufficiently supported answer."


def resolve_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--canvas-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def public_prompt(user_request: str) -> str:
    # Version 8.0: the response cue is no longer part of the prompt; the model
    # appends it internally from ``config.response_cue_ids`` (adapter metadata).
    return "\n".join(
        (
            "### Instruction",
            user_request.strip(),
            "### Relevant context",
            "- None supplied",
            "### Constraints",
            "- Answer directly.",
            "- Do not invent evidence.",
            "### Response requirements",
            "- Return only the answer.",
            "Do not repeat these instructions. Return only the answer.",
        )
    )


def encode_preserving_ends(
    tokenizer: Any, text: str, max_length: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) > max_length:
        if max_length == 1:
            ids = ids[:1]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        head = max(1, min(max_length - 1, int(round(max_length * 0.70))))
        ids = ids[:head] + ids[-(max_length - head) :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def load_adapter(
    adapter: Path, device: torch.device
) -> tuple[HLWMForConditionalGeneration, Any, Dict[str, str]]:
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    with safe_open(str(adapter), framework="pt", device="cpu") as stream:
        metadata = dict(stream.metadata() or {})
    if metadata.get("format") != "hlwm-adapter-v5":
        raise RuntimeError("unsupported adapter format: %r" % metadata.get("format"))
    config_values = json.loads(metadata["hlwm_config"])
    allowed = {item.name for item in fields(HLWMConfig)}
    overrides = {name: value for name, value in config_values.items() if name in allowed}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        metadata["base_model"],
        revision=metadata["base_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    amp_dtype = resolve_amp_dtype()
    model = HLWMForConditionalGeneration.from_pretrained(
        metadata["base_model"],
        revision=metadata["base_revision"],
        hlwm_overrides=overrides,
        dtype=amp_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    model.unfreeze_language_tail(int(metadata.get("unfreeze_tail_layers", "0")))
    model.to(device=device, dtype=amp_dtype)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    state = load_file(str(adapter), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError("unexpected adapter tensors: %s" % unexpected)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(name for name in missing if name in trainable)
    if missing_trainable:
        raise RuntimeError("adapter is missing trainable tensors: %s" % missing_trainable)
    return model.eval(), tokenizer, metadata


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HLWM inference currently expects a CUDA GPU")
    if min(args.context_tokens, args.canvas_tokens, args.max_new_tokens) <= 0:
        raise ValueError("token budgets must be positive")
    device = torch.device("cuda")
    model, tokenizer, metadata = load_adapter(args.adapter, device)
    encoded = encode_preserving_ends(
        tokenizer, public_prompt(args.prompt), args.context_tokens, device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=resolve_amp_dtype()):
        result = model.generate_hlwm(
            encoded["input_ids"],
            encoded["attention_mask"],
            canvas_length=args.canvas_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            generator=generator,
        )
    candidate = tokenizer.decode(
        result.candidate_ids[0].detach().cpu(), skip_special_tokens=True
    ).strip()
    print(
        json.dumps(
            {
                "adapter_step": int(metadata["step"]),
                "decision": result.decision,
                "answer": candidate if result.decision == "publish" else SAFE_FALLBACK,
                "answer_source": (
                    "verified_candidate" if result.decision == "publish" else "policy_fallback"
                ),
                "candidate_for_audit": candidate,
                "commit_probability": result.commit_probability,
                "risk_probability": result.risk_probability,
                "verifier_error_probability": result.verifier_error_probability,
                "private_verifier_error_probability": (
                    result.private_verifier_error_probability
                ),
                "policy_thresholds": {
                    "commitment": model.config.commitment_threshold,
                    "risk": model.config.risk_threshold,
                    "verifier_error": model.config.verifier_error_threshold,
                },
                "reverse_timesteps": result.workspace.reverse_timesteps[0].cpu().tolist(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
