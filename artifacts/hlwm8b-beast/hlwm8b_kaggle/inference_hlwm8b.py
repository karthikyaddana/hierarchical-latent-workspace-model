from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .checkpointing_hlwm8b import load_sidecar, verify_checkpoint
    from .evaluate_hlwm8b import load_backbone
    from .modeling_hlwm8b import HLWM8BConfig, QwenHLWM
except ImportError:
    from checkpointing_hlwm8b import load_sidecar, verify_checkpoint
    from evaluate_hlwm8b import load_backbone
    from modeling_hlwm8b import HLWM8BConfig, QwenHLWM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and verify one HLWM answer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--force-hlwm", action="store_true")
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def calibration_path(checkpoint: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        checkpoint / "commitment-calibration.json",
        checkpoint.parent / "commitment-calibration.json",
        checkpoint.parent / "evaluation" / "commitment-calibration.json",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError("commitment-calibration.json is required for publish gating")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B inference requires CUDA")
    verify_checkpoint(args.checkpoint)
    config = HLWM8BConfig(
        **json.loads((args.checkpoint / "hlwm-config.json").read_text(encoding="utf-8"))
    )
    calibration = json.loads(
        calibration_path(args.checkpoint, args.calibration).read_text(encoding="utf-8")
    )
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.base_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": args.prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    base = load_backbone(config.base_model, config.base_revision, args.disable_4bit)
    adapted = PeftModel.from_pretrained(base, args.checkpoint / "adapter")
    model = QwenHLWM(adapted, config)
    load_sidecar(args.checkpoint, model)
    model.sidecar.to("cuda", dtype=torch.float32)
    model.eval()
    input_ids = encoded.input_ids.cuda()
    mask = encoded.attention_mask.cuda()
    generated = model.generate_hlwm(
        input_ids,
        mask,
        max_new_tokens=args.max_new_tokens,
        force_hlwm=args.force_hlwm,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    candidate_ids = generated["generated_ids"]
    verification = model.verify_candidate(
        input_ids,
        mask,
        candidate_ids,
        torch.ones_like(candidate_ids),
    )
    candidate = tokenizer.decode(candidate_ids[0], skip_special_tokens=True).strip()
    score = float(verification["verified_score"][0].item())
    threshold = float(calibration["threshold"])
    published = score >= threshold
    result = {
        "published": published,
        "answer": candidate
        if published
        else "I’m not confident enough to publish an answer for this request.",
        "used_hlwm": bool(generated["used_hlwm"][0].item()),
    }
    if args.debug:
        result.update(
            {
                "candidate": candidate,
                "verified_score": score,
                "publish_threshold": threshold,
                "difficulty_probability": float(
                    generated["difficulty_probability"][0].item()
                ),
                "publish_probability": float(
                    verification["publish_probability"][0].item()
                ),
                "risk_probability": float(verification["risk_probability"][0].item()),
                "error_probability": float(verification["error_probability"][0].item()),
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
