from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

try:
    from .checkpointing_hlwm8b import load_sidecar, verify_checkpoint, write_json
    from .data_hlwm8b import HLWM8BCollator, TeacherPairDataset, find_split
    from .energy_hlwm8b import EnergyMeter
    from .modeling_hlwm8b import HLWM8BConfig, QwenHLWM
    from .semantic_hlwm8b import grade_row
except ImportError:
    from checkpointing_hlwm8b import load_sidecar, verify_checkpoint, write_json
    from data_hlwm8b import HLWM8BCollator, TeacherPairDataset, find_split
    from energy_hlwm8b import EnergyMeter
    from modeling_hlwm8b import HLWM8BConfig, QwenHLWM
    from semantic_hlwm8b import grade_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External Qwen versus HLWM checkpoint audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-answer-tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--disable-4bit", action="store_true")
    return parser.parse_args()


def compact_prompt(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = input_ids[0][attention_mask[0].bool()]
    if not tokens.numel():
        raise ValueError("prompt is empty")
    return tokens.unsqueeze(0), torch.ones_like(tokens).unsqueeze(0)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def aggregate_records(records: Sequence[Mapping[str, Any]], energy: Mapping[str, float], peak_gb: float) -> Dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in records]
    correct = [bool(row["correct"]) for row in records]
    token_count = sum(int(row["generated_tokens"]) for row in records)
    elapsed = sum(latencies)
    result: Dict[str, Any] = {
        "samples": len(records),
        "accuracy": sum(correct) / max(1, len(correct)),
        "prompt_leak_rate": sum(bool(row["prompt_leak"]) for row in records) / max(1, len(records)),
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 0.95),
        "generated_tokens": token_count,
        "generated_tokens_per_second": token_count / elapsed if elapsed else 0.0,
        "peak_gpu_allocated_gb": peak_gb,
        "energy_wh": float(energy.get("energy_wh", 0.0)),
        "energy_wh_per_output": float(energy.get("energy_wh", 0.0)) / max(1, len(records)),
        "mean_power_w": float(energy.get("mean_power_w", 0.0)),
    }
    if records and "verified_score" in records[0]:
        result.update(
            {
                "hlwm_usage_rate": sum(bool(row["used_hlwm"]) for row in records) / len(records),
                "mean_verified_score": sum(float(row["verified_score"]) for row in records) / len(records),
                "mean_route_count": sum(int(row["route_count"]) for row in records) / len(records),
            }
        )
    return result


def calibration_for(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, Any]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("calibration requires aligned non-empty scores and labels")
    unique = sorted(set(float(value) for value in scores))
    thresholds = [0.0, 1.0]
    thresholds.extend(unique)
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    candidates = []
    positives = sum(labels)
    negatives = len(labels) - positives
    for threshold in sorted(set(thresholds)):
        predictions = [score >= threshold for score in scores]
        tp = sum(prediction and label for prediction, label in zip(predictions, labels))
        tn = sum((not prediction) and (not label) for prediction, label in zip(predictions, labels))
        fp = sum(prediction and (not label) for prediction, label in zip(predictions, labels))
        fn = sum((not prediction) and label for prediction, label in zip(predictions, labels))
        tpr = tp / positives if positives else 0.0
        tnr = tn / negatives if negatives else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        candidates.append(
            {
                "threshold": threshold,
                "balanced_accuracy": (tpr + tnr) / 2.0 if positives and negatives else 0.0,
                "accuracy": (tp + tn) / len(labels),
                "precision": precision,
                "coverage": (tp + fp) / len(labels),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["precision"],
            row["accuracy"],
            row["threshold"],
        ),
    )
    return {
        **best,
        "records": len(labels),
        "positive_records": positives,
        "negative_records": negatives,
        "passed_research_gate": bool(
            positives
            and negatives
            and best["balanced_accuracy"] >= 0.70
            and best["precision"] >= 0.80
        ),
        "test_split_used": False,
    }


def model_kwargs(disable_4bit: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
    }
    if not disable_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    return kwargs


def load_backbone(model_name: str, revision: str, disable_4bit: bool):
    from transformers import AutoModelForCausalLM

    kwargs = model_kwargs(disable_4bit)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
    except (ImportError, ValueError):
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
    model.eval()
    model.config.use_cache = True
    return model


def selected_rows(dataset: TeacherPairDataset, count: int) -> list[Mapping[str, Any]]:
    indices = dataset.programmatic_indices[: max(1, count)]
    if not indices:
        raise ValueError("evaluation split contains no deterministic programmatic records")
    return [dataset[index] for index in indices]


@torch.no_grad()
def evaluate_base(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    collator: HLWM8BCollator,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()
    meter = EnergyMeter()
    meter.start()
    records = []
    for row in rows:
        batch = collator([row])
        input_ids, mask = compact_prompt(batch["prompt_input_ids"], batch["prompt_attention_mask"])
        input_ids, mask = input_ids.cuda(), mask.cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        sequence = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated = sequence[:, input_ids.shape[1] :]
        answer = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        records.append(
            {
                "id": row["id"],
                "task_type": row.get("task_type"),
                "answer": answer,
                "correct": grade_row(row, answer),
                "prompt_leak": bool(row["prompt"].strip() and row["prompt"].strip() in answer),
                "latency_seconds": latency,
                "generated_tokens": int(generated.shape[1]),
            }
        )
    energy = meter.stop()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return records, aggregate_records(records, energy, peak)


@torch.no_grad()
def evaluate_hlwm(
    model: QwenHLWM,
    rows: Sequence[Mapping[str, Any]],
    collator: HLWM8BCollator,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()
    meter = EnergyMeter()
    meter.start()
    records = []
    for row in rows:
        batch = collator([row])
        input_ids = batch["prompt_input_ids"].cuda()
        mask = batch["prompt_attention_mask"].cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = model.generate_hlwm(
            input_ids,
            mask,
            max_new_tokens=max_new_tokens,
            force_hlwm=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        candidate_ids = generated["generated_ids"]
        candidate_mask = torch.ones_like(candidate_ids)
        verification = model.verify_candidate(
            input_ids,
            mask,
            candidate_ids,
            candidate_mask,
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        answer = tokenizer.decode(candidate_ids[0], skip_special_tokens=True).strip()
        routes = generated["route_indices"]
        valid_routes = routes[routes >= 0]
        records.append(
            {
                "id": row["id"],
                "task_type": row.get("task_type"),
                "answer": answer,
                "correct": grade_row(row, answer),
                "prompt_leak": bool(row["prompt"].strip() and row["prompt"].strip() in answer),
                "latency_seconds": latency,
                "generated_tokens": int(candidate_ids.shape[1]),
                "used_hlwm": bool(generated["used_hlwm"][0].item()),
                "difficulty_probability": float(generated["difficulty_probability"][0].item()),
                "publish_probability": float(verification["publish_probability"][0].item()),
                "risk_probability": float(verification["risk_probability"][0].item()),
                "error_probability": float(verification["error_probability"][0].item()),
                "verified_score": float(verification["verified_score"][0].item()),
                "route_count": int(valid_routes.unique().numel()) if valid_routes.numel() else 0,
            }
        )
    energy = meter.stop()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return records, aggregate_records(records, energy, peak)


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation requires CUDA")
    verify_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = HLWM8BConfig(
        **json.loads((args.checkpoint / "hlwm-config.json").read_text(encoding="utf-8"))
    )
    from peft import PeftModel
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.base_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = HLWM8BCollator(
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_answer_tokens=args.max_answer_tokens,
    )
    validation = TeacherPairDataset(find_split(args.data_dir, "validation"))
    test = TeacherPairDataset(find_split(args.data_dir, "test"))
    validation_rows = selected_rows(validation, args.validation_samples)
    test_rows = selected_rows(test, args.test_samples)

    baseline = load_backbone(config.base_model, config.base_revision, args.disable_4bit)
    baseline_records, baseline_metrics = evaluate_base(
        baseline, test_rows, collator, tokenizer, args.max_new_tokens
    )
    del baseline
    clear_cuda()

    adapted = load_backbone(config.base_model, config.base_revision, args.disable_4bit)
    adapted = PeftModel.from_pretrained(adapted, args.checkpoint / "adapter")
    model = QwenHLWM(adapted, config)
    load_sidecar(args.checkpoint, model)
    model.sidecar.to("cuda", dtype=torch.float32)
    model.eval()
    validation_records, _ = evaluate_hlwm(
        model, validation_rows, collator, tokenizer, args.max_new_tokens
    )
    calibration = calibration_for(
        [float(row["verified_score"]) for row in validation_records],
        [bool(row["correct"]) for row in validation_records],
    )
    hlwm_records, hlwm_metrics = evaluate_hlwm(
        model, test_rows, collator, tokenizer, args.max_new_tokens
    )
    threshold = float(calibration["threshold"])
    for row in hlwm_records:
        row["published"] = float(row["verified_score"]) >= threshold
    published = [row for row in hlwm_records if row["published"]]
    hlwm_metrics.update(
        {
            "publish_coverage": len(published) / max(1, len(hlwm_records)),
            "published_precision": sum(bool(row["correct"]) for row in published)
            / max(1, len(published)),
            "policy_decision_accuracy": sum(
                bool(row["published"]) == bool(row["correct"]) for row in hlwm_records
            )
            / max(1, len(hlwm_records)),
        }
    )
    manifest_path = next(
        path
        for path in [args.data_dir / "manifest.json", *args.data_dir.rglob("manifest.json")]
        if path.exists()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = {
        "frozen_data_ready": bool(manifest.get("ready_for_main_training")),
        "validation_calibration_passed": bool(calibration["passed_research_gate"]),
        "test_not_used_for_calibration": calibration["test_split_used"] is False,
        "hlwm_accuracy_at_least_75pct": hlwm_metrics["accuracy"] >= 0.75,
        "hlwm_not_worse_than_base": hlwm_metrics["accuracy"] >= baseline_metrics["accuracy"] - 0.02,
        "published_precision_at_least_80pct": hlwm_metrics["published_precision"] >= 0.80,
        "no_prompt_leak": hlwm_metrics["prompt_leak_rate"] == 0.0,
        "routing_exercised": hlwm_metrics["mean_route_count"] >= 1.0,
    }
    gate["passed"] = all(gate.values())
    comparison = {
        "quality_delta": hlwm_metrics["accuracy"] - baseline_metrics["accuracy"],
        "median_latency_ratio": hlwm_metrics["median_latency_seconds"]
        / max(1.0e-9, baseline_metrics["median_latency_seconds"]),
        "peak_memory_delta_gb": hlwm_metrics["peak_gpu_allocated_gb"]
        - baseline_metrics["peak_gpu_allocated_gb"],
        "energy_wh_per_output_ratio": hlwm_metrics["energy_wh_per_output"]
        / max(1.0e-9, baseline_metrics["energy_wh_per_output"]),
    }
    report = {
        "status": "candidate_passed" if gate["passed"] else "candidate_failed",
        "checkpoint": str(args.checkpoint),
        "baseline": baseline_metrics,
        "hlwm": hlwm_metrics,
        "comparison": comparison,
        "calibration": calibration,
        "capability_gate": gate,
        "baseline_records": baseline_records,
        "validation_records": validation_records,
        "hlwm_records": hlwm_records,
        "scope": "Deterministic held-out behavior anchors; this is not a broad production benchmark.",
    }
    write_json(args.output_dir / "commitment-calibration.json", calibration)
    write_json(args.output_dir / "capability-gate.json", gate)
    write_json(args.output_dir / "checkpoint-evaluation.json", report)
    print(json.dumps({key: report[key] for key in ("status", "baseline", "hlwm", "comparison", "capability_gate")}, indent=2))


if __name__ == "__main__":
    main()
