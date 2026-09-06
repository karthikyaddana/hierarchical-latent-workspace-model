from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from .checkpointing_hlwm8b import (
        checkpoint_state,
        find_resume_checkpoint,
        load_training_state,
        save_checkpoint,
        write_json,
    )
    from .data_hlwm8b import (
        DeterministicDistributedBatchSampler,
        HLWM8BCollator,
        TeacherPairDataset,
        find_split,
        move_batch,
    )
    from .energy_hlwm8b import EnergyMeter
    from .modeling_hlwm8b import HLWM8BConfig, QwenHLWM
    from .semantic_hlwm8b import grade_programmatic
except ImportError:
    from checkpointing_hlwm8b import (
        checkpoint_state,
        find_resume_checkpoint,
        load_training_state,
        save_checkpoint,
        write_json,
    )
    from data_hlwm8b import (
        DeterministicDistributedBatchSampler,
        HLWM8BCollator,
        TeacherPairDataset,
        find_split,
        move_batch,
    )
    from energy_hlwm8b import EnergyMeter
    from modeling_hlwm8b import HLWM8BConfig, QwenHLWM
    from semantic_hlwm8b import grade_programmatic


PINNED_MODEL = "Qwen/Qwen3-8B"
PINNED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable dual-T4 Qwen3-8B HLWM training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/hlwm8b-output"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model", default=PINNED_MODEL)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=17029)
    parser.add_argument("--joint-updates", type=int, default=2200)
    parser.add_argument("--on-policy-updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8.0e-5)
    parser.add_argument("--sidecar-learning-rate", type=float, default=1.6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-updates", type=int, default=50)
    parser.add_argument("--max-runtime-hours", type=float, default=7.5)
    parser.add_argument("--runtime-save-buffer-minutes", type=float, default=20.0)
    parser.add_argument("--save-every-updates", type=int, default=100)
    parser.add_argument("--eval-every-updates", type=int, default=200)
    parser.add_argument("--max-validation-records", type=int, default=24)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--minimum-train-records", type=int, default=1000)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=384)
    parser.add_argument("--max-answer-tokens", type=int, default=192)
    parser.add_argument("--on-policy-new-tokens", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--canvas-tokens", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--expert-bottleneck", type=int, default=96)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=2)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--resume", default="auto", help="auto, off, or an explicit checkpoint path")
    parser.add_argument("--stop-file", type=Path, default=Path("/kaggle/working/HLWM_STOP"))
    parser.add_argument("--hf-repo", default=os.getenv("HF_REPO", ""))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-small-data", action="store_true")
    parser.add_argument("--disable-4bit", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def manifest_for(data_dir: Path) -> Dict[str, Any]:
    candidates = [data_dir / "manifest.json"] + list(data_dir.rglob("manifest.json"))
    for path in candidates:
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if value.get("name") == "hlwm-beast-teacher-snapshot":
                return value
    raise FileNotFoundError("HLWM beast teacher snapshot manifest not found")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_snapshot(data_dir: Path, manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    declared = manifest.get("files") or {}
    for split in ("train", "validation", "test"):
        path = find_split(data_dir, split)
        actual = file_sha256(path)
        entry = declared.get(split) or {}
        expected = str(entry.get("sha256") or "")
        if expected and actual != expected:
            raise ValueError("%s split checksum mismatch" % split)
        expected_rows = entry.get("rows")
        if expected_rows is not None:
            with path.open("r", encoding="utf-8") as stream:
                actual_rows = sum(1 for line in stream if line.strip())
            if actual_rows != int(expected_rows):
                raise ValueError("%s split row-count mismatch" % split)
        digest.update((split + ":" + actual + "\n").encode("utf-8"))
    return digest.hexdigest()


class StopController:
    def __init__(self, stop_file: Path, deadline: float) -> None:
        self.stop_file = stop_file
        self.deadline = deadline
        self.signal_received: Optional[int] = None

    def install(self) -> None:
        def handler(signum, _frame):
            self.signal_received = int(signum)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def reason(self) -> Optional[str]:
        if self.signal_received is not None:
            return "signal-%d" % self.signal_received
        if self.stop_file.exists():
            return "stop-file"
        if time.monotonic() >= self.deadline:
            return "runtime-budget"
        return None


def resolve_hf_resume(repo_id: str, output_dir: Path) -> Optional[Path]:
    token = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        return None
    from huggingface_hub import snapshot_download

    try:
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                token=token,
                allow_patterns=["latest/**"],
                local_dir=output_dir / "hf-resume",
            )
        )
    except Exception as exc:
        print("HF resume unavailable:", type(exc).__name__, str(exc)[:300])
        return None
    latest = root / "latest"
    return latest if (latest / "checkpoint-complete.json").exists() else None


def sync_hf_checkpoint(checkpoint: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        path_in_repo="latest",
        delete_patterns=["latest/*"],
        commit_message="HLWM resumable checkpoint %s" % checkpoint.name,
    )


def resolve_resume(args: argparse.Namespace) -> Optional[Path]:
    if args.resume == "off":
        return None
    if args.resume not in {"", "auto"}:
        path = Path(args.resume)
        if not (path / "checkpoint-complete.json").exists():
            raise FileNotFoundError("resume checkpoint is incomplete: %s" % path)
        return path
    local = find_resume_checkpoint(args.output_dir, args.input_root)
    return local or resolve_hf_resume(args.hf_repo, args.output_dir)


def load_base_model(args: argparse.Namespace, accelerator: Any, resume: Optional[Path]):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "trust_remote_code": False,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if args.disable_4bit:
        kwargs["device_map"] = {"": accelerator.local_process_index}
    else:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = {"": accelerator.local_process_index}
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except (ValueError, ImportError) as exc:
        if kwargs.get("attn_implementation") != "eager":
            print("SDPA load failed; retrying eager attention:", type(exc).__name__)
            kwargs["attn_implementation"] = "eager"
            base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        else:
            raise
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    if resume is not None:
        base = PeftModel.from_pretrained(base, resume / "adapter", is_trainable=True)
    else:
        base = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    return base


def build_hlwm_config(args: argparse.Namespace, resume: Optional[Path]) -> HLWM8BConfig:
    if resume is not None:
        value = json.loads((resume / "hlwm-config.json").read_text(encoding="utf-8"))
        return HLWM8BConfig(**value)
    return HLWM8BConfig(
        base_model=args.model,
        base_revision=args.revision,
        latent_dim=args.latent_dim,
        num_lanes=args.num_lanes,
        canvas_tokens=args.canvas_tokens,
        num_experts=args.num_experts,
        expert_bottleneck=args.expert_bottleneck,
        diffusion_steps=args.diffusion_steps,
        refinement_steps=args.refinement_steps,
        prefix_tokens=args.prefix_tokens,
        num_attention_heads=args.attention_heads,
    )


def trainable_summary(model: QwenHLWM) -> Dict[str, int]:
    base = sidecar = total = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            if name.startswith("sidecar"):
                sidecar += parameter.numel()
            else:
                base += parameter.numel()
    return {"trainable_base_adapter": base, "trainable_sidecar": sidecar, "total": total}


def configure_phase_trainability(model: QwenHLWM, phase: str) -> None:
    if phase == "joint":
        for name, parameter in model.base_model.named_parameters():
            parameter.requires_grad_("lora_" in name)
        for parameter in model.sidecar.parameters():
            parameter.requires_grad_(True)
        return
    if phase != "on_policy":
        raise ValueError("unknown phase %s" % phase)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.sidecar.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.sidecar.context_down,
        model.sidecar.token_down,
        model.sidecar.policy,
        model.sidecar.cheap_difficulty,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def optimizer_for(model: QwenHLWM, args: argparse.Namespace):
    base_parameters = []
    sidecar_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (sidecar_parameters if name.startswith("sidecar") else base_parameters).append(parameter)
    groups = [
        {"params": base_parameters, "lr": args.learning_rate},
        {"params": sidecar_parameters, "lr": args.sidecar_learning_rate},
    ]
    try:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(
            groups, weight_decay=args.weight_decay, betas=(0.9, 0.95)
        )
    except (ImportError, AttributeError):
        return torch.optim.AdamW(
            groups, weight_decay=args.weight_decay, betas=(0.9, 0.95)
        )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: TeacherPairDataset,
    collator: HLWM8BCollator,
    accelerator: Any,
    limit: int,
) -> Dict[str, float]:
    model.eval()
    values = []
    margins = []
    indices = list(range(accelerator.process_index, min(len(dataset), limit), accelerator.num_processes))
    for index in indices:
        batch = move_batch(collator([dataset[index]]), accelerator.device)
        output = model(batch, mode="joint")
        values.append(output["loss"].float())
        margins.append(
            (output["chosen_commit_probability"] - output["rejected_commit_probability"]).float()
        )
    if not values:
        values = [torch.tensor(float("nan"), device=accelerator.device)]
        margins = [torch.tensor(float("nan"), device=accelerator.device)]
    local = torch.stack((torch.stack(values).nanmean(), torch.stack(margins).nanmean()))
    gathered = accelerator.gather(local)
    gathered = gathered.reshape(-1, 2)
    model.train()
    return {
        "validation_loss": float(gathered[:, 0].nanmean().cpu()),
        "validation_commit_margin": float(gathered[:, 1].nanmean().cpu()),
    }


def pad_generated(ids: Tensor, pad_id: int, length: int) -> tuple[Tensor, Tensor]:
    ids = ids[:, :length]
    padding = length - ids.shape[1]
    if padding:
        ids = torch.cat(
            (
                ids,
                torch.full(
                    (ids.shape[0], padding), pad_id, dtype=ids.dtype, device=ids.device
                ),
            ),
            dim=1,
        )
    return ids, (ids != pad_id).long()


@torch.no_grad()
def add_on_policy_candidate(
    unwrapped: QwenHLWM,
    batch: Dict[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    max_answer_tokens: int,
) -> Dict[str, Any]:
    was_training = unwrapped.training
    unwrapped.eval()
    generation = unwrapped.generate_hlwm(
        batch["prompt_input_ids"],
        batch["prompt_attention_mask"],
        max_new_tokens=max_new_tokens,
        force_hlwm=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ids, mask = pad_generated(
        generation["generated_ids"], tokenizer.pad_token_id, max_answer_tokens
    )
    texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
    labels = []
    for task_type, candidate, reference in zip(
        batch["task_types"], texts, batch["chosen_texts"]
    ):
        labels.append(grade_programmatic(task_type, candidate, reference))
    batch["on_policy_ids"] = ids
    batch["on_policy_attention_mask"] = mask
    batch["on_policy_correct"] = torch.tensor(
        labels, dtype=torch.float32, device=ids.device
    )
    batch["on_policy_texts"] = texts
    if was_training:
        unwrapped.train()
    return batch


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = []
    for path in output_dir.glob("hlwm8b-checkpoint-*"):
        if not (path / "checkpoint-complete.json").exists():
            continue
        try:
            update = int(json.loads((path / "checkpoint-complete.json").read_text())["global_update"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        checkpoints.append((update, path))
    for _, path in sorted(checkpoints)[: max(0, len(checkpoints) - max(1, keep))]:
        shutil.rmtree(path)


def run_phase(
    *,
    phase: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataset: TeacherPairDataset,
    validation: TeacherPairDataset,
    collator: HLWM8BCollator,
    tokenizer: Any,
    accelerator: Any,
    args: argparse.Namespace,
    state: Dict[str, Any],
    controller: StopController,
    metrics_path: Path,
) -> Optional[str]:
    phase_updates = args.joint_updates if phase == "joint" else args.on_policy_updates
    state_key = phase + "_microstep"
    start_microstep = int(state.get(state_key, 0))
    total_microsteps = phase_updates * args.gradient_accumulation
    allowed = dataset.programmatic_indices if phase == "on_policy" else None
    sampler = DeterministicDistributedBatchSampler(
        len(dataset),
        args.batch_size,
        start_microstep,
        total_microsteps,
        args.seed + (90_000 if phase == "on_policy" else 0),
        accelerator.process_index,
        accelerator.num_processes,
        allowed_indices=allowed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    configure_phase_trainability(accelerator.unwrap_model(model), phase)
    model.train()
    rolling = defaultdict(lambda: deque(maxlen=50))
    for microstep_offset, raw_batch in enumerate(loader, start=start_microstep):
        batch = move_batch(raw_batch, accelerator.device)
        if phase == "on_policy":
            batch = add_on_policy_candidate(
                accelerator.unwrap_model(model),
                batch,
                tokenizer,
                args.on_policy_new_tokens,
                args.max_answer_tokens,
            )
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = model(batch, mode=phase)
                loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite %s loss at microstep %d" % (phase, microstep_offset))
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        state[state_key] = microstep_offset + 1
        for key, value in output.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                rolling[key].append(float(value.detach().float().cpu()))
        if accelerator.sync_gradients:
            state["global_update"] = int(state.get("global_update", 0)) + 1
            state["phase"] = phase
            update = state["global_update"]
            if accelerator.is_main_process:
                record = {
                    "phase": phase,
                    "global_update": update,
                    "phase_microstep": state[state_key],
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                }
                record.update(
                    {key: sum(values) / len(values) for key, values in rolling.items() if values}
                )
                append_jsonl(metrics_path, record)
            should_eval = update % args.eval_every_updates == 0
            should_save = update % args.save_every_updates == 0
            if should_eval and phase == "joint":
                evaluation = evaluate_loss(
                    model,
                    validation,
                    collator,
                    accelerator,
                    args.max_validation_records,
                )
                if accelerator.is_main_process:
                    append_jsonl(
                        metrics_path,
                        {"phase": phase, "global_update": update, **evaluation},
                    )
                    print(json.dumps({"update": update, **evaluation}, sort_keys=True))
            reason = controller.reason()
            if should_save or reason:
                state["status"] = "paused" if reason else "running"
                state["pause_reason"] = reason
                checkpoint = save_checkpoint(
                    accelerator, model, optimizer, scheduler, args.output_dir, state
                )
                if accelerator.is_main_process:
                    prune_checkpoints(args.output_dir, args.keep_checkpoints)
                    if args.hf_repo:
                        try:
                            sync_hf_checkpoint(checkpoint, args.hf_repo)
                        except Exception as exc:
                            print("HF checkpoint sync failed:", type(exc).__name__, str(exc)[:500])
                accelerator.wait_for_everyone()
            if reason:
                return reason
    return None


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from peft import PeftModel
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision="fp16",
        kwargs_handlers=[ddp],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B HLWM training requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + accelerator.process_index)
    manifest = manifest_for(args.data_dir)
    dataset_fingerprint = verify_dataset_snapshot(args.data_dir, manifest)
    train_path = find_split(args.data_dir, "train")
    validation_path = find_split(args.data_dir, "validation")
    train_data = TeacherPairDataset(train_path, args.max_train_records)
    validation_data = TeacherPairDataset(validation_path)
    data_ready = bool(manifest.get("ready_for_main_training")) and (
        len(train_data) >= args.minimum_train_records
    )
    if not data_ready and not (
        args.allow_small_data or args.preflight_only
    ):
        if accelerator.is_main_process:
            status = {
                "status": "waiting_for_data",
                "train_records": len(train_data),
                "minimum_train_records": args.minimum_train_records,
                "manifest_ready": manifest.get("ready_for_main_training"),
                "accepted_teacher_records": manifest.get("accepted_teacher_records", 0),
                "minimum_teacher_records": manifest.get("minimum_teacher_records"),
            }
            write_json(args.output_dir / "run-status.json", status)
            print(json.dumps(status, indent=2))
        return
    resume = resolve_resume(args)
    if resume:
        print("Resuming from", resume)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = HLWM8BCollator(
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_answer_tokens=args.max_answer_tokens,
    )
    base = load_base_model(args, accelerator, resume)
    config = build_hlwm_config(args, resume)
    model = QwenHLWM(base, config)
    model.sidecar.to(accelerator.device, dtype=torch.float32)
    configure_phase_trainability(model, "joint")
    if accelerator.is_main_process:
        print(json.dumps({"parameters": trainable_summary(model)}, indent=2))
    optimizer = optimizer_for(model, args)
    total_updates = args.joint_updates + args.on_policy_updates
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_updates,
        num_training_steps=total_updates,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    state: Dict[str, Any] = {
        "format": "hlwm8b-trainer-v1",
        "status": "running",
        "phase": "joint",
        "global_update": 0,
        "joint_microstep": 0,
        "on_policy_microstep": 0,
        "joint_updates_planned": args.joint_updates,
        "on_policy_updates_planned": args.on_policy_updates,
        "dataset_manifest": manifest,
        "base_model": args.model,
        "base_revision": args.revision,
        "run_signature": {
            "dataset_fingerprint": dataset_fingerprint,
            "world_size": accelerator.num_processes,
            "batch_size_per_rank": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "joint_updates": args.joint_updates,
            "on_policy_updates": args.on_policy_updates,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_answer_tokens": args.max_answer_tokens,
            "base_model": args.model,
            "base_revision": args.revision,
        },
    }
    if resume is not None:
        loaded_state = load_training_state(resume, accelerator, model, optimizer, scheduler)
        if loaded_state.get("run_signature") != state["run_signature"]:
            raise ValueError(
                "resume settings or frozen dataset do not match the checkpoint run signature"
            )
        state.update(loaded_state)
    metrics_path = args.output_dir / "metrics.jsonl"

    if args.preflight_only:
        batch = move_batch(collator([train_data[0]]), accelerator.device)
        with accelerator.autocast():
            output = model(batch, mode="joint")
        accelerator.backward(output["loss"])
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            report = {
                "status": "passed",
                "loss": float(output["loss"].detach().float().cpu()),
                "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated()
                / (1024**3),
                "route_count": int(output["route_indices"].unique().numel()),
            }
            write_json(args.output_dir / "preflight.json", report)
            print(json.dumps({"preflight": report}, indent=2))
        return

    usable_seconds = args.max_runtime_hours * 3600 - args.runtime_save_buffer_minutes * 60
    if usable_seconds <= 60:
        raise ValueError("runtime budget leaves less than one minute before checkpoint buffer")
    controller = StopController(args.stop_file, time.monotonic() + usable_seconds)
    controller.install()
    meter = EnergyMeter()
    if accelerator.is_main_process:
        meter.start()
    pause_reason = None
    if int(state.get("joint_microstep", 0)) < args.joint_updates * args.gradient_accumulation:
        pause_reason = run_phase(
            phase="joint",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=train_data,
            validation=validation_data,
            collator=collator,
            tokenizer=tokenizer,
            accelerator=accelerator,
            args=args,
            state=state,
            controller=controller,
            metrics_path=metrics_path,
        )
    if pause_reason is None and int(state.get("on_policy_microstep", 0)) < (
        args.on_policy_updates * args.gradient_accumulation
    ):
        pause_reason = run_phase(
            phase="on_policy",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=train_data,
            validation=validation_data,
            collator=collator,
            tokenizer=tokenizer,
            accelerator=accelerator,
            args=args,
            state=state,
            controller=controller,
            metrics_path=metrics_path,
        )
    state["status"] = "paused" if pause_reason else "training_complete"
    state["pause_reason"] = pause_reason
    if pause_reason:
        checkpoint = args.output_dir / (
            "hlwm8b-checkpoint-%06d" % int(state["global_update"])
        )
        if not (checkpoint / "checkpoint-complete.json").exists():
            raise RuntimeError("pause checkpoint was not completed")
    else:
        checkpoint = save_checkpoint(
            accelerator, model, optimizer, scheduler, args.output_dir, state
        )
    if accelerator.is_main_process:
        state["energy"] = meter.stop()
        write_json(args.output_dir / "run-status.json", state)
        print(json.dumps({"run_status": state, "checkpoint": str(checkpoint)}, indent=2))
        prune_checkpoints(args.output_dir, args.keep_checkpoints)
        if args.hf_repo and not pause_reason:
            try:
                sync_hf_checkpoint(checkpoint, args.hf_repo)
            except Exception as exc:
                print("HF checkpoint sync failed:", type(exc).__name__, str(exc)[:500])


if __name__ == "__main__":
    main()
