from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_checkpoint(path: Path) -> Dict[str, Any]:
    manifest_path = path / "checkpoint-complete.json"
    if not manifest_path.exists():
        raise FileNotFoundError("incomplete checkpoint: %s" % path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "hlwm8b-resumable-v1",
        "hlwm8b-deliverable-v1",
    }:
        raise ValueError("unsupported checkpoint format in %s" % path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint file manifest is empty: %s" % path)
    for entry in files:
        relative = Path(str(entry.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe checkpoint member %r" % str(relative))
        file = path / relative
        if not file.is_file():
            raise FileNotFoundError("checkpoint member is missing: %s" % file)
        if file.stat().st_size != int(entry["bytes"]):
            raise ValueError("checkpoint member size mismatch: %s" % file)
        if sha256(file) != str(entry["sha256"]):
            raise ValueError("checkpoint member checksum mismatch: %s" % file)
    return manifest


def _complete_checkpoints(root: Path) -> list[Path]:
    if not root.exists():
        return []
    result = []
    for manifest in root.rglob("checkpoint-complete.json"):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            update = int(value["global_update"])
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        try:
            verified = verify_checkpoint(manifest.parent)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if verified.get("format") == "hlwm8b-resumable-v1":
            result.append((update, manifest.parent))
    return [path for _, path in sorted(result, key=lambda item: (item[0], str(item[1])))]


def find_resume_checkpoint(output_dir: Path, input_root: Optional[Path] = None) -> Optional[Path]:
    local = _complete_checkpoints(output_dir)
    if local:
        return local[-1]
    if input_root is not None:
        external = _complete_checkpoints(input_root)
        if external:
            return external[-1]
    return None


def checkpoint_state(path: Path) -> Dict[str, Any]:
    verify_checkpoint(path)
    return json.loads((path / "trainer-state.json").read_text(encoding="utf-8"))


def save_checkpoint(
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    trainer_state: Mapping[str, Any],
) -> Path:
    update = int(trainer_state["global_update"])
    final = output_dir / ("hlwm8b-checkpoint-%06d" % update)
    partial = output_dir / (final.name + ".partial")
    if accelerator.is_main_process:
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
    accelerator.wait_for_everyone()

    rank_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(rank_state, partial / ("rng-rank-%03d.pt" % accelerator.process_index))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.base_model.save_pretrained(
            partial / "adapter",
            safe_serialization=True,
        )
        sidecar_state = {
            name: value.detach().cpu().contiguous()
            for name, value in unwrapped.sidecar.state_dict().items()
        }
        save_file(sidecar_state, str(partial / "hlwm-sidecar.safetensors"))
        torch.save(optimizer.state_dict(), partial / "optimizer.pt")
        torch.save(scheduler.state_dict(), partial / "scheduler.pt")
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            torch.save(scaler.state_dict(), partial / "scaler.pt")
        write_json(partial / "trainer-state.json", dict(trainer_state))
        write_json(partial / "hlwm-config.json", unwrapped.hlwm_config.to_dict())
        files = []
        for file in sorted(path for path in partial.rglob("*") if path.is_file()):
            if file.name == "checkpoint-complete.json":
                continue
            files.append(
                {
                    "path": str(file.relative_to(partial)),
                    "bytes": file.stat().st_size,
                    "sha256": sha256(file),
                }
            )
        complete = {
            "format": "hlwm8b-resumable-v1",
            "global_update": update,
            "phase": trainer_state.get("phase"),
            "files": files,
        }
        write_json(partial / "checkpoint-complete.json", complete)
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
        write_json(
            output_dir / "latest.json",
            {"checkpoint": final.name, "global_update": update, "phase": trainer_state.get("phase")},
        )
    accelerator.wait_for_everyone()
    return final


def load_sidecar(path: Path, model: torch.nn.Module) -> None:
    state = load_file(str(path / "hlwm-sidecar.safetensors"), device="cpu")
    missing, unexpected = model.sidecar.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError("sidecar state mismatch: missing=%s unexpected=%s" % (missing, unexpected))


def load_training_state(
    path: Path,
    accelerator: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Dict[str, Any]:
    unwrapped = accelerator.unwrap_model(model)
    load_sidecar(path, unwrapped)
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
    scaler_path = path / "scaler.pt"
    scaler = getattr(accelerator, "scaler", None)
    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=False))
    rng_path = path / ("rng-rank-%03d.pt" % accelerator.process_index)
    if not rng_path.exists():
        rng_path = path / "rng-rank-000.pt"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda"):
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except RuntimeError:
            torch.cuda.set_rng_state(rng["cuda"][0], accelerator.device)
    return checkpoint_state(path)


def package_checkpoint(checkpoint: Path, destination: Path, part_bytes: int = 190 * 1024 * 1024) -> Dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    archive = Path(shutil.make_archive(str(destination / checkpoint.name), "zip", checkpoint))
    parts = []
    with archive.open("rb") as source:
        index = 0
        while True:
            payload = source.read(part_bytes)
            if not payload:
                break
            part = destination / (archive.name + ".part-%03d" % index)
            part.write_bytes(payload)
            parts.append(
                {"name": part.name, "bytes": part.stat().st_size, "sha256": sha256(part)}
            )
            index += 1
    manifest = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "parts": parts,
    }
    write_json(destination / (archive.name + ".parts.json"), manifest)
    return manifest
