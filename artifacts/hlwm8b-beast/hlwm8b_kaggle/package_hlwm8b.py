from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

try:
    from .checkpointing_hlwm8b import package_checkpoint, sha256, verify_checkpoint, write_json
except ImportError:
    from checkpointing_hlwm8b import package_checkpoint, sha256, verify_checkpoint, write_json


RUNTIME_FILES = (
    "modeling_hlwm8b.py",
    "checkpointing_hlwm8b.py",
    "data_hlwm8b.py",
    "semantic_hlwm8b.py",
    "energy_hlwm8b.py",
    "evaluate_hlwm8b.py",
    "inference_hlwm8b.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package HLWM adapter and resumable checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--part-mb", type=int, default=190)
    return parser.parse_args()


def split_archive(archive: Path, part_bytes: int) -> Dict[str, Any]:
    parts = []
    with archive.open("rb") as source:
        for index in range(10000):
            payload = source.read(part_bytes)
            if not payload:
                break
            part = archive.with_name(archive.name + ".part-%03d" % index)
            part.write_bytes(payload)
            parts.append({"name": part.name, "bytes": part.stat().st_size, "sha256": sha256(part)})
    manifest = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "parts": parts,
    }
    write_json(archive.with_name(archive.name + ".parts.json"), manifest)
    return manifest


def main() -> None:
    args = parse_args()
    verify_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deliverable = args.output_dir / "hlwm8b-deliverable"
    if deliverable.exists():
        shutil.rmtree(deliverable)
    deliverable.mkdir()
    shutil.copytree(args.checkpoint / "adapter", deliverable / "adapter")
    for name in ("hlwm-sidecar.safetensors", "hlwm-config.json", "trainer-state.json"):
        shutil.copy2(args.checkpoint / name, deliverable / name)
    for name in RUNTIME_FILES:
        shutil.copy2(args.code_dir / name, deliverable / name)
    if args.evaluation_dir and args.evaluation_dir.exists():
        for name in (
            "commitment-calibration.json",
            "capability-gate.json",
            "checkpoint-evaluation.json",
        ):
            source = args.evaluation_dir / name
            if source.exists():
                shutil.copy2(source, deliverable / name)
    if args.data_manifest and args.data_manifest.exists():
        shutil.copy2(args.data_manifest, deliverable / "training-data-manifest.json")
    readme = """# Qwen3-8B HLWM candidate

This package contains the QLoRA adapter, native HLWM sidecar, validation-only publish
calibration, held-out evaluation, and inference runtime. It still requires the pinned
Qwen3-8B base model named in `hlwm-config.json`.

Run `inference_hlwm8b.py --checkpoint . --prompt \"...\"` on a CUDA machine. Treat the
model as production-ready only when `capability-gate.json` exists and `passed` is true.
"""
    (deliverable / "README.md").write_text(readme, encoding="utf-8")
    files = [
        {
            "path": str(path.relative_to(deliverable)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(deliverable.rglob("*"))
        if path.is_file() and path.name != "checkpoint-complete.json"
    ]
    write_json(
        deliverable / "checkpoint-complete.json",
        {
            "format": "hlwm8b-deliverable-v1",
            "global_update": verify_checkpoint(args.checkpoint)["global_update"],
            "files": files,
        },
    )
    deliverable_archive = Path(
        shutil.make_archive(str(args.output_dir / "hlwm8b-deliverable"), "zip", deliverable)
    )
    part_bytes = args.part_mb * 1024 * 1024
    deliverable_parts = split_archive(deliverable_archive, part_bytes)
    checkpoint_parts = package_checkpoint(args.checkpoint, args.output_dir, part_bytes=part_bytes)
    result = {
        "deliverable": deliverable_parts,
        "resumable_checkpoint": checkpoint_parts,
        "checkpoint_update": verify_checkpoint(args.checkpoint)["global_update"],
    }
    write_json(args.output_dir / "downloads.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
