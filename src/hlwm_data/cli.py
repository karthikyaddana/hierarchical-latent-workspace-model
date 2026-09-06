from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict

from .azure_client import AzureTeacherClient
from .beast_factory import (
    export_teacher_dataset,
    run_teacher_factory,
    run_teacher_supervisor,
    teacher_factory_status,
    validate_provider_roles,
)
from .builder_factory import (
    builder_doctor,
    builder_status,
    export_builder_dataset,
    run_builder,
)
from .teacher_providers import provider_statuses
from .config import load_pipeline_config
from .fast_generate import fast_generate_episodes
from .generate import generate_episodes, judge_episodes, repair_rejected_episodes
from .ingest import append_ingested_manifest, filter_chunk_file_by_language, ingest_manifest
from .materialize import build_dataset, validate_built_dataset


def _root(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _load(args) -> tuple:
    root = _root(args.root)
    config_path = root / args.config
    return root, load_pipeline_config(config_path)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_doctor(args) -> int:
    root, config = _load(args)
    deployments = [config["azure"]["deployment"]]
    blueprint = config["azure"].get("blueprint_deployment")
    if blueprint and blueprint not in deployments:
        deployments.append(blueprint)
    blueprint_judge = config["azure"].get("blueprint_judge_deployment")
    if blueprint_judge and blueprint_judge not in deployments:
        deployments.append(blueprint_judge)
    episode_critic = config["azure"].get("episode_critic_deployment")
    if episode_critic and episode_critic not in deployments:
        deployments.append(episode_critic)
    episode_editor = config["azure"].get("episode_editor_deployment")
    if episode_editor and episode_editor not in deployments:
        deployments.append(episode_editor)
    judge = config["azure"].get("judge_deployment")
    if judge and judge not in deployments:
        deployments.append(judge)
    secondary_judge = config["azure"].get("secondary_judge_deployment")
    if secondary_judge and secondary_judge not in deployments:
        deployments.append(secondary_judge)
    checks = []
    for deployment in deployments:
        doctor_config = dict(config["azure"])
        doctor_config["deployment"] = deployment
        # Reasoning deployments can consume a small hidden budget before they
        # emit the requested JSON, so a 128-token probe creates false failures.
        doctor_config["max_completion_tokens"] = 1024
        doctor_config["temperature"] = 0.0
        client = AzureTeacherClient(doctor_config, root / config["output"]["request_log"])
        try:
            result = client.chat_json(
                "Return one small valid JSON object only.",
                '{"status":"ok","service":"azure-openai"}',
                "doctor",
                "connection-test-%s" % deployment,
            )
        finally:
            client.close()
        checks.append({"deployment": deployment, "response": result.value, "elapsed_seconds": result.elapsed_seconds})
    _print(
        {
            "status": "ok",
            "endpoint": config["azure"]["endpoint"],
            "deployments": checks,
        }
    )
    return 0


def command_ingest(args) -> int:
    root, config = _load(args)
    ingest = append_ingested_manifest if args.append else ingest_manifest
    output_path = root / (args.output or config["output"]["chunks_file"])
    stats = ingest(
        root / args.manifest,
        output_path,
        target_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
        max_chunks_per_source_group=args.max_chunks_per_source_group,
    )
    _print(stats)
    return 0


def command_generate(args) -> int:
    root, config = _load(args)
    if args.workers:
        config["azure"]["max_workers"] = args.workers
    _print(generate_episodes(config, root, args.count, resume=not args.no_resume, offset=args.offset))
    return 0


def command_fast_generate(args) -> int:
    root, config = _load(args)
    if args.workers:
        config["azure"]["max_workers"] = args.workers
    if args.models:
        config["azure"]["fast_deployments"] = [
            value.strip() for value in args.models.split(",") if value.strip()
        ]
    result = fast_generate_episodes(
        config,
        root,
        target=args.target,
        resume=not args.no_resume,
        offset=args.offset,
    )
    _print(result)
    return 0 if result.get("status") == "complete" else 1


def command_filter_language(args) -> int:
    root, config = _load(args)
    path = root / (args.input or config["output"]["chunks_file"])
    _print(
        filter_chunk_file_by_language(
            path,
            expected="en",
            minimum_confidence=float(config["generation"].get("minimum_language_confidence", 0.78)),
        )
    )
    return 0


def command_judge(args) -> int:
    root, config = _load(args)
    if args.workers:
        config["azure"]["max_workers"] = args.workers
    _print(judge_episodes(config, root, resume=not args.no_resume))
    return 0


def command_repair(args) -> int:
    root, config = _load(args)
    if args.workers:
        config["azure"]["max_workers"] = args.workers
    _print(repair_rejected_episodes(config, root, resume=not args.no_resume))
    return 0


def command_build(args) -> int:
    root, config = _load(args)
    _print(build_dataset(config, root))
    return 0


def command_validate(args) -> int:
    root, config = _load(args)
    report = validate_built_dataset(config, root)
    _print(report)
    return 0 if report["valid"] else 1


def command_pilot(args) -> int:
    root, config = _load(args)
    if args.workers:
        config["azure"]["max_workers"] = args.workers
    result: Dict[str, Any] = {}
    result["ingest"] = ingest_manifest(
        root / args.manifest,
        root / config["output"]["chunks_file"],
        target_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
        max_chunks_per_source_group=args.max_chunks_per_source_group,
    )
    result["generate"] = generate_episodes(config, root, args.count, resume=True, offset=args.offset)
    result["judge"] = judge_episodes(config, root, resume=True)
    result["build"] = build_dataset(config, root)
    result["validate"] = validate_built_dataset(config, root)
    _print(result)
    return 0 if result["validate"]["valid"] else 1


def command_status(args) -> int:
    root, config = _load(args)
    output = config["output"]
    generated = list((root / output["generated_dir"]).glob("*.json"))
    reviewed = list((root / output["reviewed_dir"]).glob("*.json"))
    accepted = 0
    rejected = 0
    accepted_domains: Counter = Counter()
    for path in reviewed:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            accepted += int(bool(value.get("accepted")))
            rejected += int(not bool(value.get("accepted")))
            if value.get("accepted"):
                accepted_domains[str(value.get("episode", {}).get("domain", "unknown"))] += 1
        except Exception:
            rejected += 1
    prompt_tokens = completion_tokens = request_rows = 0
    request_log = root / output["request_log"]
    if request_log.exists():
        with request_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                request_rows += 1
                prompt_tokens += int(row.get("prompt_tokens") or 0)
                completion_tokens += int(row.get("completion_tokens") or 0)
    target = int(config.get("generation", {}).get("target_accepted_episodes", 0) or 0)
    _print(
        {
            "generated": len(generated),
            "reviewed": len(reviewed),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / len(reviewed) if reviewed else 0.0,
            "target_accepted": target or None,
            "remaining_accepted": max(0, target - accepted) if target else None,
            "accepted_domains": dict(sorted(accepted_domains.items())),
            "azure_usage": {
                "request_log_rows": request_rows,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "final_manifest_exists": (root / output["final_dir"] / "manifest.json").exists(),
            "progress_file_exists": (root / "data/reasoning9000/progress.json").exists(),
        }
    )
    return 0


def command_beast_doctor(args) -> int:
    root, config = _load(args)
    errors = validate_provider_roles(config)
    _print(
        {
            "status": "ok" if not errors else "blocked",
            "providers": [status.__dict__ for status in provider_statuses(config)],
            "role_errors": errors,
        }
    )
    return 0 if not errors else 1


def command_beast_run(args) -> int:
    root, config = _load(args)
    _print(
        run_teacher_factory(
            config,
            root,
            target=args.target,
            offset=args.offset,
            workers=args.workers,
            max_requests=args.max_requests,
            max_tokens=args.max_tokens,
        )
    )
    return 0


def command_beast_auto(args) -> int:
    root, config = _load(args)
    _print(
        run_teacher_supervisor(
            config,
            root,
            accepted_target=args.accepted_target,
            batch_size=args.batch_size,
            max_total_requests=args.max_total_requests,
            max_total_tokens=args.max_total_tokens,
            workers=args.workers,
            max_rounds=args.max_rounds,
        )
    )
    return 0


def command_beast_status(args) -> int:
    root, config = _load(args)
    _print(teacher_factory_status(config, root))
    return 0


def command_beast_export(args) -> int:
    root, config = _load(args)
    _print(export_teacher_dataset(config, root))
    return 0


def command_beast_stop(args) -> int:
    root, config = _load(args)
    stop = root / config["teacher_factory"].get("stop_file", "data/beast/STOP")
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    _print({"status": "stop-requested", "path": str(stop)})
    return 0


def command_beast_resume(args) -> int:
    root, config = _load(args)
    stop = root / config["teacher_factory"].get("stop_file", "data/beast/STOP")
    if stop.exists():
        stop.unlink()
    _print({"status": "resume-enabled", "path": str(stop)})
    return 0


def command_builder_doctor(args) -> int:
    root, config = _load(args)
    report = builder_doctor(config, root)
    _print(report)
    return 0 if report["status"] == "ok" else 1


def command_builder_auto(args) -> int:
    root, config = _load(args)
    _print(
        run_builder(
            config,
            root,
            accepted_target=args.accepted_target,
            workers=args.workers,
            max_total_requests=args.max_total_requests,
            max_total_tokens=args.max_total_tokens,
            domains=args.domains.split(",") if args.domains else None,
        )
    )
    return 0


def command_builder_status(args) -> int:
    root, config = _load(args)
    _print(builder_status(config, root))
    return 0


def command_builder_export(args) -> int:
    root, config = _load(args)
    _print(export_builder_dataset(config, root))
    return 0


def command_builder_stop(args) -> int:
    root, config = _load(args)
    stop = root / (config.get("builder_factory") or {}).get("stop_file", "data/builder/STOP")
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.touch()
    _print({"status": "stop-requested", "path": str(stop)})
    return 0


def command_builder_resume(args) -> int:
    root, config = _load(args)
    stop = root / (config.get("builder_factory") or {}).get("stop_file", "data/builder/STOP")
    if stop.exists():
        stop.unlink()
    _print({"status": "resume-enabled", "path": str(stop)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hlwm-data",
        description="Create verified, leakage-safe HLWM training datasets with Azure-hosted teacher models.",
    )
    parser.add_argument("--root", default=".", help="Project root (default: current directory)")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config relative to root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Verify Azure authentication and deployment access")
    doctor.set_defaults(func=command_doctor)

    ingest = subparsers.add_parser("ingest", help="Validate and chunk local source material")
    ingest.add_argument("--manifest", default="config/sources.yaml")
    ingest.add_argument("--output", help="Optional chunk JSONL output path relative to the project root")
    ingest.add_argument("--chunk-chars", type=int, default=4200)
    ingest.add_argument("--overlap-chars", type=int, default=400)
    ingest.add_argument("--min-chunk-chars", type=int, default=300)
    ingest.add_argument("--max-chunks-per-source-group", type=int, default=0)
    ingest.add_argument(
        "--append",
        action="store_true",
        help="Atomically append a manifest to an existing chunk corpus and remove exact collisions",
    )
    ingest.set_defaults(func=command_ingest)

    generate = subparsers.add_parser("generate", help="Generate master episodes concurrently")
    generate.add_argument("--count", type=int, required=True)
    generate.add_argument("--workers", type=int)
    generate.add_argument("--offset", type=int, default=0, help="Stable global job offset for resumable batches")
    generate.add_argument("--no-resume", action="store_true")
    generate.set_defaults(func=command_generate)

    fast_generate = subparsers.add_parser(
        "fast-generate",
        help="Generate schema-valid English episodes directly with parallel Azure models and no judges",
    )
    fast_generate.add_argument("--target", type=int, required=True)
    fast_generate.add_argument("--workers", type=int)
    fast_generate.add_argument("--offset", type=int, default=0)
    fast_generate.add_argument(
        "--models", help="Optional comma-separated Azure deployment pool"
    )
    fast_generate.add_argument("--no-resume", action="store_true")
    fast_generate.set_defaults(func=command_fast_generate)

    filter_language = subparsers.add_parser(
        "filter-language", help="Atomically remove non-English and disallowed-script chunks"
    )
    filter_language.add_argument("--input", help="Optional chunk JSONL path relative to the project root")
    filter_language.set_defaults(func=command_filter_language)

    judge = subparsers.add_parser("judge", help="Independently grade generated episodes")
    judge.add_argument("--workers", type=int)
    judge.add_argument("--no-resume", action="store_true")
    judge.set_defaults(func=command_judge)

    repair = subparsers.add_parser("repair", help="Repair rejected episodes using judge feedback")
    repair.add_argument("--workers", type=int)
    repair.add_argument("--no-resume", action="store_true")
    repair.set_defaults(func=command_repair)

    build = subparsers.add_parser("build", help="Deduplicate, split and materialize training views")
    build.set_defaults(func=command_build)

    validate = subparsers.add_parser("validate", help="Check the built dataset and leakage boundaries")
    validate.set_defaults(func=command_validate)

    pilot = subparsers.add_parser("pilot", help="Run ingest, generation, judging, build and validation")
    pilot.add_argument("--count", type=int, default=20)
    pilot.add_argument("--workers", type=int)
    pilot.add_argument("--offset", type=int, default=0, help="Stable global job offset")
    pilot.add_argument("--manifest", default="config/sources.yaml")
    pilot.add_argument("--chunk-chars", type=int, default=4200)
    pilot.add_argument("--overlap-chars", type=int, default=400)
    pilot.add_argument("--min-chunk-chars", type=int, default=300)
    pilot.add_argument("--max-chunks-per-source-group", type=int, default=0)
    pilot.set_defaults(func=command_pilot)

    status = subparsers.add_parser("status", help="Show resumable pipeline progress")
    status.set_defaults(func=command_status)

    beast_doctor = subparsers.add_parser(
        "beast-doctor", help="Show which Azure, NIM and Ollama teacher roles are ready"
    )
    beast_doctor.set_defaults(func=command_beast_doctor)

    beast_run = subparsers.add_parser(
        "beast-run", help="Run the resumable multi-teacher distillation factory"
    )
    beast_run.add_argument("--target", type=int, required=True)
    beast_run.add_argument("--offset", type=int, default=0)
    beast_run.add_argument("--workers", type=int)
    beast_run.add_argument("--max-requests", type=int)
    beast_run.add_argument("--max-tokens", type=int)
    beast_run.set_defaults(func=command_beast_run)

    beast_auto = subparsers.add_parser(
        "beast-auto",
        help="Run bounded resumable teacher batches and export after every round",
    )
    beast_auto.add_argument("--accepted-target", type=int, required=True)
    beast_auto.add_argument("--batch-size", type=int, default=8)
    beast_auto.add_argument("--workers", type=int)
    beast_auto.add_argument("--max-total-requests", type=int, required=True)
    beast_auto.add_argument("--max-total-tokens", type=int, required=True)
    beast_auto.add_argument("--max-rounds", type=int, default=0)
    beast_auto.set_defaults(func=command_beast_auto)

    beast_status = subparsers.add_parser(
        "beast-status", help="Show resumable teacher-factory progress"
    )
    beast_status.set_defaults(func=command_beast_status)

    beast_export = subparsers.add_parser(
        "beast-export", help="Freeze a checksummed Kaggle teacher-data snapshot"
    )
    beast_export.set_defaults(func=command_beast_export)

    beast_stop = subparsers.add_parser("beast-stop", help="Request a graceful local stop")
    beast_stop.set_defaults(func=command_beast_stop)

    beast_resume = subparsers.add_parser("beast-resume", help="Clear the graceful-stop marker")
    beast_resume.set_defaults(func=command_beast_resume)

    builder_doctor_cmd = subparsers.add_parser(
        "builder-doctor",
        help="Show provider roles, judge independence and local verification capabilities",
    )
    builder_doctor_cmd.set_defaults(func=command_builder_doctor)

    builder_auto = subparsers.add_parser(
        "builder-auto",
        help="Run the resumable verified product/code/copy teacher factory within a budget",
    )
    builder_auto.add_argument("--accepted-target", type=int, required=True)
    builder_auto.add_argument("--workers", type=int, default=2)
    builder_auto.add_argument("--max-total-requests", type=int, required=True)
    builder_auto.add_argument("--max-total-tokens", type=int, required=True)
    builder_auto.add_argument(
        "--domains", default="", help="Comma-separated domain filter (default: all)"
    )
    builder_auto.set_defaults(func=command_builder_auto)

    builder_status_cmd = subparsers.add_parser(
        "builder-status", help="Show resumable builder-factory progress and budget"
    )
    builder_status_cmd.set_defaults(func=command_builder_status)

    builder_export = subparsers.add_parser(
        "builder-export", help="Export builder episodes to master/SFT/DPO splits with checksums"
    )
    builder_export.set_defaults(func=command_builder_export)

    builder_stop = subparsers.add_parser("builder-stop", help="Request a graceful builder stop")
    builder_stop.set_defaults(func=command_builder_stop)

    builder_resume = subparsers.add_parser(
        "builder-resume", help="Clear the builder graceful-stop marker"
    )
    builder_resume.set_defaults(func=command_builder_resume)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
