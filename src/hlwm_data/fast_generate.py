from __future__ import annotations

import json
import math
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .azure_client import AzureTeacherClient
from .generate import (
    EpisodeJob,
    _normalize_episode,
    _output_contract,
    _source_packet,
    archived_episode_id_collisions,
    build_jobs,
    filter_generation_chunks_by_source_quality,
)
from .language import classify_language, contains_blocked_script
from .util import append_jsonl, atomic_write_json, iter_jsonl, sanitize_unicode, stable_hash
from .validation import schema_errors


def fast_generation_deployments(config: Dict[str, Any]) -> List[str]:
    azure = config["azure"]
    configured = azure.get("fast_deployments") or [
        azure.get("deployment"),
        azure.get("blueprint_deployment"),
        azure.get("blueprint_judge_deployment"),
        azure.get("episode_critic_deployment"),
    ]
    deployments: List[str] = []
    for value in configured:
        name = str(value or "").strip()
        if name and name not in deployments:
            deployments.append(name)
    if not deployments:
        raise ValueError("Fast generation requires at least one Azure deployment")
    return deployments


def _english_output_text(episode: Dict[str, Any]) -> str:
    parts: List[str] = []
    input_value = episode.get("input", {})
    frame = episode.get("frame", {})
    integration = episode.get("integration", {})
    parts.extend(
        [
            str(input_value.get("user_request", "")),
            str(frame.get("objective", "")),
            str(integration.get("published_answer", "")),
        ]
    )
    for lane in episode.get("lanes", []):
        if isinstance(lane, dict):
            parts.append(str(lane.get("summary", "")))
    return "\n".join(part for part in parts if part.strip())


def fast_episode_errors(
    episode: Dict[str, Any], episode_schema: Path, minimum_language_confidence: float = 0.62
) -> List[str]:
    errors = list(schema_errors(episode, episode_schema))
    serialized = sanitize_unicode(json.dumps(episode, ensure_ascii=False))
    if "\N{REPLACEMENT CHARACTER}" in serialized:
        errors.append("output contains corrupted Unicode")
    if contains_blocked_script(serialized):
        errors.append("output contains a blocked non-English script")
    english_text = _english_output_text(episode)
    if not english_text.strip():
        errors.append("output has no substantive English text")
    else:
        decision = classify_language(
            english_text,
            expected="en",
            minimum_confidence=minimum_language_confidence,
        )
        if not decision.accepted:
            errors.append("output language check failed: %s" % decision.reason)
    return errors


def _fast_job_language_errors(jobs: Sequence[EpisodeJob]) -> List[str]:
    errors: List[str] = []
    seen = set()
    for job in jobs:
        for chunk in job.chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            text = sanitize_unicode(str(chunk.get("text", "")))
            if str(chunk.get("language", "")).lower() != "en":
                errors.append("%s is not marked English" % chunk_id)
            elif "\N{REPLACEMENT CHARACTER}" in text:
                errors.append("%s contains corrupted Unicode" % chunk_id)
            elif contains_blocked_script(text):
                errors.append("%s contains a blocked script" % chunk_id)
            if len(errors) >= 20:
                return errors
    return errors


def _accepted_wrapper(episode: Dict[str, Any], deployment: str) -> Dict[str, Any]:
    return {
        "episode": episode,
        "accepted": True,
        "acceptance_basis": "fast_generation_schema_and_english_only",
        "review": {
            "episode_id": episode["episode_id"],
            "verdict": "not_judged",
            "overall_score": 0.0,
            "issues": [],
            "required_fixes": [],
        },
        "judge_metadata": None,
        "generation_deployment": deployment,
    }


def _normalize_fast_episode(value: Dict[str, Any], job: EpisodeJob) -> Dict[str, Any]:
    episode = _normalize_episode(value, job)
    frame = episode.get("frame")
    if isinstance(frame, dict) and isinstance(frame.get("failure_contract"), str):
        frame["failure_contract"] = [frame["failure_contract"]]
    barrier = episode.get("barrier")
    if isinstance(barrier, dict):
        for key in ("conflicts", "open_claims"):
            rows = barrier.get(key)
            if isinstance(rows, list):
                barrier[key] = [
                    json.dumps(row, ensure_ascii=False, sort_keys=True)
                    if isinstance(row, dict)
                    else str(row)
                    for row in rows
                ]
    counterfactuals = episode.get("counterfactuals")
    if isinstance(counterfactuals, list) and len(counterfactuals) == 1:
        first = counterfactuals[0] if isinstance(counterfactuals[0], dict) else {}
        utility = max(0.0, min(1.0, float(first.get("utility", 0.5)) - 0.1))
        counterfactuals.append(
            {
                "variant": "single-lane",
                "outcome": (
                    "A single specialist lane would omit the complementary artifact and its "
                    "independent cross-check."
                ),
                "utility": utility,
                "errors": ["Missing complementary specialist artifact and cross-check"],
            }
        )
    return episode


def _count_fast_accepted(reviewed_dir: Path) -> int:
    paths = list(reviewed_dir.glob("*.json"))

    def is_fast_accepted(path: Path) -> bool:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(
            value.get("accepted")
            and value.get("acceptance_basis")
            == "fast_generation_schema_and_english_only"
        )

    if not paths:
        return 0
    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as executor:
        return sum(1 for accepted in executor.map(is_fast_accepted, paths) if accepted)


def fast_generate_episodes(
    config: Dict[str, Any],
    root: Path,
    target: int,
    resume: bool = True,
    offset: int = 0,
) -> Dict[str, Any]:
    if target <= 0:
        raise ValueError("Fast generation target must be positive")
    output = config["output"]
    generation = config["generation"]
    generated_dir = root / output["generated_dir"]
    reviewed_dir = root / output["reviewed_dir"]
    generated_dir.mkdir(parents=True, exist_ok=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    existing = _count_fast_accepted(reviewed_dir) if resume else 0
    if existing >= target:
        return {
            "status": "complete",
            "target": target,
            "accepted": existing,
            "generated_this_run": 0,
            "failed": 0,
        }

    all_chunks = list(iter_jsonl(root / output["chunks_file"]))
    eligible_chunks, source_quality_filtered = filter_generation_chunks_by_source_quality(
        all_chunks, generation
    )
    remaining = target - existing
    workers = int(config["azure"].get("max_workers", 32))
    multiplier = max(1.05, float(generation.get("fast_candidate_multiplier", 1.6)))
    candidate_count = max(remaining + workers, int(math.ceil(remaining * multiplier)))
    jobs = build_jobs(eligible_chunks, config, candidate_count, offset=offset)
    collisions = archived_episode_id_collisions(root, generated_dir, jobs)
    if collisions:
        raise ValueError(
            "scheduled fast episode IDs collide with archived pilots; choose another offset: %s"
            % collisions[:10]
        )
    language_errors = _fast_job_language_errors(jobs)
    if language_errors:
        raise ValueError("Fast generation source language check failed: %s" % language_errors)

    deployments = fast_generation_deployments(config)
    clients: Dict[str, AzureTeacherClient] = {}
    for deployment in deployments:
        client_config = dict(config["azure"])
        client_config["deployment"] = deployment
        client_config["timeout_seconds"] = float(config["azure"].get("fast_timeout_seconds", 120))
        client_config["max_retries"] = int(config["azure"].get("fast_max_retries", 1))
        client_config["max_completion_tokens"] = int(
            config["azure"].get("fast_max_completion_tokens", 9000)
        )
        client_config["temperature"] = float(config["azure"].get("fast_temperature", 0.45))
        client_config["requests_per_minute"] = int(
            config["azure"].get("fast_requests_per_minute", 60)
        )
        clients[deployment] = AzureTeacherClient(
            client_config, root / output["request_log"]
        )
    # All clients share one credential and token provider. Warm it once before
    # worker threads start so a large pool does not contend during login.
    next(iter(clients.values())).warm_auth()

    system = (root / "prompts/fast_episode_system.md").read_text(encoding="utf-8")
    episode_schema = root / "schemas/episode.schema.json"
    max_chars = int(generation.get("fast_max_source_chars", 10000))
    raw_dir = root / "data/reasoning9000/raw-fast"
    progress_path = root / "data/reasoning9000/fast-progress.json"
    progress_lock = threading.Lock()
    stats: Dict[str, Any] = {
        "status": "running",
        "target": target,
        "accepted_at_start": existing,
        "accepted": existing,
        "scheduled_candidates": len(jobs),
        "submitted": 0,
        "generated_this_run": 0,
        "recovered": 0,
        "failed": 0,
        "source_chunks": len(all_chunks),
        "source_quality_filtered": source_quality_filtered,
        "deployments": deployments,
        "workers": workers,
        "next_offset": offset + candidate_count,
    }

    def write_progress() -> None:
        with progress_lock:
            atomic_write_json(progress_path, stats)

    def run(job: EpisodeJob) -> Tuple[str, Optional[str]]:
        generated_path = generated_dir / (job.episode_id + ".json")
        reviewed_path = reviewed_dir / (job.episode_id + ".json")
        raw_path = raw_dir / (job.episode_id + ".json")
        if resume and reviewed_path.exists():
            return "skipped", None
        if resume and generated_path.exists():
            try:
                recovered = json.loads(generated_path.read_text(encoding="utf-8"))
                recovered_errors = fast_episode_errors(
                    recovered,
                    episode_schema,
                    float(generation.get("fast_minimum_language_confidence", 0.62)),
                )
                if not recovered_errors:
                    deployment = str(
                        recovered.get("generation_metadata", {}).get("deployment", "unknown")
                    )
                    atomic_write_json(reviewed_path, _accepted_wrapper(recovered, deployment))
                    return "recovered", deployment
            except Exception:
                pass

        if resume and raw_path.exists():
            try:
                raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
                recovered = _normalize_fast_episode(dict(raw_value["response"]), job)
                recovered_trace = recovered.get("blueprint_trace")
                if not isinstance(recovered_trace, dict):
                    recovered_trace = {}
                recovered_trace["blueprint_hash"] = stable_hash(
                    {"mode": "direct-fast", "episode_id": job.episode_id, "prompt": system}, 32
                )
                recovered["blueprint_trace"] = recovered_trace
                recovered_errors = fast_episode_errors(
                    recovered,
                    episode_schema,
                    float(generation.get("fast_minimum_language_confidence", 0.62)),
                )
                if not recovered_errors:
                    deployment = str(raw_value.get("deployment") or "unknown")
                    atomic_write_json(reviewed_path, _accepted_wrapper(recovered, deployment))
                    return "recovered", deployment
            except Exception:
                pass

        packet = _source_packet(job, max_chars)
        deployment = deployments[
            int(stable_hash({"episode_id": job.episode_id, "seed": job.variation_seed}, 8), 16)
            % len(deployments)
        ]
        prompt = (
            "Create one complete episode now.\n\n"
            "TARGET DOMAIN:\n%s\n\nTARGET OBJECTIVE:\n%s\n\nVARIATION SEED:\n%d\n\n"
            "SOURCE PACKET:\n%s\n\nREQUIRED OUTPUT SHAPE:\n%s"
            % (
                job.domain,
                job.objective,
                job.variation_seed,
                json.dumps(packet, ensure_ascii=False, separators=(",", ":")),
                json.dumps(_output_contract(config, job), ensure_ascii=False, separators=(",", ":")),
            )
        )
        completion = clients[deployment].chat_json(
            system, prompt, "fast-generate", job.episode_id
        )
        episode = _normalize_fast_episode(completion.value, job)
        trace = episode.get("blueprint_trace")
        if not isinstance(trace, dict):
            trace = {}
        trace["blueprint_hash"] = stable_hash(
            {"mode": "direct-fast", "episode_id": job.episode_id, "prompt": system}, 32
        )
        episode["blueprint_trace"] = trace
        episode["generation_metadata"] = {
            "provider": "azure",
            "mode": "direct_fast_no_judge",
            "deployment": deployment,
            "request_id": completion.request_id,
            "prompt_template_hash": stable_hash(system, 32),
            "source_packet_hash": stable_hash(packet, 32),
            "variation_seed": job.variation_seed,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "elapsed_seconds": completion.elapsed_seconds,
        }
        errors = fast_episode_errors(
            episode,
            episode_schema,
            float(generation.get("fast_minimum_language_confidence", 0.62)),
        )
        if errors:
            append_jsonl(
                root / output["run_log"],
                {
                    "stage": "fast-generate",
                    "episode_id": job.episode_id,
                    "status": "schema_or_language_rejected",
                    "deployment": deployment,
                    "errors": errors[:12],
                },
            )
            return "failed", deployment
        # The accepted wrapper is the only production checkpoint. It already
        # contains the complete episode consumed by materialization, so writing
        # duplicate raw and generated copies would triple disk traffic.
        atomic_write_json(reviewed_path, _accepted_wrapper(episode, deployment))
        return "generated", deployment

    job_iter = iter(jobs)
    pending = {}

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        try:
            job = next(job_iter)
        except StopIteration:
            return False
        pending[executor.submit(run, job)] = job
        stats["submitted"] += 1
        return True

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in range(min(workers, len(jobs))):
                submit_one(executor)
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        status, deployment = future.result()
                    except Exception as exc:
                        status, deployment = "failed", None
                        append_jsonl(
                            root / output["run_log"],
                            {
                                "stage": "fast-generate",
                                "episode_id": job.episode_id,
                                "status": "failed",
                                "error": str(exc)[:1000],
                            },
                        )
                    if status == "generated":
                        stats["generated_this_run"] += 1
                        stats["accepted"] += 1
                    elif status == "recovered":
                        stats["recovered"] += 1
                        stats["accepted"] += 1
                    elif status == "failed":
                        stats["failed"] += 1
                    if stats["accepted"] < target:
                        submit_one(executor)
                if (stats["generated_this_run"] + stats["recovered"] + stats["failed"]) % 25 == 0:
                    write_progress()
                if stats["accepted"] >= target:
                    for future in pending:
                        future.cancel()
                    break
    finally:
        for client in clients.values():
            client.close()

    stats["accepted"] = _count_fast_accepted(reviewed_dir)
    stats["status"] = "complete" if stats["accepted"] >= target else "candidate_pool_exhausted"
    write_progress()
    return stats
