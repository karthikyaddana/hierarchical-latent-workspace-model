import json
from pathlib import Path

import pytest

from hlwm_data.beast_factory import (
    RequestBudget,
    _assign_validation,
    _programmatic_test_rows,
    _task_from_episode,
    _validate_single_teacher_value,
    _workflow_coverage_failures,
    export_teacher_dataset,
    load_teacher_tasks,
    validate_provider_roles,
)
from hlwm_data.azure_client import BudgetExceededError


def test_task_factory_never_reuses_the_old_published_answer():
    episode = {
        "episode_id": "episode-1",
        "domain": "coding",
        "difficulty": 3,
        "source_group": "group-1",
        "lineage_component_id": "lineage-1",
        "input": {
            "user_request": "Fix the timeout bug.",
            "context": ["The request hangs."],
            "constraints": ["Keep the API stable."],
        },
        "integration": {"published_answer": "This known-bad answer must not be copied."},
    }
    task = _task_from_episode(episode)
    assert task["prompt"] == "Fix the timeout bug."
    assert "known-bad" not in json.dumps(task)


def test_task_factory_preserves_observable_professional_frame_only():
    episode = {
        "episode_id": "episode-frame",
        "domain": "product",
        "difficulty": 5,
        "input": {"user_request": "Build a creator portal.", "context": [], "constraints": []},
        "frame": {
            "objective": "Ship a creator portal",
            "requirements": ["Cover roles"],
            "unknowns": ["Budget"],
            "failure_contract": ["Do not expose private drafts"],
        },
        "routing": {"candidate_briefs": [{"scope": "Map roles and surfaces"}]},
        "lanes": [{"artifacts": [{"content": "old generated answer"}]}],
        "integration": {"published_answer": "old published answer"},
    }
    task = _task_from_episode(episode)
    assert task["objective"] == "Ship a creator portal"
    assert task["requirements"] == ["Cover roles"]
    assert task["unknowns"] == ["Budget"]
    assert task["failure_contract"] == ["Do not expose private drafts"]
    assert task["candidate_briefs"] == ["Map roles and surfaces"]
    assert "old generated answer" not in json.dumps(task)
    assert "old published answer" not in json.dumps(task)


def test_validation_assignment_groups_by_lineage():
    first = {"packet_id": "a", "task": {"lineage_component_id": "same"}}
    second = {"packet_id": "b", "task": {"lineage_component_id": "same"}}
    assert _assign_validation(first, 0.2) == _assign_validation(second, 0.2)


def test_programmatic_test_is_deterministic_and_balanced():
    rows = _programmatic_test_rows(17, 40)
    assert rows == _programmatic_test_rows(17, 40)
    assert {row["task_type"] for row in rows} == {
        "numeric",
        "unit",
        "ordering",
        "abstention",
    }
    assert all(row["chosen"] != row["rejected"] for row in rows)


def test_provider_role_validation_requires_independent_judges():
    config = {
        "teacher_factory": {
            "minimum_candidate_models": 2,
            "minimum_judges": 2,
            "providers": {
                "a": {"type": "azure", "model": "a", "roles": ["candidate"]},
                "b": {"type": "azure", "model": "b", "roles": ["candidate"]},
                "c": {"type": "azure", "model": "c", "roles": ["critic"]},
                "j1": {"type": "azure", "model": "j1", "roles": ["judge"]},
                "j2": {"type": "azure", "model": "j2", "roles": ["judge"]},
            },
        }
    }
    assert validate_provider_roles(config) == []
    config["teacher_factory"]["providers"]["j2"]["model"] = "j1"
    assert any("judges" in error for error in validate_provider_roles(config))
    config["teacher_factory"]["providers"]["j2"]["model"] = "j2"
    config["teacher_factory"]["variants_per_episode"] = 2
    assert any("task-author" in error for error in validate_provider_roles(config))
    config["teacher_factory"]["providers"]["author"] = {
        "type": "azure",
        "model": "author",
        "roles": ["author"],
    }
    assert validate_provider_roles(config) == []


def test_single_teacher_mode_rejects_more_than_one_available_teacher():
    config = {
        "teacher_factory": {
            "mode": "single_teacher",
            "teacher_models_per_task": 1,
            "providers": {
                "one": {"type": "azure", "model": "teacher-a", "roles": ["teacher"]}
            },
        }
    }
    assert validate_provider_roles(config) == []
    config["teacher_factory"]["providers"]["two"] = {
        "type": "azure",
        "model": "teacher-b",
        "roles": ["teacher"],
    }
    assert any("exactly one" in error for error in validate_provider_roles(config))


def test_single_teacher_local_gate_requires_complete_expert_workflow():
    source = {
        "task_id": "project-1",
        "prompt": "Build a professional music streaming application.",
        "context": [],
        "constraints": ["Cover artist and listener roles."],
        "domain": "product_and_ui_engineering",
        "difficulty": 5,
        "variation_index": 0,
    }
    sections = {
        "## Outcome": "Deliver an original music service with clear listener and artist value, measurable activation, and an intentionally bounded first release. Define non-goals for social features and label-scale rights tooling so the first scope can be evaluated honestly.",
        "## Assumptions": "Assume licensed audio, one launch region, card billing, responsive web delivery, and a small team; validate each assumption before commitment. Keep geography, catalogue size, budget, brand direction, and rights policy in a visible decision log.",
        "## Research and reuse": "Compare permissive open-source playback, search, analytics, authentication, and component libraries; record licences and reject proprietary brand assets. Document build-versus-reuse evidence, maintenance risk, accessibility maturity, and the fallback when a dependency fails review.",
        "## Users and flows": "Map listener discovery, playback and subscription journeys alongside artist upload, release management, analytics, support, and administrator review. Add a role and permission matrix, route inventory, empty states, drafts, approvals, disputes, and account recovery.",
        "## Execution plan": "Define routes, surfaces, components, states, API contracts, data ownership and architecture, then ship vertical slices for identity, catalog, playback, creator operations and billing. Each phase produces tested artifacts, migration notes, observability, and a reversible release candidate.",
        "## Edge cases": "Cover missing artwork, failed transcodes, duplicate tracks, expired rights, loading and error states, offline clients, payment retries, abuse reports and account recovery. Include quota exhaustion, inaccessible media, permission mistakes, partial writes, webhook replay and safe operational recovery.",
        "## Verification and delivery": "Test accessibility, permissions, streaming recovery, data integrity and billing; launch gradually with acceptance evidence, monitoring, alerts and rollback steps. Hand off source, design tokens, API schema, runbooks, analytics definitions, known risks, deliverables and signed acceptance criteria.",
    }
    answer = "\n\n".join(heading + "\n" + detail for heading, detail in sections.items())
    value = {
        "task": {
            "prompt": source["prompt"],
            "context": [],
            "constraints": source["constraints"],
            "domain": source["domain"],
            "difficulty": 5,
        },
        "response_mode": "expert_workflow",
        "answer": answer,
        "verification_checks": ["Check routes", "Check roles", "Check edge cases"],
        "uncertainties": ["Brand direction"],
        "rejected_answer": "Start coding the home page immediately and decide everything later.",
        "rejected_defect": "It ignores roles, failure states, architecture, testing, and delivery evidence.",
        "self_score": 0.91,
    }
    result = _validate_single_teacher_value(value, source)
    assert result["response_mode"] == "expert_workflow"
    broken = dict(value)
    broken["answer"] = answer.replace("## Edge cases", "## Happy path")
    with pytest.raises(ValueError, match="missing headings"):
        _validate_single_teacher_value(broken, source)


def test_expert_workflow_gate_requires_observable_professional_coverage():
    complete = " ".join(
        [
            "Define the outcome, scope, non-goals, success metric and assumptions.",
            "Research reusable open-source components and record licence evidence.",
            "Map each user role, permission, journey and workflow.",
            "Inventory routes, surfaces, components, data ownership, APIs and artifacts.",
            "Cover empty, loading, offline, error, abuse, failure and recovery states.",
            "Set tests, acceptance evidence, rollout monitoring, rollback and deliverables.",
        ]
    )
    assert _workflow_coverage_failures(complete) == []
    assert "failure_and_recovery" in _workflow_coverage_failures(
        "Outcome scope research reuse roles flows routes APIs tests rollout"
    )


def test_request_budget_counts_every_physical_attempt():
    budget = RequestBudget(max_requests=2, max_tokens=100)
    budget.reserve()
    budget.record(10, 20)
    budget.reserve()
    assert budget.snapshot()["requests"] == 2
    assert budget.snapshot()["tokens"] == 30
    with pytest.raises(BudgetExceededError):
        budget.reserve()


def test_task_variants_expand_source_without_reusing_old_answer(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "episode_id": "episode-variant",
                "input": {"user_request": "Diagnose a failing queue.", "context": [], "constraints": []},
                "integration": {"published_answer": "Never reuse this answer."},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = load_teacher_tasks([source], seed=17, variants_per_episode=10)
    assert len(tasks) == 10
    assert len({task["task_id"] for task in tasks}) == 10
    assert {task["variation_index"] for task in tasks} == set(range(10))
    assert "Never reuse" not in json.dumps(tasks)


def test_export_manifest_hashes_splits_and_requires_real_teacher_volume(tmp_path):
    packet_dir = tmp_path / "data" / "beast" / "checkpoints" / "task-1"
    packet_dir.mkdir(parents=True)
    packet = {
        "packet_id": "task-1",
        "accepted": True,
        "task": {
            "prompt": "What is 2 + 2?",
            "context": [],
            "constraints": [],
            "domain": "math",
            "difficulty": 1,
            "source_group": "one",
            "lineage_component_id": "one",
        },
        "chosen": "4",
        "rejected": "5",
        "rejected_defect": "Incorrect sum.",
        "mean_judge_score": 1.0,
        "judges": [{"model": "judge-a"}, {"model": "judge-b"}],
        "candidate_provenance": [{"model": "candidate-a"}, {"model": "candidate-b"}],
        "critic": {"model": "critic-a"},
        "content_sha256": "a" * 64,
    }
    packet_dir.joinpath("packet.json").write_text(json.dumps(packet), encoding="utf-8")
    config = {
        "project": {"seed": 17},
        "teacher_factory": {
            "output_dir": "data/beast",
            "validation_fraction": 0.1,
            "minimum_main_train_records": 25_000,
        },
    }
    result = export_teacher_dataset(config, tmp_path)
    manifest = result["manifest"]
    assert manifest["accepted_teacher_records"] == 1
    assert manifest["ready_for_main_training"] is False
    assert all(manifest["files"][split]["sha256"] for split in ("train", "validation", "test"))
