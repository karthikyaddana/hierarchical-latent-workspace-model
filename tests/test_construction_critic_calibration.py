import json
from pathlib import Path

import yaml

from hlwm_data.generate import (
    _construction_critic_candidate_errors,
    _construction_critic_protocol_errors,
    construction_critic_calibration_errors,
)
from hlwm_data.util import stable_hash
from scripts.assess_construction_critic import (
    build_report,
    calibration_prompt,
    validate_cases,
)


def valid_reject():
    return {
        "verdict": "reject",
        "overall_score": 0.72,
        "expertise_uplift": 0.83,
        "blocking_defects": [
            {
                "category": "correctness",
                "location": "artifact-total row 4",
                "contract_reference": "constraint-total",
                "defect": "The displayed total is 19, but the visible addends sum to 21.",
                "required_correction": "Recompute the total and every conclusion that depends on it.",
            }
        ],
    }


def test_strict_construction_critic_protocol_accepts_bounded_blocker():
    review = valid_reject()

    assert _construction_critic_protocol_errors(review, 0.84, 0.80) == []
    errors = _construction_critic_candidate_errors(review, 0.84, 0.80)
    assert errors[0] == "construction critic rejected the candidate"
    assert any("constraint-total" in error for error in errors)


def test_strict_construction_critic_protocol_rejects_old_freeform_shape():
    review = {
        "verdict": "reject",
        "overall_score": 0.78,
        "expertise_uplift": 0.75,
        "issues": ["A long speculative concern."],
    }

    errors = _construction_critic_protocol_errors(review, 0.84, 0.80)

    assert any("missing critic fields" in error for error in errors)
    assert any("unexpected critic fields" in error for error in errors)


def test_strict_construction_critic_protocol_rejects_self_contradictory_blocker():
    review = valid_reject()
    review["blocking_defects"][0]["defect"] = (
        "The extra checkpoint is not harmful and not a blocking defect."
    )

    errors = _construction_critic_protocol_errors(review, 0.84, 0.80)

    assert any("non-blocking or ambiguous" in error for error in errors)


def test_calibration_dataset_is_balanced_and_provenance_backed():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data/reasoning9000/critic-calibration/cases.jsonl"
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]

    assert validate_cases(cases, root) == []
    assert len(cases) == 14
    assert {case["gold_label"] for case in cases} == {"blocking", "non_blocking"}
    for case in cases:
        prompt = calibration_prompt(case)
        assert "gold_label" not in prompt
        assert "gold_rationale" not in prompt
        assert case["gold_rationale"] not in prompt


def test_calibration_report_requires_zero_false_decisions():
    cases = [
        {
            "case_id": "blocking-case",
            "gold_label": "blocking",
        },
        {
            "case_id": "non-blocking-case",
            "gold_label": "non_blocking",
        },
    ]
    predictions = {
        "blocking-case": {
            "case_id": "blocking-case",
            "classification": "blocking",
            "reason": "The required total is wrong.",
        },
        "non-blocking-case": {
            "case_id": "non-blocking-case",
            "classification": "blocking",
            "reason": "An optional table could be clearer.",
        },
    }

    report = build_report(cases, predictions, "critic", "prompt", "test")

    assert report["passed"] is False
    assert report["confusion"]["false_blocking"] == 1
    assert report["confusion"]["missed_blocking"] == 0


def test_generation_calibration_lock_binds_prompt_dataset_and_deployment(tmp_path):
    prompt = "strict critic prompt"
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/episode_critic_system.md").write_text(prompt, encoding="utf-8")
    dataset_path = tmp_path / "data/cases.jsonl"
    dataset_path.parent.mkdir()
    case = {"case_id": "case-1", "gold_label": "blocking"}
    dataset_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    report_path = tmp_path / "reports/calibration.json"
    report_path.parent.mkdir()
    report = {
        "passed": True,
        "deployment": "critic",
        "prompt_hash": stable_hash(prompt, 32),
        "dataset_hash": stable_hash([case], 32),
        "case_count": 1,
        "invalid_predictions": 0,
        "confusion": {"false_blocking": 0, "missed_blocking": 0},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "azure": {
            "deployment": "constructor",
            "episode_critic_deployment": "critic",
        },
        "generation": {
            "require_construction_critic_calibration": True,
            "construction_critic_calibration_dataset": "data/cases.jsonl",
            "construction_critic_calibration_report": "reports/calibration.json",
            "minimum_construction_critic_calibration_cases": 1,
        },
    }

    assert construction_critic_calibration_errors(config, tmp_path) == []

    report["prompt_hash"] = "stale"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    errors = construction_critic_calibration_errors(config, tmp_path)
    assert any("stale for the current prompt" in error for error in errors)


def test_reasoning_config_keeps_scale_locked_behind_critic_calibration():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/pipeline.reasoning9000.yaml").read_text(encoding="utf-8")
    )

    assert config["generation"]["require_construction_critic_calibration"] is True
    assert (root / "data/reasoning9000/STOP").is_file()
