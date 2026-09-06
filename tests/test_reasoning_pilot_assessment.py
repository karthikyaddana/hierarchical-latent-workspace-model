import copy
import json

from scripts.assess_reasoning_pilot import accepted_corpus_hash, assess
from tests.test_pipeline import sample_episode


def review_wrapper(episode, *, accepted=True, expertise=0.90, overall=0.90):
    review = {
        "episode_id": episode["episode_id"],
        "verdict": "accept" if accepted else "reject",
        "overall_score": overall,
        "scores": {"expertise_uplift": expertise},
    }
    return {
        "episode": episode,
        "review": review,
        "primary_review": copy.deepcopy(review),
        "secondary_review": copy.deepcopy(review),
        "accepted": accepted,
        "static_errors": [],
        "adversarial_errors": [],
        "judge_errors": [],
        "judge_consensus": accepted,
        "judge_metadata": {"deployment": "judge-primary"},
        "secondary_judge_metadata": {"deployment": "judge-secondary"},
    }


def write_wrapper(directory, name, wrapper):
    (directory / name).write_text(json.dumps(wrapper, ensure_ascii=False), encoding="utf-8")


def valid_manual_audit(*wrappers):
    accepted_ids = [
        item["episode"]["episode_id"] for item in wrappers if item.get("accepted")
    ]
    return {
        "decision": "pass",
        "reviewer": "test-reviewer",
        "completed_at": "2026-08-18T00:00:00Z",
        "accepted_corpus_hash": accepted_corpus_hash(list(wrappers)),
        "false_accept_episode_ids": [],
        "inspections": [
            {"episode_id": episode_id, "verdict": "pass", "notes": "Inspected."}
            for episode_id in accepted_ids
        ],
    }


def test_reasoning_pilot_requires_first_pass_quality(tmp_path):
    episode = sample_episode()
    episode["repair_metadata"] = {"round": 1}
    write_wrapper(tmp_path, "repaired.json", review_wrapper(episode))

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=0.50,
        minimum_domains=1,
    )

    assert report["accepted"] == 1
    assert report["first_pass_accepted"] == 0
    assert not report["gates"]["first_pass_acceptance_rate"]
    assert not report["scale_allowed"]


def test_reasoning_pilot_directly_detects_language_and_private_reasoning(tmp_path):
    episode = copy.deepcopy(sample_episode())
    episode["input"]["user_request"] = (
        "Encuentra el defecto, comprueba cada restriccion y publica solamente una respuesta respaldada."
    )
    episode["frame"]["objective"] = "Verificar la solucion con evidencia suficiente y casos limite concretos."
    episode["integration"]["published_answer"] = (
        "Internal reasoning: primero se revisa el codigo y despues se elige la correccion propuesta."
    )
    write_wrapper(tmp_path, "bad.json", review_wrapper(episode))

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=1.0,
        minimum_domains=1,
    )

    assert report["language_failures"] == 1
    assert report["private_reasoning_failures"] == 1
    assert not report["gates"]["english_only"]
    assert not report["gates"]["private_reasoning_free"]


def test_reasoning_pilot_accepts_consistent_clean_first_pass(tmp_path):
    wrapper = review_wrapper(sample_episode())
    write_wrapper(tmp_path, "clean.json", wrapper)

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=1.0,
        minimum_domains=1,
        manual_audit=valid_manual_audit(wrapper),
    )

    assert report["scale_allowed"]


def test_reasoning_pilot_rejects_accepted_wrapper_without_second_judge(tmp_path):
    wrapper = review_wrapper(sample_episode())
    wrapper["secondary_review"] = None
    wrapper["secondary_judge_metadata"] = None
    wrapper["judge_consensus"] = False
    write_wrapper(tmp_path, "missing-secondary.json", wrapper)

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=1.0,
        minimum_domains=1,
    )

    assert report["accepted_dual_judge_failures"] == 1
    assert not report["gates"]["dual_judge_consensus"]
    assert not report["scale_allowed"]


def test_reasoning_pilot_rejects_missing_manual_adversarial_inspection(tmp_path):
    wrapper = review_wrapper(sample_episode())
    write_wrapper(tmp_path, "clean-but-uninspected.json", wrapper)

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=1.0,
        minimum_domains=1,
    )

    assert not report["gates"]["manual_adversarial_inspection"]
    assert not report["scale_allowed"]


def test_empty_accepted_set_cannot_receive_vacuous_manual_pass(tmp_path):
    wrapper = review_wrapper(sample_episode(), accepted=False, expertise=0.70, overall=0.70)
    write_wrapper(tmp_path, "rejected.json", wrapper)

    report = assess(
        tmp_path,
        minimum_reviewed=1,
        minimum_acceptance=0.0,
        minimum_expertise=0.0,
        minimum_first_pass_acceptance=0.0,
        minimum_domains=0,
        minimum_overall=0.0,
        minimum_accepted_expertise=0.0,
        manual_audit=valid_manual_audit(wrapper),
    )

    assert not report["gates"]["manual_adversarial_inspection"]
    assert not report["scale_allowed"]


def test_mean_expertise_target_is_not_misused_as_per_episode_floor(tmp_path):
    first = sample_episode()
    second = copy.deepcopy(first)
    second["episode_id"] = "episode-0002"
    second["input"]["user_request"] = "Design and verify a second deterministic timeout correction."
    second["integration"]["published_answer"] = "Use a second bounded-wait design and verify its deterministic test."
    first_wrapper = review_wrapper(first, expertise=0.84)
    second_wrapper = review_wrapper(second, expertise=0.90)
    write_wrapper(tmp_path, "first.json", first_wrapper)
    write_wrapper(tmp_path, "second.json", second_wrapper)

    report = assess(
        tmp_path,
        minimum_reviewed=2,
        minimum_acceptance=1.0,
        minimum_expertise=0.85,
        minimum_first_pass_acceptance=1.0,
        minimum_domains=1,
        minimum_accepted_expertise=0.80,
        manual_audit=valid_manual_audit(first_wrapper, second_wrapper),
    )

    assert report["accepted_mean_expertise_uplift"] == 0.87
    assert report["accepted_review_inconsistencies"] == 0
    assert report["gates"]["accepted_expertise_floor"]
    assert report["scale_allowed"]
