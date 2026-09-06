from pathlib import Path

from hlwm_data.fast_generate import (
    _normalize_fast_episode,
    fast_episode_errors,
    fast_generation_deployments,
)
from hlwm_data.generate import EpisodeJob, _output_contract
from tests.test_pipeline import sample_episode


ROOT = Path(__file__).resolve().parents[1]


def test_fast_generation_uses_unique_parallel_model_pool():
    config = {
        "azure": {
            "deployment": "deepseek",
            "fast_deployments": ["deepseek", "luna", "deepseek", "kimi"],
        }
    }
    assert fast_generation_deployments(config) == ["deepseek", "luna", "kimi"]


def test_fast_episode_checks_schema_and_english_only():
    episode = sample_episode()
    assert fast_episode_errors(episode, ROOT / "schemas/episode.schema.json") == []


def test_fast_episode_rejects_blocked_non_english_script():
    episode = sample_episode()
    episode["integration"]["published_answer"] += " Проверка завершена."
    errors = fast_episode_errors(episode, ROOT / "schemas/episode.schema.json")
    assert any("blocked non-English script" in error for error in errors)


def test_fast_prompt_contains_direct_quality_preinstructions():
    prompt = (ROOT / "prompts/fast_episode_system.md").read_text(encoding="utf-8")
    required = [
        "complete synthetic reasoning-training episode",
        "genuinely complementary parallel lanes",
        "Do not reveal private chain-of-thought",
        "Keep every natural-language field in English",
        "fully deliver what it asks for",
    ]
    assert all(fragment in prompt for fragment in required)


def test_fast_normalizer_repairs_common_one_call_shape_differences():
    episode = sample_episode()
    episode["frame"]["failure_contract"] = "Do not publish an unsupported result."
    episode["barrier"]["conflicts"] = [{"description": "Resolve the lane mismatch."}]
    episode["counterfactuals"] = episode["counterfactuals"][:1]
    job = EpisodeJob(
        episode_id=episode["episode_id"],
        domain=episode["domain"],
        objective=episode["subdomain"],
        source_group=episode["source_group"],
        lineage_component_id=episode["lineage_component_id"],
        chunks=(
            {
                "source_id": "source-1",
                "chunk_id": "chunk-1",
            },
        ),
        variation_seed=1,
    )
    normalized = _normalize_fast_episode(episode, job)
    assert normalized["frame"]["failure_contract"] == [
        "Do not publish an unsupported result."
    ]
    assert isinstance(normalized["barrier"]["conflicts"][0], str)
    assert len(normalized["counterfactuals"]) == 2


def test_output_contract_requests_two_counterfactual_controls():
    job = EpisodeJob("episode-1", "coding", "debugging", "group", "lineage", tuple(), 1)
    config = {"generation": {"lanes_max": 4}}
    assert len(_output_contract(config, job)["counterfactuals"]) == 2
