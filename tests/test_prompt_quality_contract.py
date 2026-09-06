from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def test_blueprint_prompt_requires_feasible_parallel_evidence_contract():
    prompt = _prompt("blueprint_system.md")

    for requirement in (
        "support_quote",
        "evidence_mode",
        "marginal_necessity",
        "central_reconstruction",
        "At least two lanes must independently solve materially different parts",
    ):
        assert requirement in prompt

    quality_gate = _prompt("blueprint_judge_system.md")
    assert "expertise_uplift" in quality_gate
    assert "consumes a sibling's artifact" in quality_gate
    assert "Recomputed outcome" in quality_gate


def test_episode_and_repair_prompts_fail_closed_on_unresolved_claims():
    episode = _prompt("episode_system.md")
    repair = _prompt("repair_system.md")

    assert "cannot be `publish`" in episode
    for label in ("Inputs:", "Operation:", "Reconstructed result:", "Falsification:", "Verdict:"):
        assert label in episode
        assert label in repair
    assert "Any refuted/insufficient or open claim forbids `publish`" in repair
    assert "Episode-authored prose is never new evidence" in repair
