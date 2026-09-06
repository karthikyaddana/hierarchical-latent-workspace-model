import json
import sys
from pathlib import Path

import pytest

from hlwm_data.builder_factory import (
    BuilderFactory,
    _investor_metric_defects,
    _plan_consistency_defects,
    _split_for,
    numbers_grounded,
    parse_file_blocks,
    run_pytest,
    verify_code_bundle,
)
from hlwm_data.builder_seeds import BuilderTask, expand_tasks

MIX = {
    "fullstack_product": 2,
    "coding_debugging": 3,
    "website_frontend": 2,
    "copywriting": 2,
    "pitch_deck": 1,
    "investor_analysis": 2,
    "project_planning": 2,
    "devops": 2,
}


def test_expand_tasks_is_deterministic_and_covers_the_completeness_bar():
    first = expand_tasks(MIX, seed=60309)
    second = expand_tasks(MIX, seed=60309)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert {task.domain for task in first} == set(MIX)
    fullstack = [task for task in first if task.domain == "fullstack_product"][0]
    combined = " ".join(fullstack.context).lower()
    for surface in ("close-account", "advertising preferences", "conditions of use", "signup"):
        assert surface in combined or surface in fullstack.user_request.lower()
    different_seed = expand_tasks(MIX, seed=7)
    invest_a = [t for t in first if t.domain == "investor_analysis"][0]
    invest_repeat = [t for t in second if t.domain == "investor_analysis"][0]
    invest_b = [t for t in different_seed if t.domain == "investor_analysis"][0]
    assert invest_a.facts == invest_repeat.facts
    assert invest_a.task_id != invest_b.task_id


def test_parse_file_blocks_roundtrip_and_path_safety():
    text = (
        'preamble\n<<<FILE path="core/x.py">>>\nprint("hi")\n<<<END FILE>>>\n'
        '<<<FILE path="tests/test_x.py">>>\nimport core.x\n<<<END FILE>>>'
    )
    files = parse_file_blocks(text)
    assert set(files) == {"core/x.py", "tests/test_x.py"}
    with pytest.raises(ValueError):
        parse_file_blocks('<<<FILE path="../evil.py">>>\nx\n<<<END FILE>>>')
    with pytest.raises(ValueError):
        parse_file_blocks('<<<FILE path="/abs.py">>>\nx\n<<<END FILE>>>')
    with pytest.raises(ValueError):
        parse_file_blocks("no blocks at all")


GREEN_BUNDLE = {
    "core/adder.py": "def add(a: int, b: int) -> int:\n    return a + b\n",
    "tests/test_adder.py": (
        "from core.adder import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    ),
    "core/__init__.py": "",
    "tests/__init__.py": "",
}


def test_verify_code_bundle_green_then_red():
    report = verify_code_bundle(GREEN_BUNDLE)
    assert report["defects"] == []
    assert report["tests"]["green"] is True

    red = dict(GREEN_BUNDLE)
    red["core/adder.py"] = "def add(a, b):\n    return a - b\n"
    report = verify_code_bundle(red)
    assert any("pytest failed" in defect for defect in report["defects"])
    assert report["tests"]["failed"] >= 1


def test_verify_code_bundle_flags_policy_violations():
    bundle = dict(GREEN_BUNDLE)
    bundle["core/net.py"] = "import requests\n"
    bundle["core/danger.py"] = "import os\nos.system('ls')\n"
    bundle["config.py"] = 'KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"\n'
    report = verify_code_bundle(bundle, run_tests=False)
    joined = "\n".join(report["defects"])
    assert "forbidden import 'requests'" in joined
    assert "forbidden call pattern" in joined
    assert "literal secret" in joined


def test_run_pytest_reports_counts():
    result = run_pytest(GREEN_BUNDLE)
    assert result["green"] and result["passed"] == 1


def test_truncated_css_and_missing_core_import_are_flagged():
    bundle = {
        "web/index.html": "<html><head><title>x</title></head><body>hero features</body></html>",
        "web/styles.css": ":root { --x: 1; }\n.hero { color: #2aa",
        "api/routes.py": "from core.auth import login\nfrom core import cart\n",
        "core/__init__.py": "",
    }
    report = verify_code_bundle(bundle, required_sections=["hero"], run_tests=False)
    joined = "\n".join(report["defects"])
    assert "ends mid-rule" in joined or "unbalanced CSS braces" in joined
    assert "imports core.auth but core/auth.py does not exist" in joined
    assert "core/cart.py does not exist" in joined


def test_numbers_grounded_allows_facts_and_flags_inventions():
    facts = {"price_usd": 249, "coverage_sqft": 540, "arr_usd": 420000}
    text = "Only $249 for 540 sq ft of coverage. ARR is 420K (from 420000). Established 2024."
    assert numbers_grounded(text, [facts]) == []
    flagged = numbers_grounded("93% of customers agree and NPS is 71.", [facts])
    assert 93.0 in flagged and 71.0 in flagged


def test_investor_metric_check_accepts_exact_and_rejects_wrong():
    facts = {
        "revenue_by_year_usd": [400000, 800000, 1600000],
        "gross_margin_pct": 75,
        "monthly_burn_usd": 100000,
        "cash_on_hand_usd": 2400000,
        "paying_customers": 100,
        "acv_usd": 16000,
        "monthly_logo_churn_pct": 2.0,
        "cac_usd": 6000,
    }
    good = {
        "revenue_cagr_pct": 100.0,
        "gross_profit_latest_usd": 1200000.0,
        "runway_months": 24.0,
        "arpa_usd": 16000.0,
        "ltv_usd": 50000.0,
        "ltv_to_cac": 50000.0 / 6000.0,
    }
    assert _investor_metric_defects(facts, good) == []
    bad = dict(good, runway_months=36.0)
    assert any("runway_months" in defect for defect in _investor_metric_defects(facts, bad))


def test_plan_consistency_detects_cycles_and_bad_critical_path():
    good = {
        "milestones": [
            {"id": "m1", "name": "a", "duration_days": 5, "depends_on": []},
            {"id": "m2", "name": "b", "duration_days": 10, "depends_on": ["m1"]},
        ],
        "critical_path": ["m1", "m2"],
        "acceptance_criteria": ["m1 done", "m2 shipped"],
    }
    assert _plan_consistency_defects(good) == []
    cyclic = json.loads(json.dumps(good))
    cyclic["milestones"][0]["depends_on"] = ["m2"]
    assert any("cycle" in defect for defect in _plan_consistency_defects(cyclic))
    wrong_path = json.loads(json.dumps(good))
    wrong_path["critical_path"] = ["m2", "m1"]
    assert _plan_consistency_defects(wrong_path)


def test_split_assignment_is_stable_per_lineage():
    for lineage in ("product-ecommerce-vendora", "debug-lru_ttl_cache", "invest-003"):
        assert _split_for(lineage) == _split_for(lineage)
        assert _split_for(lineage) in ("train", "validation", "test")


def _offline_factory(tmp_path: Path) -> BuilderFactory:
    config = {
        "project": {"seed": 60309},
        "builder_factory": {"output_dir": str(tmp_path / "builder"), "mix": MIX},
    }
    return BuilderFactory(config, tmp_path)


def test_assembled_episode_loads_through_the_v55_trainer(tmp_path):
    factory = _offline_factory(tmp_path)
    task = [t for t in expand_tasks(MIX, seed=60309) if t.domain == "coding_debugging"][0]
    state = {
        "task_id": task.task_id,
        "stage": "built",
        "files": dict(GREEN_BUNDLE),
        "attempts": [{"kind": "files", "defects": ["pytest failed: boom"]}],
        "builder": "nvidia_nemotron_ultra",
        "bug": {
            "module_path": "core/adder.py",
            "buggy_source": "def add(a, b):\n    return a - b\n",
            "bug_category": "comparison direction",
            "bug_summary": "subtraction instead of addition",
            "failing_output": "1 failed",
            "failed": 1,
            "injector": "azure_deepseek_v4_flash",
        },
    }
    deterministic = {"defects": [], "tests": {"green": True, "passed": 1, "failed": 0}}
    episode = factory._assemble_episode(task, state, deterministic)
    assert episode["commitment"]["decision"] == "publish"
    assert episode["generation_metadata"]["independently_adjudicated"] is True
    assert episode["barrier"]["open_claims"] == []
    assert episode["debug_pair"]["fixed_source"].startswith("def add")

    trainer_dir = Path(__file__).resolve().parents[1] / "experiments" / "kaggle_hlwm"
    sys.path.insert(0, str(trainer_dir))
    try:
        from data import normalize_episode

        normalized = normalize_episode(episode, num_lanes=2)
    finally:
        sys.path.remove(str(trainer_dir))
    assert normalized["public_target"], "published answer must survive trainer normalization"
    assert len(normalized["lane_briefs"]) == 2
    assert normalized["commitment_target"] == 1.0
    assert normalized["risk_target"] == 0.0
    assert normalized["policy_supervision_eligible"] is True


def test_abstention_episode_when_tests_fail(tmp_path):
    factory = _offline_factory(tmp_path)
    task = [t for t in expand_tasks(MIX, seed=60309) if t.domain == "coding_debugging"][0]
    state = {"task_id": task.task_id, "stage": "built", "files": dict(GREEN_BUNDLE), "attempts": []}
    deterministic = {
        "defects": ["pytest failed: assertion"],
        "tests": {"green": False, "passed": 0, "failed": 1},
    }
    episode = factory._assemble_episode(task, state, deterministic)
    assert episode["commitment"]["decision"] == "abstain"
    assert episode["integration"]["published_answer"] == ""
    assert episode["barrier"]["open_claims"]
