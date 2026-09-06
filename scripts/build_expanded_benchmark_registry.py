#!/usr/bin/env python3
"""Build a transparent English benchmark registry without copying benchmark rows.

The registry records discoverable suites and their execution requirements.  It does
not download questions or make a dataset eligible for training.  Official training
splits are handled separately from protected evaluation splits.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LM_EVAL_TASKS: Dict[str, List[str]] = {
    "reasoning_and_knowledge": [
        "agieval", "anli", "arc", "babi", "babilong", "bbh", "bigbench",
        "commonsense_qa", "drop", "fld", "gpqa", "graphwalks", "hellaswag",
        "logiqa", "logiqa2", "mastermind", "mmlu", "mmlu_pro",
        "mmlu-pro-plus", "mmlu-redux", "mmlusr", "moral_stories", "mutual",
        "openbookqa", "piqa", "prost", "race", "sciq", "siqa", "storycloze",
        "super_glue", "swag", "triviaqa", "winogrande", "wsc273",
    ],
    "mathematics": [
        "aime", "arithmetic", "asdiv", "gsm8k", "gsm8k_platinum", "gsm_plus",
        "hendrycks_math", "mathqa", "mc_taco", "minerva_math", "putnam_axiom",
    ],
    "coding_and_structure": [
        "code_x_glue", "cruxeval", "humaneval", "humaneval_infilling", "jsonschema_bench", "mbpp",
    ],
    "instruction_truth_and_safety": [
        "bbq", "crows_pairs", "discrim_eval", "hendrycks_ethics", "ifeval",
        "inverse_scaling", "model_written_evals", "realtoxicityprompts", "toxigen",
        "truthfulqa", "winogender", "wmdp",
    ],
    "long_context_and_qa": [
        "coqa", "infinitebench", "lambada", "longbench", "longbench2", "nq_open",
        "qasper", "ruler", "scrolls", "squadv2", "webqs",
    ],
    "language_and_generation_quality": [
        "blimp", "eq_bench", "legalbench", "paloma", "tinyBenchmarks", "unscramble", "wikitext",
    ],
}


# These are benchmark code or dataset landing pages. Reachability is checked live;
# dataset licences and split policies still require a separate audit.
SPECIALIZED: List[Tuple[str, str, str, str]] = [
    # Coding, debugging and repository work.
    ("EvalPlus", "coding", "https://github.com/evalplus/evalplus", "code_execution"),
    ("LiveCodeBench", "coding", "https://github.com/LiveCodeBench/LiveCodeBench", "code_execution"),
    ("BigCodeBench", "coding", "https://github.com/bigcode-project/bigcodebench", "code_execution"),
    ("APPS", "coding", "https://github.com/hendrycks/apps", "code_execution"),
    ("CodeContests", "coding", "https://github.com/google-deepmind/code_contests", "code_execution"),
    ("DS-1000", "data_science_code", "https://github.com/xlang-ai/DS-1000", "code_execution"),
    ("SciCode", "scientific_code", "https://github.com/scicode-bench/SciCode", "code_execution"),
    ("SWE-bench", "repository_debugging", "https://github.com/SWE-bench/SWE-bench", "container_replay"),
    ("RepoBench", "repository_code", "https://github.com/Leolty/repobench", "repository_context"),
    ("CrossCodeEval", "repository_code", "https://github.com/amazon-science/cceval", "repository_context"),
    ("ClassEval", "coding", "https://github.com/FudanSELab/ClassEval", "code_execution"),
    ("DevEval", "coding", "https://github.com/seketeam/DevEval", "code_execution"),
    ("CoderEval", "coding", "https://github.com/CoderEval/CoderEval", "code_execution"),
    ("NaturalCodeBench", "coding", "https://github.com/THUDM/NaturalCodeBench", "code_execution"),
    ("MLE-bench", "ml_engineering", "https://github.com/openai/mle-bench", "container_replay"),
    ("MLAgentBench", "ml_engineering", "https://github.com/snap-stanford/MLAgentBench", "container_replay"),
    ("Defects4J", "repository_debugging", "https://github.com/rjust/defects4j", "container_replay"),
    ("BugsInPy", "repository_debugging", "https://github.com/soarsmu/BugsInPy", "container_replay"),
    # SQL, tables, analytics and spreadsheets.
    ("Spider", "text_to_sql", "https://github.com/taoyds/spider", "database_execution"),
    ("Spider-2.0", "text_to_sql", "https://github.com/xlang-ai/Spider2", "database_execution"),
    ("BIRD", "text_to_sql", "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird", "database_execution"),
    ("WikiSQL", "text_to_sql", "https://github.com/salesforce/WikiSQL", "database_execution"),
    ("SParC", "text_to_sql", "https://github.com/taoyds/sparc", "database_execution"),
    ("CoSQL", "text_to_sql", "https://github.com/taoyds/cosql", "database_execution"),
    ("Dr.Spider", "text_to_sql", "https://github.com/awslabs/diagnostic-robustness-text-to-sql", "database_execution"),
    ("KaggleDBQA", "text_to_sql", "https://github.com/Chia-Hsuan-Lee/KaggleDBQA", "database_execution"),
    ("TableBench", "table_reasoning", "https://github.com/TableBench/TableBench", "table_execution"),
    ("SpreadsheetBench", "spreadsheet", "https://github.com/RUCKBReasoning/SpreadsheetBench", "spreadsheet_execution"),
    ("InsightBench", "data_analysis", "https://github.com/ServiceNow/insight-bench", "artifact_and_human_judge"),
    # Finance and business analysis.
    ("FinQA", "financial_reasoning", "https://github.com/czyssrs/FinQA", "deterministic_calculation"),
    ("ConvFinQA", "financial_reasoning", "https://github.com/czyssrs/ConvFinQA", "deterministic_calculation"),
    ("TAT-QA", "financial_reasoning", "https://github.com/NExTplusplus/TAT-QA", "deterministic_calculation"),
    ("MultiHiertt", "financial_reasoning", "https://github.com/psunlpgroup/MultiHiertt", "deterministic_calculation"),
    ("FinanceBench", "financial_analysis", "https://github.com/patronus-ai/financebench", "source_grounded_qa"),
    ("FinBen", "financial_analysis", "https://github.com/The-FinAI/PIXIU", "mixed_finance_eval"),
    ("BizBench", "business_reasoning", "https://huggingface.co/datasets/kensho/bizbench", "mixed_business_eval"),
    # Tools, agents and interactive environments.
    ("BFCL", "function_calling", "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", "tool_schema_execution"),
    ("ToolBench", "tool_use", "https://github.com/OpenBMB/ToolBench", "tool_replay"),
    ("API-Bank", "tool_use", "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank", "tool_replay"),
    ("ToolAlpaca", "tool_use", "https://github.com/tangqiaoyu/ToolAlpaca", "tool_replay"),
    ("StableToolBench", "tool_use", "https://github.com/THUNLP-MT/StableToolBench", "tool_replay"),
    ("ToolSandbox", "tool_use", "https://github.com/apple/ToolSandbox", "tool_replay"),
    ("tau-bench", "tool_use", "https://github.com/sierra-research/tau-bench", "environment_execution"),
    ("tau2-bench", "tool_use", "https://github.com/sierra-research/tau2-bench", "environment_execution"),
    ("AgentBench", "agents", "https://github.com/THUDM/AgentBench", "environment_execution"),
    ("AgentBoard", "agents", "https://github.com/hkust-nlp/agentboard", "environment_execution"),
    ("GAIA", "agents", "https://huggingface.co/datasets/gaia-benchmark/GAIA", "tool_environment"),
    ("AssistantBench", "agents", "https://github.com/oriyor/assistantbench", "web_environment"),
    ("WebArena", "browser_agents", "https://github.com/web-arena-x/webarena", "web_environment"),
    ("WorkArena", "browser_agents", "https://github.com/ServiceNow/WorkArena", "web_environment"),
    ("BrowserGym", "browser_agents", "https://github.com/ServiceNow/BrowserGym", "web_environment"),
    ("Mind2Web", "browser_agents", "https://github.com/OSU-NLP-Group/Mind2Web", "web_environment"),
    ("WebLINX", "browser_agents", "https://github.com/McGill-NLP/weblinx", "web_environment"),
    ("WebShop", "browser_agents", "https://github.com/princeton-nlp/WebShop", "web_environment"),
    ("ALFWorld", "agents", "https://github.com/alfworld/alfworld", "environment_execution"),
    ("ScienceWorld", "agents", "https://github.com/allenai/ScienceWorld", "environment_execution"),
    ("InterCode", "agents", "https://github.com/princeton-nlp/intercode", "environment_execution"),
    ("AppWorld", "agents", "https://github.com/StonyBrookNLP/appworld", "environment_execution"),
    ("MINT", "agents", "https://github.com/xingyaoww/mint-bench", "tool_environment"),
    ("AgentDojo", "agent_safety", "https://github.com/ethz-spylab/agentdojo", "tool_environment"),
    ("OSWorld", "computer_use", "https://github.com/xlang-ai/OSWorld", "computer_environment"),
    ("AndroidWorld", "computer_use", "https://github.com/google-research/android_world", "computer_environment"),
    ("BrowseComp-Plus", "browser_agents", "https://github.com/texttron/BrowseComp-Plus", "web_environment"),
    ("MCP-Universe", "mcp_agents", "https://github.com/SalesforceAIResearch/MCP-Universe", "mcp_environment"),
    ("ACEBench", "function_calling", "https://github.com/ACEBench/ACEBench", "tool_schema_execution"),
    ("xLAM", "function_calling", "https://github.com/SalesforceAIResearch/xLAM", "tool_schema_execution"),
    # Instruction following, factuality and open-ended quality.
    ("FollowBench", "instruction_following", "https://github.com/YJiangcm/FollowBench", "deterministic_and_judge"),
    ("InfoBench", "instruction_following", "https://github.com/qinyiwei/InfoBench", "deterministic_and_judge"),
    ("ComplexBench", "instruction_following", "https://github.com/thu-coai/ComplexBench", "deterministic_and_judge"),
    ("WildBench", "open_ended_quality", "https://github.com/allenai/WildBench", "human_or_model_judge"),
    ("Arena-Hard-Auto", "open_ended_quality", "https://github.com/lm-sys/arena-hard-auto", "model_judge"),
    ("AlpacaEval", "open_ended_quality", "https://github.com/tatsu-lab/alpaca_eval", "model_judge"),
    ("MT-Bench", "open_ended_quality", "https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge", "model_judge"),
    ("LiveBench", "reasoning", "https://github.com/LiveBench/LiveBench", "mixed_eval"),
    ("SimpleQA", "factuality", "https://github.com/openai/simple-evals", "exact_and_judge"),
    ("FreshQA", "factuality", "https://github.com/freshllms/freshqa", "source_grounded_qa"),
    ("HaluEval", "hallucination", "https://github.com/RUCAIBox/HaluEval", "classification_and_generation"),
    ("RAGTruth", "hallucination", "https://github.com/ParticleMedia/RAGTruth", "span_and_claim_eval"),
    ("RewardBench", "preference", "https://github.com/allenai/reward-bench", "preference_accuracy"),
    ("JudgeBench", "judging", "https://github.com/ScalerLab/JudgeBench", "judge_accuracy"),
    ("VibeEval", "open_ended_quality", "https://github.com/vibeeval/vibeeval", "human_or_model_judge"),
]


def get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "hlwm-benchmark-registry/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def verify_url(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "hlwm-benchmark-registry/1.0"}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {"reachable": True, "http_status": response.status, "resolved_url": response.geturl()}
    except urllib.error.HTTPError as exc:
        return {"reachable": False, "http_status": exc.code, "verification_error": str(exc)[:300]}
    except Exception as exc:  # network state is recorded rather than hidden
        return {"reachable": False, "http_status": None, "verification_error": str(exc)[:300]}


def lm_eval_entries() -> Tuple[List[Dict[str, Any]], List[str]]:
    api = "https://api.github.com/repos/EleutherAI/lm-evaluation-harness/contents/lm_eval/tasks"
    rows = get_json(api)
    present = {row.get("name") for row in rows if row.get("type") == "dir"}
    entries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for capability, task_names in LM_EVAL_TASKS.items():
        for task in task_names:
            if task not in present:
                missing.append(task)
                continue
            entries.append({
                "name": task,
                "registry": "EleutherAI/lm-evaluation-harness",
                "capability": capability,
                "url": "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/%s" % task,
                "language_scope": "English configuration only",
                "execution": "lm_eval_harness",
                "registry_verified": True,
                "dataset_license_status": "audit_each_underlying_dataset",
            })
    return entries, missing


def specialized_entries(workers: int) -> List[Dict[str, Any]]:
    verifications: List[Dict[str, Any]]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        verifications = list(executor.map(lambda row: verify_url(row[2]), SPECIALIZED))
    entries: List[Dict[str, Any]] = []
    for (name, capability, url, execution), verification in zip(SPECIALIZED, verifications):
        entries.append({
            "name": name,
            "registry": "specialized",
            "capability": capability,
            "url": url,
            "language_scope": "English subset or configuration required",
            "execution": execution,
            "dataset_license_status": "audit_before_download_or_training_split_use",
            **verification,
        })
    return entries


def count_by(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    lm_rows, missing = lm_eval_entries()
    specialized_rows = specialized_entries(args.workers)
    entries = lm_rows + specialized_rows
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "English-capable benchmark registry; no benchmark rows are copied by this file",
        "policy": {
            "official_training_splits_may_feed_training_after_license_and_contamination_audit": True,
            "validation_test_hidden_challenge_and_future_splits_are_evaluation_only": True,
            "paraphrasing_evaluation_items_for_training_is_forbidden": True,
            "generated_analogues_must_change_entities_numbers_structure_and_solution_path": True,
            "base_model_pretraining_contamination_is_reported_as_unknown": True,
        },
        "counts": {
            "total": len(entries),
            "lm_eval_registry": len(lm_rows),
            "specialized": len(specialized_rows),
            "specialized_reachable": sum(bool(row.get("reachable")) for row in specialized_rows),
            "by_capability": count_by(entries, "capability"),
        },
        "missing_lm_eval_tasks": missing,
        "benchmarks": entries,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["counts"], "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
