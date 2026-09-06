from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .util import resolve_env


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping in %s" % path)
    return value


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    config = load_yaml(path)
    azure = config.setdefault("azure", {})
    azure["endpoint"] = resolve_env(azure.get("endpoint"), "HLWM_AZURE_ENDPOINT")
    azure["deployment"] = resolve_env(azure.get("deployment"), "HLWM_AZURE_DEPLOYMENT")
    azure["blueprint_deployment"] = resolve_env(
        azure.get("blueprint_deployment", azure.get("deployment")),
        "HLWM_AZURE_BLUEPRINT_DEPLOYMENT",
    )
    azure["judge_deployment"] = resolve_env(
        azure.get("judge_deployment", azure.get("deployment")),
        "HLWM_AZURE_JUDGE_DEPLOYMENT",
    )
    azure["blueprint_judge_deployment"] = resolve_env(
        azure.get("blueprint_judge_deployment", azure.get("judge_deployment")),
        "HLWM_AZURE_BLUEPRINT_JUDGE_DEPLOYMENT",
    )
    azure["episode_critic_deployment"] = resolve_env(
        azure.get("episode_critic_deployment"),
        "HLWM_AZURE_EPISODE_CRITIC_DEPLOYMENT",
    )
    azure["episode_editor_deployment"] = resolve_env(
        azure.get("episode_editor_deployment", azure.get("blueprint_deployment")),
        "HLWM_AZURE_EPISODE_EDITOR_DEPLOYMENT",
    )
    azure["secondary_judge_deployment"] = resolve_env(
        azure.get("secondary_judge_deployment"),
        "HLWM_AZURE_SECONDARY_JUDGE_DEPLOYMENT",
    )
    azure["scope"] = resolve_env(azure.get("scope"), "HLWM_AZURE_SCOPE")
    azure["max_workers"] = resolve_env(azure.get("max_workers", 8), "HLWM_MAX_WORKERS", int)
    azure["requests_per_minute"] = resolve_env(
        azure.get("requests_per_minute", 60), "HLWM_REQUESTS_PER_MINUTE", int
    )
    required = ["endpoint", "deployment", "scope"]
    missing = [key for key in required if not azure.get(key)]
    if missing:
        raise ValueError("Missing Azure configuration: %s" % ", ".join(missing))
    return config
