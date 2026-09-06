from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from hlwm_data.azure_client import AzureTeacherClient
from hlwm_data.config import load_pipeline_config
from hlwm_data.generate import generation_model_roles
from hlwm_data.util import atomic_write_json, iter_jsonl, stable_hash


LABELS = {"blocking", "non_blocking"}


def historical_prediction(case: Dict[str, Any]) -> str:
    if case.get("origin") == "pilot-003 confirmed false accept":
        return "non_blocking"
    return "blocking"


def calibration_prompt(case: Dict[str, Any]) -> str:
    return (
        "This is one blinded severity-calibration item extracted from a prior episode. "
        "Classify only the supplied allegation; do not infer unseen defects.\n\n"
        "CASE ID:\n%s\n\n"
        "EXACT CONTRACT:\n%s\n\n"
        "VISIBLE CANDIDATE EVIDENCE:\n%s\n\n"
        "ALLEGED DEFECT:\n%s\n\n"
        "Return exactly: "
        '{"case_id":"...","classification":"blocking|non_blocking","reason":"one concise evidence-based sentence"}'
        % (
            case["case_id"],
            case["contract"],
            case["candidate_evidence"],
            case["allegation"],
        )
    )


def calibration_system(production_prompt: str) -> str:
    return (
        production_prompt
        + "\n\nCALIBRATION MODE OVERRIDE: You are classifying one supplied allegation, not "
        "reviewing a full episode. Apply the same four-condition blocking test. Return only the "
        "three calibration fields requested by the user message. Do not expose private reasoning."
    )


def prediction_errors(case: Dict[str, Any], response: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if set(response) != {"case_id", "classification", "reason"}:
        errors.append("prediction must contain exactly case_id, classification, and reason")
    if response.get("case_id") != case.get("case_id"):
        errors.append("prediction case_id does not match")
    if response.get("classification") not in LABELS:
        errors.append("classification must be blocking or non_blocking")
    reason = str(response.get("reason", "")).strip()
    if not reason:
        errors.append("prediction reason is empty")
    if len(reason) > 420:
        errors.append("prediction reason exceeds 420 characters")
    lowered_reason = reason.lower()
    required_terms = [str(term).lower() for term in case.get("required_reason_terms", [])]
    if required_terms and any(term not in lowered_reason for term in required_terms):
        errors.append("prediction reason omits required visible evidence")
    forbidden_phrases = [
        str(phrase).lower() for phrase in case.get("forbidden_reason_phrases", [])
    ]
    if any(phrase in lowered_reason for phrase in forbidden_phrases):
        errors.append("prediction reason invents a fact excluded by the calibration case")
    return errors


def validate_cases(cases: Iterable[Dict[str, Any]], root: Path) -> List[str]:
    errors: List[str] = []
    seen = set()
    label_counts = {label: 0 for label in LABELS}
    for index, case in enumerate(cases, 1):
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            errors.append("case %d has a missing or duplicate case_id" % index)
        seen.add(case_id)
        label = case.get("gold_label")
        if label not in LABELS:
            errors.append("case %s has an invalid gold label" % case_id)
        else:
            label_counts[label] += 1
        for field in ("contract", "candidate_evidence", "allegation", "gold_rationale"):
            if not str(case.get(field, "")).strip():
                errors.append("case %s is missing %s" % (case_id, field))
        evidence_path = root / str(case.get("evidence_path", ""))
        if not evidence_path.is_file():
            errors.append("case %s evidence path does not exist" % case_id)
    for label, count in label_counts.items():
        if count < 6:
            errors.append("calibration set needs at least six %s cases" % label)
    return errors


def score_predictions(
    cases: List[Dict[str, Any]], predictions: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, int], List[Dict[str, Any]], int]:
    confusion = {
        "correct_blocking": 0,
        "correct_non_blocking": 0,
        "false_blocking": 0,
        "missed_blocking": 0,
    }
    rows: List[Dict[str, Any]] = []
    invalid = 0
    for case in cases:
        case_id = case["case_id"]
        response = predictions.get(case_id, {})
        errors = prediction_errors(case, response)
        prediction = response.get("classification") if not errors else None
        gold = case["gold_label"]
        if errors:
            invalid += 1
        elif prediction == gold == "blocking":
            confusion["correct_blocking"] += 1
        elif prediction == gold == "non_blocking":
            confusion["correct_non_blocking"] += 1
        elif prediction == "blocking":
            confusion["false_blocking"] += 1
        else:
            confusion["missed_blocking"] += 1
        rows.append(
            {
                "case_id": case_id,
                "gold_label": gold,
                "prediction": prediction,
                "correct": prediction == gold and not errors,
                "reason": response.get("reason"),
                "errors": errors,
            }
        )
    return confusion, rows, invalid


def build_report(
    cases: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    deployment: str,
    prompt_text: str,
    mode: str,
) -> Dict[str, Any]:
    confusion, rows, invalid = score_predictions(cases, predictions)
    correct = confusion["correct_blocking"] + confusion["correct_non_blocking"]
    passed = (
        len(cases) >= 12
        and invalid == 0
        and confusion["false_blocking"] == 0
        and confusion["missed_blocking"] == 0
    )
    return {
        "schema_version": "1.0",
        "mode": mode,
        "passed": passed,
        "deployment": deployment,
        "prompt_hash": stable_hash(prompt_text, 32),
        "dataset_hash": stable_hash(cases, 32),
        "case_count": len(cases),
        "accuracy": correct / len(cases) if cases else 0.0,
        "invalid_predictions": invalid,
        "confusion": confusion,
        "requirements": {
            "minimum_cases": 12,
            "false_blocking": 0,
            "missed_blocking": 0,
            "invalid_predictions": 0,
        },
        "cases": rows,
    }


def live_predictions(
    cases: List[Dict[str, Any]], config: Dict[str, Any], root: Path, prompt_text: str, workers: int
) -> Dict[str, Dict[str, Any]]:
    critic_config = dict(config["azure"])
    critic_config["deployment"] = generation_model_roles(config)["episode_construction_critic"]
    critic_config["temperature"] = 0.0
    critic_config["max_completion_tokens"] = 2048
    client = AzureTeacherClient(
        critic_config, root / "logs/construction-critic-calibration-azure-requests.jsonl"
    )
    system = calibration_system(prompt_text)

    def classify(case: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        completion = client.chat_json(
            system,
            calibration_prompt(case),
            "construction-critic-calibration",
            case["case_id"],
        )
        response = dict(completion.value)
        response["_provenance"] = {
            "request_id": completion.request_id,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "elapsed_seconds": completion.elapsed_seconds,
        }
        return case["case_id"], response

    predictions: Dict[str, Dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(classify, case): case["case_id"] for case in cases}
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    returned_id, response = future.result()
                    provenance = response.pop("_provenance")
                    response["provenance"] = provenance
                    predictions[returned_id] = response
                except Exception as exc:
                    predictions[case_id] = {
                        "case_id": case_id,
                        "classification": None,
                        "reason": "",
                        "error": str(exc),
                    }
    finally:
        client.close()
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure construction-critic blocking/non-blocking discrimination."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="config/pipeline.reasoning9000.yaml")
    parser.add_argument(
        "--dataset", default="data/reasoning9000/critic-calibration/cases.jsonl"
    )
    parser.add_argument(
        "--output", default="reports/reasoning9000-construction-critic-calibration.json"
    )
    parser.add_argument("--mode", choices=("historical", "live"), default="historical")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config = load_pipeline_config(root / args.config)
    cases = list(iter_jsonl(root / args.dataset))
    case_errors = validate_cases(cases, root)
    if case_errors:
        raise SystemExit("Invalid calibration dataset: %s" % "; ".join(case_errors))
    prompt_text = (root / "prompts/episode_critic_system.md").read_text(encoding="utf-8")
    deployment = generation_model_roles(config)["episode_construction_critic"]
    if args.mode == "historical":
        predictions = {
            case["case_id"]: {
                "case_id": case["case_id"],
                "classification": historical_prediction(case),
                "reason": "Historical decision reconstructed from the archived run outcome.",
            }
            for case in cases
        }
    else:
        raw_predictions = live_predictions(cases, config, root, prompt_text, args.workers)
        predictions = {}
        for case_id, response in raw_predictions.items():
            provenance = response.pop("provenance", None)
            predictions[case_id] = response
            if provenance:
                predictions[case_id]["_provenance"] = provenance

    scoring_predictions = {
        case_id: {key: value for key, value in response.items() if not key.startswith("_")}
        for case_id, response in predictions.items()
    }
    report = build_report(
        cases, scoring_predictions, deployment, prompt_text, args.mode
    )
    for row in report["cases"]:
        provenance = predictions.get(row["case_id"], {}).get("_provenance")
        if provenance:
            row["provenance"] = provenance
    atomic_write_json(root / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
