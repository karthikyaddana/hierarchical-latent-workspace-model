#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from hlwm_data.util import stable_hash, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "sources" / "public" / "expert-curated" / "kaggle-verified-analytics.jsonl"
AB_PATH = ROOT / "sources" / "public" / "kaggle" / "faviovaz__marketing-ab-testing" / "files" / "marketing_AB.csv"
STARTUP_PATH = ROOT / "sources" / "public" / "kaggle" / "dhrubangtalukdar__startup-funding-and-outcome-dataset" / "files" / "startup_success_dataset.csv"


def pair(
    pair_id: str,
    prompt: str,
    chosen: str,
    rejected: str,
    defect: str,
    domain: str,
    lineage: str,
    task_type: str,
) -> Dict[str, Any]:
    return {
        "id": pair_id,
        "prompt": prompt,
        "context": [],
        "constraints": ["Use only the supplied aggregates and state the limits of inference."],
        "chosen": chosen,
        "rejected": rejected,
        "rejected_defect": defect,
        "domain": domain,
        "difficulty": 3,
        "response_mode": "brief",
        "source_group": lineage,
        "lineage_component_id": lineage,
        "mean_judge_score": 1.0,
        "judge_models": ["deterministic-arithmetic-grader"],
        "task_type": task_type,
        "policy_labels": {"chosen_publish": 1, "chosen_risk": 0, "rejected_publish": 0, "rejected_risk": 1},
        "provenance_sha256": stable_hash({"id": pair_id, "prompt": prompt, "chosen": chosen}, 64),
    }


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def ab_pairs() -> List[Dict[str, Any]]:
    aggregates: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: {"ad": [0, 0], "psa": [0, 0]}
    )
    with AB_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            group = str(row["test group"])
            converted = str(row["converted"]).lower() == "true"
            user_id = int(row["user id"])
            total_ads = int(row["total ads"])
            if total_ads <= 5:
                exposure = "0-5"
            elif total_ads <= 10:
                exposure = "6-10"
            elif total_ads <= 20:
                exposure = "11-20"
            elif total_ads <= 50:
                exposure = "21-50"
            else:
                exposure = "51+"
            segments = (
                ("all", "all participants"),
                ("day", str(row["most ads day"])),
                ("hour", str(row["most ads hour"])),
                ("exposure", exposure),
                ("stable_cohort", str(user_id % 20)),
            )
            for kind, value in segments:
                totals = aggregates[(kind, value)][group]
                totals[0] += 1
                totals[1] += int(converted)
    rows: List[Dict[str, Any]] = []
    for index, ((kind, value), groups) in enumerate(sorted(aggregates.items())):
        ad_total, ad_converted = groups["ad"]
        psa_total, psa_converted = groups["psa"]
        if min(ad_total, psa_total) < 20:
            continue
        ad_rate = percent(ad_converted, ad_total)
        psa_rate = percent(psa_converted, psa_total)
        absolute = ad_rate - psa_rate
        relative = 100.0 * (ad_rate / psa_rate - 1.0) if psa_rate else 0.0
        prompt = (
            "An A/B test segment (%s=%s) contains these deterministic aggregates:\n"
            "ad: %d conversions from %d users\npsa control: %d conversions from %d users\n\n"
            "Calculate both conversion rates, absolute percentage-point difference and relative uplift. "
            "Give a bounded interpretation; do not claim statistical significance from aggregates alone."
            % (kind, value, ad_converted, ad_total, psa_converted, psa_total)
        )
        chosen = (
            "Ad conversion = %d/%d = %.4f%%. Control conversion = %d/%d = %.4f%%. "
            "The observed absolute difference is %.4f percentage points and the relative uplift is %.2f%%. "
            "This is a descriptive segment result, not proof of significance or causality; validate randomization, "
            "sample-ratio mismatch, confidence intervals and guardrail metrics before deciding to ship."
            % (ad_converted, ad_total, ad_rate, psa_converted, psa_total, psa_rate, absolute, relative)
        )
        rejected = (
            "The ad conversion rate is %.4f%% and the campaign definitely caused a %.2f%% increase, so ship it immediately."
            % (psa_rate, abs(relative) + 10.0)
        )
        rows.append(
            pair(
                "kaggle-ab-%03d" % index,
                prompt,
                chosen,
                rejected,
                "It swaps a rate, invents a different uplift and claims causal certainty without statistical checks.",
                "experimentation_and_analytics",
                "kaggle-faviovaz-marketing-ab-v1",
                "ab_test_arithmetic",
            )
        )
    return rows


def startup_group(rows: Iterable[Mapping[str, str]], field: str) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "positive": 0.0, "revenue": 0.0, "burn": 0.0, "traction": 0.0}
    )
    for row in rows:
        key = str(row[field])
        item = values[key]
        item["count"] += 1
        item["positive"] += int(str(row["outcome"]) in {"Acquired", "IPO"})
        item["revenue"] += float(row["revenue_million"])
        item["burn"] += float(row["burn_rate_million"])
        item["traction"] += float(row["product_traction_users"])
    return values


def startup_pairs() -> List[Dict[str, Any]]:
    with STARTUP_PATH.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    output: List[Dict[str, Any]] = []
    index = 0
    for field in ("sector", "investor_type", "founder_background", "funding_rounds"):
        groups = startup_group(source_rows, field)
        keys = sorted(groups)
        for left, right in itertools.combinations(keys, 2):
            a, b = groups[left], groups[right]
            if min(a["count"], b["count"]) < 100:
                continue
            a_rate = percent(int(a["positive"]), int(a["count"]))
            b_rate = percent(int(b["positive"]), int(b["count"]))
            a_revenue = a["revenue"] / a["count"]
            b_revenue = b["revenue"] / b["count"]
            a_burn = a["burn"] / a["count"]
            b_burn = b["burn"] / b["count"]
            prompt = (
                "This is a synthetic startup dataset for arithmetic practice, not real market evidence. Compare %s groups:\n"
                "%s: n=%d, IPO-or-acquired=%d, total revenue=%.3f, total burn=%.3f\n"
                "%s: n=%d, IPO-or-acquired=%d, total revenue=%.3f, total burn=%.3f\n\n"
                "Compute success rates and average revenue and burn. State what cannot be inferred."
                % (
                    field,
                    left,
                    int(a["count"]),
                    int(a["positive"]),
                    a["revenue"],
                    a["burn"],
                    right,
                    int(b["count"]),
                    int(b["positive"]),
                    b["revenue"],
                    b["burn"],
                )
            )
            chosen = (
                "%s: %.3f%% synthetic success rate, %.3f average revenue and %.3f average burn. "
                "%s: %.3f%% synthetic success rate, %.3f average revenue and %.3f average burn. "
                "The descriptive success-rate difference is %.3f percentage points. Because the rows are simulated and "
                "not randomized, this comparison cannot establish real-world performance, causality, investability or a forecast."
                % (left, a_rate, a_revenue, a_burn, right, b_rate, b_revenue, b_burn, a_rate - b_rate)
            )
            rejected = (
                "%s startups are guaranteed to outperform %s startups in the real world, so investors should fund them without further diligence."
                % (left, right)
            )
            output.append(
                pair(
                    "kaggle-startup-%03d" % index,
                    prompt,
                    chosen,
                    rejected,
                    "It turns synthetic descriptive data into a causal investment recommendation and omits the requested calculations.",
                    "finance_and_analysis",
                    "kaggle-synthetic-startup-outcomes-v1",
                    "synthetic_finance_arithmetic",
                )
            )
            index += 1
            if index >= 96:
                return output
    return output


def main() -> None:
    if not AB_PATH.exists() or not STARTUP_PATH.exists():
        raise FileNotFoundError("Run scripts/fetch_expert_public_data.py first")
    rows = ab_pairs() + startup_pairs()
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate derived pair ids")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT, rows)
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows), "sha256": stable_hash(OUTPUT.read_bytes().hex(), 64)}, indent=2))


if __name__ == "__main__":
    main()
