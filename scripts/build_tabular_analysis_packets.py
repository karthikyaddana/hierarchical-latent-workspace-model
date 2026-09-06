#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return round(value, 6) if isinstance(value, float) else value


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=clean) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create analytical source packets from a licensed tabular dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    frame = pd.read_csv(source, sep=None, engine="python")
    target = args.target
    if target not in frame.columns:
        raise ValueError("target %r not found in %s" % (target, list(frame.columns)))
    positive = frame[target].astype(str).str.lower().isin({"yes", "true", "1", "positive"})
    rows = []
    rows.append({
        "packet_type": "dataset_overview",
        "dataset": args.dataset_name,
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": list(frame.columns),
        "target": target,
        "positive_count": int(positive.sum()),
        "positive_rate": clean(float(positive.mean())),
        "missing_counts": {name: int(frame[name].isna().sum()) for name in frame.columns},
        "unknown_string_counts": {name: int(frame[name].astype(str).str.lower().eq("unknown").sum()) for name in frame.columns},
        "analysis_boundary": "Descriptive evidence only; no causal effect is established by this observational dataset.",
    })
    numeric = list(frame.select_dtypes(include="number").columns)
    categorical = [name for name in frame.columns if name not in numeric and name != target]
    for name in numeric:
        series = frame[name].dropna()
        rows.append({
            "packet_type": "numeric_profile",
            "dataset": args.dataset_name,
            "feature": name,
            "count": int(series.count()),
            "mean": clean(series.mean()),
            "std": clean(series.std()),
            "min": clean(series.min()),
            "p25": clean(series.quantile(0.25)),
            "median": clean(series.median()),
            "p75": clean(series.quantile(0.75)),
            "max": clean(series.max()),
            "analysis_boundary": "Distribution summary; association with the target requires a separate grouped packet.",
        })
        try:
            bins = pd.qcut(frame[name], q=5, duplicates="drop")
            grouped = pd.DataFrame({"bin": bins.astype(str), "positive": positive}).groupby("bin", observed=True)["positive"].agg(["count", "mean"])
            rows.append({
                "packet_type": "numeric_target_association",
                "dataset": args.dataset_name,
                "feature": name,
                "groups": [{"bin": str(index), "count": int(item["count"]), "positive_rate": clean(item["mean"])} for index, item in grouped.iterrows()],
                "analysis_boundary": "Observed association by quantile; not a causal estimate and vulnerable to confounding.",
            })
        except (ValueError, TypeError):
            pass
    for name in categorical:
        grouped = pd.DataFrame({"value": frame[name].astype(str), "positive": positive}).groupby("value")["positive"].agg(["count", "mean"]).sort_values("count", ascending=False).head(25)
        rows.append({
            "packet_type": "categorical_target_association",
            "dataset": args.dataset_name,
            "feature": name,
            "groups": [{"value": str(index), "count": int(item["count"]), "positive_rate": clean(item["mean"])} for index, item in grouped.iterrows()],
            "analysis_boundary": "Observed association only; small groups are unstable and no causal conclusion is permitted.",
        })
    for left, right in list(combinations(categorical, 2))[:20]:
        grouped = pd.DataFrame({"left": frame[left].astype(str), "right": frame[right].astype(str), "positive": positive}).groupby(["left", "right"])["positive"].agg(["count", "mean"])
        grouped = grouped[grouped["count"] >= max(50, int(len(frame) * 0.002))].sort_values("count", ascending=False).head(20)
        if grouped.empty:
            continue
        rows.append({
            "packet_type": "two_feature_segment",
            "dataset": args.dataset_name,
            "features": [left, right],
            "segments": [{"values": [str(index[0]), str(index[1])], "count": int(item["count"]), "positive_rate": clean(item["mean"])} for index, item in grouped.iterrows()],
            "analysis_boundary": "Exploratory segmentation only; validate on held-out data before using a segment operationally.",
        })
    count = write_rows(output, rows)
    print(json.dumps({"input_rows": len(frame), "analysis_packets": count, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
