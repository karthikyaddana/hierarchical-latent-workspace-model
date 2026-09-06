"""Materialize pinned external-benchmark EVALUATION subsets for session C.

These packets are **evaluation-only** artifacts (GSM8K / ARC-Challenge /
MMLU). They must never be written under ``data/`` or enter any training
bundle; the repository policy (`config/benchmarks.reasoning.yaml`) marks all
three suites `allowed_for_training: false`, and this script exists precisely
so the eval items live in a separate, checksummed artifact with an explicit
`eval_only` marker.

Subsets are deterministic: a fixed per-suite item count sampled with a fixed
seed from the official test split at a pinned dataset revision. Every packet
carries the suite, a normalized prompt, a structured `answer_spec` the
deterministic graders understand (`numeric` for GSM8K, `multiple_choice` for
ARC/MMLU), the license, and the revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "sources" / "public" / "benchmark-eval-subsets"

LETTERS = ["A", "B", "C", "D", "E"]

SUITES = {
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "license": "mit",
    },
    "arc-challenge": {
        "dataset_id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "license": "cc-by-sa-4.0",
    },
    "mmlu": {
        "dataset_id": "cais/mmlu",
        "config": "all",
        "split": "test",
        "license": "mit",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-suite", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def gsm8k_expected(answer_text: str) -> float:
    match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", str(answer_text))
    if not match:
        raise ValueError("gsm8k answer without #### marker: %r" % answer_text[:80])
    return float(match.group(1).replace(",", ""))


def gsm8k_packet(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    prompt = (
        str(row["question"]).strip()
        + "\n\nSolve step by step, then state the final numeric answer on the "
        "last line as: The answer is <number>."
    )
    return {
        "suite": "gsm8k",
        "item_id": "gsm8k-test-%04d" % index,
        "prompt": prompt,
        "answer_spec": {
            "type": "numeric",
            "expected": gsm8k_expected(row["answer"]),
            "absolute_tolerance": 1.0e-6,
            "relative_tolerance": 1.0e-6,
        },
    }


def choice_packet(
    suite: str,
    index: int,
    question: str,
    choice_texts: List[str],
    answer_letter: str,
) -> Dict[str, Any]:
    lines = [str(question).strip(), "", "Options:"]
    for letter, text in zip(LETTERS, choice_texts):
        lines.append("%s) %s" % (letter, str(text).strip()))
    lines.append("")
    lines.append("Answer with the letter of the correct option.")
    return {
        "suite": suite,
        "item_id": "%s-test-%04d" % (suite, index),
        "prompt": "\n".join(lines),
        "answer_spec": {
            "type": "multiple_choice",
            "expected": answer_letter,
            "choices": [str(text).strip() for text in choice_texts],
        },
    }


def arc_packet(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    labels = [str(label) for label in row["choices"]["label"]]
    texts = [str(text) for text in row["choices"]["text"]]
    key = str(row["answerKey"]).strip()
    if key not in labels:
        raise ValueError("arc answer key %r not in labels %r" % (key, labels))
    # Normalize 1/2/3/4-style labels to letters so every packet is A-E.
    letter = LETTERS[labels.index(key)]
    return choice_packet("arc-challenge", index, row["question"], texts, letter)


def mmlu_packet(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    letter = LETTERS[int(row["answer"])]
    packet = choice_packet("mmlu", index, row["question"], list(row["choices"]), letter)
    packet["subject"] = str(row.get("subject", ""))
    return packet


BUILDERS = {"gsm8k": gsm8k_packet, "arc-challenge": arc_packet, "mmlu": mmlu_packet}


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": (
            "evaluation-only artifact; forbidden for training per "
            "config/benchmarks.reasoning.yaml; never place under data/"
        ),
        "eval_only": True,
        "per_suite": args.per_suite,
        "sample_seed": args.seed,
        "suites": {},
    }
    for suite, spec in SUITES.items():
        revision = api.dataset_info(spec["dataset_id"]).sha
        dataset = load_dataset(
            spec["dataset_id"],
            spec["config"],
            split=spec["split"],
            revision=revision,
        )
        rng = random.Random("%s:%d:%s" % (suite, args.seed, revision))
        indices = sorted(rng.sample(range(len(dataset)), min(args.per_suite, len(dataset))))
        packets = []
        skipped = 0
        for index in indices:
            try:
                packet = BUILDERS[suite](dataset[index], index)
            except (KeyError, ValueError, IndexError):
                skipped += 1
                continue
            packet["dataset_id"] = spec["dataset_id"]
            packet["revision"] = revision
            packet["license"] = spec["license"]
            packet["eval_only"] = True
            packets.append(packet)
        path = args.output / (suite + ".jsonl")
        with path.open("w", encoding="utf-8") as stream:
            for packet in packets:
                stream.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["suites"][suite] = {
            "dataset_id": spec["dataset_id"],
            "revision": revision,
            "license": spec["license"],
            "split": spec["split"],
            "source_split_size": len(dataset),
            "rows": len(packets),
            "skipped": skipped,
            "sha256": digest,
        }
        print(suite, len(packets), "rows", "sha", digest[:12])
    (args.output / "fetch-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["suites"], indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
