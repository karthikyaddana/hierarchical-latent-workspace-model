"""Build the HLWM v5.6 master dataset.

Combines, into trainer-native master episodes (``normalize_episode`` schema):

1. Reasoning9000 + builder-factory episodes from ``data/combined/master``
   (behavior anchors are excluded here; the bundle builder regenerates them
   deterministically and leakage-checks them per split).
2. Benchmark train-split packets (MBPP, APPS, TACO-verified, CodeContests,
   Spider) converted to episodes. Coding packets are **execution-verified at
   build time**: the official reference solution is run against the official
   checks (assert lists or io pairs) in an isolated subprocess. Only rows
   whose reference passes are marked ``programmatically_verified`` (policy
   supervision eligible); rows whose reference fails or times out are dropped
   entirely (a label we could not verify is not shipped). Spider rows keep an
   exact-match ``sql_exact`` spec but stay policy-ineligible because we do not
   execute the gold SQL against the Spider databases.
3. Audited OpenThoughts/DeepCoder packets from ``data/combined/sft`` as
   causal-only episodes (no checkable spec, policy-ineligible).

SWE-bench rows are deferred to the 8B candidate: they rarely fit the 0.6B
prototype budgets and their gold patches are not checkable without a repo
harness.

Dev slices: every converted external source is split 92/3/5 into
train/validation/test by a stable hash of the episode id, so v5.6 audits can
report in-domain generalization without touching any official eval split.

Output: ``data/hlwm-v5.6/master/{train,validation,test}.jsonl`` + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from kaggle_hlwm.semantic_grading import (  # noqa: E402
    grade_io_tests,
    grade_python_tests,
)

COMBINED = ROOT / "data" / "combined"
PACKETS = ROOT / "sources" / "public" / "benchmark-train-splits"
OUTPUT = ROOT / "data" / "hlwm-v5.6"

MAX_ESTIMATED_TOKENS = 1600  # chars/4; both-ends truncation covers the tail
EXEC_TIMEOUT_SECONDS = 8.0
IO_CASES = 3

SOURCES = {
    "mbpp-train": {
        "domain": "python-function",
        "subdomain": "mbpp",
        "spec_type": "python_tests",
        "objective": "Implement the requested Python function so the official tests pass.",
        "constraints": [
            "Return complete runnable Python in a fenced code block.",
            "Do not print explanations inside the code block.",
        ],
    },
    "apps-train": {
        "domain": "competitive-programming",
        "subdomain": "apps",
        "spec_type": "io_tests",
        "objective": "Write a Python program that reads stdin and prints the required answer.",
        "constraints": [
            "Return complete runnable Python in a fenced code block.",
            "Read input from stdin and write only the answer to stdout.",
        ],
    },
    "taco-train": {
        "domain": "competitive-programming",
        "subdomain": "taco",
        "spec_type": "io_tests",
        "objective": "Write a Python program that reads stdin and prints the required answer.",
        "constraints": [
            "Return complete runnable Python in a fenced code block.",
            "Read input from stdin and write only the answer to stdout.",
        ],
    },
    "code-contests-train": {
        "domain": "competitive-programming",
        "subdomain": "code-contests",
        "spec_type": "io_tests",
        "objective": "Write a Python program that reads stdin and prints the required answer.",
        "constraints": [
            "Return complete runnable Python in a fenced code block.",
            "Read input from stdin and write only the answer to stdout.",
        ],
    },
    "spider-train": {
        "domain": "text-to-sql",
        "subdomain": "spider",
        "spec_type": "sql_exact",
        "objective": "Write the single SQL query that answers the question over the named schema.",
        "constraints": [
            "Return exactly one SQL query in a fenced code block.",
            "Use only tables and columns from the referenced database.",
        ],
    },
}

AUDITED_DATASETS = {
    "open-thoughts/OpenThoughts-114k",
    "agentica-org/DeepCoder-Preview-Dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Mark no packet verified (fast dry runs / tests only).",
    )
    parser.add_argument("--limit-per-source", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_fraction(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def assign_split(episode_id: str) -> str:
    value = stable_fraction("v5.6-split:" + episode_id)
    if value < 0.05:
        return "test"
    if value < 0.08:
        return "validation"
    return "train"


def parse_verification_material(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def packet_episode_id(source_key: str, packet: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"problem": packet.get("problem"), "reference": packet.get("reference_answer")},
        sort_keys=True,
    )
    return "%s-%s" % (source_key, hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16])


def fence(language: str, body: str) -> str:
    return "```%s\n%s\n```" % (language, str(body).strip())


def build_answer_spec(source_key: str, material: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    spec_type = SOURCES[source_key]["spec_type"]
    if spec_type == "python_tests":
        tests = [str(item) for item in material.get("test_list", []) if str(item).strip()]
        if not tests:
            return None
        return {
            "type": "python_tests",
            "tests": tests,
            "setup": str(material.get("test_setup_code", "") or ""),
            "time_limit_seconds": EXEC_TIMEOUT_SECONDS,
        }
    if spec_type == "io_tests":
        if material.get("fn_name"):
            return None  # call-based checks need a different harness; skip
        def coerce(item: Any) -> str:
            if isinstance(item, (list, tuple)):
                return "\n".join(str(part) for part in item)
            return str(item)

        inputs = material.get("inputs") or material.get("public_test_inputs") or []
        outputs = material.get("outputs") or material.get("public_test_outputs") or []
        inputs = [coerce(item) for item in inputs][:IO_CASES]
        outputs = [coerce(item) for item in outputs][:IO_CASES]
        if not inputs or len(inputs) != len(outputs):
            return None
        return {
            "type": "io_tests",
            "inputs": inputs,
            "outputs": outputs,
            "max_cases": IO_CASES,
            "time_limit_seconds": EXEC_TIMEOUT_SECONDS,
        }
    if spec_type == "sql_exact":
        gold = str(material.get("gold_sql", "") or "").strip()
        if not gold:
            return None
        return {
            "type": "sql_exact",
            "expected": gold,
            "db_id": str(material.get("db_id", "") or ""),
        }
    raise ValueError("unknown spec type for %s" % source_key)


def verify_reference(task: Tuple[str, str, Dict[str, Any]]) -> Tuple[bool, str]:
    """Execute one official reference answer against its own spec."""

    spec_type, reference, spec = task
    if spec_type == "python_tests":
        outcome = grade_python_tests(
            reference,
            spec["tests"],
            setup=spec.get("setup", ""),
            timeout_seconds=spec["time_limit_seconds"],
        )
    elif spec_type == "io_tests":
        outcome = grade_io_tests(
            reference,
            spec["inputs"],
            spec["outputs"],
            timeout_seconds=spec["time_limit_seconds"],
            max_cases=spec["max_cases"],
        )
    else:
        return False, "not executable"
    return bool(outcome.get("correct")), str(outcome.get("detail", ""))


def convert_packet(
    source_key: str,
    packet: Mapping[str, Any],
    spec: Dict[str, Any],
    verified: bool,
    verification_method: str,
) -> Dict[str, Any]:
    profile = SOURCES[source_key]
    episode_id = packet_episode_id(source_key, packet)
    problem = str(packet.get("problem", "")).strip()
    reference = str(packet.get("reference_answer", "")).strip()
    language = "sql" if profile["spec_type"] == "sql_exact" else "python"
    published = fence(language, reference)
    curation = dict(packet.get("curation") or {})
    if profile["spec_type"] == "python_tests":
        check_summary = "Official tests:\n" + "\n".join(spec["tests"][:3])
    elif profile["spec_type"] == "io_tests":
        check_summary = "Official io check:\ninput:\n%s\nexpected output:\n%s" % (
            spec["inputs"][0][:400],
            spec["outputs"][0][:400],
        )
    else:
        check_summary = "Official database: %s. The gold query is exact-match graded." % (
            spec.get("db_id", "unknown")
        )
    verdict = "supported" if verified or profile["spec_type"] == "sql_exact" else "insufficient"
    return {
        "episode_id": episode_id,
        "schema_version": "1.0.0",
        "source_group": "external-verified-v5.6",
        "domain": profile["domain"],
        "subdomain": profile["subdomain"],
        "difficulty": str(curation.get("difficulty", "unspecified")),
        "input": {
            "user_request": problem,
            "context": [],
            "constraints": list(profile["constraints"]),
        },
        "frame": {
            "objective": profile["objective"],
            "failure_contract": [
                "An answer that fails the official checks fails the task.",
            ],
        },
        "lanes": [
            {
                "lane_id": "lane-solution",
                "route": ["root", profile["domain"], profile["subdomain"]],
                "route_windows": [{"window": 0, "decision": "halt"}],
                "brief": {
                    "scope": profile["objective"],
                    "assumptions": [],
                    "deliverable": "A complete answer in a fenced code block.",
                    "rejection_tests": ["Fails the official checks."],
                },
                "artifacts": [
                    {"artifact_id": "solution", "type": "code", "content": published}
                ],
                "claims": [
                    {
                        "claim_id": episode_id + "-solution",
                        "statement": "The solution satisfies the official checks.",
                    }
                ],
                "checkpoints": [],
                "summary": "Produced a candidate solution for the official checks.",
            },
            {
                "lane_id": "lane-check",
                "route": ["root", profile["domain"], "verification"],
                "route_windows": [{"window": 0, "decision": "halt"}],
                "brief": {
                    "scope": "Restate the checks the solution must pass.",
                    "assumptions": [],
                    "deliverable": "The concrete acceptance checks.",
                    "rejection_tests": ["Checks omitted or altered."],
                },
                "artifacts": [
                    {"artifact_id": "checks", "type": "text", "content": check_summary}
                ],
                "claims": [],
                "checkpoints": [],
                "summary": "Acceptance checks recorded for verification.",
            },
        ],
        "verification": [
            {
                "claim_id": episode_id + "-solution",
                "verdict": verdict,
                "method": verification_method,
            }
        ],
        "barrier": {"open_claims": []},
        "integration": {"decision": "publish", "published_answer": published},
        "commitment": {"decision": "publish"},
        "evaluation": {
            "answer_spec": spec,
            "expected_action": "answer",
            "expected_commit": True,
            "negative_answer": "",
            "required_phrases": [],
            "forbidden_phrases": [],
        },
        "generation_metadata": {
            "method": "benchmark_train_split_conversion_v5.6",
            "programmatically_verified": bool(verified),
            "independently_adjudicated": False,
            "dataset_id": str(curation.get("dataset_id", "")),
            "source_curation": curation,
            "license": str(packet.get("license", "")),
        },
    }


def convert_audited_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    messages = row.get("messages") or []
    user = next((m for m in messages if m.get("role") == "user"), None)
    assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if user is None or assistant is None:
        return None
    provenance = dict(row.get("provenance") or {})
    episode_id = "audited-" + hashlib.sha1(
        str(user.get("content", "")).encode("utf-8")
    ).hexdigest()[:16]
    answer = str(assistant.get("content", "")).strip()
    return {
        "episode_id": episode_id,
        "schema_version": "1.0.0",
        "source_group": "audited-packets-v5.6",
        "domain": str(row.get("domain", "software-reasoning")),
        "subdomain": provenance.get("dataset_id", "audited"),
        "difficulty": "unspecified",
        "input": {
            "user_request": str(user.get("content", "")).strip(),
            "context": [],
            "constraints": ["Answer completely and verifiably."],
        },
        "frame": {
            "objective": "Solve the software-reasoning task completely.",
            "failure_contract": ["An incomplete or incorrect solution fails."],
        },
        "lanes": [
            {
                "lane_id": "lane-solution",
                "route": ["root", "software-reasoning"],
                "route_windows": [{"window": 0, "decision": "halt"}],
                "brief": {
                    "scope": "Solve the task.",
                    "assumptions": [],
                    "deliverable": "A complete answer.",
                    "rejection_tests": ["Solution incomplete."],
                },
                "artifacts": [
                    {"artifact_id": "solution", "type": "text", "content": answer}
                ],
                "claims": [],
                "checkpoints": [],
                "summary": "Produced a candidate solution.",
            }
        ],
        "verification": [],
        "barrier": {"open_claims": []},
        "integration": {"decision": "publish", "published_answer": answer},
        "commitment": {"decision": "publish"},
        "evaluation": {
            "answer_spec": {},
            "expected_action": "answer",
            "expected_commit": True,
            "negative_answer": "",
            "required_phrases": [],
            "forbidden_phrases": [],
        },
        "generation_metadata": {
            "method": "audited_packet_conversion_v5.6",
            "programmatically_verified": False,
            "independently_adjudicated": False,
            "dataset_id": str(provenance.get("dataset_id", "")),
            "revision": str(provenance.get("revision", "")),
        },
    }


def estimated_tokens(packet: Mapping[str, Any]) -> int:
    return (len(str(packet.get("problem", ""))) + len(str(packet.get("reference_answer", "")))) // 4


def main() -> None:
    args = parse_args()
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    stats: Dict[str, Any] = {"per_source": {}, "dropped": {}}

    # 1) combined master passthrough (drop regenerated behavior anchors)
    passthrough = {"train": 0, "validation": 0, "test": 0}
    anchors_dropped = 0
    for split in splits:
        for record in read_jsonl(COMBINED / "master" / (split + ".jsonl")):
            if record.get("domain") == "behavior-anchor":
                anchors_dropped += 1
                continue
            record.pop("bundle_source", None)
            splits[split].append(record)
            passthrough[split] += 1
    stats["combined_master_passthrough"] = passthrough
    stats["behavior_anchors_excluded_for_regeneration"] = anchors_dropped

    # 2) benchmark packets
    for source_key, profile in SOURCES.items():
        path = PACKETS / (source_key + ".jsonl")
        packets = list(read_jsonl(path))
        if args.limit_per_source:
            packets = packets[: args.limit_per_source]
        candidates: List[Tuple[Mapping[str, Any], Dict[str, Any]]] = []
        dropped_length = dropped_spec = 0
        for packet in packets:
            if estimated_tokens(packet) > MAX_ESTIMATED_TOKENS:
                dropped_length += 1
                continue
            spec = build_answer_spec(source_key, parse_verification_material(packet.get("verification_material")))
            if spec is None:
                dropped_spec += 1
                continue
            candidates.append((packet, spec))

        executable = profile["spec_type"] in ("python_tests", "io_tests")
        verified_flags: List[bool] = []
        details: List[str] = []
        if executable and not args.skip_execution and candidates:
            tasks = [
                (profile["spec_type"], str(packet.get("reference_answer", "")), spec)
                for packet, spec in candidates
            ]
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for verified, detail in pool.map(verify_reference, tasks, chunksize=8):
                    verified_flags.append(verified)
                    details.append(detail)
        else:
            verified_flags = [False] * len(candidates)
            details = [""] * len(candidates)

        kept = 0
        dropped_exec = 0
        episodes: List[Dict[str, Any]] = []
        for (packet, spec), verified in zip(candidates, verified_flags):
            if executable and not args.skip_execution and not verified:
                dropped_exec += 1  # unverifiable label: do not ship
                continue
            method = (
                "executed_reference_tests"
                if executable and verified
                else ("official_gold_label" if profile["spec_type"] == "sql_exact" else "unexecuted")
            )
            episodes.append(convert_packet(source_key, packet, spec, verified, method))
            kept += 1

        # cross-problem hard negatives for policy-eligible rows
        verified_answers = [
            episode["integration"]["published_answer"]
            for episode in episodes
            if episode["generation_metadata"]["programmatically_verified"]
        ]
        if len(verified_answers) >= 2:
            index = 0
            for episode in episodes:
                if episode["generation_metadata"]["programmatically_verified"]:
                    swap = verified_answers[(index + 1) % len(verified_answers)]
                    if swap != episode["integration"]["published_answer"]:
                        episode["evaluation"]["negative_answer"] = swap
                    index += 1

        split_counts = {"train": 0, "validation": 0, "test": 0}
        for episode in episodes:
            split = assign_split(episode["episode_id"])
            splits[split].append(episode)
            split_counts[split] += 1
        stats["per_source"][source_key] = {
            "packets": len(packets),
            "kept": kept,
            "verified": int(sum(verified_flags[: len(candidates)])) if executable else 0,
            "dropped_length": dropped_length,
            "dropped_missing_spec": dropped_spec,
            "dropped_reference_failed_execution": dropped_exec,
            "splits": split_counts,
        }

    # 3) audited packets from the combined sft split
    audited_counts = {"train": 0, "validation": 0, "test": 0}
    audited_seen = 0
    for row in read_jsonl(COMBINED / "sft" / "train.jsonl"):
        provenance = row.get("provenance") or {}
        if provenance.get("dataset_id") not in AUDITED_DATASETS:
            continue
        episode = convert_audited_row(row)
        if episode is None:
            continue
        audited_seen += 1
        split = assign_split(episode["episode_id"])
        splits[split].append(episode)
        audited_counts[split] += 1
    stats["audited_packets"] = {"converted": audited_seen, "splits": audited_counts}

    output_master = args.output / "master"
    output_master.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split, rows in splits.items():
        path = output_master / (split + ".jsonl")
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        totals[split] = len(rows)
    stats["totals"] = totals
    stats["policy"] = {
        "behavior_anchors": "regenerated by the bundle builder, not stored here",
        "policy_supervision": (
            "only executed_reference_tests rows and independently adjudicated builder "
            "episodes are policy-eligible; sql_exact and audited packets are causal/"
            "workspace supervision only"
        ),
        "swebench": "deferred to the 8B candidate (budget + unexecutable gold patches)",
        "official_eval_splits": "never read; dev slices are carved from train material only",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
