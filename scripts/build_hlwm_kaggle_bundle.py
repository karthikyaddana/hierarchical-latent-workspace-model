#!/usr/bin/env python3
"""Build the Version 5.5 single-device (A100 MIG 1g.5gb) HLWM package.

Version 5.5 adds the on-policy policy-head phase, the router entropy
regularizer, preregistered routing gates, BF16 support, and a 3.5-hour
single-GPU session plan sized for a 5 GB A100 partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "kaggle_hlwm"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from semantic_grading import grade_semantic_answer


DEFAULT_OUTPUT = ROOT / "artifacts" / "kaggle" / "hlwm-v5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--train", type=int, default=None, help="Optional balanced train subset size"
    )
    parser.add_argument(
        "--validation", type=int, default=None, help="Optional balanced validation subset size"
    )
    parser.add_argument(
        "--test", type=int, default=None, help="Optional balanced test subset size"
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def stable_score(seed: int, episode_id: str) -> str:
    return hashlib.sha256((str(seed) + ":" + episode_id).encode("utf-8")).hexdigest()


def alphabetic_code(value: int) -> str:
    """Encode a non-negative integer without digits for prompt-isolation tags."""

    if value < 0:
        raise ValueError("alphabetic codes require a non-negative value")
    letters = []
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("a") + remainder))
        if value == 0:
            return "".join(reversed(letters))


def balanced_rows(path: Path, count: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                groups[str(row.get("domain", "unknown"))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: stable_score(seed, str(row.get("episode_id", ""))))
    domains = sorted(groups)
    selected: List[Dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and domains:
        domain = domains[cursor % len(domains)]
        rows = groups[domain]
        if rows:
            selected.append(rows.pop(0))
        else:
            domains.remove(domain)
            cursor -= 1
        cursor += 1
    if len(selected) < count:
        raise ValueError("requested %d rows but selected %d from %s" % (count, len(selected), path))
    return selected


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def behavior_anchor_rows(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    """Create split-isolated tasks with structured semantic answer specs.

    Every fourth record is arithmetic, unit conversion, ordering, or missing
    evidence. Operands share one distribution across splits while prompts,
    entity identifiers, and exact examples remain split-isolated. Multiple
    templates prevent the policy from learning one fixed surface form.
    """

    rows: List[Dict[str, Any]] = []
    for index in range(count):
        key = int(
            hashlib.sha256(
                ("%d:%s:%d" % (seed, split, index)).encode("utf-8")
            ).hexdigest()[:12],
            16,
        )
        kind = index % 4
        if kind == 0:
            operation = (index // 4) % 3
            if operation == 0:
                left = 1_000 + key % 89_000
                right = 100 + (key >> 13) % 8_900
                value = left + right
                templates = (
                    "Calculate %d + %d. Give one short sentence with the result.",
                    "What is the exact sum of %d and %d? Return only a concise answer.",
                    "Add %d to %d and state the checked total.",
                )
                request = templates[(key >> 7) % len(templates)] % (left, right)
                answer = "The result is %d." % value
                negative = "The result is %d." % (value + 1)
                reasoning = "%d plus %d equals %d." % (left, right, value)
            elif operation == 1:
                smaller = 100 + key % 8_000
                value = 200 + (key >> 11) % 70_000
                larger = smaller + value
                templates = (
                    "Calculate %d - %d and state the exact result.",
                    "Subtract %d from %d. Give only the verified difference.",
                    "What is the checked difference between %d and %d?",
                )
                template_index = (key >> 5) % len(templates)
                if template_index == 1:
                    request = templates[template_index] % (smaller, larger)
                else:
                    request = templates[template_index] % (larger, smaller)
                answer = "The verified difference is %d." % value
                negative = "The verified difference is %d." % (value - 1)
                reasoning = "%d minus %d equals %d." % (larger, smaller, value)
            else:
                left = 12 + key % 888
                right = 2 + (key >> 17) % 97
                value = left * right
                templates = (
                    "Multiply %d by %d and state the verified result directly.",
                    "What is %d times %d? Give the checked product only.",
                    "Compute the exact product of %d and %d.",
                )
                request = templates[(key >> 9) % len(templates)] % (left, right)
                answer = "The verified result is %d." % value
                negative = "The verified result is %d." % (value + left)
                reasoning = "%d multiplied by %d equals %d." % (left, right, value)
            answer_spec = {"type": "numeric", "expected": value}
            route = "arithmetic"
        elif kind == 1:
            conversion = (index // 4) % 3
            quantity = 2 + key % 998
            if conversion == 0:
                factor, source_unit, target_unit = 60, "minutes", "seconds"
            elif conversion == 1:
                factor, source_unit, target_unit = 60, "hours", "minutes"
            else:
                factor, source_unit, target_unit = 24, "days", "hours"
            converted = quantity * factor
            templates = (
                "Convert %d %s to %s. Answer without restating the task.",
                "How many %s are in %d %s? Give the exact conversion.",
                "State the checked number of %s equivalent to %d %s.",
            )
            template_index = (key >> 8) % len(templates)
            if template_index == 0:
                request = templates[template_index] % (
                    quantity, source_unit, target_unit
                )
            else:
                request = templates[template_index] % (
                    target_unit, quantity, source_unit
                )
            answer = "%d %s equals %d %s." % (
                quantity, source_unit, converted, target_unit
            )
            negative = "%d %s equals %d %s." % (
                quantity, source_unit, converted + factor, target_unit
            )
            answer_spec = {
                "type": "unit", "expected": converted, "unit": target_unit
            }
            route = "unit_conversion"
            reasoning = "Multiply %s by %d: %d x %d = %d." % (
                source_unit, factor, quantity, factor, converted
            )
        elif kind == 2:
            base = 100 + key % 90_000
            gaps = [
                1 + (key >> 9) % 7,
                2 + (key >> 17) % 11,
                3 + (key >> 25) % 13,
            ]
            ordered = [
                base,
                base + gaps[0],
                base + gaps[0] + gaps[1],
                base + sum(gaps),
            ]
            permutations = (
                [ordered[2], ordered[0], ordered[3], ordered[1]],
                [ordered[3], ordered[1], ordered[0], ordered[2]],
                [ordered[1], ordered[3], ordered[2], ordered[0]],
            )
            values = permutations[key % len(permutations)]
            rendered_values = ", ".join(str(value) for value in values)
            templates = (
                "Sort these integers in ascending order: %s.",
                "Put the following values from smallest to largest: %s.",
                "Return only the ascending sequence for these numbers: %s.",
            )
            request = templates[(key >> 6) % len(templates)] % rendered_values
            answer = "Ascending order: %s." % ", ".join(str(value) for value in ordered)
            negative_order = list(reversed(ordered))
            negative = "Ascending order: %s." % ", ".join(
                str(value) for value in negative_order
            )
            answer_spec = {"type": "ordering", "expected": ordered}
            route = "ordering"
            reasoning = "Comparing the four values gives %s." % ", ".join(
                str(value) for value in ordered
            )
        else:
            entity = "%s-%s-%04d" % (
                ("sensor", "shipment", "release")[(index // 4) % 3],
                split[:3],
                index,
            )
            missing_templates = (
                (
                    "What exact battery percentage did %s report? If it is absent, say that it cannot be determined.",
                    "battery percentage",
                    "The battery percentage cannot be determined from the provided information.",
                    "The battery percentage is 80%.",
                ),
                (
                    "What was the exact delivery temperature for %s? Do not guess if no measurement is supplied.",
                    "delivery temperature",
                    "There is insufficient information to determine the delivery temperature.",
                    "The delivery temperature was 4 degrees.",
                ),
                (
                    "Which exact software version was deployed for %s? State when the value was not provided.",
                    "software version",
                    "The software version was not provided, so it cannot be determined.",
                    "Software version 3 was deployed.",
                ),
            )
            request_template, missing_value, answer, negative = missing_templates[
                (key >> 10) % len(missing_templates)
            ]
            request = request_template % entity
            answer_spec = {"type": "abstention", "forbid_numbers": True}
            route = "evidence_check"
            reasoning = "No %s is present, so any specific value would be a guess." % (
                missing_value
            )

        request = "%s [case %s-%s]" % (
            request.rstrip(), split, alphabetic_code(index)
        )
        expected_action = "abstain" if kind == 3 else "answer"
        verdict = "supported"
        decision = "publish"
        claim_id = "anchor-claim-%s-%04d" % (split, index)
        episode_id = "behavior-anchor-%s-%04d" % (split, index)
        rows.append(
            {
                "schema_version": "1.0.0",
                "episode_id": episode_id,
                "domain": "behavior-anchor",
                "subdomain": route,
                "difficulty": "deterministic",
                "source_group": "programmatic-v5.4",
                "input": {
                    "user_request": request,
                    "context": [
                        "This is a deterministic behavior check. Use only facts explicitly supplied."
                    ],
                    "constraints": ["Answer directly.", "Do not invent missing facts."],
                },
                "frame": {
                    "objective": "Produce the checked answer or abstain when evidence is absent.",
                    "requirements": ["Do not repeat the prompt.", "Keep the answer concise."],
                    "failure_contract": ["A guessed or numerically incorrect answer fails."],
                },
                "lanes": [
                    {
                        "lane_id": "constructor",
                        "route": ["root", route, "solver"],
                        "brief": {
                            "scope": "Derive the requested result.",
                            "assumptions": [],
                            "deliverable": "A direct candidate answer.",
                            "rejection_tests": ["Recompute the result."],
                        },
                        "artifacts": [
                            {"artifact_id": "derivation", "type": "text", "content": reasoning}
                        ],
                        "claims": [
                            {
                                "claim_id": claim_id,
                                "statement": answer,
                                "evidence_refs": ["derivation"],
                            }
                        ],
                        "checkpoints": [
                            {"step": 1, "observable_update": reasoning, "decision": "halt"}
                        ],
                        "route_windows": [{"decision": "halt"}],
                        "summary": answer,
                    },
                    {
                        "lane_id": "critic",
                        "route": ["root", "verification", "checker"],
                        "brief": {
                            "scope": "Independently verify evidence and reject guessing.",
                            "assumptions": [],
                            "deliverable": "A verification verdict.",
                            "rejection_tests": ["Reject the supplied counterfactual: " + negative],
                        },
                        "artifacts": [
                            {
                                "artifact_id": "check",
                                "type": "text",
                                "content": reasoning + " Verdict: " + verdict + ".",
                            }
                        ],
                        "claims": [
                            {
                                "claim_id": claim_id + "-check",
                                "statement": "The candidate is %s." % verdict,
                                "evidence_refs": ["check"],
                            }
                        ],
                        "checkpoints": [
                            {"step": 1, "observable_update": verdict, "decision": "halt"}
                        ],
                        "route_windows": [{"decision": "halt"}],
                        "summary": "Verification: %s. %s" % (verdict, reasoning),
                    },
                ],
                "barrier": {"open_claims": []},
                "verification": [
                    {"claim_id": claim_id, "verdict": verdict},
                    {"claim_id": claim_id + "-check", "verdict": "supported"},
                ],
                "integration": {"decision": decision, "published_answer": answer},
                "commitment": {"decision": decision},
                "evaluation": {
                    "expected_commit": True,
                    "expected_action": expected_action,
                    "answer_spec": answer_spec,
                    "required_phrases": [],
                    "forbidden_phrases": [],
                    "negative_answer": negative,
                },
                "generation_metadata": {
                    "method": "programmatic_verified_behavior_anchor",
                    "programmatically_verified": True,
                    "independently_adjudicated": False,
                },
            }
        )
    return rows


def write_anchors_then_source(
    destination: Path, anchors: Iterable[Mapping[str, Any]], source: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for row in anchors:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with source.open("r", encoding="utf-8") as input_stream:
            shutil.copyfileobj(input_stream, output)


def validate_behavior_anchors(rows: List[Dict[str, Any]], split: str) -> None:
    ids = [str(row.get("episode_id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("behavior anchor ids are missing or duplicated in %s" % split)
    prompts = [str((row.get("input") or {}).get("user_request", "")) for row in rows]
    if len(prompts) != len(set(prompts)) or any(not value for value in prompts):
        raise ValueError("behavior anchor prompts are missing or duplicated in %s" % split)
    task_types: Counter[str] = Counter()
    for row in rows:
        evaluation = row.get("evaluation") or {}
        answer = str((row.get("integration") or {}).get("published_answer", ""))
        negative = str(evaluation.get("negative_answer", ""))
        answer_spec = dict(evaluation.get("answer_spec") or {})
        if not answer or answer == negative:
            raise ValueError("anchor answer/negative pair is invalid: %s" % row.get("episode_id"))
        if not grade_semantic_answer(answer, answer_spec)["correct"]:
            raise ValueError("anchor semantic grader rejects its clean answer")
        if grade_semantic_answer(negative, answer_spec)["correct"]:
            raise ValueError("anchor semantic grader accepts its negative answer")
        if not evaluation.get("expected_commit"):
            raise ValueError("safe anchor answers must be public commitment targets")
        metadata = row.get("generation_metadata") or {}
        if not metadata.get("programmatically_verified") or metadata.get(
            "independently_adjudicated"
        ):
            raise ValueError("anchor provenance flags are invalid")
        task_types[str(answer_spec.get("type", "missing"))] += 1
    if len(task_types) != 4 or max(task_types.values()) - min(task_types.values()) > 1:
        raise ValueError("anchor split is not balanced across semantic task types: %r" % task_types)


def _legacy_v4_notebook_document() -> Dict[str, Any]:
    def markdown(text: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

    def code(text: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(True),
        }

    cells = [
        markdown(
            "# Embel HLWM Version 4 - reproducible Qwen3-0.6B experiment\n\n"
            "This clean notebook trains the native categorical-diffusion HLWM prototype on the "
            "complete Reasoning9000 master splits. It has deterministic resume ordering, stable "
            "mixed precision, checksummed checkpoints, adapter export, validation diagnostics, "
            "checkpoint reload and direct output comparisons. It tests the executable research "
            "architecture; it does not claim model superiority or calibrated safety.\n"
        ),
        code(
            "import platform, subprocess, sys\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(
            "!python -m pip install -q transformers==4.56.2 accelerate==1.10.1 safetensors==0.6.2 sentencepiece==0.2.1 pytest==8.4.1\n"
        ),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "WORK = Path('/kaggle/working/hlwm-v4')\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "archives = list(Path('/kaggle/input').rglob('hlwm-v4-full-bundle.zip'))\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for path in Path('/kaggle/input').rglob('manifest.json'):\n"
            "        try:\n"
            "            if json.loads(path.read_text()).get('name') == 'hlwm-reasoning9000-v4-full': manifests.append(path)\n"
            "        except (OSError, json.JSONDecodeError):\n"
            "            pass\n"
            "    if not manifests:\n"
            "        raise FileNotFoundError('Attach the HLWM Version 4 full dataset first.')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "    shutil.copytree(manifests[0].parent, PROJECT)\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "print('project', PROJECT)\n"
            "print('dataset counts', manifest['counts'])\n"
            "assert manifest['counts'] == {'train': 3310, 'validation': 585, 'test': 405}\n"
        ),
        code(
            "import py_compile, subprocess, sys\n"
            "for name in ('modeling_hlwm.py','data.py','train_kaggle.py','evaluate_checkpoint.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')], cwd=PROJECT, check=True)\n"
            "print('static compilation and tiny-model tests passed')\n"
        ),
        markdown(
            "## Stabilized 250-step baseline\n\n"
            "This is a fresh run from the pinned base, not a continuation of the manually repaired "
            "Version 3 checkpoint. The trainer rejects non-empty output directories and excessive "
            "loss-scaler skips rather than silently mixing runs.\n"
        ),
        code(
            "import subprocess, sys\n"
            "STEPS = 250\n"
            "OUTPUT = Path('/kaggle/working/hlwm-v4-output')\n"
            "command = [\n"
            "    sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(OUTPUT),\n"
            "    '--steps', str(STEPS), '--batch-size', '1',\n"
            "    '--gradient-accumulation', '4', '--learning-rate', '0.0001',\n"
            "    '--warmup-updates', '10', '--initial-loss-scale', '1024',\n"
            "    '--max-skipped-updates', '2', '--num-lanes', '2',\n"
            "    '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--context-tokens', '96', '--canvas-tokens', '96',\n"
            "    '--brief-tokens', '64', '--causal-tokens', '192',\n"
            "    '--max-validation', '64', '--eval-every', '50', '--save-every', '50',\n"
            "]\n"
            "print(' '.join(command))\n"
            "subprocess.run(command, cwd=PROJECT, check=True)\n"
        ),
        code(
            "summary = json.loads((OUTPUT/'summary.json').read_text())\n"
            "assert summary['steps'] == 250\n"
            "assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "assert Path(summary['checkpoint']).exists()\n"
            "assert Path(summary['adapter']['path']).exists()\n"
            "summary\n"
        ),
        markdown(
            "## Reload and inspect actual outputs\n\n"
            "This reloads the saved checkpoint into a fresh model, verifies its checksum, generates "
            "pinned-base and HLWM causal controls, then runs the private diffusion path while logging "
            "routes, lane similarity, verifier scores, halting and commitment.\n"
        ),
        code(
            "CHECKPOINT = Path(summary['checkpoint'])\n"
            "evaluation_command = [\n"
            "    sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "    '--checkpoint', str(CHECKPOINT), '--data-dir', str(DATA),\n"
            "    '--output-dir', str(OUTPUT), '--samples', '4',\n"
            "    '--context-tokens', '96', '--canvas-tokens', '64', '--max-new-tokens', '48',\n"
            "]\n"
            "subprocess.run(evaluation_command, cwd=PROJECT, check=True)\n"
        ),
        code(
            "evaluation = json.loads((OUTPUT/'checkpoint-evaluation.json').read_text())\n"
            "print(json.dumps(evaluation['aggregate'], indent=2))\n"
            "for record in evaluation['records']:\n"
            "    print('\\n', record['episode_id'], record['domain'], 'committed=', record['committed'])\n"
            "    print('BASE:', record['base_causal'][:500])\n"
            "    print('HLWM CAUSAL:', record['hlwm_causal'][:500])\n"
            "    print('HLWM PRIVATE:', record['hlwm_candidate'][:500])\n"
        ),
        code(
            "artifacts = {\n"
            "    str(path.relative_to(OUTPUT)): path.stat().st_size\n"
            "    for path in sorted(OUTPUT.rglob('*')) if path.is_file()\n"
            "}\n"
            "print(json.dumps(artifacts, indent=2))\n"
            "assert any(name.endswith('.safetensors') for name in artifacts)\n"
            "assert 'checkpoint-evaluation.json' in artifacts\n"
        ),
        markdown(
            "## Gate before longer training\n\n"
            "Save this notebook version only after every assertion passes. A 2,000-step run is "
            "authorized only if skipped optimizer updates remain zero, checkpoint reload succeeds, "
            "causal outputs remain coherent, and the diagnostic report shows non-degenerate routing "
            "and commitment. The next architecture phase adds the connected expert graph, route "
            "refresh, recurrent verifier lanes and multiple macrocycles.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_document() -> Dict[str, Any]:
    """Return the clean Version 5 Kaggle notebook."""

    def markdown(text: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

    def code(text: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(True),
        }

    cells = [
        markdown(
            "# Embel HLWM Version 5 - correctness-gated Qwen prototype\n\n"
            "This notebook tests the complete Version 5 execution path: one-pass local categorical "
            "denoising, short joint tuning, full reverse diffusion at inference, connected expert "
            "paths, a summary-only barrier, Qwen autoregressive answer generation and conservative "
            "commit/risk/verifier gating. The current Reasoning9000 release remains unreviewed "
            "synthetic architecture data, so passing this notebook proves wiring and learnability, "
            "not production capability. The correctness run intentionally uses one T4; dual-T4 "
            "optimization begins only after these gates pass.\n"
        ),
        code(
            "import platform, subprocess, sys\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(
            "!python -m pip install -q transformers==4.56.2 accelerate==1.10.1 safetensors==0.6.2 sentencepiece==0.2.1 pytest==8.4.1\n"
        ),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "WORK = Path('/kaggle/working/hlwm-v5')\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "archives = list(Path('/kaggle/input').rglob('hlwm-v5-candidate-bundle.zip'))\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for path in Path('/kaggle/input').rglob('manifest.json'):\n"
            "        try:\n"
            "            if json.loads(path.read_text()).get('name') == 'hlwm-reasoning9000-v5-candidate': manifests.append(path)\n"
            "        except (OSError, json.JSONDecodeError):\n"
            "            pass\n"
            "    if not manifests:\n"
            "        raise FileNotFoundError('Attach the HLWM Version 5 candidate dataset first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "quality = json.loads((PROJECT/'data-quality.json').read_text())\n"
            "print('writable project', PROJECT)\n"
            "print('dataset counts', manifest['counts'])\n"
            "print(json.dumps(quality, indent=2))\n"
            "assert manifest['counts'] == {'train': 3310, 'validation': 585, 'test': 405}\n"
            "assert quality['independently_adjudicated_episodes'] == 0\n"
        ),
        code(
            "import py_compile, subprocess, sys\n"
            "for name in ('modeling_hlwm.py','data.py','train_kaggle.py','evaluate_checkpoint.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')], cwd=PROJECT, check=True)\n"
            "print('Version 5 static checks and tiny-model invariants passed')\n"
        ),
        markdown(
            "## Progressive pilot\n\n"
            "The first 48 microsteps repeatedly train a 24-example slice. A fixed-noise loss "
            "measurement must improve before the run can spend time on the 120-step local stage "
            "and 80-step joint stage. A runtime guard saves a resumable checkpoint before the "
            "Kaggle session budget is at risk.\n"
        ),
        code(
            "import subprocess, sys\n"
            "OUTPUT = Path('/kaggle/working/hlwm-v5-output')\n"
            "command = [\n"
            "    sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(OUTPUT),\n"
            "    '--overfit-steps', '48', '--local-steps', '120', '--joint-steps', '80',\n"
            "    '--overfit-examples', '24', '--overfit-min-improvement', '0.0',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4',\n"
            "    '--learning-rate', '0.0001', '--warmup-updates', '10',\n"
            "    '--initial-loss-scale', '1024', '--max-skipped-updates', '2',\n"
            "    '--num-lanes', '2', '--num-experts', '6',\n"
            "    '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--context-tokens', '96', '--canvas-tokens', '96',\n"
            "    '--brief-tokens', '64', '--causal-tokens', '192',\n"
            "    '--max-validation', '32', '--num-workers', '2',\n"
            "    '--eval-every', '48', '--save-every', '48',\n"
            "    '--max-runtime-hours', '7.5',\n"
            "]\n"
            "print(' '.join(command))\n"
            "subprocess.run(command, cwd=PROJECT, check=True)\n"
        ),
        code(
            "summary = json.loads((OUTPUT/'summary.json').read_text())\n"
            "assert summary['status'] == 'stable_prototype_training_complete', summary\n"
            "assert summary['steps'] == summary['planned_steps'] == 248, summary\n"
            "assert summary['overfit_gate']['passed'] is True, summary\n"
            "assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "assert Path(summary['checkpoint']).exists()\n"
            "assert Path(summary['adapter']['path']).exists()\n"
            "summary\n"
        ),
        markdown(
            "## Fresh reload, full reverse schedule and public-output audit\n\n"
            "This reloads the checksummed checkpoint, compares the pinned Qwen control, executes "
            "all four reverse timesteps, generates through Qwen rather than private canvas argmax, "
            "and measures rejection of mechanically corrupted candidates.\n"
        ),
        code(
            "CHECKPOINT = Path(summary['checkpoint'])\n"
            "evaluation_command = [\n"
            "    sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "    '--checkpoint', str(CHECKPOINT), '--data-dir', str(DATA),\n"
            "    '--output-dir', str(OUTPUT), '--samples', '3',\n"
            "    '--context-tokens', '96', '--canvas-tokens', '64', '--max-new-tokens', '32',\n"
            "]\n"
            "subprocess.run(evaluation_command, cwd=PROJECT, check=True)\n"
        ),
        code(
            "evaluation = json.loads((OUTPUT/'checkpoint-evaluation.json').read_text())\n"
            "aggregate = evaluation['aggregate']\n"
            "assert all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in evaluation['records'])\n"
            "assert all(row['hlwm_published'] != row.get('private_canvas', None) for row in evaluation['records'])\n"
            "capability_gate = {\n"
            "    'nonempty_candidates': all(bool(row['hlwm_candidate'].strip()) for row in evaluation['records']),\n"
            "    'corrupt_commit_lower': aggregate['mean_corrupt_commit_probability'] < aggregate['mean_commit_probability'],\n"
            "    'corrupt_risk_higher': aggregate['mean_corrupt_risk_probability'] > aggregate['mean_risk_probability'],\n"
            "    'multiple_routes_used': sum(value > 0.01 for value in aggregate['route_load']) >= 2,\n"
            "}\n"
            "capability_gate['passed'] = all(capability_gate.values())\n"
            "(OUTPUT/'v5-capability-gate.json').write_text(json.dumps(capability_gate, indent=2)+'\\n')\n"
            "print(json.dumps(aggregate, indent=2))\n"
            "print(json.dumps(capability_gate, indent=2))\n"
            "for row in evaluation['records']:\n"
            "    print('\\n', row['episode_id'], 'decision=', 'publish' if row['committed'] else 'abstain')\n"
            "    print('QWEN CONTROL:', row['base_causal'][:500])\n"
            "    print('HLWM QWEN OUTPUT:', row['hlwm_candidate'][:500])\n"
        ),
        code(
            "artifacts = {\n"
            "    str(path.relative_to(OUTPUT)): path.stat().st_size\n"
            "    for path in sorted(OUTPUT.rglob('*')) if path.is_file()\n"
            "}\n"
            "print(json.dumps(artifacts, indent=2))\n"
            "assert any(name.endswith('.safetensors') for name in artifacts)\n"
            "assert 'checkpoint-evaluation.json' in artifacts\n"
            "assert 'v5-capability-gate.json' in artifacts\n"
        ),
        markdown(
            "## Decision boundary\n\n"
            "Do not start a long or dual-GPU run merely because training completed. Continue only "
            "if the capability gate passes and manual inspection finds coherent outputs. Even then, "
            "production claims require an independently adjudicated correction/abstention set, "
            "external held-out benchmarks, calibration, multi-seed replication and systems tests.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_document_v54() -> Dict[str, Any]:
    """Return the Version 5.4 generated-emission-gated dual-seed notebook."""

    def markdown(source: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code(source: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    cells = [
        markdown(
            "# Embel HLWM Version 5.4 - dual-T4 generated-emission-gated prototype\n\n"
            "This run keeps the stable FP32 LoRA and full-vocabulary numerical path, expands the "
            "multi-template split-isolated anchors to 1,536 examples, adds a candidate-level "
            "verification head, and calibrates the complete publication gate on semantically "
            "graded generated validation emissions. Both T4s run concurrently for independent seeds. "
            "Passing is a research gate, not a production-readiness claim.\n\n"
            "**Important:** start this notebook with **Save Version → Save & Run All**. A Quick "
            "Version does not persist the trained files. The last cell verifies and links every "
            "deliverable that Kaggle must save.\n"
        ),
        code(
            "import platform, subprocess, sys\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(
            "!python -m pip install -q transformers==4.56.2 accelerate==1.10.1 safetensors==0.6.2 sentencepiece==0.2.1 pytest==8.4.1\n"
        ),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "WORK = Path('/kaggle/working/hlwm-v5.4')\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "archives = list(Path('/kaggle/input').rglob('hlwm-v5.4-candidate-bundle.zip'))\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for path in Path('/kaggle/input').rglob('manifest.json'):\n"
            "        try:\n"
            "            if json.loads(path.read_text()).get('name') == 'hlwm-reasoning9000-v5-4-candidate': manifests.append(path)\n"
            "        except (OSError, json.JSONDecodeError): pass\n"
            "    if not manifests: raise FileNotFoundError('Attach the HLWM Version 5.4 candidate dataset first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "quality = json.loads((PROJECT/'data-quality.json').read_text())\n"
            "assert manifest['package_version'] == '5.4.0'\n"
            "assert manifest['reasoning9000_counts'] == {'train': 3310, 'validation': 585, 'test': 405}\n"
            "assert manifest['behavior_anchor_counts'] == {'train': 1024, 'validation': 256, 'test': 256}\n"
            "assert quality['independently_adjudicated_episodes'] == 0\n"
            "print(json.dumps({'project': str(PROJECT), 'counts': manifest['counts'], 'boundary': quality['behavior_anchor_boundary']}, indent=2))\n"
        ),
        code(
            "import py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.device_count() == 2, f'Expected two T4 GPUs, found {torch.cuda.device_count()}'\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')], cwd=PROJECT, check=True)\n"
            "print('Version 5.4 architecture, generated-emission calibration and data tests passed')\n"
        ),
        code(
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])\n"
            "print('Pinned Qwen snapshot cached before the dual launch')\n"
        ),
        markdown(
            "## Exact-length real-Qwen numerical preflight\n\n"
            "This performs a forward and backward pass with the real 151,936-token vocabulary, "
            "the production context/canvas/brief lengths and named finiteness hooks. The dual run "
            "does not launch unless this passes.\n"
        ),
        code(
            "import os, subprocess, sys\n"
            "PREFLIGHT_OUTPUT = Path('/kaggle/working/hlwm-v5.4-real-qwen-preflight')\n"
            "if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)\n"
            "preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '17', '--preflight-only',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',\n"
            "    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "    '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--context-tokens', '192', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '384',\n"
            "    '--max-validation', '1', '--num-workers', '0']\n"
            "preflight_environment = os.environ.copy(); preflight_environment['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "subprocess.run(preflight_command, cwd=PROJECT, env=preflight_environment, check=True)\n"
            "print('Exact-length real-Qwen forward/backward preflight passed on GPU 0')\n"
        ),
        markdown(
            "## Concurrent two-seed training\n\n"
            "GPU 0 trains seed 17 and GPU 1 trains seed 29. Each run must pass a 5% fixed-noise "
            "gate, complete 4,224 microsteps with no skipped updates, calibrate policy thresholds "
            "on frozen generated validation emissions only, and export a checksummed adapter.\n"
        ),
        code(
            "import os, subprocess, sys, time\n"
            "SEEDS = [17, 29]\n"
            "OUTPUTS = {seed: Path(f'/kaggle/working/hlwm-v5.4-seed-{seed}') for seed in SEEDS}\n"
            "processes, handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    if output.exists(): shutil.rmtree(output)\n"
            "    command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),\n"
            "        '--overfit-steps', '128', '--local-steps', '1024', '--joint-steps', '3072',\n"
            "        '--overfit-examples', '64', '--overfit-min-improvement', '0.05',\n"
            "        '--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32',\n"
            "        '--causal-ratio', '0.40', '--anchor-ratio', '0.75', '--unfreeze-tail-layers', '0',\n"
            "        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "        '--skip-real-qwen-preflight',\n"
            "        '--initial-loss-scale', '1024', '--max-skipped-updates', '2',\n"
            "        '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "        '--context-tokens', '192', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '384',\n"
            "        '--max-validation', '256', '--calibration-records', '64', '--calibration-new-tokens', '96',\n"
            "        '--num-workers', '2', '--eval-every', '512', '--save-every', '1024',\n"
            "        '--max-runtime-hours', '10.0']\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = Path(f'/kaggle/working/hlwm-v5.4-seed-{seed}-training.log').open('w'); handles[seed] = handle\n"
            "    processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "    print('launched seed', seed, 'on physical GPU', gpu)\n"
            "while any(process.poll() is None for process in processes.values()):\n"
            "    time.sleep(15)\n"
            "    print({seed: {'returncode': process.poll(), 'metrics_bytes': (OUTPUTS[seed]/'metrics.jsonl').stat().st_size if (OUTPUTS[seed]/'metrics.jsonl').exists() else 0} for seed, process in processes.items()})\n"
            "for handle in handles.values(): handle.close()\n"
            "failures = {seed: process.returncode for seed, process in processes.items() if process.returncode != 0}\n"
            "if failures:\n"
            "    for seed in failures: print(Path(f'/kaggle/working/hlwm-v5.4-seed-{seed}-training.log').read_text()[-5000:])\n"
            "    raise RuntimeError(f'dual training failed: {failures}')\n"
            "print('Both seeds finished')\n"
        ),
        code(
            "summaries = {seed: json.loads((output/'summary.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "for seed, summary in summaries.items():\n"
            "    assert summary['status'] == 'stable_prototype_training_complete', summary\n"
            "    assert summary['steps'] == summary['planned_steps'] == 4224, summary\n"
            "    assert summary['overfit_gate']['passed'] and summary['overfit_gate']['relative_improvement'] > 0.05, summary\n"
            "    assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "    assert summary['mode_counts']['causal'] >= 1000, summary\n"
            "    assert summary['commitment_calibration']['verified_anchor_records'] == 64, summary\n"
            "    assert summary['commitment_calibration']['method'] == 'validation_generated_candidate_joint_threshold_v2', summary\n"
            "    assert summary['commitment_calibration']['test_split_used'] is False, summary\n"
            "    assert Path(summary['checkpoint']).exists() and Path(summary['adapter']['path']).exists()\n"
            "print(json.dumps({seed: {'gate': s['overfit_gate'], 'modes': s['mode_counts'], 'peak_memory_gb': s['peak_gpu_allocated_gb']} for seed, s in summaries.items()}, indent=2))\n"
        ),
        markdown(
            "## Fresh reload and 128-output audit\n\n"
            "Each seed is tested on 32 unseen behavior probes and 32 held-out Reasoning9000 "
            "episodes. The audit checks the full reverse schedule, prompt leakage, answer completion, "
            "semantic content, safe abstention, lane collapse, routing and clean/corrupt score margins.\n"
        ),
        code(
            "evaluation_processes, evaluation_handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'), '--checkpoint', summaries[seed]['checkpoint'],\n"
            "        '--data-dir', str(DATA), '--output-dir', str(output), '--samples', '64',\n"
            "        '--context-tokens', '192', '--canvas-tokens', '128', '--max-new-tokens', '96', '--seed', str(seed + 1000)]\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = (output/'evaluation.log').open('w'); evaluation_handles[seed] = handle\n"
            "    evaluation_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "for seed, process in evaluation_processes.items():\n"
            "    if process.wait() != 0:\n"
            "        for handle in evaluation_handles.values(): handle.close()\n"
            "        print((OUTPUTS[seed]/'evaluation.log').read_text()[-5000:])\n"
            "        raise RuntimeError(f'evaluation failed for seed {seed}')\n"
            "for handle in evaluation_handles.values(): handle.close()\n"
            "print('Both fresh-process evaluations finished')\n"
        ),
        code(
            "evaluations = {seed: json.loads((output/'checkpoint-evaluation.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "gates = {}\n"
            "for seed, evaluation in evaluations.items():\n"
            "    aggregate, records = evaluation['aggregate'], evaluation['records']\n"
            "    assert len(records) == 64 and all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in records)\n"
            "    assert all(bool(row['hlwm_candidate'].strip()) for row in records)\n"
            "    gate = {\n"
            "        'no_prompt_leak': aggregate['prompt_leak_rate'] == 0.0,\n"
            "        'quality_pass_rate_at_least_75pct': aggregate['quality_pass_rate'] >= 0.75,\n"
            "        'complete_answer_rate_at_least_50pct': aggregate['complete_answer_rate'] >= 0.50,\n"
            "        'semantic_probe_accuracy_at_least_75pct': aggregate['probe_content_accuracy'] >= 0.75,\n"
            "        'safe_abstention_accuracy_at_least_75pct': aggregate['safe_abstention_accuracy'] >= 0.75,\n"
            "        'safe_abstention_commit_rate_at_least_70pct': aggregate['safe_abstention_commit_rate'] >= 0.70,\n"
            "        'probe_commit_accuracy_at_least_70pct': aggregate['probe_commit_accuracy'] >= 0.70,\n"
            "        'validation_calibration_passed': bool(evaluation['validation_calibration'] and evaluation['validation_calibration']['passed_research_gate']),\n"
            "        'clean_commit_ranked_above_corrupt': aggregate['mean_commit_margin'] > 0.0,\n"
            "        'corrupt_risk_ranked_above_clean': aggregate['mean_risk_margin'] > 0.0,\n"
            "        'corrupt_candidate_verifier_ranked_above_clean': aggregate['mean_verifier_error_margin'] > 0.0,\n"
            "        'multiple_routes_used': sum(value > 0.01 for value in aggregate['route_load']) >= 2,\n"
            "        'lanes_materially_distinct': abs(aggregate['mean_lane_summary_cosine']) < 0.90,\n"
            "        'candidate_f1_not_worse_than_80pct_base': aggregate['mean_hlwm_candidate_token_f1'] >= 0.8 * aggregate['mean_base_token_f1']}\n"
            "    gate['passed'] = all(gate.values()); gates[seed] = gate\n"
            "    (OUTPUTS[seed]/'v5.4-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')\n"
            "    print('SEED', seed, json.dumps(aggregate, indent=2), json.dumps(gate, indent=2))\n"
            "    for row in records[:4]:\n"
            "        print('\\n', row['episode_id'], 'expected_action=', row['expected_action'], 'committed=', row['committed'])\n"
            "        print('OUTPUT:', row['hlwm_candidate'][:700])\n"
            "cross_seed_gate = {'both_seeds_passed': all(gate['passed'] for gate in gates.values())}\n"
            "print(json.dumps(cross_seed_gate, indent=2))\n"
        ),
        code(
            "def seed_score(seed):\n"
            "    a = evaluations[seed]['aggregate']\n"
            "    return a['probe_content_accuracy'] + a['safe_abstention_accuracy'] + a['probe_commit_accuracy'] + a['quality_pass_rate'] + a['mean_hlwm_candidate_token_f1'] - a['prompt_leak_rate']\n"
            "BEST_SEED = max(SEEDS, key=seed_score)\n"
            "DELIVERABLE = Path('/kaggle/working/hlwm-v5.4-best-deliverable')\n"
            "if DELIVERABLE.exists(): shutil.rmtree(DELIVERABLE)\n"
            "DELIVERABLE.mkdir(); best = OUTPUTS[BEST_SEED]\n"
            "for source in [Path(summaries[BEST_SEED]['adapter']['path']), best/'summary.json', best/'commitment-calibration.json', best/'checkpoint-evaluation.json', best/'v5.4-capability-gate.json', best/'metrics.jsonl', PROJECT/'manifest.json', PROJECT/'README.md', PROJECT/'modeling_hlwm.py', PROJECT/'semantic_grading.py', PROJECT/'inference_hlwm.py']:\n"
            "    shutil.copy2(source, DELIVERABLE/source.name)\n"
            "BEST_CHECKPOINT = Path('/kaggle/working/hlwm-v5.4-best-resumable.pt')\n"
            "if BEST_CHECKPOINT.exists(): BEST_CHECKPOINT.unlink()\n"
            "source_checkpoint = Path(summaries[BEST_SEED]['checkpoint'])\n"
            "shutil.move(str(source_checkpoint), BEST_CHECKPOINT)\n"
            "source_sidecar = source_checkpoint.with_suffix(source_checkpoint.suffix+'.sha256')\n"
            "if source_sidecar.exists(): source_sidecar.unlink()\n"
            "for output in OUTPUTS.values():\n"
            "    for stale in output.glob('checkpoint-*.pt*'):\n"
            "        if stale.is_file(): stale.unlink()\n"
            "(DELIVERABLE/'best-seed.json').write_text(json.dumps({'best_seed': BEST_SEED, 'scores': {seed: seed_score(seed) for seed in SEEDS}, 'resumable_checkpoint': str(BEST_CHECKPOINT), 'cross_seed_gate': cross_seed_gate}, indent=2)+'\\n')\n"
            "archive = Path(shutil.make_archive('/kaggle/working/hlwm-v5.4-best-deliverable', 'zip', DELIVERABLE))\n"
            "import hashlib\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "for artifact in (archive, BEST_CHECKPOINT):\n"
            "    assert artifact.exists() and artifact.stat().st_size > 1_000_000, artifact\n"
            "    artifact.with_suffix(artifact.suffix+'.sha256').write_text(file_sha256(artifact)+'  '+artifact.name+'\\n')\n"
            "CHECKPOINT_PARTS = []\n"
            "with BEST_CHECKPOINT.open('rb') as source:\n"
            "    part_index = 0\n"
            "    while True:\n"
            "        payload = source.read(190 * 1024 * 1024)\n"
            "        if not payload: break\n"
            "        part = Path(str(BEST_CHECKPOINT) + f'.part-{part_index:03d}')\n"
            "        part.write_bytes(payload)\n"
            "        part.with_suffix(part.suffix+'.sha256').write_text(file_sha256(part)+'  '+part.name+'\\n')\n"
            "        CHECKPOINT_PARTS.append(part); part_index += 1\n"
            "parts_manifest = Path(str(BEST_CHECKPOINT) + '.parts.json')\n"
            "parts_manifest.write_text(json.dumps({'checkpoint': BEST_CHECKPOINT.name, 'checkpoint_bytes': BEST_CHECKPOINT.stat().st_size, 'checkpoint_sha256': file_sha256(BEST_CHECKPOINT), 'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': file_sha256(part)} for part in CHECKPOINT_PARTS]}, indent=2)+'\\n')\n"
            "PERSISTENCE = Path('/kaggle/working/HLWM-V5.4-DOWNLOADS.txt')\n"
            "PERSISTENCE.write_text('Created by a Save & Run All execution. Download the deliverable ZIP. For resumable training, download either the full checkpoint or every numbered part plus the parts manifest.\\n'+str(archive)+'\\n'+str(BEST_CHECKPOINT)+'\\n'+'\\n'.join(str(part) for part in CHECKPOINT_PARTS)+'\\n'+str(parts_manifest)+'\\n')\n"
            "from IPython.display import FileLink, display\n"
            "print('BEST SEED', BEST_SEED, 'cross-seed gate', cross_seed_gate)\n"
            "print('Verified deliverable bytes', archive.stat().st_size, 'checkpoint bytes', BEST_CHECKPOINT.stat().st_size)\n"
            "display(FileLink(str(archive)))\n"
            "display(FileLink(str(archive.with_suffix(archive.suffix+'.sha256'))))\n"
            "display(FileLink(str(BEST_CHECKPOINT)))\n"
            "display(FileLink(str(BEST_CHECKPOINT.with_suffix(BEST_CHECKPOINT.suffix+'.sha256'))))\n"
            "display(FileLink(str(parts_manifest)))\n"
            "for part in CHECKPOINT_PARTS:\n"
            "    display(FileLink(str(part)))\n"
            "    display(FileLink(str(part.with_suffix(part.suffix+'.sha256'))))\n"
        ),
        markdown(
            "## Decision boundary\n\n"
            "A passing gate authorizes the next research-scale run, not deployment. Production "
            "still requires a human-adjudicated correction and abstention set, external benchmarks, "
            "red-team testing, longer multi-seed training and deployment reliability tests. After "
            "this cell succeeds, wait for the Save & Run All version to finish and download the "
            "linked ZIP plus resumable checkpoint from that saved version's Output tab.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


TRAIN_ARGS_V55 = [
    "'--seed', '29'",
    "'--overfit-steps', '128', '--local-steps', '768', '--joint-steps', '1664'",
    "'--overfit-examples', '64', '--overfit-min-improvement', '0.05'",
    "'--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32'",
    "'--causal-ratio', '0.40', '--anchor-ratio', '0.75', '--unfreeze-tail-layers', '0'",
    "'--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8'",
    "'--precision', 'auto', '--router-entropy-weight', '0.02'",
    "'--skip-real-qwen-preflight'",
    "'--initial-loss-scale', '1024', '--max-skipped-updates', '2'",
    "'--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4'",
    "'--context-tokens', '160', '--canvas-tokens', '80', '--brief-tokens', '96', '--causal-tokens', '256'",
    "'--max-validation', '256', '--calibration-records', '64', '--calibration-new-tokens', '96'",
    "'--policy-records', '96', '--policy-new-tokens', '96', '--policy-epochs', '400', '--policy-learning-rate', '0.001'",
    "'--num-workers', '2', '--eval-every', '512', '--save-every', '512'",
    "'--max-runtime-hours', '2.3'",
]


def runner_script_v55() -> str:
    """Return a shell runner for plain (non-notebook) single-GPU sessions."""

    train_arguments = " ".join(
        line.replace("'", "").replace(", ", " ") for line in TRAIN_ARGS_V55
    )
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "# HLWM Version 5.5 single-GPU session (A100 MIG 1g.5gb, 3.5-hour budget).",
            "# Run from inside the extracted hlwm_kaggle bundle directory.",
            "set -euo pipefail",
            'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}"',
            'WORK="${HLWM_WORK:-$(pwd)/../hlwm-v5.5-work}"',
            'RESUME="${HLWM_RESUME:-}"',
            'mkdir -p "$WORK"',
            "python -m pytest -q test_modeling_hlwm.py || python test_modeling_hlwm.py",
            "python train_kaggle.py --data-dir data \\",
            '  --output-dir "$WORK/preflight" --seed 29 --preflight-only \\',
            "  --batch-size 1 --gradient-accumulation 4 --unfreeze-tail-layers 0 \\",
            "  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 --lora-tail-layers 8 \\",
            "  --num-lanes 2 --num-experts 6 --refinement-steps 2 --diffusion-steps 4 \\",
            "  --precision auto --router-entropy-weight 0.02 \\",
            "  --context-tokens 160 --canvas-tokens 80 --brief-tokens 96 --causal-tokens 256 \\",
            "  --max-validation 1 --num-workers 0",
            'RESUME_ARGS=()',
            'if [ -n "$RESUME" ]; then RESUME_ARGS=(--resume "$RESUME"); fi',
            "python train_kaggle.py --data-dir data \\",
            '  --output-dir "$WORK/seed-29" \\',
            "  " + train_arguments + ' ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}',
            'CHECKPOINT="$(python -c \'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])\' "$WORK/seed-29/summary.json")"',
            "python evaluate_checkpoint.py \\",
            '  --checkpoint "$CHECKPOINT" --data-dir data --output-dir "$WORK/seed-29" \\',
            "  --samples 48 --context-tokens 160 --canvas-tokens 80 --max-new-tokens 96 --seed 1029",
            'echo "Outputs in $WORK/seed-29 (summary.json, commitment-calibration.json,"',
            'echo "policy-head-training.json, checkpoint-evaluation.json, adapter, checkpoint)."',
            "",
        )
    )


PIP_CELL_V55 = (
    "import importlib, importlib.metadata, site, subprocess, sys\n"
    "def ensure(spec):\n"
    "    name, version = spec.split('==')\n"
    "    try:\n"
    "        if importlib.metadata.version(name) == version:\n"
    "            print('already pinned:', spec); return\n"
    "    except importlib.metadata.PackageNotFoundError: pass\n"
    "    try:\n"
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', spec], check=True)\n"
    "    except subprocess.CalledProcessError:\n"
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--user',\n"
    "                        '--no-warn-script-location', spec], check=True)\n"
    "    print('installed:', spec)\n"
    "for spec in ('transformers==4.56.2', 'accelerate==1.10.1', 'safetensors==0.6.2', 'pytest==8.4.1'):\n"
    "    ensure(spec)\n"
    "user_site = site.getusersitepackages()\n"
    "if user_site not in sys.path: sys.path.insert(0, user_site)\n"
    "importlib.invalidate_caches()\n"
    "import transformers\n"
    "print('transformers', transformers.__version__, '| user site', user_site)\n"
)


def notebook_document_v55_t4() -> Dict[str, Any]:
    """Return the Version 5.5 dual-T4 two-seed replication notebook (Kaggle)."""

    def markdown(source: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code(source: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    cells = [
        markdown(
            "# Embel HLWM Version 5.5 - dual-T4 two-seed replication\n\n"
            "This session runs the Version 5.5 architecture (on-policy policy-head phase, "
            "router marginal-entropy regularizer, preregistered routing gates) on two T4s "
            "concurrently, one independent seed per GPU, at the Version 5.4-proven T4 scale: "
            "4,224 microsteps and context 192 / canvas 128 / brief 96 / causal 384. The T4 "
            "path is FP16 with the GradScaler; zero skipped updates remain required. "
            "Plan: `reports/hlwm-v5.5-t4-replication-plan-2026-08-30.md`. Passing both seeds "
            "is a replication result, not a production-readiness claim.\n\n"
            "**Important:** start with **Save Version -> Save & Run All**; a Quick Version "
            "does not persist outputs. Session 1 runs seeds 17 and 29. For session 2, set "
            "the `HLWM_SEEDS` environment variable to `41,73` (or edit the default in the "
            "training cell) and attach nothing new; per-seed resumable checkpoints attached "
            "as input datasets are picked up automatically by name.\n"
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')\n"
            "import platform, subprocess, sys, time\n"
            "SESSION_START = time.time()\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(PIP_CELL_V55),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "KAGGLE = Path('/kaggle/input').exists()\n"
            "WORK = Path('/kaggle/working/hlwm-v5.5') if KAGGLE else Path.cwd() / 'hlwm-v5.5-work'\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
            "archives = [p for root in search_roots for p in root.rglob('hlwm-v5*candidate-bundle.zip')]\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for root in search_roots:\n"
            "        for path in root.rglob('manifest.json'):\n"
            "            try:\n"
            "                if json.loads(path.read_text()).get('name') == 'hlwm-reasoning9000-v5-5-candidate': manifests.append(path)\n"
            "            except (OSError, json.JSONDecodeError): pass\n"
            "    if not manifests: raise FileNotFoundError('Attach the HLWM Version 5.5 candidate bundle first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "quality = json.loads((PROJECT/'data-quality.json').read_text())\n"
            "assert manifest['package_version'] == '5.5.0'\n"
            "assert manifest['reasoning9000_counts'] == {'train': 3310, 'validation': 585, 'test': 405}\n"
            "assert manifest['behavior_anchor_counts'] == {'train': 1024, 'validation': 256, 'test': 256}\n"
            "assert quality['independently_adjudicated_episodes'] == 0\n"
            "print(json.dumps({'project': str(PROJECT), 'counts': manifest['counts']}, indent=2))\n"
        ),
        code(
            "import importlib.util, py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.device_count() == 2, f'expected two T4 GPUs, found {torch.cuda.device_count()} (select GPU T4 x2)'\n"
            "for index in range(2):\n"
            "    properties = torch.cuda.get_device_properties(index)\n"
            "    print(index, properties.name, round(properties.total_memory/2**30, 2), 'GB')\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "if importlib.util.find_spec('pytest') is not None:\n"
            "    test_command = [sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')]\n"
            "else:\n"
            "    print('pytest unavailable; falling back to direct execution')\n"
            "    test_command = [sys.executable, str(PROJECT/'test_modeling_hlwm.py')]\n"
            "result = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)\n"
            "print((result.stdout or '')[-4000:])\n"
            "if result.returncode != 0:\n"
            "    print((result.stderr or '')[-4000:])\n"
            "    raise RuntimeError('Version 5.5 tests failed')\n"
            "print('Version 5.5 architecture, on-policy head, routing and data tests passed')\n"
        ),
        code(
            "os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))\n"
            "Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])\n"
            "print('Pinned Qwen snapshot cached before the dual launch')\n"
        ),
        markdown(
            "## Real-Qwen numerical and memory preflight (GPU 0)\n\n"
            "Local **and joint** forward/backward passes at the T4 production lengths with the "
            "full 151,936-token vocabulary, plus the 92% device-memory headroom limit. Both "
            "T4s are identical, so one preflight covers the dual launch.\n"
        ),
        code(
            "import os, subprocess, sys\n"
            "PREFLIGHT_OUTPUT = WORK / 'preflight'\n"
            "if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)\n"
            "preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '17', '--preflight-only',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',\n"
            "    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "    '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--precision', 'auto', '--router-entropy-weight', '0.02',\n"
            "    '--context-tokens', '192', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '384',\n"
            "    '--max-validation', '1', '--num-workers', '0']\n"
            "preflight_environment = os.environ.copy(); preflight_environment['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "subprocess.run(preflight_command, cwd=PROJECT, env=preflight_environment, check=True)\n"
            "print('Local+joint real-Qwen preflight and memory headroom check passed on GPU 0')\n"
        ),
        markdown(
            "## Concurrent two-seed Version 5.5 training\n\n"
            "Each seed runs 128 overfit + 1,024 local + 3,072 joint microsteps (5% fixed-noise "
            "gate, zero skipped updates required), then generates 96 train-anchor emissions, "
            "grades them, fits only the three candidate heads on emissions plus semantic and "
            "mechanical hard negatives, and calibrates thresholds on 64 generated validation "
            "emissions. The 8.5-hour training cap checkpoints exact resumable state; a "
            "truncated seed resumes next session via its attached "
            "`hlwm-v5.5-seed-<seed>-resumable.pt`.\n"
        ),
        code(
            "import os, subprocess, sys, time\n"
            "SEEDS = [int(part) for part in os.environ.get('HLWM_SEEDS', '17,29').split(',')]\n"
            "assert len(SEEDS) == 2, 'this notebook schedules exactly two seeds, one per T4'\n"
            "OUTPUTS = {seed: Path(f'/kaggle/working/hlwm-v5.5-seed-{seed}') for seed in SEEDS}\n"
            "processes, handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    if output.exists(): shutil.rmtree(output)\n"
            "    resume = [p for root in search_roots for p in root.rglob(f'hlwm-v5.5-seed-{seed}-resumable.pt')]\n"
            "    command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),\n"
            "        '--overfit-steps', '128', '--local-steps', '1024', '--joint-steps', '3072',\n"
            "        '--overfit-examples', '64', '--overfit-min-improvement', '0.05',\n"
            "        '--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32',\n"
            "        '--causal-ratio', '0.40', '--anchor-ratio', '0.75', '--unfreeze-tail-layers', '0',\n"
            "        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "        '--precision', 'auto', '--router-entropy-weight', '0.02',\n"
            "        '--skip-real-qwen-preflight',\n"
            "        '--initial-loss-scale', '1024', '--max-skipped-updates', '2',\n"
            "        '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "        '--context-tokens', '192', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '384',\n"
            "        '--max-validation', '256', '--calibration-records', '64', '--calibration-new-tokens', '96',\n"
            "        '--policy-records', '96', '--policy-new-tokens', '96', '--policy-epochs', '400', '--policy-learning-rate', '0.001',\n"
            "        '--num-workers', '2', '--eval-every', '512', '--save-every', '1024',\n"
            "        '--max-runtime-hours', '8.5']\n"
            "    if resume:\n"
            "        command += ['--resume', str(resume[0])]\n"
            "        print('seed', seed, 'resuming from', resume[0])\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = Path(f'/kaggle/working/hlwm-v5.5-seed-{seed}-training.log').open('w'); handles[seed] = handle\n"
            "    processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "    print('launched seed', seed, 'on physical GPU', gpu)\n"
            "while any(process.poll() is None for process in processes.values()):\n"
            "    time.sleep(30)\n"
            "    print({seed: {'returncode': process.poll(),\n"
            "                  'elapsed_min': round((time.time()-SESSION_START)/60, 1),\n"
            "                  'metrics_bytes': (OUTPUTS[seed]/'metrics.jsonl').stat().st_size if (OUTPUTS[seed]/'metrics.jsonl').exists() else 0}\n"
            "           for seed, process in processes.items()})\n"
            "for handle in handles.values(): handle.close()\n"
            "failures = {seed: process.returncode for seed, process in processes.items() if process.returncode != 0}\n"
            "if failures:\n"
            "    for seed in failures: print(Path(f'/kaggle/working/hlwm-v5.5-seed-{seed}-training.log').read_text()[-8000:])\n"
            "    raise RuntimeError(f'dual training failed: {failures}')\n"
            "print('Both seeds finished')\n"
        ),
        code(
            "summaries = {seed: json.loads((output/'summary.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "RUN_COMPLETE = {}\n"
            "for seed, summary in summaries.items():\n"
            "    assert summary['status'] in ('stable_prototype_training_complete', 'time_budget_checkpoint_saved'), summary['status']\n"
            "    RUN_COMPLETE[seed] = summary['status'] == 'stable_prototype_training_complete'\n"
            "    assert summary['planned_steps'] == 4224, summary['planned_steps']\n"
            "    assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "    assert summary['overfit_gate']['passed'], summary['overfit_gate']\n"
            "    assert summary['on_policy_policy_head'] is not None and summary['on_policy_policy_head']['trained'], summary['on_policy_policy_head']\n"
            "    calibration = summary['commitment_calibration']\n"
            "    assert calibration['method'] == 'validation_generated_candidate_joint_threshold_v2'\n"
            "    assert calibration['test_split_used'] is False\n"
            "    assert calibration['policy_heads_trained_on_policy'] is True\n"
            "    assert Path(summary['checkpoint']).exists() and Path(summary['adapter']['path']).exists()\n"
            "    print(seed, json.dumps({'run_complete': RUN_COMPLETE[seed], 'steps': summary['steps'],\n"
            "        'precision': summary['precision'], 'peak_gb': summary['peak_gpu_allocated_gb'],\n"
            "        'policy_separation_after': summary['on_policy_policy_head'].get('separation_after'),\n"
            "        'calibration': {k: calibration[k] for k in ('positive_accept_rate','negative_reject_rate','balanced_accuracy','passed_research_gate')}}, indent=2))\n"
        ),
        markdown(
            "## Fresh reload and per-seed 64-output audit with routing interventions\n\n"
            "Each seed reloads its checksummed checkpoint in a fresh process on its own GPU, "
            "runs the full reverse schedule on unseen test probes and Reasoning9000 episodes, "
            "and measures normalized route entropy, second-route load, and the policy-score "
            "movement when routing is pinned to the least-used expert.\n"
        ),
        code(
            "evaluation_processes, evaluation_handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "        '--checkpoint', summaries[seed]['checkpoint'], '--data-dir', str(DATA), '--output-dir', str(output),\n"
            "        '--samples', '64', '--context-tokens', '192', '--canvas-tokens', '128',\n"
            "        '--max-new-tokens', '96', '--seed', str(seed + 1000)]\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = (output/'evaluation.log').open('w'); evaluation_handles[seed] = handle\n"
            "    evaluation_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "for seed, process in evaluation_processes.items():\n"
            "    if process.wait() != 0:\n"
            "        for handle in evaluation_handles.values(): handle.close()\n"
            "        print((OUTPUTS[seed]/'evaluation.log').read_text()[-5000:])\n"
            "        raise RuntimeError(f'evaluation failed for seed {seed}')\n"
            "for handle in evaluation_handles.values(): handle.close()\n"
            "print('Both fresh-process evaluations finished')\n"
        ),
        code(
            "evaluations = {seed: json.loads((output/'checkpoint-evaluation.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "gates = {}\n"
            "for seed, evaluation in evaluations.items():\n"
            "    aggregate, records = evaluation['aggregate'], evaluation['records']\n"
            "    assert len(records) == 64 and all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in records)\n"
            "    gate = {\n"
            "        'training_complete': RUN_COMPLETE[seed],\n"
            "        'no_prompt_leak': aggregate['prompt_leak_rate'] == 0.0,\n"
            "        'quality_pass_rate_at_least_75pct': aggregate['quality_pass_rate'] >= 0.75,\n"
            "        'complete_answer_rate_at_least_50pct': aggregate['complete_answer_rate'] >= 0.50,\n"
            "        'semantic_probe_accuracy_at_least_75pct': aggregate['probe_content_accuracy'] >= 0.75,\n"
            "        'safe_abstention_accuracy_at_least_75pct': aggregate['safe_abstention_accuracy'] >= 0.75,\n"
            "        'safe_abstention_commit_rate_at_least_70pct': aggregate['safe_abstention_commit_rate'] >= 0.70,\n"
            "        'probe_commit_accuracy_at_least_70pct': aggregate['probe_commit_accuracy'] >= 0.70,\n"
            "        'validation_calibration_passed': bool(evaluation['validation_calibration'] and evaluation['validation_calibration']['passed_research_gate']),\n"
            "        'policy_heads_trained_on_policy': bool(evaluation['validation_calibration'] and evaluation['validation_calibration'].get('policy_heads_trained_on_policy')),\n"
            "        'clean_commit_ranked_above_corrupt': aggregate['mean_commit_margin'] > 0.0,\n"
            "        'corrupt_risk_ranked_above_clean': aggregate['mean_risk_margin'] > 0.0,\n"
            "        'corrupt_candidate_verifier_ranked_above_clean': aggregate['mean_verifier_error_margin'] > 0.0,\n"
            "        'second_route_load_at_least_10pct': aggregate['second_route_load'] >= 0.10,\n"
            "        'route_entropy_normalized_at_least_0p25': aggregate['route_entropy_normalized'] >= 0.25,\n"
            "        'routing_causally_live': aggregate['routing_intervention']['mean_abs_commit_delta'] >= 0.01,\n"
            "        'lanes_materially_distinct': abs(aggregate['mean_lane_summary_cosine']) < 0.90,\n"
            "        'candidate_f1_not_worse_than_80pct_base': aggregate['mean_hlwm_candidate_token_f1'] >= 0.8 * aggregate['mean_base_token_f1'],\n"
            "    }\n"
            "    gate['passed'] = all(gate.values()); gates[seed] = gate\n"
            "    (OUTPUTS[seed]/'v5.5-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')\n"
            "    print('SEED', seed); print(json.dumps(aggregate, indent=2)); print(json.dumps(gate, indent=2))\n"
            "    for row in records[:2]:\n"
            "        print('\\n', row['episode_id'], 'expected_action=', row['expected_action'], 'committed=', row['committed'])\n"
            "        print('OUTPUT:', row['hlwm_candidate'][:500])\n"
            "replication = {'seeds': SEEDS,\n"
            "    'per_seed_passed': {str(seed): gates[seed]['passed'] for seed in SEEDS},\n"
            "    'replicated': all(gates[seed]['passed'] for seed in SEEDS),\n"
            "    'planned_steps': 4224,\n"
            "    'key_rates': {str(seed): {key: evaluations[seed]['aggregate'][key] for key in (\n"
            "        'probe_content_accuracy','safe_abstention_accuracy','probe_commit_accuracy',\n"
            "        'second_route_load','route_entropy_normalized')} for seed in SEEDS}}\n"
            "Path('/kaggle/working/hlwm-v5.5-replication-verdict.json').write_text(json.dumps(replication, indent=2)+'\\n')\n"
            "print(json.dumps(replication, indent=2))\n"
        ),
        code(
            "import hashlib\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "ARTIFACTS = [Path('/kaggle/working/hlwm-v5.5-replication-verdict.json')]\n"
            "for seed in SEEDS:\n"
            "    output = OUTPUTS[seed]; summary = summaries[seed]\n"
            "    staging = Path(f'/kaggle/working/hlwm-v5.5-seed-{seed}-deliverable')\n"
            "    if staging.exists(): shutil.rmtree(staging)\n"
            "    staging.mkdir()\n"
            "    for source in [Path(summary['adapter']['path']), output/'summary.json', output/'commitment-calibration.json',\n"
            "                   output/'policy-head-training.json', output/'checkpoint-evaluation.json',\n"
            "                   output/'v5.5-capability-gate.json', output/'metrics.jsonl',\n"
            "                   PROJECT/'manifest.json', PROJECT/'README.md', PROJECT/'modeling_hlwm.py',\n"
            "                   PROJECT/'semantic_grading.py', PROJECT/'inference_hlwm.py']:\n"
            "        shutil.copy2(source, staging/source.name)\n"
            "    archive = Path(shutil.make_archive(f'/kaggle/working/hlwm-v5.5-seed-{seed}-deliverable', 'zip', staging))\n"
            "    resumable = Path(f'/kaggle/working/hlwm-v5.5-seed-{seed}-resumable.pt')\n"
            "    if resumable.exists(): resumable.unlink()\n"
            "    source_checkpoint = Path(summary['checkpoint'])\n"
            "    shutil.move(str(source_checkpoint), resumable)\n"
            "    sidecar = source_checkpoint.with_suffix(source_checkpoint.suffix + '.sha256')\n"
            "    if sidecar.exists(): sidecar.unlink()\n"
            "    for stale in output.glob('checkpoint-*.pt*'):\n"
            "        if stale.is_file(): stale.unlink()\n"
            "    for artifact in (archive, resumable):\n"
            "        assert artifact.exists() and artifact.stat().st_size > 1_000_000, artifact\n"
            "        artifact.with_suffix(artifact.suffix+'.sha256').write_text(file_sha256(artifact)+'  '+artifact.name+'\\n')\n"
            "    parts = []\n"
            "    with resumable.open('rb') as source:\n"
            "        part_index = 0\n"
            "        while True:\n"
            "            payload = source.read(190 * 1024 * 1024)\n"
            "            if not payload: break\n"
            "            part = Path(str(resumable) + f'.part-{part_index:03d}')\n"
            "            part.write_bytes(payload)\n"
            "            part.with_suffix(part.suffix+'.sha256').write_text(file_sha256(part)+'  '+part.name+'\\n')\n"
            "            parts.append(part); part_index += 1\n"
            "    parts_manifest = Path(str(resumable) + '.parts.json')\n"
            "    parts_manifest.write_text(json.dumps({'checkpoint': resumable.name, 'checkpoint_bytes': resumable.stat().st_size,\n"
            "        'checkpoint_sha256': file_sha256(resumable),\n"
            "        'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': file_sha256(part)} for part in parts]}, indent=2)+'\\n')\n"
            "    ARTIFACTS += [archive, archive.with_suffix(archive.suffix+'.sha256'), resumable,\n"
            "                  resumable.with_suffix(resumable.suffix+'.sha256'), parts_manifest, *parts]\n"
            "Path('/kaggle/working/HLWM-V5.5-DOWNLOADS.txt').write_text('Save & Run All output. Download every line below.\\n'+'\\n'.join(str(path) for path in ARTIFACTS)+'\\n')\n"
            "print('replicated:', replication['replicated'])\n"
            "for seed in SEEDS: print(seed, 'run_complete', RUN_COMPLETE[seed], 'gate_passed', gates[seed]['passed'])\n"
            "try:\n"
            "    from IPython.display import FileLink, display\n"
            "    for path in ARTIFACTS: display(FileLink(str(path)))\n"
            "except Exception as error:\n"
            "    print('links unavailable outside a notebook UI:', error)\n"
            "import time\n"
            "print('total session minutes:', round((time.time()-SESSION_START)/60, 1))\n"
        ),
        markdown(
            "## Decision boundary\n\n"
            "Both seeds passing replicates the Version 5.5 research gate at T4 scale; it "
            "authorizes the external-benchmark step (GSM8K / ARC / MMLU subsets against "
            "same-size instruct models), not deployment. A time-truncated seed is valid "
            "evidence and must be resumed (attach its `hlwm-v5.5-seed-<seed>-resumable.pt`) "
            "before any gate claim for that seed. Session 2 with `HLWM_SEEDS=41,73` extends "
            "the replication to four seeds. Production still requires independently "
            "adjudicated correction/abstention data, red-teaming, and serving tests.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_document_v55() -> Dict[str, Any]:
    """Return the Version 5.5 single-GPU (A100 MIG 1g.5gb) notebook."""

    def markdown(source: str) -> Dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code(source: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    train_arguments = ",\n".join("        " + line for line in TRAIN_ARGS_V55)
    cells = [
        markdown(
            "# Embel HLWM Version 5.5 - single-GPU on-policy-gated prototype\n\n"
            "This session targets one A100 MIG `1g.5gb` partition (about 4.75 GB) inside a "
            "3.5-hour budget; it also runs on a single T4. Version 5.5 trains the three "
            "candidate policy heads on generated train-anchor emissions with graded hard "
            "negatives, regularizes routing toward a live multi-expert marginal, and audits "
            "routing with preregistered entropy, minimum-load, and intervention checks. "
            "Passing is a research gate, not a production-readiness claim.\n\n"
            "**Session plan (3:30 total):** setup+preflight ~15 min, training capped at "
            "2 h 18 min, on-policy head phase + calibration ~25 min, 48-output audit ~20 min, "
            "packaging ~5 min. A truncated run saves an exact resumable checkpoint; a second "
            "session resumes it via `HLWM_RESUME`/attached checkpoint without repeating steps.\n"
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')\n"
            "import platform, subprocess, sys, time\n"
            "SESSION_START = time.time()\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)\n"
        ),
        code(PIP_CELL_V55),
        code(
            "from pathlib import Path\n"
            "import json, shutil, zipfile\n"
            "KAGGLE = Path('/kaggle/input').exists()\n"
            "WORK = Path('/kaggle/working/hlwm-v5.5') if KAGGLE else Path.cwd() / 'hlwm-v5.5-work'\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
            "archives = [p for root in search_roots for p in root.rglob('hlwm-v5*candidate-bundle.zip')]\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "else:\n"
            "    manifests = []\n"
            "    for root in search_roots:\n"
            "        for path in root.rglob('manifest.json'):\n"
            "            try:\n"
            "                if json.loads(path.read_text()).get('name') == 'hlwm-reasoning9000-v5-5-candidate': manifests.append(path)\n"
            "            except (OSError, json.JSONDecodeError): pass\n"
            "    if not manifests: raise FileNotFoundError('Attach or copy the HLWM Version 5.5 candidate bundle first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            "quality = json.loads((PROJECT/'data-quality.json').read_text())\n"
            "assert manifest['package_version'] == '5.5.0'\n"
            "assert manifest['reasoning9000_counts'] == {'train': 3310, 'validation': 585, 'test': 405}\n"
            "assert manifest['behavior_anchor_counts'] == {'train': 1024, 'validation': 256, 'test': 256}\n"
            "assert quality['independently_adjudicated_episodes'] == 0\n"
            "print(json.dumps({'project': str(PROJECT), 'counts': manifest['counts']}, indent=2))\n"
        ),
        code(
            "import importlib.util, py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.is_available(), 'a CUDA device is required'\n"
            "properties = torch.cuda.get_device_properties(0)\n"
            "total_gb = properties.total_memory / 2**30\n"
            "FREE_GB = torch.cuda.mem_get_info()[0] / 2**30\n"
            "print({'device': properties.name, 'total_memory_gb': round(total_gb, 2),\n"
            "       'free_memory_gb': round(FREE_GB, 2),\n"
            "       'bf16': torch.cuda.is_bf16_supported(), 'devices': torch.cuda.device_count()})\n"
            "assert total_gb >= 4.0, f'need at least a 5 GB partition, found {total_gb:.2f} GB'\n"
            "assert FREE_GB >= 3.2, f'only {FREE_GB:.2f} GB free; another process is holding the partition'\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "if importlib.util.find_spec('pytest') is not None:\n"
            "    test_command = [sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')]\n"
            "else:\n"
            "    print('pytest unavailable; falling back to direct execution')\n"
            "    test_command = [sys.executable, str(PROJECT/'test_modeling_hlwm.py')]\n"
            "result = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)\n"
            "print((result.stdout or '')[-4000:])\n"
            "if result.returncode != 0:\n"
            "    print((result.stderr or '')[-4000:])\n"
            "    raise RuntimeError('Version 5.5 tests failed')\n"
            "print('Version 5.5 architecture, on-policy head, routing and data tests passed')\n"
        ),
        code(
            "os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))\n"
            "Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])\n"
            "print('Pinned Qwen snapshot cached under', os.environ['HF_HOME'])\n"
        ),
        markdown(
            "## Real-Qwen numerical and memory preflight\n\n"
            "This runs local **and joint** forward/backward passes at the production lengths "
            "with the full 151,936-token vocabulary, then enforces the 92% device-memory "
            "headroom limit. On a 5 GB MIG partition an oversized configuration fails here, "
            "in minutes, not mid-session.\n"
        ),
        code(
            "import os, subprocess, sys\n"
            "PREFLIGHT_OUTPUT = WORK / 'preflight'\n"
            "if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)\n"
            "preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '29', '--preflight-only',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',\n"
            "    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "    '--num-lanes', '2', '--num-experts', '6', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--precision', 'auto', '--router-entropy-weight', '0.02',\n"
            "    '--context-tokens', '160', '--canvas-tokens', '80', '--brief-tokens', '96', '--causal-tokens', '256',\n"
            "    '--max-validation', '1', '--num-workers', '0']\n"
            "subprocess.run(preflight_command, cwd=PROJECT, check=True)\n"
            "print('Local+joint real-Qwen preflight and memory headroom check passed')\n"
        ),
        markdown(
            "## Single-seed training with the on-policy policy-head phase\n\n"
            "Seed 29 trains 128 overfit + 768 local + 1,664 joint microsteps (5% fixed-noise "
            "gate, zero skipped updates required), then generates 96 train-anchor emissions, "
            "grades them, fits only the three candidate heads on emissions plus semantic and "
            "mechanical hard negatives, and finally calibrates thresholds on 64 generated "
            "validation emissions. The 2.3-hour training cap checkpoints resumable state "
            "before the session budget is at risk.\n"
        ),
        code(
            "import os, subprocess, sys\n"
            "OUTPUT = WORK / 'seed-29'\n"
            "if OUTPUT.exists(): shutil.rmtree(OUTPUT)\n"
            "RESUME = os.environ.get('HLWM_RESUME', '')\n"
            "if not RESUME:\n"
            "    found = [p for root in search_roots for p in root.rglob('hlwm-v5.5-resumable.pt')]\n"
            "    RESUME = str(found[0]) if found else ''\n"
            "command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(OUTPUT),\n"
            + train_arguments
            + "]\n"
            "if RESUME:\n"
            "    command += ['--resume', RESUME]\n"
            "    print('resuming from', RESUME)\n"
            "print(' '.join(command))\n"
            "log_path = WORK / 'seed-29-training.log'\n"
            "with log_path.open('w') as handle:\n"
            "    process = subprocess.Popen(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)\n"
            "    import time\n"
            "    while process.poll() is None:\n"
            "        time.sleep(30)\n"
            "        metrics = OUTPUT / 'metrics.jsonl'\n"
            "        print({'elapsed_min': round((time.time()-SESSION_START)/60, 1),\n"
            "               'metrics_bytes': metrics.stat().st_size if metrics.exists() else 0})\n"
            "if process.returncode != 0:\n"
            "    print(log_path.read_text()[-8000:])\n"
            "    raise RuntimeError('training failed')\n"
            "print('Training session finished')\n"
        ),
        code(
            "summary = json.loads((OUTPUT/'summary.json').read_text())\n"
            "RUN_COMPLETE = summary['status'] == 'stable_prototype_training_complete'\n"
            "assert summary['status'] in ('stable_prototype_training_complete', 'time_budget_checkpoint_saved'), summary['status']\n"
            "assert summary['planned_steps'] == 2560, summary['planned_steps']\n"
            "assert summary['skipped_optimizer_updates'] == 0, summary\n"
            "assert summary['overfit_gate']['passed'], summary['overfit_gate']\n"
            "assert summary['on_policy_policy_head'] is not None and summary['on_policy_policy_head']['trained'], summary['on_policy_policy_head']\n"
            "calibration = summary['commitment_calibration']\n"
            "assert calibration['method'] == 'validation_generated_candidate_joint_threshold_v2'\n"
            "assert calibration['test_split_used'] is False\n"
            "assert calibration['policy_heads_trained_on_policy'] is True\n"
            "assert Path(summary['checkpoint']).exists() and Path(summary['adapter']['path']).exists()\n"
            "print(json.dumps({'run_complete': RUN_COMPLETE, 'steps': summary['steps'],\n"
            "    'precision': summary['precision'], 'peak_gb': summary['peak_gpu_allocated_gb'],\n"
            "    'policy_separation_after': summary['on_policy_policy_head'].get('separation_after'),\n"
            "    'calibration': {k: calibration[k] for k in ('positive_accept_rate','negative_reject_rate','balanced_accuracy','passed_research_gate')}}, indent=2))\n"
        ),
        markdown(
            "## Fresh reload and 48-output audit with routing interventions\n\n"
            "The audit reloads the checksummed checkpoint in a fresh process, runs the full "
            "reverse schedule on unseen test behavior probes and Reasoning9000 episodes, and "
            "additionally measures normalized route entropy, second-route load, and the "
            "policy-score movement when routing is pinned to the least-used expert.\n"
        ),
        code(
            "evaluation_command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "    '--checkpoint', summary['checkpoint'], '--data-dir', str(DATA), '--output-dir', str(OUTPUT),\n"
            "    '--samples', '48', '--context-tokens', '160', '--canvas-tokens', '80',\n"
            "    '--max-new-tokens', '96', '--seed', '1029']\n"
            "subprocess.run(evaluation_command, cwd=PROJECT, check=True)\n"
            "print('Fresh-process evaluation finished')\n"
        ),
        code(
            "evaluation = json.loads((OUTPUT/'checkpoint-evaluation.json').read_text())\n"
            "aggregate, records = evaluation['aggregate'], evaluation['records']\n"
            "assert len(records) == 48 and all(row['reverse_timesteps'] == [4, 3, 2, 1] for row in records)\n"
            "gate = {\n"
            "    'training_complete': RUN_COMPLETE,\n"
            "    'no_prompt_leak': aggregate['prompt_leak_rate'] == 0.0,\n"
            "    'quality_pass_rate_at_least_75pct': aggregate['quality_pass_rate'] >= 0.75,\n"
            "    'complete_answer_rate_at_least_50pct': aggregate['complete_answer_rate'] >= 0.50,\n"
            "    'semantic_probe_accuracy_at_least_75pct': aggregate['probe_content_accuracy'] >= 0.75,\n"
            "    'safe_abstention_accuracy_at_least_75pct': aggregate['safe_abstention_accuracy'] >= 0.75,\n"
            "    'safe_abstention_commit_rate_at_least_70pct': aggregate['safe_abstention_commit_rate'] >= 0.70,\n"
            "    'probe_commit_accuracy_at_least_70pct': aggregate['probe_commit_accuracy'] >= 0.70,\n"
            "    'validation_calibration_passed': bool(evaluation['validation_calibration'] and evaluation['validation_calibration']['passed_research_gate']),\n"
            "    'policy_heads_trained_on_policy': bool(evaluation['validation_calibration'] and evaluation['validation_calibration'].get('policy_heads_trained_on_policy')),\n"
            "    'clean_commit_ranked_above_corrupt': aggregate['mean_commit_margin'] > 0.0,\n"
            "    'corrupt_risk_ranked_above_clean': aggregate['mean_risk_margin'] > 0.0,\n"
            "    'corrupt_candidate_verifier_ranked_above_clean': aggregate['mean_verifier_error_margin'] > 0.0,\n"
            "    'second_route_load_at_least_10pct': aggregate['second_route_load'] >= 0.10,\n"
            "    'route_entropy_normalized_at_least_0p25': aggregate['route_entropy_normalized'] >= 0.25,\n"
            "    'routing_causally_live': aggregate['routing_intervention']['mean_abs_commit_delta'] >= 0.01,\n"
            "    'lanes_materially_distinct': abs(aggregate['mean_lane_summary_cosine']) < 0.90,\n"
            "    'candidate_f1_not_worse_than_80pct_base': aggregate['mean_hlwm_candidate_token_f1'] >= 0.8 * aggregate['mean_base_token_f1'],\n"
            "}\n"
            "gate['passed'] = all(gate.values())\n"
            "(OUTPUT/'v5.5-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')\n"
            "print(json.dumps(aggregate, indent=2))\n"
            "print(json.dumps(gate, indent=2))\n"
            "for row in records[:4]:\n"
            "    print('\\n', row['episode_id'], 'expected_action=', row['expected_action'], 'committed=', row['committed'])\n"
            "    print('OUTPUT:', row['hlwm_candidate'][:700])\n"
        ),
        code(
            "DELIVERABLE = WORK / 'hlwm-v5.5-deliverable'\n"
            "if DELIVERABLE.exists(): shutil.rmtree(DELIVERABLE)\n"
            "DELIVERABLE.mkdir()\n"
            "for source in [Path(summary['adapter']['path']), OUTPUT/'summary.json', OUTPUT/'commitment-calibration.json',\n"
            "               OUTPUT/'policy-head-training.json', OUTPUT/'checkpoint-evaluation.json',\n"
            "               OUTPUT/'v5.5-capability-gate.json', OUTPUT/'metrics.jsonl',\n"
            "               PROJECT/'manifest.json', PROJECT/'README.md', PROJECT/'modeling_hlwm.py',\n"
            "               PROJECT/'semantic_grading.py', PROJECT/'inference_hlwm.py']:\n"
            "    shutil.copy2(source, DELIVERABLE/source.name)\n"
            "RESUMABLE = WORK.parent / 'hlwm-v5.5-resumable.pt' if KAGGLE else WORK / 'hlwm-v5.5-resumable.pt'\n"
            "if RESUMABLE.exists(): RESUMABLE.unlink()\n"
            "source_checkpoint = Path(summary['checkpoint'])\n"
            "shutil.move(str(source_checkpoint), RESUMABLE)\n"
            "sidecar = source_checkpoint.with_suffix(source_checkpoint.suffix + '.sha256')\n"
            "if sidecar.exists(): sidecar.unlink()\n"
            "for stale in OUTPUT.glob('checkpoint-*.pt*'):\n"
            "    if stale.is_file(): stale.unlink()\n"
            "archive = Path(shutil.make_archive(str(WORK.parent / 'hlwm-v5.5-deliverable') if KAGGLE else str(WORK / 'hlwm-v5.5-deliverable-archive'), 'zip', DELIVERABLE))\n"
            "import hashlib\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "for artifact in (archive, RESUMABLE):\n"
            "    assert artifact.exists() and artifact.stat().st_size > 1_000_000, artifact\n"
            "    artifact.with_suffix(artifact.suffix+'.sha256').write_text(file_sha256(artifact)+'  '+artifact.name+'\\n')\n"
            "CHECKPOINT_PARTS = []\n"
            "with RESUMABLE.open('rb') as source:\n"
            "    part_index = 0\n"
            "    while True:\n"
            "        payload = source.read(190 * 1024 * 1024)\n"
            "        if not payload: break\n"
            "        part = Path(str(RESUMABLE) + f'.part-{part_index:03d}')\n"
            "        part.write_bytes(payload)\n"
            "        part.with_suffix(part.suffix+'.sha256').write_text(file_sha256(part)+'  '+part.name+'\\n')\n"
            "        CHECKPOINT_PARTS.append(part); part_index += 1\n"
            "parts_manifest = Path(str(RESUMABLE) + '.parts.json')\n"
            "parts_manifest.write_text(json.dumps({'checkpoint': RESUMABLE.name, 'checkpoint_bytes': RESUMABLE.stat().st_size,\n"
            "    'checkpoint_sha256': file_sha256(RESUMABLE),\n"
            "    'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': file_sha256(part)} for part in CHECKPOINT_PARTS]}, indent=2)+'\\n')\n"
            "print('run_complete', RUN_COMPLETE, 'gate_passed', gate['passed'])\n"
            "print('deliverable', archive, archive.stat().st_size, 'bytes')\n"
            "print('resumable', RESUMABLE, RESUMABLE.stat().st_size, 'bytes')\n"
            "try:\n"
            "    from IPython.display import FileLink, display\n"
            "    for path in [archive, archive.with_suffix(archive.suffix+'.sha256'), RESUMABLE,\n"
            "                 RESUMABLE.with_suffix(RESUMABLE.suffix+'.sha256'), parts_manifest, *CHECKPOINT_PARTS]:\n"
            "        display(FileLink(str(path)))\n"
            "except Exception as error:\n"
            "    print('links unavailable outside a notebook UI:', error)\n"
            "import time\n"
            "print('total session minutes:', round((time.time()-SESSION_START)/60, 1))\n"
        ),
        markdown(
            "## Decision boundary\n\n"
            "A passing gate authorizes the next research-scale run, not deployment. A "
            "time-truncated run is valid evidence and must be resumed (attach "
            "`hlwm-v5.5-resumable.pt` or set `HLWM_RESUME`) before any gate claim. Production "
            "still requires an independently adjudicated correction and abstention set, "
            "external benchmarks, red-team testing, multi-seed replication, and serving "
            "reliability tests.\n"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_quality_report(source_master: Path) -> Dict[str, Any]:
    commitments: Counter[str] = Counter()
    verification: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    episodes = 0
    lanes = 0
    adjudicated = 0
    for split in ("train", "validation", "test"):
        with (source_master / (split + ".jsonl")).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                episodes += 1
                metadata = record.get("generation_metadata") or {}
                adjudicated += int(bool(metadata.get("independently_adjudicated", False)))
                commitments[str((record.get("commitment") or {}).get("decision", "missing"))] += 1
                for item in record.get("verification", []):
                    if isinstance(item, Mapping):
                        verification[str(item.get("verdict", "missing"))] += 1
                for lane in record.get("lanes", []):
                    if not isinstance(lane, Mapping):
                        continue
                    lanes += 1
                    route = " -> ".join(str(node) for node in lane.get("route", []))
                    routes[route or "missing"] += 1
    return {
        "episodes": episodes,
        "lanes": lanes,
        "independently_adjudicated_episodes": adjudicated,
        "commitment_decisions": dict(sorted(commitments.items())),
        "verification_verdicts": dict(sorted(verification.items())),
        "top_routes": dict(routes.most_common(10)),
        "generation_method": "direct_fast_no_judge",
        "policy_label_use": (
            "masked for direct_fast_no_judge rows; enabled only for independently adjudicated "
            "or narrow programmatically verified behavior anchors"
        ),
        "production_boundary": (
            "This release can test architecture and optimization but cannot establish factual "
            "quality, abstention calibration or production readiness."
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    project = output / "bundle" / "hlwm_kaggle"
    if output.exists():
        shutil.rmtree(output)
    project.mkdir(parents=True)

    for name in (
        "__init__.py",
        "modeling_hlwm.py",
        "data.py",
        "semantic_grading.py",
        "train_kaggle.py",
        "evaluate_checkpoint.py",
        "inference_hlwm.py",
        "test_modeling_hlwm.py",
        "README.md",
    ):
        shutil.copy2(SOURCE / name, project / name)

    requested_counts = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    anchor_counts = {"train": 1024, "validation": 256, "test": 256}
    counts: Dict[str, int] = {}
    reasoning_counts: Dict[str, int] = {}
    anchor_requests: Dict[str, set[str]] = {}
    source_master = ROOT / "data" / "reasoning9000" / "final" / "master"
    for index, (split, requested_count) in enumerate(requested_counts.items()):
        source_path = source_master / (split + ".jsonl")
        destination = project / "data" / "master" / (split + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        anchors = behavior_anchor_rows(split, anchor_counts[split], args.seed)
        validate_behavior_anchors(anchors, split)
        anchor_requests[split] = {
            str((row.get("input") or {}).get("user_request", "")) for row in anchors
        }
        if requested_count is None:
            reasoning_counts[split] = line_count(source_path)
            write_anchors_then_source(destination, anchors, source_path)
        else:
            rows = balanced_rows(source_path, requested_count, args.seed + index)
            reasoning_counts[split] = len(rows)
            write_jsonl(destination, list(anchors) + rows)
        counts[split] = reasoning_counts[split] + anchor_counts[split]

    split_names = sorted(anchor_requests)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = anchor_requests[left] & anchor_requests[right]
            if overlap:
                raise ValueError("behavior anchor prompt leakage between %s and %s" % (left, right))

    quality = data_quality_report(source_master)
    quality["programmatically_verified_behavior_anchors"] = sum(anchor_counts.values())
    quality["behavior_anchor_counts"] = anchor_counts
    quality["behavior_anchor_task_types"] = [
        "numeric",
        "unit",
        "ordering",
        "abstention",
    ]
    quality["safe_abstention_is_public_target"] = True
    quality["policy_calibration_split"] = "generated_validation_emissions_only"
    quality["behavior_anchor_boundary"] = (
        "Deterministic semantic arithmetic, conversion, ordering and missing-evidence checks; "
        "all prompts are split-isolated and task-balanced, but they are not a substitute for "
        "independent human adjudication."
    )
    (project / "data-quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "name": "hlwm-reasoning9000-v5-5-candidate",
        "package_version": "5.5.0",
        "purpose": (
            "on-policy-head-calibrated single-device HLWM architecture evaluation "
            "sized for an A100 MIG 1g.5gb partition inside a 3.5-hour session"
        ),
        "review_status": (
            "unreviewed direct_fast_no_judge synthetic development release; the stronger "
            "50-episode pilot accepted 0 and the scaling gate is false; not authorized as "
            "factual-quality or calibrated-policy evidence"
        ),
        "counts": counts,
        "reasoning9000_counts": reasoning_counts,
        "behavior_anchor_counts": anchor_counts,
        "base_model": "Qwen/Qwen3-0.6B-Base",
        "base_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "seed": args.seed,
        "training_defaults": {
            "overfit_steps": 128,
            "local_steps": 768,
            "joint_steps": 1664,
            "gradient_accumulation": 4,
            "learning_rate": 0.00008,
            "initial_loss_scale": 1024.0,
            "warmup_updates": 32,
            "causal_ratio": 0.40,
            "anchor_ratio": 0.75,
            "unfreeze_tail_layers": 0,
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "lora_dropout": 0.05,
            "lora_tail_layers": 8,
            "precision": "auto",
            "router_entropy_weight": 0.02,
            "context_tokens": 160,
            "canvas_tokens": 96,
            "brief_tokens": 96,
            "causal_tokens": 320,
            "policy_records": 96,
            "policy_new_tokens": 96,
            "policy_epochs": 400,
            "policy_learning_rate": 0.001,
            "calibration_records": 64,
            "calibration_new_tokens": 96,
            "max_runtime_hours": 2.3,
        },
        "session_plan_hours": 3.5,
        "target_hardware": "one A100 MIG 1g.5gb partition (~4.75 GB) or one T4",
        "v5_5_changes": [
            "on-policy policy-head phase on generated train-anchor emissions with "
            "semantic and mechanical hard negatives before validation threshold fitting",
            "router marginal-entropy regularizer against the observed 5.4 routing collapse",
            "preregistered routing gates: normalized entropy, second-route load >= 0.10, "
            "and a least-used-expert intervention on frozen candidates",
            "BF16 autocast with disabled gradient scaling on supporting devices",
            "joint-stage memory preflight with a 92% device headroom limit for 5 GB partitions",
            "single-seed 2,560-microstep session sized to a 3.5-hour wall clock with exact resume",
        ],
    }
    code_hashes = {
        path.name: sha256(path)
        for path in sorted(project.glob("*.py"))
    }
    manifest["code_sha256"] = code_hashes
    (project / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runner = project / "run_v55_a100_5gb.sh"
    runner.write_text(runner_script_v55(), encoding="utf-8")
    runner.chmod(0o755)

    notebook = output / "embel-hlwm-v5.5-a100-mig.ipynb"
    notebook.write_text(
        json.dumps(notebook_document_v55(), indent=1) + "\n", encoding="utf-8"
    )
    notebook_t4 = output / "embel-hlwm-v5.5-kaggle-2xt4.ipynb"
    notebook_t4.write_text(
        json.dumps(notebook_document_v55_t4(), indent=1) + "\n", encoding="utf-8"
    )
    archive = output / "hlwm-v5.5-candidate-bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted((output / "bundle").rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output / "bundle"))

    report = {
        "notebook": str(notebook),
        "notebook_sha256": sha256(notebook),
        "notebook_t4": str(notebook_t4),
        "notebook_t4_sha256": sha256(notebook_t4),
        "bundle": str(archive),
        "bundle_sha256": sha256(archive),
        "bundle_bytes": archive.stat().st_size,
        "counts": counts,
        "data_quality": quality,
        "code_sha256": code_hashes,
    }
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
