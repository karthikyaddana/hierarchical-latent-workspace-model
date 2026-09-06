#!/usr/bin/env python3
"""Build the HLWM Version 9.0 candidate bundle: necessity by construction.

Version 9.0 responds to the Session F result (all repairs worked, all
mechanisms measured null) and its post-session code audit (RC-1: the ablation
never removed the whole prefix and the gate was structurally silent; RC-2: the
harvest generated from a padded surface the audit never uses; RC-3: anchor
difficulty was never a parameter). The design is preregistered in
reports/hlwm-v9.0-plan-2026-09-02.md.

The builder:
  1. regenerates the behavior-anchor rows with difficulty parameters, masked
     (withheld-premise) variants and premise literals (RC-3 + Claim L);
  2. merges them with the non-anchor rows of the Version 8.0 master;
  3. runs the offline unit suite (69 tests);
  4. writes the manifest, the dual-T4 notebook (with the in-session banding
     cell), and a deterministic zip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = ROOT / "artifacts" / "kaggle" / "hlwm-v9.0"
PROJECT = BUNDLE_ROOT / "bundle" / "hlwm_kaggle"
V8_MASTER = ROOT / "artifacts" / "kaggle" / "hlwm-v8.0" / "bundle" / "hlwm_kaggle" / "data" / "master"

PACKAGE_NAME = "hlwm-necessity-v9-0-candidate"
PACKAGE_VERSION = "9.0"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

ANCHOR_COUNTS = {"train": 1024, "validation": 384, "test": 576}

CODE_FILES = (
    "__init__.py",
    "data.py",
    "modeling_hlwm.py",
    "semantic_grading.py",
    "train_kaggle.py",
    "evaluate_checkpoint.py",
    "inference_hlwm.py",
    "test_modeling_hlwm.py",
    "README.md",
)

V9_CHANGES = [
    "M1 information-asymmetric channels: masked anchor rows condition the answer "
    "channel on a withheld-premise prompt while the workspace reads the full "
    "context; the latent prefix is the only path from the operands to the answer. "
    "Leak integrity at four layers (generator, normalization assert, collator "
    "subsequence assert, audit leak gate).",
    "M2 dense latent supervision: a latent-only probe decodes the withheld premise "
    "tokens from the prefix positions (never the gold answer); weight 0.1, "
    "restricted candidate scoring, masked rows only.",
    "M3 attribution control: a gist prefix (mean-pool projection, same 25-token "
    "budget, no canvas/diffusion) trained on alternating masked batches and "
    "audited as the canvas's killer baseline.",
    "M4 harvest/verifier repair (RC-2): one shared unpadded encode primitive for "
    "harvest, calibration and audit; validity floor is now a real gate; harvest "
    "wall-clock guard.",
    "M5 difficulty banding (RC-3): anchor generators parameterized by chain "
    "length, operand digits and distractors; the audit anchors are selected by an "
    "in-session banding pass against a frozen estimator before training starts.",
    "M6 abstention realignment: conformal target coverage 0.40 in band "
    "[0.20, 0.55]; three-way calibration split (weights on A, score form on B1, "
    "threshold on B2); five-feature publish rule nests plurality agreement; "
    "headline is coverage-restricted partial AUGRC in [0.05, 0.50] at >=80% of "
    "paired bootstraps; risk UCB <= 0.15 with an n>=20 feasibility precondition.",
    "RC-1 instrument fixes: full-prefix ablation shipped (the causal arm is the "
    "workspace arm with prefix removed; masked rows recompute it as the floor "
    "arm); the memory-only slice is retained as a descriptive second arm; the "
    "prefix gate is telemetry only and gates nothing.",
    "Single lane (the preregistered unconditional lane exit executed after "
    "failures eight and nine); prefix = 8 synthesis + 16 windows + 1 summary = 25.",
]

TRAINING_DEFAULTS = {
    "steps": "128 overfit + 768 local + 2304 joint = 3200",
    "num_lanes": 1,
    "num_experts": 1,
    "workspace_memory_windows": 16,
    "synthesis_kl_weight": 0.02,
    "premise_aux_weight": 0.1,
    "gist_prefix_tokens": 25,
    "conformal_target_coverage": 0.40,
    "policy_max_attempts": 512,
    "harvest_max_hours": 1.0,
    "candidate_temperatures": "0.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8",
    "audit_rows": "224 banded anchors + 64 SQL = 288",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def stable_key(seed: int, split: str, index: int) -> int:
    return int(
        hashlib.sha256(("%d:%s:%d" % (seed, split, index)).encode("utf-8")).hexdigest()[:14],
        16,
    )


def alphabetic_code(value: int) -> str:
    letters = []
    value += 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def _operand(key: int, digits: int, salt: int) -> int:
    low = 10 ** (digits - 1)
    span = 9 * low
    return low + ((key >> salt) % span)


def _distractor_values(key: int, count: int, forbidden: List[str]) -> List[int]:
    values: List[int] = []
    cursor = 0
    while len(values) < count and cursor < 64:
        candidate = 100 + ((key >> (cursor + 5)) % 90_000)
        text = str(candidate)
        cursor += 1
        # No withheld literal may be a substring of a distractor (or vice
        # versa): the normalization-time leak assert is substring-based.
        if any(text in literal or literal in text for literal in forbidden):
            continue
        if text in [str(v) for v in values]:
            continue
        values.append(candidate)
    return values


def behavior_anchor_rows_v9(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    """Difficulty-parameterized, maskable behavior anchors (Version 9.0).

    Difficulty knobs (RC-3): chain length 1-3, operand digit count 2-6, and
    0-2 irrelevant distractor values in the context. Maskable kinds (numeric,
    unit, ordering) emit a withheld-premise request variant and the withheld
    literals; roughly half of maskable rows are flagged ``masked``.
    """

    rows: List[Dict[str, Any]] = []
    for index in range(count):
        key = stable_key(seed, split, index)
        kind = index % 4
        chain = 1 + (key >> 3) % 3
        digits = 2 + (key >> 21) % 5
        distractor_count = (key >> 27) % 3
        masked = kind != 3 and ((key >> 15) % 2 == 0)
        withheld: List[str] = []
        masked_request = ""
        if kind == 0:
            operands = [_operand(key, digits, 31)]
            operators: List[str] = []
            value = operands[0]
            for step in range(chain):
                op_pick = (key >> (7 + 5 * step)) % 3
                if op_pick == 2:
                    factor = 2 + ((key >> (11 + 5 * step)) % 11)
                    operators.append("*")
                    operands.append(factor)
                    value = value * factor
                elif op_pick == 1 and value > 2:
                    term = 1 + ((key >> (13 + 5 * step)) % max(2, min(value - 1, 10 ** digits)))
                    operators.append("-")
                    operands.append(term)
                    value = value - term
                else:
                    term = _operand(key, max(2, digits - 1), 17 + 5 * step)
                    operators.append("+")
                    operands.append(term)
                    value = value + term
            rendered = str(operands[0])
            masked_rendered = "[?]"
            for operator, operand in zip(operators, operands[1:]):
                symbol = {"+": "+", "-": "-", "*": "x"}[operator]
                rendered += " %s %d" % (symbol, operand)
                masked_rendered += " %s [?]" % symbol
            templates = (
                "Calculate %s and state the exact result.",
                "What is the exact value of %s? Return only a concise answer.",
                "Compute %s and state the checked result.",
            )
            template = templates[(key >> 9) % len(templates)]
            request = template % rendered
            masked_request = template % masked_rendered
            answer = "The result is %d." % value
            negative = "The result is %d." % (value + 1)
            reasoning = "%s equals %d." % (rendered, value)
            answer_spec = {"type": "numeric", "expected": value}
            route = "arithmetic"
            withheld = [str(operand) for operand in operands]
        elif kind == 1:
            hops = {
                1: (("minutes", "seconds", 60), ("hours", "minutes", 60), ("days", "hours", 24)),
                2: (("hours", "seconds", 3600), ("days", "minutes", 1440)),
                3: (("days", "seconds", 86400),),
            }[chain]
            source_unit, target_unit, factor = hops[(key >> 8) % len(hops)]
            quantity = 2 + key % (10 ** min(digits, 4))
            converted = quantity * factor
            templates = (
                "Convert %s %s to %s. Answer without restating the task.",
                "How many %s are in %s %s? Give the exact conversion.",
            )
            template_index = (key >> 6) % len(templates)
            if template_index == 0:
                request = templates[0] % (quantity, source_unit, target_unit)
                masked_request = templates[0] % ("[?]", source_unit, target_unit)
            else:
                request = templates[1] % (target_unit, quantity, source_unit)
                masked_request = templates[1] % (target_unit, "[?]", source_unit)
            answer = "%d %s equals %d %s." % (quantity, source_unit, converted, target_unit)
            negative = "%d %s equals %d %s." % (
                quantity, source_unit, converted + factor, target_unit
            )
            reasoning = "Multiply %d by %d: %d." % (quantity, factor, converted)
            answer_spec = {"type": "unit", "expected": converted, "unit": target_unit}
            route = "unit_conversion"
            withheld = [str(quantity)]
        elif kind == 2:
            length = {1: 4, 2: 6, 3: 8}[chain]
            base = _operand(key, digits, 29)
            values = [base]
            for step in range(length - 1):
                values.append(values[-1] + 1 + ((key >> (9 + 3 * step)) % 13))
            ordered = list(values)
            shuffled = list(ordered)
            for position in range(length - 1, 0, -1):
                swap = (key >> (position * 2 + 1)) % (position + 1)
                shuffled[position], shuffled[swap] = shuffled[swap], shuffled[position]
            if shuffled == ordered:
                shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
            rendered_values = ", ".join(str(value) for value in shuffled)
            masked_values = ", ".join("[?]" for _ in shuffled)
            templates = (
                "Sort these integers in ascending order: %s.",
                "Return only the ascending sequence for these numbers: %s.",
            )
            template = templates[(key >> 6) % len(templates)]
            request = template % rendered_values
            masked_request = template % masked_values
            answer = "Ascending order: %s." % ", ".join(str(value) for value in ordered)
            negative = "Ascending order: %s." % ", ".join(
                str(value) for value in reversed(ordered)
            )
            reasoning = "Comparing the values gives %s." % ", ".join(
                str(value) for value in ordered
            )
            answer_spec = {"type": "ordering", "expected": ordered}
            route = "ordering"
            withheld = [str(value) for value in shuffled]
        else:
            entity = "%s-%s-%04d" % (
                ("sensor", "shipment", "release")[(index // 4) % 3], split[:3], index
            )
            missing_templates = (
                (
                    "What exact battery percentage did %s report? If it is absent, say that it cannot be determined.",
                    "The battery percentage cannot be determined from the provided information.",
                    "The battery percentage is 80%.",
                ),
                (
                    "What was the exact delivery temperature for %s? Do not guess if no measurement is supplied.",
                    "There is insufficient information to determine the delivery temperature.",
                    "The delivery temperature was 4 degrees.",
                ),
                (
                    "Which exact software version was deployed for %s? State when the value was not provided.",
                    "The software version was not provided, so it cannot be determined.",
                    "Software version 3 was deployed.",
                ),
            )
            request_template, answer, negative = missing_templates[(key >> 10) % 3]
            request = request_template % entity
            masked_request = ""
            reasoning = "No such value is present, so any specific value would be a guess."
            answer_spec = {"type": "abstention", "forbid_numbers": True}
            route = "evidence_check"
            withheld = []
            masked = False

        case_suffix = " [case %s-%s]" % (split, alphabetic_code(index))
        request = request.rstrip() + case_suffix
        if masked_request:
            masked_request = masked_request.rstrip() + case_suffix

        context_lines = [
            "This is a deterministic behavior check. Use only facts explicitly supplied."
        ]
        distractors = _distractor_values(key, distractor_count, withheld)
        if distractors:
            context_lines.append(
                "Unrelated reference values seen elsewhere (not relevant to this task): %s."
                % ", ".join(str(value) for value in distractors)
            )

        expected_action = "abstain" if kind == 3 else "answer"
        claim_id = "anchor-claim-%s-%04d" % (split, index)
        episode: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "episode_id": "behavior-anchor-%s-%04d" % (split, index),
            "domain": "behavior-anchor",
            "subdomain": route,
            "difficulty": "banded",
            "difficulty_params": {
                "kind": ("numeric", "unit", "ordering", "abstention")[kind],
                "chain": chain,
                "digits": digits,
                "distractors": distractor_count,
            },
            "source_group": "programmatic-v9.0",
            "input": {
                "user_request": request,
                "context": context_lines,
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
                        {"claim_id": claim_id, "statement": answer, "evidence_refs": ["derivation"]}
                    ],
                    "checkpoints": [
                        {"step": 1, "observable_update": reasoning, "decision": "halt"}
                    ],
                    "route_windows": [{"decision": "halt"}],
                    "summary": answer,
                }
            ],
            "verification": [{"claim_id": claim_id, "verdict": "supported"}],
            "barrier": {"open_claims": []},
            "integration": {"published_answer": answer},
            "commitment": {"decision": "publish"},
            "evaluation": {
                "answer_spec": answer_spec,
                "expected_action": expected_action,
                "expected_commit": True,
                "negative_answer": negative,
                "withheld_literals": withheld,
                "masked": bool(masked),
            },
            "generation_metadata": {"programmatically_verified": True},
        }
        if masked_request:
            episode["input"]["user_request_masked"] = masked_request
        rows.append(episode)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rebuild_master(seed: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source = V8_MASTER / ("%s.jsonl" % split)
        non_anchor: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("domain")) != "behavior-anchor":
                    non_anchor.append(row)
        anchors = behavior_anchor_rows_v9(split, ANCHOR_COUNTS[split], seed)
        rows = anchors + non_anchor
        write_jsonl(PROJECT / "data" / "master" / ("%s.jsonl" % split), rows)
        counts[split] = len(rows)

        # Build-time self-checks: masked fraction, leak absence, and premise
        # presence, done on the raw rows (the string-level layer of the
        # four-layer leak battery; normalization re-asserts at load time and
        # the collator re-asserts at token level).
        maskable = [row for row in anchors if row["evaluation"]["withheld_literals"]]
        flagged = [row for row in maskable if row["evaluation"]["masked"]]
        for row in flagged:
            masked_request = row["input"]["user_request_masked"]
            for literal in row["evaluation"]["withheld_literals"]:
                if literal in masked_request:
                    raise SystemExit(
                        "build-time leak: %r in masked request of %s"
                        % (literal, row["episode_id"])
                    )
        kinds = Counter(row["difficulty_params"]["kind"] for row in anchors)
        print(
            json.dumps(
                {
                    "split": split,
                    "rows": len(rows),
                    "anchors": len(anchors),
                    "maskable": len(maskable),
                    "masked": len(flagged),
                    "kinds": dict(kinds),
                }
            )
        )
    return counts


# --------------------------------------------------------------------------
# Notebook


def code(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def notebook_document(expected_counts: Dict[str, int]) -> Dict[str, Any]:
    train_flags = """
        '--data-dir', str(DATA), '--output-dir', str(output), '--seed', str(seed),
        '--skip-real-qwen-preflight',
        '--overfit-steps', '128', '--local-steps', '768', '--joint-steps', '2304',
        '--overfit-examples', '64', '--overfit-min-improvement', '0.05',
        '--batch-size', '1', '--gradient-accumulation', '4', '--learning-rate', '0.00008', '--warmup-updates', '32',
        '--causal-ratio', '0.40', '--anchor-ratio', '0.50', '--unfreeze-tail-layers', '0',
        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',
        '--precision', 'auto', '--router-entropy-weight', '0.0',
        '--router-aux-weight', '0.0', '--expert-diversity-weight', '0.0',
        '--expert-init-scale', '0.01', '--workspace-memory-windows', '16',
        '--synthesis-kl-weight', '0.02', '--prefix-gate-init', '0.05',
        '--premise-aux-weight', '0.1', '--gist-prefix-tokens', '25',
        '--initial-loss-scale', '1024', '--max-skipped-updates', '2',
        '--num-lanes', '1', '--num-experts', '1', '--refinement-steps', '2', '--diffusion-steps', '4',
        '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',
        '--max-validation', '256', '--calibration-records', '256', '--calibration-new-tokens', '96',
        '--conformal-target-coverage', '0.40',
        '--policy-records', '128', '--policy-new-tokens', '96', '--policy-epochs', '120',
        '--policy-label-smoothing', '0.05', '--policy-learning-rate', '0.001',
        '--policy-min-valid-per-family', '16', '--policy-max-attempts', '512',
        '--harvest-max-hours', '1.0',
        '--candidate-temperatures', '0.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8',
        '--canary-anchors', '8', '--canary-new-tokens', '96',
        '--num-workers', '2', '--eval-every', '512', '--save-every', '1024',
        '--max-runtime-hours', '6.5'"""

    cells = [
        markdown(
            "# Embel HLWM Version 9.0 - necessity by construction\n\n"
            "Two independent seeds, one per T4, single lane, workspace-only. Version 9.0 exists\n"
            "because Session F (v8.0) proved every repair and falsified every mechanism: the\n"
            "latent read-out was REDUNDANT BY CONSTRUCTION - both channels saw the full context,\n"
            "so the model could minimize loss while ignoring the workspace, and it did.\n\n"
            "Version 9.0 makes ignoring the latent COST loss:\n\n"
            "- **Claim L (latent channel)** - on masked rows the answer channel sees a\n"
            "  withheld-premise prompt; only the workspace sees the operands. The masked floor\n"
            "  (prefix removed) must crater (<=0.10) and the workspace arm must recover >=0.30\n"
            "  over it, on both seeds. A gist control (same budget, no canvas) guards attribution.\n"
            "- **Claim V (verifier integrity)** - on-policy verifier margin must stay positive on\n"
            "  generated audit emissions; the publish score must be continuous (>=100 distinct).\n"
            "- **Claim A2 (calibrated abstention)** - split-conformal coverage 0.40 in [0.20,0.55];\n"
            "  headline = coverage-restricted partial AUGRC in [0.05,0.50] vs mean-logprob,\n"
            "  >=80% of 1,000 paired bootstraps; risk UCB <= 0.15 with an n>=20 precondition.\n\n"
            "Difficulty banding runs IN-SESSION against a frozen estimator BEFORE training; the\n"
            "audit consumes the frozen anchor ids. The preregistration is\n"
            "`reports/hlwm-v9.0-plan-2026-09-02.md` (with build-time amendments in section 8);\n"
            "the binding ladder is the final cell."
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8')\n"
            "import platform, subprocess, sys, time\n"
            "SESSION_START = time.time()\n"
            "print(platform.platform())\n"
            "subprocess.run(['nvidia-smi'], check=True)\n"
            "print('python', sys.version)"
        ),
        code(
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
            "print('transformers', transformers.__version__, '| user site', user_site)"
        ),
        code(
            "from pathlib import Path\n"
            "import hashlib, json, shutil, zipfile\n"
            "KAGGLE = Path('/kaggle/input').exists()\n"
            "WORK = Path('/kaggle/working/hlwm-v9.0') if KAGGLE else Path.cwd() / 'hlwm-v9.0-work'\n"
            "if WORK.exists(): shutil.rmtree(WORK)\n"
            "search_roots = [Path('/kaggle/input')] if KAGGLE else [Path.cwd()]\n"
            "archives = [p for root in search_roots for p in root.rglob('hlwm-v9.0*candidate-bundle.zip')]\n"
            "archives += [p for root in search_roots for p in root.rglob('hlwm-v90*candidate-bundle.zip')]\n"
            "if archives:\n"
            "    with zipfile.ZipFile(archives[0]) as bundle: bundle.extractall(WORK)\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "    print('bundle source:', archives[0])\n"
            "else:\n"
            "    # Kaggle auto-extracts uploaded zips into the dataset tree, so the\n"
            "    # archive itself may not exist: locate the extracted bundle by its\n"
            "    # manifest instead.\n"
            f"    wanted = '{PACKAGE_NAME}'\n"
            "    manifests = []\n"
            "    for root in search_roots:\n"
            "        for path in root.rglob('manifest.json'):\n"
            "            try:\n"
            "                if json.loads(path.read_text()).get('name') == wanted: manifests.append(path)\n"
            "            except (OSError, json.JSONDecodeError): pass\n"
            "    if not manifests:\n"
            "        for root in search_roots:\n"
            "            print('ATTACHED under', root, ':')\n"
            "            for path in sorted(root.glob('*')):\n"
            "                print('  ', path)\n"
            "                for child in sorted(path.glob('*'))[:12]: print('     ', child.name)\n"
            "        raise FileNotFoundError('Attach the HLWM Version 9.0 candidate bundle (zip or extracted tree) first.')\n"
            "    shutil.copytree(manifests[0].parent, WORK/'hlwm_kaggle')\n"
            "    PROJECT = WORK / 'hlwm_kaggle'\n"
            "    print('bundle source (extracted tree):', manifests[0].parent)\n"
            "DATA = PROJECT / 'data'\n"
            "manifest = json.loads((PROJECT/'manifest.json').read_text())\n"
            f"assert manifest['package_version'] == '{PACKAGE_VERSION}', manifest['package_version']\n"
            f"assert manifest['name'] == '{PACKAGE_NAME}', manifest['name']\n"
            f"assert manifest['counts'] == {json.dumps(expected_counts)}, manifest['counts']\n"
            "def file_sha256(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with open(path, 'rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1 << 20), b''): digest.update(chunk)\n"
            "    return digest.hexdigest()\n"
            "# Verify the shipped code BEFORE committing GPU hours (v6.0 shipped stale shas).\n"
            "for name, expected in manifest['code_sha256'].items():\n"
            "    actual = file_sha256(PROJECT / name)\n"
            "    assert actual == expected, f'code sha mismatch for {name}'\n"
            "print('bundle verified: all', len(manifest['code_sha256']), 'code shas match at', PROJECT)"
        ),
        code(
            "import importlib.util, py_compile, subprocess, sys, torch\n"
            "assert torch.cuda.device_count() == 2, f'expected two T4 GPUs, found {torch.cuda.device_count()} (select GPU T4 x2)'\n"
            "for index in range(2):\n"
            "    properties = torch.cuda.get_device_properties(index)\n"
            "    print(index, properties.name, round(properties.total_memory/2**30, 2), 'GB')\n"
            "for name in ('modeling_hlwm.py','data.py','semantic_grading.py','train_kaggle.py','evaluate_checkpoint.py','inference_hlwm.py','test_modeling_hlwm.py'):\n"
            "    py_compile.compile(str(PROJECT/name), doraise=True)\n"
            "result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT/'test_modeling_hlwm.py')],\n"
            "                        cwd=PROJECT, capture_output=True, text=True)\n"
            "print((result.stdout or '')[-3000:])\n"
            "if result.returncode != 0:\n"
            "    print((result.stderr or '')[-3000:])\n"
            "    raise RuntimeError('Version 9.0 tests failed')\n"
            "print('Version 9.0 masked-channel, premise-probe, gist, conformal and gate tests passed')"
        ),
        code(
            "os.environ.setdefault('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))\n"
            "Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)\n"
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id=manifest['base_model'], revision=manifest['base_revision'])\n"
            "print('Pinned Qwen snapshot cached before the dual launch')"
        ),
        markdown(
            "## Tokenizer assertions\n\n"
            "The v6.0 root cause (no BOS; `bos_token_id` aliases to end-of-text) is asserted\n"
            "directly, plus the Version 9.0 mask-marker sanity checks."
        ),
        code(
            "from transformers import AutoTokenizer\n"
            "_tokenizer = AutoTokenizer.from_pretrained(manifest['base_model'], revision=manifest['base_revision'], trust_remote_code=False)\n"
            "print('bos_token', _tokenizer.bos_token, '| bos_token_id', _tokenizer.bos_token_id)\n"
            "assert _tokenizer.bos_token_id in (None, _tokenizer.eos_token_id), (\n"
            "    'base revision now defines a distinct BOS; revisit the no-seed decision')\n"
            "sys.path.insert(0, str(PROJECT))\n"
            "from data import RESPONSE_CUE_TEXT, build_public_prompt, encode_preserving_ends\n"
            "cue_ids = _tokenizer.encode(RESPONSE_CUE_TEXT, add_special_tokens=False)\n"
            "print('response cue', repr(RESPONSE_CUE_TEXT), '->', cue_ids)\n"
            "_probe_prompt = build_public_prompt({'input': {'user_request': 'ping'}})\n"
            "assert RESPONSE_CUE_TEXT not in _probe_prompt, 'cue is back in the prompt'\n"
            "# Masked-row spot check straight from the shipped test split.\n"
            "from data import Reasoning9000Dataset\n"
            "_test = Reasoning9000Dataset(DATA/'master'/'test.jsonl', num_lanes=1)\n"
            "_masked = [row for row in _test.rows if row.get('masked')]\n"
            "print('test rows', len(_test.rows), '| masked anchors', len(_masked))\n"
            "assert len(_masked) >= 150, 'expected a large masked stratum'\n"
            "for row in _masked[:64]:\n"
            "    for literal in row['withheld_literals']:\n"
            "        assert literal not in row['masked_prompt'], row['episode_id']\n"
            "print('masked-prompt leak spot-check passed (normalization already asserts every row)')"
        ),
        markdown(
            "## Real-Qwen numerical and memory preflight (GPU 0)\n\n"
            "Local and joint forward/backward at production lengths with the full vocabulary,\n"
            "including the masked answer channel, the premise probe and the gist arm."
        ),
        code(
            "import os, subprocess, sys, shutil\n"
            "PREFLIGHT_OUTPUT = WORK / 'preflight'\n"
            "if PREFLIGHT_OUTPUT.exists(): shutil.rmtree(PREFLIGHT_OUTPUT)\n"
            "preflight_command = [sys.executable, str(PROJECT/'train_kaggle.py'),\n"
            "    '--data-dir', str(DATA), '--output-dir', str(PREFLIGHT_OUTPUT), '--seed', '17', '--preflight-only',\n"
            "    '--batch-size', '1', '--gradient-accumulation', '4', '--unfreeze-tail-layers', '0',\n"
            "    '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--lora-tail-layers', '8',\n"
            "    '--num-lanes', '1', '--num-experts', '1', '--refinement-steps', '2', '--diffusion-steps', '4',\n"
            "    '--precision', 'auto', '--router-entropy-weight', '0.0',\n"
            "    '--router-aux-weight', '0.0', '--expert-diversity-weight', '0.0',\n"
            "    '--expert-init-scale', '0.01', '--workspace-memory-windows', '16',\n"
            "    '--synthesis-kl-weight', '0.02', '--prefix-gate-init', '0.05',\n"
            "    '--premise-aux-weight', '0.1', '--gist-prefix-tokens', '25',\n"
            "    '--candidate-temperatures', '0.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8',\n"
            "    '--context-tokens', '256', '--canvas-tokens', '128', '--brief-tokens', '96', '--causal-tokens', '1024',\n"
            "    '--max-validation', '1', '--num-workers', '0']\n"
            "preflight_environment = os.environ.copy(); preflight_environment['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "subprocess.run(preflight_command, cwd=PROJECT, env=preflight_environment, check=True)\n"
            "print('Local+joint real-Qwen preflight and memory headroom check passed on GPU 0')"
        ),
        markdown(
            "## In-session difficulty banding (frozen estimator, before training)\n\n"
            "RC-3: v8.0's audit difficulty was bimodal because anchor difficulty was never a\n"
            "parameter. The 576-anchor test pool is banded by pass@8 under a FROZEN estimator\n"
            "before any v9.0 training step: the attached Version 8.0 seed-17 checkpoint when\n"
            "present, otherwise the pinned base model (disclosed in the strata file). Selection\n"
            "rule (frozen): all medium rows up to 168, then hard, then easy fill, total 224.\n"
            "The audit consumes exactly these ids; the strata cannot drift with the new model."
        ),
        code(
            "import gc, torch\n"
            "from evaluate_checkpoint import encode_preserving_ends as audit_encode, difficulty_bin, load_hlwm\n"
            "from semantic_grading import grade_semantic_answer\n"
            "BAND_DEVICE = torch.device('cuda:0')\n"
            "anchor_rows = [row for row in _test.rows if row['is_behavior_anchor']]\n"
            "print('banding pool:', len(anchor_rows))\n"
            "v8_checkpoints = [p for root in search_roots for p in root.rglob('checkpoint-step-004224.pt')]\n"
            "v8_checkpoints += [p for root in search_roots for p in root.rglob('hlwm-v8.0-seed-17-resumable.pt')]\n"
            "estimator_name = 'pinned-base'\n"
            "banding_model = None\n"
            "if v8_checkpoints:\n"
            "    try:\n"
            "        payload = torch.load(v8_checkpoints[0], map_location='cpu', weights_only=False)\n"
            "        banding_model = load_hlwm(payload, BAND_DEVICE)\n"
            "        estimator_name = 'v8.0-checkpoint:' + v8_checkpoints[0].name\n"
            "    except Exception as error:\n"
            "        print('checkpoint estimator unavailable, falling back to base:', error)\n"
            "if banding_model is None:\n"
            "    from transformers import AutoModelForCausalLM\n"
            "    base = AutoModelForCausalLM.from_pretrained(manifest['base_model'], revision=manifest['base_revision'],\n"
            "        dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=False).to(BAND_DEVICE).eval()\n"
            "cue_tensor = torch.tensor([cue_ids], dtype=torch.long, device=BAND_DEVICE)\n"
            "strata, band_started = {}, time.time()\n"
            "with torch.no_grad():\n"
            "    for position, row in enumerate(anchor_rows):\n"
            "        encoded = audit_encode(_tokenizer, str(row['public_prompt']), 256, BAND_DEVICE)\n"
            "        flags = []\n"
            "        if banding_model is not None:\n"
            "            generator = torch.Generator(device=BAND_DEVICE).manual_seed(9000 + position)\n"
            "            with torch.autocast(device_type='cuda', dtype=torch.float16):\n"
            "                for arm in range(8):\n"
            "                    ids = banding_model._decode_candidate(encoded['input_ids'], encoded['attention_mask'], None,\n"
            "                        max_new_tokens=96, temperature=0.0 if arm == 0 else 0.8, generator=generator)\n"
            "                    text = _tokenizer.decode(ids[0].cpu(), skip_special_tokens=True).strip()\n"
            "                    flags.append(bool(grade_semantic_answer(text, row['answer_spec'])['correct']))\n"
            "        else:\n"
            "            cued = torch.cat((encoded['input_ids'], cue_tensor), dim=1)\n"
            "            mask = torch.ones_like(cued)\n"
            "            greedy = base.generate(input_ids=cued, attention_mask=mask, max_new_tokens=96,\n"
            "                do_sample=False, pad_token_id=_tokenizer.pad_token_id, eos_token_id=_tokenizer.eos_token_id)\n"
            "            sampled = base.generate(input_ids=cued, attention_mask=mask, max_new_tokens=96,\n"
            "                do_sample=True, temperature=0.8, num_return_sequences=7,\n"
            "                pad_token_id=_tokenizer.pad_token_id, eos_token_id=_tokenizer.eos_token_id)\n"
            "            outputs = [greedy[0]] + list(sampled)\n"
            "            for generated in outputs:\n"
            "                text = _tokenizer.decode(generated[cued.shape[1]:], skip_special_tokens=True).strip()\n"
            "                flags.append(bool(grade_semantic_answer(text, row['answer_spec'])['correct']))\n"
            "        pass_rate = sum(flags) / len(flags)\n"
            "        strata[row['episode_id']] = {'pass8': pass_rate, 'bin': difficulty_bin(pass_rate), 'masked': bool(row.get('masked'))}\n"
            "        if (position + 1) % 64 == 0:\n"
            "            print(json.dumps({'banded': position + 1, 'elapsed_s': round(time.time() - band_started, 1)}), flush=True)\n"
            "if banding_model is not None: del banding_model\n"
            "else: del base\n"
            "gc.collect(); torch.cuda.empty_cache()\n"
            "by_bin = {'medium': [], 'hard': [], 'easy': []}\n"
            "for episode_id, entry in strata.items(): by_bin[entry['bin']].append(episode_id)\n"
            "selected = by_bin['medium'][:168]\n"
            "selected += by_bin['hard'][: max(0, 224 - len(selected))]\n"
            "selected += by_bin['easy'][: max(0, 224 - len(selected))]\n"
            "banding_report = {'estimator': estimator_name, 'pool': len(anchor_rows),\n"
            "    'bins': {name: len(ids) for name, ids in by_bin.items()},\n"
            "    'selected': len(selected), 'selected_medium': sum(1 for i in selected if strata[i]['bin'] == 'medium'),\n"
            "    'medium_floor_met': sum(1 for i in selected if strata[i]['bin'] == 'medium') >= 120,\n"
            "    'elapsed_s': round(time.time() - band_started, 1)}\n"
            "print(json.dumps(banding_report, indent=1))\n"
            "Path('/kaggle/working/hlwm-v9.0-banded-anchor-ids.json').write_text(json.dumps(selected))\n"
            "Path('/kaggle/working/hlwm-v9.0-strata.json').write_text(json.dumps({'report': banding_report, 'strata': strata}, indent=1))"
        ),
        markdown(
            "## Concurrent two-seed Version 9.0 training\n\n"
            "3,200 microsteps per seed (128 overfit + 768 local + 2,304 joint). Masked anchor\n"
            "rows train the information-asymmetric channel with pure-noise full-reverse\n"
            "canvases, the premise probe, and no KL anchor; gist batches alternate on masked\n"
            "rows so the attribution control is equally trained. Harvest and calibration run\n"
            "on the unpadded audit surface (RC-2) with a 512-attempt cap, a 1h wall-clock\n"
            "guard, and the 16/family validity floor now a real gate. Conformal target 0.40,\n"
            "three-way split 96/64/96."
        ),
        code(
            "import os, re, subprocess, sys, time, shutil\n"
            "SEEDS = [int(part) for part in os.environ.get('HLWM_SEEDS', '17,29').split(',')]\n"
            "assert len(SEEDS) == 2, 'this notebook schedules exactly two seeds, one per T4'\n"
            "OUTPUTS = {seed: Path(f'/kaggle/working/hlwm-v9.0-seed-{seed}') for seed in SEEDS}\n"
            "processes, handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    if output.exists(): shutil.rmtree(output)\n"
            "    resume = [p for root in search_roots + [WORK] for p in root.rglob(f'hlwm-v9.0-seed-{seed}-resumable.pt')]\n"
            "    if not resume:\n"
            "        # A run that crashed after training but before packaging (rung 0:\n"
            "        # infrastructure) leaves raw checkpoint-*.pt files instead of the\n"
            "        # renamed -resumable.pt; resume from the highest step among them.\n"
            "        raw = [p for root in search_roots for p in root.rglob(f'hlwm-v9.0-seed-{seed}/checkpoint-*.pt')]\n"
            "        resume = sorted(raw, key=lambda p: int(re.findall(r'(\\d+)', p.stem)[-1]))[-1:]\n"
            "    command = [sys.executable, str(PROJECT/'train_kaggle.py')," + train_flags + "]\n"
            "    if resume:\n"
            "        command += ['--resume', str(resume[0])]\n"
            "        print('seed', seed, 'resuming from', resume[0])\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = Path(f'/kaggle/working/hlwm-v9.0-seed-{seed}-training.log').open('w'); handles[seed] = handle\n"
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
            "    for seed in failures: print(Path(f'/kaggle/working/hlwm-v9.0-seed-{seed}-training.log').read_text()[-8000:])\n"
            "    raise RuntimeError(f'dual training failed: {failures}')\n"
            "print('Both seeds finished')"
        ),
        code(
            "summaries = {seed: json.loads((output/'summary.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "RUN_COMPLETE, ZERO_SKIPPED = {}, {}\n"
            "for seed, summary in summaries.items():\n"
            "    assert summary.get('status') in ('stable_prototype_training_complete', 'time_budget_checkpoint_saved'), summary.get('status')\n"
            "    RUN_COMPLETE[seed] = summary.get('status') == 'stable_prototype_training_complete'\n"
            "    ZERO_SKIPPED[seed] = summary.get('skipped_optimizer_updates') == 0\n"
            "    assert summary.get('planned_steps') == 3200, summary.get('planned_steps')\n"
            "    assert (summary.get('overfit_gate') or {}).get('passed'), summary.get('overfit_gate')\n"
            "    head = summary.get('on_policy_policy_head') or {}\n"
            "    assert head.get('trained'), head\n"
            "    print(seed, 'validity_floor_met', head.get('validity_floor_met'),\n"
            "          'family_valid', head.get('family_valid_positives'),\n"
            "          'harvest_capped', head.get('wall_clock_capped'), 'unpadded', head.get('unpadded_prompt_surface'))\n"
            "    calibration = summary.get('commitment_calibration') or {}\n"
            "    assert calibration.get('method') == 'validation_generated_nbest_conformal_publish_v5', calibration.get('method')\n"
            "    assert calibration.get('test_split_used') is False\n"
            "    conformal = calibration.get('conformal') or {}\n"
            "    assert conformal.get('fitted'), conformal\n"
            "    splits = calibration.get('split_sizes') or {}\n"
            "    assert splits.get('holdout_anchors', 0) >= 16 and splits.get('form_anchors', 0) >= 8, splits\n"
            "    print(seed, 'conformal', json.dumps(conformal, sort_keys=True))\n"
            "    print(seed, 'form_selection', json.dumps(calibration.get('form_selection') or {}, sort_keys=True))\n"
            "    print(seed, 'prefix_gate(telemetry)', json.dumps(summary.get('prefix_gate') or {}))"
        ),
        markdown(
            "## Fresh reload and 288-row banded audit per seed\n\n"
            "224 banded anchors (frozen ids from the banding cell) + 64 text-to-SQL rows.\n"
            "Masked rows run the Claim L arms: masked workspace, masked floor (full prefix\n"
            "removed - the fixed RC-1 instrument), and the gist control. Selection statistics\n"
            "are anchors-only; SQL stays in the risk population. 3h wall-clock guard drops SQL\n"
            "rows first, never masked anchors."
        ),
        code(
            "evaluation_processes, evaluation_handles = {}, {}\n"
            "for gpu, seed in enumerate(SEEDS):\n"
            "    output = OUTPUTS[seed]\n"
            "    command = [sys.executable, str(PROJECT/'evaluate_checkpoint.py'),\n"
            "        '--checkpoint', summaries[seed]['checkpoint'], '--data-dir', str(DATA), '--output-dir', str(output),\n"
            "        '--samples', '288', '--context-tokens', '256', '--canvas-tokens', '128',\n"
            "        '--max-new-tokens', '288', '--candidate-temperatures', '0.0,0.8,0.8,0.8,0.8,0.8,0.8,0.8',\n"
            "        '--anchor-ids-file', '/kaggle/working/hlwm-v9.0-banded-anchor-ids.json',\n"
            "        '--max-sql-rows', '64', '--max-audit-hours', '3.0',\n"
            "        '--seed', str(seed + 1000)]\n"
            "    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
            "    handle = (output/'evaluation.log').open('w'); evaluation_handles[seed] = handle\n"
            "    evaluation_processes[seed] = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=handle, stderr=subprocess.STDOUT)\n"
            "for seed, process in evaluation_processes.items():\n"
            "    if process.wait() != 0:\n"
            "        for handle in evaluation_handles.values(): handle.close()\n"
            "        print((OUTPUTS[seed]/'evaluation.log').read_text()[-5000:])\n"
            "        raise RuntimeError(f'evaluation failed for seed {seed}')\n"
            "for handle in evaluation_handles.values(): handle.close()\n"
            "print('Both fresh-process evaluations finished')"
        ),
        code(
            "# Post-science cell: every field access uses .get so a schema drift can\n"
            "# never void the session after the audits are on disk (Session F lesson).\n"
            "evaluations = {seed: json.loads((output/'checkpoint-evaluation.json').read_text()) for seed, output in OUTPUTS.items()}\n"
            "gates = {}\n"
            "for seed, evaluation in evaluations.items():\n"
            "    aggregate, records = evaluation.get('aggregate') or {}, evaluation.get('records') or []\n"
            "    gate = dict(evaluation.get('gates') or {})\n"
            "    gate['training_complete'] = bool(RUN_COMPLETE.get(seed))\n"
            "    gate['zero_skipped_updates'] = bool(ZERO_SKIPPED.get(seed))\n"
            "    gate['passed'] = all(bool(value) for key, value in gate.items() if key != 'passed')\n"
            "    gates[seed] = gate\n"
            "    (OUTPUTS[seed]/'v9.0-capability-gate.json').write_text(json.dumps(gate, indent=2)+'\\n')\n"
            "    print('SEED', seed, '| audited rows', len(records))\n"
            "    print(json.dumps({k: v for k, v in aggregate.items() if k not in ('domain_metrics',)}, indent=2)[:9000])\n"
            "    print(json.dumps(gate, indent=2))\n"
            "    print('FAILED GATES:', [k for k, v in gate.items() if k != 'passed' and not v])\n"
            "    print('PER-DOMAIN (report-only):'); print(json.dumps(aggregate.get('domain_metrics') or {}, indent=2))\n"
            "    for row in records[:2]:\n"
            "        print('\\n', row.get('episode_id'), 'masked=', row.get('masked'), 'committed=', row.get('committed'))\n"
            "        print('OUTPUT:', str(row.get('hlwm_published', ''))[:400])"
        ),
        code(
            "# Replication verdict: per-seed claim blocks plus the binding-rung readout.\n"
            "CLAIM_L = ('masked_leak_zero','masked_causal_floor','latent_channel_live','full_ablation_consistent','unmasked_parity','aux_premise_decodable')\n"
            "ATTRIB = ('gist_control_run','canvas_beats_gist')\n"
            "CLAIM_V = ('verifier_margin_positive_on_generated','clean_commit_ranked_above_corrupt','publish_score_continuous')\n"
            "CLAIM_A2 = ('coverage_in_band','risk_ucb_015','abstention_beats_logprob_partial_augrc','safe_abstention_probes')\n"
            "INFRA = ('training_complete','zero_skipped_updates','truncation_integrity','validity_floor_met','calibration_fitted','canary_clean')\n"
            "def block(seed, keys): return {key: bool(gates[seed].get(key)) for key in keys}\n"
            "verdict = {'package_version': '9.0', 'seeds': {}, 'binding_rung': None}\n"
            "for seed in SEEDS:\n"
            "    verdict['seeds'][str(seed)] = {\n"
            "        'passed': bool(gates[seed].get('passed')),\n"
            "        'failed_gates': [k for k, v in gates[seed].items() if k != 'passed' and not v],\n"
            "        'infra': block(seed, INFRA), 'claim_L': block(seed, CLAIM_L),\n"
            "        'attribution': block(seed, ATTRIB), 'claim_V': block(seed, CLAIM_V), 'claim_A2': block(seed, CLAIM_A2),\n"
            "        'masked_channel': (evaluations[seed].get('aggregate') or {}).get('masked_channel'),\n"
            "        'abstention': (evaluations[seed].get('aggregate') or {}).get('abstention'),\n"
            "        'gate_notes': (evaluations[seed].get('aggregate') or {}).get('gate_notes'),\n"
            "    }\n"
            "leak_ok = all(verdict['seeds'][str(s)]['claim_L']['masked_leak_zero'] and verdict['seeds'][str(s)]['infra']['truncation_integrity'] for s in SEEDS)\n"
            "live = [bool(verdict['seeds'][str(s)]['claim_L']['latent_channel_live'] and verdict['seeds'][str(s)]['claim_L']['masked_causal_floor']) for s in SEEDS]\n"
            "gist_beaten = [bool(verdict['seeds'][str(s)]['attribution']['canvas_beats_gist']) for s in SEEDS]\n"
            "a2 = [all(verdict['seeds'][str(s)]['claim_A2'].values()) for s in SEEDS]\n"
            "if not leak_ok:\n"
            "    verdict['binding_rung'] = 'rung 0: leak/integrity failure - infrastructure failure, no mechanism verdict; fix and rerun authorized'\n"
            "elif not any(live):\n"
            "    verdict['binding_rung'] = 'rung 1: latent_channel_live failed on both seeds - the latent channel cannot carry structurally necessary information at 0.6B; TERMINAL for the workspace generative mechanism'\n"
            "elif not all(live):\n"
            "    verdict['binding_rung'] = 'rung 2: Claim L supported on one seed - not replicated; one preregistered rerun of the failing seed is authorized'\n"
            "elif not all(gist_beaten):\n"
            "    verdict['binding_rung'] = 'rung 3: canvas does not beat gist - canvas retired; claim re-scopes to the latent memory channel at matched budget'\n"
            "elif not all(a2):\n"
            "    verdict['binding_rung'] = 'rung 4: Claim L passed, A2 failed - abstention re-scopes to coverage-control-without-superiority'\n"
            "else:\n"
            "    verdict['binding_rung'] = 'rung 5: FULL PASS - the matched plain-LoRA control (full-context rows only) is authorized IMMEDIATELY, then benchmarks'\n"
            "print(json.dumps(verdict, indent=1)[:6000])\n"
            "Path('/kaggle/working/hlwm-v9.0-replication-verdict.json').write_text(json.dumps(verdict, indent=2)+'\\n')"
        ),
        code(
            "ARTIFACTS = [Path('/kaggle/working/hlwm-v9.0-replication-verdict.json'),\n"
            "             Path('/kaggle/working/hlwm-v9.0-banded-anchor-ids.json'),\n"
            "             Path('/kaggle/working/hlwm-v9.0-strata.json')]\n"
            "for seed in SEEDS:\n"
            "    output = OUTPUTS[seed]; summary = summaries[seed]\n"
            "    staging = Path(f'/kaggle/working/hlwm-v9.0-seed-{seed}-deliverable')\n"
            "    if staging.exists(): shutil.rmtree(staging)\n"
            "    staging.mkdir()\n"
            "    for source in [Path((summary.get('adapter') or {}).get('path', '')), output/'summary.json',\n"
            "                   output/'commitment-calibration.json', output/'policy-head-training.json',\n"
            "                   output/'checkpoint-evaluation.json', output/'v9.0-capability-gate.json', output/'metrics.jsonl',\n"
            "                   PROJECT/'manifest.json', PROJECT/'README.md']:\n"
            "        if source and Path(source).exists(): shutil.copy2(source, staging/Path(source).name)\n"
            "    archive = Path(shutil.make_archive(f'/kaggle/working/hlwm-v9.0-seed-{seed}-deliverable', 'zip', staging))\n"
            "    resumable = Path(f'/kaggle/working/hlwm-v9.0-seed-{seed}-resumable.pt')\n"
            "    if resumable.exists(): resumable.unlink()\n"
            "    source_checkpoint = Path(summary.get('checkpoint', ''))\n"
            "    if source_checkpoint.exists(): shutil.move(str(source_checkpoint), resumable)\n"
            "    for stale in output.glob('checkpoint-*.pt*'):\n"
            "        if stale.is_file(): stale.unlink()\n"
            "    ARTIFACTS.append(archive); ARTIFACTS.append(resumable)\n"
            "print('deliverables:'); [print(' ', p) for p in ARTIFACTS if Path(p).exists()]"
        ),
        markdown(
            "## Binding ladder (preregistered; read the rung off the verdict, do not improvise)\n\n"
            "0. **Leak/integrity gates fail** - infrastructure failure; no mechanism verdict;\n"
            "   fix and rerun authorized (does not consume the mechanism's last chance).\n"
            "1. **latent_channel_live fails on both seeds** (leak clean, floor confirmed) - the\n"
            "   latent channel cannot carry structurally necessary information through the\n"
            "   frozen LM at 0.6B. TERMINAL for the workspace generative mechanism.\n"
            "2. **L passes one seed only** - supported-not-replicated; ONE preregistered rerun\n"
            "   of the failing seed is authorized.\n"
            "3. **L passes both, canvas_beats_gist fails** - the canvas is retired; the claim\n"
            "   re-scopes to a latent memory channel at matched budget (compression suffices).\n"
            "4. **A2 headline fails both seeds with V passing** - abstention re-scopes\n"
            "   permanently to coverage-control-without-superiority.\n"
            "5. **L core + A2 pass on both seeds** - the matched plain-LoRA control runs\n"
            "   IMMEDIATELY (full-context rows only; masked rows would rig it), then the\n"
            "   preregistered external benchmarks.\n\n"
            "A pass is a research gate, not deployment. The masked-row setting is a constructed\n"
            "mechanism demonstration and is reported as such."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PROJECT.exists():
        raise SystemExit(f"v9.0 bundle sources not found at {PROJECT}")
    missing = [name for name in CODE_FILES if not (PROJECT / name).exists()]
    if missing:
        raise SystemExit(f"bundle is missing code files: {missing}")
    if not V8_MASTER.exists():
        raise SystemExit(f"v8.0 master data not found at {V8_MASTER}")

    counts = rebuild_master(args.seed)

    if not args.skip_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_modeling_hlwm.py"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        print((result.stdout or "").strip()[-2000:])
        if result.returncode != 0:
            print((result.stderr or "")[-2000:])
            raise SystemExit("v9.0 tests failed; refusing to build a bundle")

    manifest = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "behavior_anchor_counts": ANCHOR_COUNTS,
        "code_sha256": {name: sha256(PROJECT / name) for name in CODE_FILES},
        "counts": counts,
        "name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "purpose": (
            "necessity by construction: information-asymmetric masked channel with "
            "dense latent supervision and a gist attribution control; repaired "
            "harvest surface; difficulty-banded audit; conformal abstention with a "
            "coverage-restricted headline; sized for a Kaggle dual-T4 two-seed session"
        ),
        "review_status": (
            "Reasoning9000 rows remain unreviewed and policy-masked. The masked-row "
            "setting is a constructed mechanism demonstration, not a capability "
            "claim. Not authorized as production or factual-quality evidence until "
            "the preregistered gate passes."
        ),
        "seed": args.seed,
        "session_plan_hours": 9.5,
        "supersedes": "hlwm-repaired-channel-v8-0-candidate",
        "target_hardware": "Kaggle T4 x2, one independent seed per GPU",
        "training_defaults": TRAINING_DEFAULTS,
        "v9_0_changes": V9_CHANGES,
    }
    manifest_path = PROJECT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    notebook_path = BUNDLE_ROOT / "embel-hlwm-v9.0-kaggle-2xt4.ipynb"
    notebook_path.write_text(
        json.dumps(notebook_document(counts), indent=1) + "\n", encoding="utf-8"
    )

    archive_path = BUNDLE_ROOT / "hlwm-v9.0-candidate-bundle.zip"
    if archive_path.exists():
        archive_path.unlink()
    members: List[Path] = []
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        members.append(path)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in members:
            info = zipfile.ZipInfo(
                str(Path("hlwm_kaggle") / path.relative_to(PROJECT)),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    report = {
        "built": "2026-09-02",
        "package_version": PACKAGE_VERSION,
        "bundle_sha256": sha256(archive_path),
        "notebook_sha256": sha256(notebook_path),
        "counts": counts,
        "anchor_counts": ANCHOR_COUNTS,
        "code_sha256": manifest["code_sha256"],
    }
    (BUNDLE_ROOT / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
