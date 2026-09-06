from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.])")
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do", "does",
    "each", "for", "from", "has", "have", "how", "if", "in", "is", "it", "its", "must",
    "of", "on", "only", "or", "that", "the", "their", "them", "then", "these", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
}
_NUMERIC_LITERAL = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_EXPRESSION = rf"-?{_NUMERIC_LITERAL}(?:\s*(?:\+|-|\*{{1,2}}|/{{1,2}}|%|×|÷)\s*-?{_NUMERIC_LITERAL})*"
_EQUATION_RE = re.compile(
    rf"(?<![\w.^%*/+\-])({_EXPRESSION})\s*=\s*({_EXPRESSION})(?![\w^%*/+\-]|\.\d)"
)
_EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:tests?|experiments?|surveys?|benchmarks?|training(?:\s+runs?)?|replications?)\s+"
    r"(?:show|showed|confirm|confirmed|pass|passed|fail|failed|yield|yielded|found|demonstrate|demonstrated)\b"
    r"|\b(?:was|were|has been|have been)\s+(?:performed|executed|run|completed|replicated)\b"
    r"|\b(?:p[- ]?value|t[- ]?statistic)\s*(?:=|<|>)"
    r"|\b(?:results?|measurements?)\s+(?:show|showed|confirm|confirmed|yield|yielded)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:if|when|propose|proposes|proposed|proposal|planned|design|expected|expects|hypothetical|illustrative|not executed|not run|would|could|should)\b",
    re.IGNORECASE,
)
_NEGATED_EXECUTION_RE = re.compile(
    r"\bno\s+(?:actual\s+)?(?:[a-z-]+\s+){0,4}(?:was|were|has been|have been)\s+"
    r"(?:performed|executed|run|completed|replicated)\b"
    r"|\b(?:was|were|has|have)\s+not\s+(?:been\s+)?(?:performed|executed|run|completed|replicated)\b"
    r"|\b(?:not|never)\s+(?:performed|executed|run|completed|replicated)\b",
    re.IGNORECASE,
)
_EXTERNAL_METHOD_RE = re.compile(
    r"\b(?:survey|controlled experiment|benchmark|independent replication|replication (?:run|experiment)|training run|cargo (?:build|test)|pytest|"
    r"compile|execute|simulation|production logs?|user study|a/b test)\b",
    re.IGNORECASE,
)
_SELF_CERTIFY_RE = re.compile(
    r"^(?:the\s+)?(?:constraint\s+)?(?:is\s+|was\s+)?(?:satisfied|verified|confirmed|met|compliant|passed)"
    r"(?:\s+successfully)?[.!]?$",
    re.IGNORECASE,
)
_PROPOSED_OUTCOME_RE = re.compile(
    r"\b(?:all\s+)?(?:tests?|checks?|builds?|deployments?|pipelines?|scripts?|experiments?|surveys?|"
    r"benchmarks?|commands?|queries?|requests?)\s+(?:passed|failed|succeeded|completed|executed|"
    r"returned|produced)\b"
    r"|\b(?:execution|deployment|build|pipeline)\s+(?:was\s+)?successful\b"
    r"|\bexit\s+code\s*[:=]?\s*0\b",
    re.IGNORECASE,
)


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _all_strings(child)


def _artifact_index(episode: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(artifact.get("artifact_id")): artifact
        for lane in episode.get("lanes", [])
        for artifact in lane.get("artifacts", [])
        if artifact.get("artifact_id")
    }


def _safe_numeric_expression(expression: str) -> Optional[float]:
    value = expression.replace("×", "*").replace("÷", "/").replace(",", "").replace(" ", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:[+\-*/%]{1,2}-?\d+(?:\.\d+)?)*", value):
        return None
    tokens = re.findall(r"-?\d+(?:\.\d+)?|//|\*\*|[+\-*/%]", value)
    if "**" in tokens:
        return None
    try:
        # The strict token grammar above permits numeric literals and basic operators only.
        result = eval(value, {"__builtins__": {}}, {})  # noqa: S307
    except (ArithmeticError, SyntaxError, ValueError):
        return None
    if isinstance(result, (int, float)) and math.isfinite(float(result)):
        return float(result)
    return None


def _display_rounding_tolerance(expression: str) -> float:
    value = expression.replace(",", "").strip()
    match = re.fullmatch(r"-?\d+\.(\d+)", value)
    if not match:
        return 0.0
    return 0.5 * (10 ** -len(match.group(1))) + 1e-12


def _arithmetic_errors(episode: Mapping[str, Any]) -> List[str]:
    texts = [str(episode.get("integration", {}).get("published_answer", ""))]
    texts.extend(str(item.get("content", "")) for item in _artifact_index(episode).values())
    texts.extend(str(item.get("result", "")) for item in episode.get("verification", []))
    texts.extend(str(item.get("result", "")) for item in episode.get("tool_runs", []))
    errors: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    for text in texts:
        for match in _EQUATION_RE.finditer(text):
            prefix = text[max(0, match.start() - 20) : match.start()]
            suffix = text[match.end() : match.end() + 8]
            if re.search(r"[+\-*/%^×÷]\s*\$?\s*$", prefix) or re.match(r"\s*(?:[+\-*/%^]|=>)", suffix):
                continue
            left_text, right_text = match.group(1), match.group(2)
            key = (left_text.replace(" ", ""), right_text.replace(" ", ""))
            if key in seen:
                continue
            seen.add(key)
            left = _safe_numeric_expression(left_text)
            right = _safe_numeric_expression(right_text)
            if left is None or right is None:
                continue
            tolerance = max(
                1e-9,
                1e-9 * max(abs(left), abs(right), 1.0),
                _display_rounding_tolerance(right_text),
            )
            if abs(left - right) > tolerance:
                errors.append("deterministic arithmetic check failed: %s = %s" % key)
    return errors


def _declared_word_count_errors(episode: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    declaration = re.compile(
        r"\((?P<count>\d+)\s+words?\)",
        re.IGNORECASE,
    )
    for artifact_id, artifact in _artifact_index(episode).items():
        content = str(artifact.get("content", ""))
        match = declaration.search(content)
        if not match:
            continue
        claimed = int(match.group("count"))
        quoted = re.search(r'["“](.+?)["”](?:\s*\n|$)', content, re.DOTALL)
        if quoted:
            measured_text = quoted.group(1)
        else:
            measured_text = content[: match.start()] + content[match.end() :]
        measured = _word_count(measured_text)
        if measured != claimed:
            errors.append(
                "deterministic declared word-count check failed for %s: claimed %d, measured %d"
                % (artifact_id, claimed, measured)
            )
    return errors


def _word_count_consistency_errors(episode: Mapping[str, Any]) -> List[str]:
    """Reject internally contradictory reconstruction counts before model judging."""
    surfaces: List[Tuple[str, str]] = []
    surfaces.extend(
        ("artifact %s" % artifact_id, str(artifact.get("content", "")))
        for artifact_id, artifact in _artifact_index(episode).items()
    )
    surfaces.extend(
        ("lane %s summary" % lane.get("lane_id", "unknown"), str(lane.get("summary", "")))
        for lane in episode.get("lanes", [])
    )
    surfaces.extend(
        (
            "verification %s" % item.get("claim_id", "unknown"),
            str(item.get("result", "")),
        )
        for item in episode.get("verification", [])
    )
    surfaces.append(
        ("published answer", str(episode.get("integration", {}).get("published_answer", "")))
    )
    errors: List[str] = []
    declaration = re.compile(r"\b(?P<count>\d+)\s+words?\b", re.IGNORECASE)
    numbered_token = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*\((\d+)\)")
    for label, text in surfaces:
        declared = {int(match.group("count")) for match in declaration.finditer(text)}
        enumerated = [int(value) for value in numbered_token.findall(text)]
        if len(declared) > 1:
            errors.append(
                "deterministic word-count consistency check failed for %s: conflicting declarations %s"
                % (label, sorted(declared))
            )
        if enumerated and declared and max(enumerated) not in declared:
            errors.append(
                "deterministic word-count reconstruction failed for %s: declared %s, enumeration ends at %d"
                % (label, sorted(declared), max(enumerated))
            )
    return errors


def _declared_token_count_errors(episode: Mapping[str, Any]) -> List[str]:
    artifacts = _artifact_index(episode)
    errors: List[str] = []
    declaration = re.compile(
        r"\b(?:contains?|has|count(?:ed)?(?:\s+is)?|totals?)\s+(\d+)\s+"
        r"(?:whitespace[- ]separated\s+)?tokens?\b",
        re.IGNORECASE,
    )
    for verification in episode.get("verification", []):
        result = str(verification.get("result", ""))
        match = declaration.search(result)
        if not match:
            continue
        cited = [
            str(ref)
            for ref in verification.get("evidence_refs", [])
            if str(ref) in artifacts
        ]
        if len(cited) != 1:
            continue
        artifact_id = cited[0]
        claimed = int(match.group(1))
        measured = len(str(artifacts[artifact_id].get("content", "")).split())
        if claimed != measured:
            errors.append(
                "deterministic whitespace-token count failed for %s: claimed %d, measured %d"
                % (artifact_id, claimed, measured)
            )
    return errors


def _central_reconstruction_errors(
    episode: Mapping[str, Any], blueprint: Optional[Mapping[str, Any]]
) -> List[str]:
    if not blueprint or episode.get("commitment", {}).get("decision") != "publish":
        return []
    required = ("inputs", "operation", "reconstructed result", "falsification", "verdict")
    for verification in episode.get("verification", []):
        if verification.get("verdict") != "supported":
            continue
        result = str(verification.get("result", "")).replace("\\n", "\n")
        positions: List[int] = []
        complete = True
        for label in required:
            match = re.search(
                r"(?:^|[.\n;])\s*%s\s*:\s*\S" % re.escape(label),
                result,
                re.IGNORECASE,
            )
            if not match:
                complete = False
                break
            positions.append(match.start())
        if complete and positions == sorted(positions):
            falsification_start = re.search(
                r"(?:^|[.\n;])\s*Falsification\s*:", result, re.IGNORECASE
            )
            verdict_start = re.search(
                r"(?:^|[.\n;])\s*Verdict\s*:", result, re.IGNORECASE
            )
            if falsification_start and verdict_start and falsification_start.end() < verdict_start.start():
                falsification = result[falsification_start.end() : verdict_start.start()]
                required_falsification_fields = (
                    "Mutation:",
                    "Recomputed outcome:",
                    "Rejection rule:",
                )
                if all(field.casefold() in falsification.casefold() for field in required_falsification_fields):
                    return []
                return [
                    "central falsification must apply one concrete Mutation, Recomputed outcome, and Rejection rule"
                ]
    return [
        "central reconstruction is missing one ordered visible chain: "
        "Inputs -> Operation -> Reconstructed result -> Falsification -> Verdict"
    ]


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _word_requirement(statement: str) -> Optional[Tuple[int, Optional[int]]]:
    lower = statement.lower()
    match = re.search(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+words?\b", lower)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"\b(?:exactly|must be)\s+(\d+)\s*(?:-|\s)words?\b", lower)
    if match:
        value = int(match.group(1))
        return value, value
    match = re.search(r"\b(\d+)[- ]word\b", lower)
    if match:
        value = int(match.group(1))
        return value, value
    match = re.search(r"\b(?:at least|minimum(?: of)?)\s+(\d+)\s+words?\b", lower)
    if match:
        return int(match.group(1)), None
    match = re.search(r"\b(?:at most|no more than|maximum(?: of)?|under)\s+(\d+)\s+words?\b", lower)
    if match:
        maximum = int(match.group(1)) - (1 if "under" in match.group(0) else 0)
        return 0, maximum
    return None


def _trace_artifact_texts(
    episode: Mapping[str, Any], constraint_id: str, artifacts: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    rows = episode.get("blueprint_trace", {}).get("constraint_trace", [])
    row = next((item for item in rows if str(item.get("constraint_id")) == constraint_id), None)
    if row is None:
        return []
    return [
        str(artifacts[artifact_id].get("content", ""))
        for artifact_id in map(str, row.get("artifact_ids", []))
        if artifact_id in artifacts
    ]


def _percentage_sums(text: str) -> List[float]:
    sums: List[float] = []
    first_per_line: List[float] = []
    for line in text.splitlines():
        if re.search(r"\b(?:total|sum)\b", line, re.IGNORECASE):
            continue
        found = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", line)
        if found:
            first_per_line.append(float(found[0]))
    if len(first_per_line) >= 2:
        sums.append(sum(first_per_line))

    table = [
        [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.strip().startswith("|")
    ]
    if table:
        header = table[0]
        candidate_columns = [
            index
            for index, cell in enumerate(header)
            if "%" in cell and ("revised" in cell.lower() or "proposed" in cell.lower())
        ]
        for column in candidate_columns:
            values: List[float] = []
            for row in table[1:]:
                if column >= len(row) or not row:
                    continue
                if set("".join(row)) <= {"-", ":", " "}:
                    continue
                if "total" in row[0].lower():
                    break
                match = re.fullmatch(r"[+\-]?(\d+(?:\.\d+)?)%?", row[column].strip())
                if match:
                    values.append(float(match.group(1)))
            if len(values) >= 2:
                sums.append(sum(values))
    return sums


def _constraint_evidence_errors(
    episode: Mapping[str, Any], blueprint: Optional[Mapping[str, Any]]
) -> List[str]:
    artifacts = _artifact_index(episode)
    selected = set(map(str, episode.get("integration", {}).get("selected_artifacts", [])))
    rows = episode.get("blueprint_trace", {}).get("constraint_trace", [])
    errors: List[str] = []
    for row in rows:
        constraint_id = str(row.get("constraint_id", "unknown"))
        artifact_ids = list(map(str, row.get("artifact_ids", [])))
        visible = [artifacts[item] for item in artifact_ids if item in artifacts]
        if not artifact_ids or not visible:
            errors.append("constraint %s is self-certified without a visible artifact" % constraint_id)
            continue
        if not any(item in selected for item in artifact_ids):
            errors.append("constraint %s has no selected acceptance artifact" % constraint_id)
        if not any(len(str(item.get("content", "")).strip()) >= 20 for item in visible):
            errors.append("constraint %s cites no substantive artifact evidence" % constraint_id)
        result = str(row.get("result", "")).strip()
        if _SELF_CERTIFY_RE.fullmatch(result):
            errors.append("constraint %s uses a self-certifying result" % constraint_id)

    constraints: List[Tuple[str, str]] = []
    if blueprint:
        constraints = [
            (str(item.get("constraint_id", "constraint-%d" % index)), str(item.get("statement", "")))
            for index, item in enumerate(blueprint.get("task", {}).get("constraints", []))
        ]
    else:
        constraints = [
            ("constraint-%d" % index, str(statement))
            for index, statement in enumerate(episode.get("input", {}).get("constraints", []))
        ]

    for constraint_id, statement in constraints:
        requirement = _word_requirement(statement)
        candidates = _trace_artifact_texts(episode, constraint_id, artifacts)
        if not candidates:
            candidates = [str(episode.get("integration", {}).get("published_answer", ""))]
        if requirement is not None:
            minimum, maximum = requirement
            counts = [_word_count(text) for text in candidates if text.strip()]
            valid = any(count >= minimum and (maximum is None or count <= maximum) for count in counts)
            if not valid:
                errors.append(
                    "deterministic word-count check failed for %s: observed %s, required %s..%s"
                    % (constraint_id, counts, minimum, maximum if maximum is not None else "unbounded")
                )

        if "sum to exactly 100%" in statement.lower() or "sum to 100%" in statement.lower():
            sums: List[float] = []
            for text in candidates:
                sums.extend(_percentage_sums(text))
            if not sums or not any(abs(value - 100.0) <= 1e-6 for value in sums):
                errors.append(
                    "deterministic percentage check failed for %s: first-per-line sums %s"
                    % (constraint_id, sums)
                )
    return errors


def _execution_errors(
    episode: Mapping[str, Any], trusted_execution_run_ids: Set[str]
) -> List[str]:
    errors: List[str] = []
    for run in episode.get("tool_runs", []):
        run_id = str(run.get("run_id", "unknown"))
        status = str(run.get("status", ""))
        if status in {"executed", "provided_evidence"} and run_id not in trusted_execution_run_ids:
            errors.append(
                "tool run %s claims %s without pipeline-issued execution attestation"
                % (run_id, status)
            )
        if status in {"executed", "provided_evidence"} and not bool(run.get("replayable")):
            errors.append("tool run %s claims inaccessible non-replayable evidence" % run_id)
        result = str(run.get("result", ""))
        if (
            status in {"proposed", "not_run"}
            and _PROPOSED_OUTCOME_RE.search(result)
            and not _HYPOTHETICAL_RE.search(result)
            and not _NEGATED_EXECUTION_RE.search(result)
        ):
            errors.append(
                "tool run %s reports an outcome despite status %s" % (run_id, status)
            )

    integration = episode.get("integration", {})
    selected = set(map(str, integration.get("selected_artifacts", [])))
    surfaces = [("published answer", str(integration.get("published_answer", "")))]
    surfaces.extend(
        (
            "selected artifact %s" % artifact_id,
            re.sub(r"```.*?```", "", str(artifact.get("content", "")), flags=re.DOTALL),
        )
        for artifact_id, artifact in _artifact_index(episode).items()
        if artifact_id in selected
    )
    for label, text in surfaces:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            clauses = re.split(
                r"\b(?:but|however|although|whereas)\b|[;]",
                sentence,
                flags=re.IGNORECASE,
            )
            for clause in clauses:
                if (
                    _EXECUTION_CLAIM_RE.search(clause)
                    and not _HYPOTHETICAL_RE.search(clause)
                    and not _NEGATED_EXECUTION_RE.search(clause)
                    and not any(run_id in clause for run_id in trusted_execution_run_ids)
                ):
                    errors.append(
                        "%s makes an unattested execution claim: %s"
                        % (label, clause.strip()[:180])
                    )

    for verification in episode.get("verification", []):
        if verification.get("verdict") != "supported":
            continue
        method = str(verification.get("method", ""))
        evidence_refs = set(map(str, verification.get("evidence_refs", [])))
        if (
            _EXTERNAL_METHOD_RE.search(method)
            and not _HYPOTHETICAL_RE.search(method)
            and not evidence_refs.intersection(trusted_execution_run_ids)
        ):
            errors.append(
                "claim %s relies on an unexecuted or inaccessible verification method"
                % verification.get("claim_id", "unknown")
            )
    return errors


def _numbers(text: str) -> Set[str]:
    values = {item.lstrip("+") for item in _NUMBER_RE.findall(text)}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _NUMBER_WORDS:
            values.add(_NUMBER_WORDS[word])
    return values


def _arithmetic_concepts(text: str) -> Set[str]:
    lower = text.lower()
    concepts: Set[str] = set()
    if re.search(r"\b(?:equal|equally|same amount|each)\b", lower):
        concepts.add("equal-allocation")
    if re.search(r"\b(?:leftover|left over|remainder|remain)\b", lower):
        concepts.add("remainder")
    if re.search(r"\b(?:distribut|divid|share|feed)\w*\b", lower):
        concepts.add("distribution")
    if re.search(r"\b(?:maximum|maximal|as many|all .* can)\b", lower):
        concepts.add("maximization")
    return concepts


def _source_prompt(text: str) -> str:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(value, dict):
        for key in ("question", "problem", "prompt", "instruction", "input"):
            if isinstance(value.get(key), str):
                return value[key]
    return text


def _lexical_similarity(left: str, right: str) -> float:
    def tokens(text: str) -> List[str]:
        return [
            item for item in re.findall(r"[a-z0-9]+", text.lower())
            if item not in _STOPWORDS
        ]

    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_set, right_set = set(left_tokens), set(right_tokens)
    jaccard = len(left_set & right_set) / max(1, len(left_set | right_set))
    sequence = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    return max(jaccard, sequence)


def _source_similarity_errors(
    episode: Mapping[str, Any], source_packet: Optional[Mapping[str, Any]]
) -> List[str]:
    if not source_packet:
        return []
    source_group = str(episode.get("source_group", ""))
    chunks = source_packet.get("chunks", [])
    benchmark_like = source_group.startswith("hf-") or any(
        str(item.get("data_role", "")) == "verification_material" for item in chunks
    )
    if not benchmark_like:
        return []
    task = str(episode.get("input", {}).get("user_request", ""))
    task_numbers = _numbers(task)
    task_concepts = _arithmetic_concepts(task)
    errors: List[str] = []
    for chunk in chunks:
        source = _source_prompt(str(chunk.get("text", "")))
        source_numbers = _numbers(source)
        source_concepts = _arithmetic_concepts(source)
        numeric_copy = (
            len(task_numbers) >= 2
            and len(task_numbers & source_numbers) / max(1, len(task_numbers)) >= 0.80
            and len(task_concepts & source_concepts) >= 2
        )
        lexical_copy = _lexical_similarity(task, source) >= 0.72
        if numeric_copy or lexical_copy:
            errors.append(
                "source-task similarity gate detected a cosmetic benchmark transformation from chunk %s"
                % chunk.get("chunk_id", "unknown")
            )
            break
    return errors


def _complexity_errors(episode: Mapping[str, Any]) -> List[str]:
    domain = str(episode.get("domain", ""))
    if domain != "mathematics_and_optimization":
        return []
    task = str(episode.get("input", {}).get("user_request", ""))
    concepts = _arithmetic_concepts(task)
    advanced = re.search(
        r"\b(?:optimization|probability|distributional|matrix|calculus|gradient|graph|recurrence|"
        r"combinator|asymptotic|integer program|linear program|stochastic|bayesian|convex)\w*\b",
        task,
        re.IGNORECASE,
    )
    if len(concepts) >= 2 and len(_numbers(task)) <= 4 and not advanced:
        return [
            "expertise complexity floor failed: basic arithmetic is not made expert-level by formal-proof decoration"
        ]
    return []


def _verification_specificity_errors(episode: Mapping[str, Any]) -> List[str]:
    seen: Dict[str, str] = {}
    errors: List[str] = []
    for row in episode.get("verification", []):
        claim_id = str(row.get("claim_id", "unknown"))
        result = str(row.get("result", ""))
        before_verdict = re.split(r"\bVerdict\s*:", result, maxsplit=1, flags=re.IGNORECASE)[0]
        signature = re.sub(r"\s+", " ", before_verdict).strip().casefold()
        if not signature:
            continue
        previous = seen.get(signature)
        if previous is not None and previous != claim_id:
            errors.append(
                "verification rows %s and %s reuse the same reconstruction instead of claim-specific evidence"
                % (previous, claim_id)
            )
        else:
            seen[signature] = claim_id
    return errors


def adversarial_acceptance_errors(
    episode: Dict[str, Any],
    blueprint: Optional[Dict[str, Any]] = None,
    source_packet: Optional[Dict[str, Any]] = None,
    trusted_execution_run_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Apply deterministic, fail-closed checks before any model-judge acceptance.

    ``trusted_execution_run_ids`` must come from the pipeline, never from model-authored
    episode fields. The current generator supplies no such attestations, so execution
    claims are rejected until a real tool runner is integrated.
    """

    trusted = set(trusted_execution_run_ids or set())
    errors: List[str] = []
    errors.extend(_arithmetic_errors(episode))
    errors.extend(_declared_word_count_errors(episode))
    errors.extend(_word_count_consistency_errors(episode))
    errors.extend(_declared_token_count_errors(episode))
    errors.extend(_constraint_evidence_errors(episode, blueprint))
    errors.extend(_central_reconstruction_errors(episode, blueprint))
    errors.extend(_verification_specificity_errors(episode))
    errors.extend(_execution_errors(episode, trusted))
    errors.extend(_source_similarity_errors(episode, source_packet))
    errors.extend(_complexity_errors(episode))
    return list(dict.fromkeys(errors))
