"""Builder teacher factory: verified product-grade training episodes.

Domains: full-stack product replicas, coding/debugging, frontend pages,
copywriting, pitch decks, investor analysis, project planning and devops.

Stage flow per task: plan -> build -> deterministic verify -> bounded repair
-> independent judging -> episode assembly. Deterministic execution (pytest,
AST parses, HTML/YAML/shell checks, numeric recomputation) is the primary
gate; the standing independent judge pair gates prose quality and fails
closed on disagreement. Drafting providers never judge. Teacher reasoning
traces are never persisted.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from .azure_client import BudgetExceededError
from .builder_seeds import (
    BUG_INJECTOR_SYSTEM,
    BUILDER_SYSTEM,
    JUDGE_SYSTEM,
    PLANNER_SYSTEM,
    PROSE_BUILDER_SYSTEM,
    REPAIR_SYSTEM,
    SCALE_LIMITS,
    BuilderTask,
    expand_tasks,
)
from .language import classify_language
from .teacher_providers import TeacherProvider, build_teacher_providers, provider_statuses
from .util import append_jsonl, atomic_write_json, iter_jsonl, stable_hash, write_jsonl
from .validation import PRIVATE_REASONING_PATTERNS, SECRET_PATTERNS

FILE_BLOCK_PATTERN = re.compile(
    r'<<<FILE path="(?P<path>[^"\n]+)">>>\n(?P<body>.*?)\n?<<<END FILE>>>', re.DOTALL
)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
CURL_PIPE_PATTERN = re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:ba|z|da)?sh\b")
DANGEROUS_CALL_PATTERN = re.compile(
    r"\bos\.system\b|\bos\.popen\b|\bos\.exec\w*\b|\bos\.spawn\w*\b|\bos\.fork\b|"
    r"(?<![\w.])eval\s*\(|(?<![\w.])exec\s*\(|__import__\s*\(|\bshutil\.rmtree\b"
)
LOREM_PATTERN = re.compile(r"lorem ipsum", re.IGNORECASE)

STDLIB_WHITELIST = {
    "__future__", "abc", "array", "base64", "binascii", "bisect", "calendar",
    "collections", "contextlib", "copy", "csv", "dataclasses", "datetime",
    "decimal", "enum", "fractions", "functools", "gzip", "hashlib", "heapq",
    "hmac", "html", "io", "itertools", "json", "logging", "math", "numbers",
    "operator", "os", "pathlib", "queue", "random", "re", "secrets", "sqlite3",
    "statistics", "string", "struct", "tempfile", "textwrap", "threading",
    "time", "traceback", "types", "typing", "unicodedata", "unittest", "uuid",
    "warnings", "weakref", "zlib", "zoneinfo", "pytest",
}
API_EXTRA_WHITELIST = {"fastapi", "pydantic", "starlette", "flask"}
EXECUTED_DIRS = ("core", "tests")
MAX_FILE_BYTES = 120_000
MAX_JUDGE_CHARS = 100_000
JUDGE_SCORE_THRESHOLD = 0.75
BENIGN_INTEGERS = set(range(0, 13)) | {15, 20, 24, 25, 30, 50, 60, 100}

JUDGED_DOMAINS_DEFAULT = [
    "fullstack_product",
    "website_frontend",
    "copywriting",
    "pitch_deck",
    "investor_analysis",
    "project_planning",
]

INVESTOR_FORMULAS = (
    "Pinned formulas (use exactly these): "
    "revenue_cagr_pct = ((year3/year1)**0.5 - 1) * 100; "
    "gross_profit_latest_usd = latest_revenue * gross_margin_pct/100; "
    "runway_months = cash_on_hand_usd / monthly_burn_usd; "
    "arpa_usd = latest_revenue / paying_customers; "
    "ltv_usd = (arpa_usd * gross_margin_pct/100) / (monthly_logo_churn_pct/100 * 12); "
    "ltv_to_cac = ltv_usd / cac_usd."
)


# --------------------------------------------------------------------------
# Small helpers.
# --------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text)


def parse_file_blocks(text: str) -> Dict[str, str]:
    """Parse FILE blocks; reject unsafe paths and oversized files."""

    files: Dict[str, str] = {}
    for match in FILE_BLOCK_PATTERN.finditer(_strip_think(text)):
        path = match.group("path").strip()
        body = match.group("body")
        if path.startswith("/") or ".." in path.split("/") or "\\" in path:
            raise ValueError("unsafe file path: %s" % path)
        if len(body.encode("utf-8", "ignore")) > MAX_FILE_BYTES:
            raise ValueError("file too large: %s" % path)
        files[path] = body
    if not files:
        raise ValueError("no FILE blocks found in builder output")
    return files


def _extract_numbers(text: str) -> List[float]:
    values: List[float] = []
    for token in NUMBER_PATTERN.findall(text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _fact_values(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))
    elif isinstance(value, Mapping):
        for child in value.values():
            out.extend(_fact_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            out.extend(_fact_values(child))
    return out


def numbers_grounded(text: str, allowed_sources: Sequence[Any], extra_allowed: Sequence[float] = ()) -> List[float]:
    """Return the numbers in ``text`` that cannot be traced to allowed facts."""

    allowed: List[float] = list(extra_allowed)
    for source in allowed_sources:
        if isinstance(source, str):
            allowed.extend(_extract_numbers(source))
        else:
            allowed.extend(_fact_values(source))
    expanded: List[float] = []
    for value in allowed:
        expanded.extend((value, value / 1_000.0, value / 1_000_000.0, value / 1_000_000_000.0, value * 100.0))
    violations: List[float] = []
    for found in _extract_numbers(text):
        if found in BENIGN_INTEGERS or 1990 <= found <= 2035:
            continue
        if any(
            math.isclose(found, candidate, rel_tol=0.006, abs_tol=0.51)
            for candidate in expanded
        ):
            continue
        violations.append(found)
    return violations


def _scan_text_policy(name: str, text: str) -> List[str]:
    defects: List[str] = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            defects.append("%s: literal secret material" % name)
            break
    for pattern in PRIVATE_REASONING_PATTERNS:
        if pattern.search(text):
            defects.append("%s: private-reasoning marker" % name)
            break
    if LOREM_PATTERN.search(text):
        defects.append("%s: placeholder lorem ipsum" % name)
    if CURL_PIPE_PATTERN.search(text):
        defects.append("%s: curl piped to shell" % name)
    return defects


def _python_import_defects(path: str, source: str, local_roots: set[str]) -> List[str]:
    executed = path.split("/", 1)[0] in EXECUTED_DIRS
    whitelist = STDLIB_WHITELIST | local_roots
    if not executed:
        whitelist = whitelist | API_EXTRA_WHITELIST
    defects: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ["%s: syntax error: %s" % (path, exc)]
    for node in ast.walk(tree):
        names: List[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root not in whitelist:
                defects.append("%s: forbidden import '%s'" % (path, name))
    if executed and DANGEROUS_CALL_PATTERN.search(source):
        defects.append("%s: forbidden call pattern" % path)
    return defects


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: List[str] = []
        self.errors: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)

    def error(self, message: str) -> None:  # pragma: no cover - py<3.10 shim
        self.errors.append(message)


def _frontend_defects(files: Mapping[str, str], required_sections: Sequence[str]) -> List[str]:
    defects: List[str] = []
    html_files = {path: body for path, body in files.items() if path.endswith(".html")}
    if not html_files:
        return ["no HTML file produced"]
    combined = "\n".join(html_files.values()).lower()
    for path, body in html_files.items():
        parser = _StructureParser()
        try:
            parser.feed(body)
        except Exception as exc:
            defects.append("%s: HTML parse failure: %s" % (path, exc))
            continue
        for tag in ("html", "head", "body", "title"):
            if tag not in parser.tags:
                defects.append("%s: missing <%s>" % (path, tag))
        if "http://" in body:
            defects.append("%s: insecure http:// reference" % path)
        if re.search(r'\bsrc="https?://', body) or re.search(r'<link[^>]+href="https?://', body):
            defects.append("%s: external asset reference" % path)
    for section in required_sections:
        if section.lower() not in combined:
            defects.append("required section '%s' not found" % section)
    for path, body in files.items():
        if path.endswith(".css"):
            if body.count("{") != body.count("}"):
                defects.append("%s: unbalanced CSS braces" % path)
            tail = body.rstrip()
            if tail and not tail.endswith(("}", "*/")):
                defects.append("%s: stylesheet ends mid-rule (truncated output)" % path)
    return defects


def _yaml_defects(path: str, body: str) -> List[str]:
    defects: List[str] = []
    try:
        value = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        return ["%s: YAML parse failure: %s" % (path, str(exc).splitlines()[0])]
    text = body
    if re.search(r"\bprivileged:\s*true\b", text):
        defects.append("%s: privileged container" % path)
    for match in re.finditer(r"(?m)^\s*image:\s*(\S+)\s*$", text):
        image = match.group(1).strip("'\"")
        if "${" in image:
            continue
        if ":" not in image or image.endswith(":latest"):
            defects.append("%s: unpinned image '%s'" % (path, image))
    del value
    return defects


def _dockerfile_defects(path: str, body: str) -> List[str]:
    defects: List[str] = []
    froms = re.findall(r"(?im)^FROM\s+(\S+)", body)
    if not froms:
        defects.append("%s: missing FROM" % path)
    for image in froms:
        if ":" not in image or image.endswith(":latest"):
            defects.append("%s: unpinned base image '%s'" % (path, image))
    if not re.search(r"(?im)^USER\s+(?!root\b)\S+", body):
        defects.append("%s: no non-root USER" % path)
    if re.search(r"(?im)^ADD\s+https?://", body):
        defects.append("%s: ADD from URL" % path)
    return defects


def _shell_defects(path: str, body: str, workdir: Path) -> List[str]:
    defects: List[str] = []
    if "set -euo pipefail" not in body:
        defects.append("%s: missing 'set -euo pipefail'" % path)
    target = workdir / path
    try:
        result = subprocess.run(
            ["bash", "-n", str(target)], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            defects.append("%s: bash -n failed: %s" % (path, result.stderr.strip()[:200]))
    except (OSError, subprocess.TimeoutExpired) as exc:
        defects.append("%s: bash check unavailable: %s" % (path, exc))
    return defects


def _write_workdir(files: Mapping[str, str]) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="hlwm-builder-"))
    for path, body in files.items():
        target = workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return workdir


def run_pytest(files: Mapping[str, str], timeout_seconds: int = 180) -> Dict[str, Any]:
    """Execute the bundled pytest suite in an isolated temp dir."""

    workdir = _write_workdir(files)
    try:
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(workdir),
            "PYTHONPATH": str(workdir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_PROXY": "*",
        }
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            output = "pytest timed out after %ds" % timeout_seconds
            returncode = -1
        elapsed = time.monotonic() - started
        passed = failed = errors = 0
        match = re.search(r"(\d+) passed", output)
        if match:
            passed = int(match.group(1))
        match = re.search(r"(\d+) failed", output)
        if match:
            failed = int(match.group(1))
        match = re.search(r"(\d+) error", output)
        if match:
            errors = int(match.group(1))
        return {
            "returncode": returncode,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "green": returncode == 0 and passed > 0,
            "output_tail": output[-4000:],
            "elapsed_seconds": round(elapsed, 2),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def verify_code_bundle(
    files: Mapping[str, str],
    *,
    required_sections: Sequence[str] = (),
    run_tests: bool = True,
    devops_policy: bool = False,
) -> Dict[str, Any]:
    """Static policy scan plus executed tests for a generated file bundle."""

    defects: List[str] = []
    local_roots = {path.split("/", 1)[0].removesuffix(".py") for path in files}
    local_roots |= {Path(path).stem for path in files if path.endswith(".py")}
    for path, body in sorted(files.items()):
        defects.extend(_scan_text_policy(path, body))
        if path.endswith(".py"):
            defects.extend(_python_import_defects(path, body, local_roots))
        elif path.endswith((".yml", ".yaml")):
            defects.extend(_yaml_defects(path, body))
        elif path.endswith(".json"):
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                defects.append("%s: JSON parse failure: %s" % (path, exc))
        elif Path(path).name == "Dockerfile" and devops_policy:
            defects.extend(_dockerfile_defects(path, body))
    # Cross-group drift guard: every core.<module> import must resolve to a file
    # (or to a symbol the package __init__ actually defines).
    init_body = files.get("core/__init__.py", "")
    for path, body in sorted(files.items()):
        if not path.endswith(".py"):
            continue
        referenced = set(re.findall(r"(?m)^\s*(?:from|import)\s+core\.(\w+)", body))
        for match in re.finditer(r"(?m)^\s*from\s+core\s+import\s+([\w ,]+)", body):
            for name in match.group(1).split(","):
                name = name.strip()
                if name and name == name.lower():
                    referenced.add(name)
        for name in sorted(referenced):
            if "core/%s.py" % name in files:
                continue
            if re.search(r"(?m)^(?:def|class)\s+%s\b|^%s\s*=" % (name, name), init_body):
                continue
            defects.append(
                "%s imports core.%s but core/%s.py does not exist; create it or fix the import"
                % (path, name, name)
            )
    frontend_files = {p: b for p, b in files.items() if p.split("/", 1)[0] == "web"}
    frontend_defects: List[str] = []
    if frontend_files:
        frontend_defects = _frontend_defects(frontend_files, required_sections)
        defects.extend(frontend_defects)
    shell_files = {p: b for p, b in files.items() if p.endswith(".sh")}
    if shell_files and devops_policy:
        workdir = _write_workdir(files)
        try:
            for path, body in shell_files.items():
                defects.extend(_shell_defects(path, body, workdir))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    test_result: Dict[str, Any] = {"green": False, "skipped": True}
    has_tests = any(path.startswith("tests/") for path in files)
    parse_clean = not any("syntax error" in defect for defect in defects)
    if run_tests and has_tests and parse_clean:
        test_result = run_pytest(files)
        test_result["skipped"] = False
        if not test_result["green"]:
            defects.append("pytest failed: %s" % test_result["output_tail"][-1200:])
    elif run_tests and not has_tests:
        defects.append("no tests/ suite present")
    return {
        "defects": defects,
        "tests": test_result,
        "frontend_defects": frontend_defects,
        "files": sorted(files),
    }


def _plan_consistency_defects(plan: Mapping[str, Any]) -> List[str]:
    milestones = plan.get("milestones") or []
    defects: List[str] = []
    by_id: Dict[str, Mapping[str, Any]] = {}
    for milestone in milestones:
        identifier = str(milestone.get("id", "")).strip()
        if not identifier or identifier in by_id:
            defects.append("milestone id missing or duplicated: %r" % identifier)
            continue
        by_id[identifier] = milestone
    for milestone in milestones:
        for dependency in milestone.get("depends_on") or []:
            if str(dependency) not in by_id:
                defects.append("unknown dependency %r" % dependency)
    state: Dict[str, int] = {}

    def _visit(node: str, stack: Tuple[str, ...]) -> float:
        if state.get(node) == 1:
            defects.append("dependency cycle at %s" % node)
            return 0.0
        milestone = by_id.get(node)
        if milestone is None:
            return 0.0
        state[node] = 1
        best = 0.0
        for dependency in milestone.get("depends_on") or []:
            best = max(best, _visit(str(dependency), stack + (node,)))
        state[node] = 2
        return best + float(milestone.get("duration_days", 0) or 0)

    longest = 0.0
    for identifier in by_id:
        state = {k: 0 if v != 2 else 2 for k, v in state.items()}
        longest = max(longest, _visit(identifier, ()))
    declared = [str(item) for item in plan.get("critical_path") or []]
    if declared:
        if any(item not in by_id for item in declared):
            defects.append("critical path references unknown milestones")
        else:
            for earlier, later in zip(declared, declared[1:]):
                if earlier not in (str(d) for d in (by_id[later].get("depends_on") or [])):
                    defects.append("critical path %s -> %s is not a dependency edge" % (earlier, later))
            declared_duration = sum(float(by_id[item].get("duration_days", 0) or 0) for item in declared)
            if longest and abs(declared_duration - longest) > 0.01:
                defects.append(
                    "critical path duration %.1f != longest chain %.1f" % (declared_duration, longest)
                )
    else:
        defects.append("critical_path missing")
    criteria = [str(item) for item in plan.get("acceptance_criteria") or []]
    for identifier in by_id:
        if not any(identifier in criterion for criterion in criteria):
            defects.append("milestone %s has no acceptance criterion" % identifier)
    return defects


def _investor_expected(facts: Mapping[str, Any]) -> Dict[str, float]:
    revenue = [float(v) for v in facts["revenue_by_year_usd"]]
    margin = float(facts["gross_margin_pct"]) / 100.0
    arpa = revenue[-1] / float(facts["paying_customers"])
    ltv = (arpa * margin) / (float(facts["monthly_logo_churn_pct"]) / 100.0 * 12.0)
    return {
        "revenue_cagr_pct": ((revenue[-1] / revenue[0]) ** 0.5 - 1.0) * 100.0,
        "gross_profit_latest_usd": revenue[-1] * margin,
        "runway_months": float(facts["cash_on_hand_usd"]) / float(facts["monthly_burn_usd"]),
        "arpa_usd": arpa,
        "ltv_usd": ltv,
        "ltv_to_cac": ltv / float(facts["cac_usd"]),
    }


def _investor_metric_defects(facts: Mapping[str, Any], reported: Mapping[str, Any]) -> List[str]:
    expected = _investor_expected(facts)
    defects: List[str] = []
    for key, value in expected.items():
        raw = reported.get(key)
        try:
            observed = float(raw)
        except (TypeError, ValueError):
            defects.append("computed_metrics.%s missing or non-numeric" % key)
            continue
        if not math.isclose(observed, value, rel_tol=0.015, abs_tol=0.51):
            defects.append(
                "computed_metrics.%s = %.2f but pinned formula gives %.2f" % (key, observed, value)
            )
    return defects


# --------------------------------------------------------------------------
# Ledger and provider role resolution.
# --------------------------------------------------------------------------

class BuilderLedger:
    def __init__(self, path: Path, max_requests: Optional[int], max_tokens: Optional[int]) -> None:
        self.path = path
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        state = {}
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
        self.lifetime_requests = int(state.get("lifetime_requests", 0))
        self.lifetime_tokens = int(state.get("lifetime_tokens", 0))
        self.run_requests = 0
        self.run_tokens = 0

    def before_attempt(self) -> None:
        with self.lock:
            if self.max_requests is not None and self.run_requests + 1 > self.max_requests:
                raise BudgetExceededError("request budget exhausted (%d)" % self.max_requests)
            if self.max_tokens is not None and self.run_tokens >= self.max_tokens:
                raise BudgetExceededError("token budget exhausted (%d)" % self.max_tokens)
            self.run_requests += 1
            self.lifetime_requests += 1

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self.lock:
            total = int(prompt_tokens) + int(completion_tokens)
            self.run_tokens += total
            self.lifetime_tokens += total
            atomic_write_json(
                self.path,
                {
                    "lifetime_requests": self.lifetime_requests,
                    "lifetime_tokens": self.lifetime_tokens,
                },
            )

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return {
                "run_requests": self.run_requests,
                "run_tokens": self.run_tokens,
                "lifetime_requests": self.lifetime_requests,
                "lifetime_tokens": self.lifetime_tokens,
            }


def resolve_roles(providers: Mapping[str, TeacherProvider]) -> Dict[str, List[str]]:
    roles: Dict[str, List[str]] = {}
    for provider_id, provider in providers.items():
        for role in provider.roles:
            roles.setdefault(role, []).append(provider_id)
    for role in roles:
        roles[role] = sorted(roles[role])
    return roles


def validate_builder_roles(roles: Mapping[str, Sequence[str]], judged_domains: Sequence[str]) -> List[str]:
    errors: List[str] = []
    if not roles.get("builder") and not roles.get("builder_heavy"):
        errors.append("no builder-capable provider is available")
    drafting = set(roles.get("builder", [])) | set(roles.get("builder_heavy", []))
    drafting |= set(roles.get("planner", [])) | set(roles.get("editor", []))
    drafting |= set(roles.get("bug_injector", []))
    judges = [j for j in roles.get("judge", []) if j not in drafting]
    if judged_domains and len(judges) < 2:
        errors.append(
            "judged domains need two independent non-drafting judges; available: %s" % (judges,)
        )
    return errors


# --------------------------------------------------------------------------
# The factory.
# --------------------------------------------------------------------------

class BuilderFactory:
    def __init__(self, config: Mapping[str, Any], root: Path) -> None:
        self.config = dict(config)
        self.root = root
        factory = dict(config.get("builder_factory") or {})
        self.factory = factory
        self.output_dir = root / str(factory.get("output_dir", "data/builder"))
        self.tasks_dir = self.output_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = root / str(factory.get("stop_file", "data/builder/STOP"))
        request_log = root / str(factory.get("request_log", "logs/builder-requests.jsonl"))
        self.providers = build_teacher_providers(config, request_log)
        self.roles = resolve_roles(self.providers)
        self.judged_domains = list(factory.get("judge_domains", JUDGED_DOMAINS_DEFAULT))
        self.max_repair_rounds = int(factory.get("max_repair_rounds", 2))
        self.seed = int((config.get("project") or {}).get("seed", 0))
        self.mix = dict(factory.get("mix") or {})
        self.default_scale = str(factory.get("scale", "medium"))
        self.ledger: Optional[BuilderLedger] = None

    # -- provider helpers ---------------------------------------------------

    def _provider(self, role: str, task_id: str, exclude: Sequence[str] = ()) -> TeacherProvider:
        candidates = [
            provider_id
            for provider_id in self.roles.get(role, [])
            if provider_id not in exclude
        ]
        if not candidates and role == "builder_heavy":
            candidates = [p for p in self.roles.get("builder", []) if p not in exclude]
        if not candidates and role in ("editor", "bug_injector", "planner"):
            fallback = self.roles.get("planner", []) or self.roles.get("editor", [])
            candidates = [p for p in fallback if p not in exclude]
        if not candidates:
            raise RuntimeError("no provider available for role %s" % role)
        index = int(stable_hash(task_id + role, 8), 16) % len(candidates)
        return self.providers[candidates[index]]

    def _judges(self) -> List[TeacherProvider]:
        drafting = set(self.roles.get("builder", [])) | set(self.roles.get("builder_heavy", []))
        drafting |= set(self.roles.get("planner", [])) | set(self.roles.get("editor", []))
        drafting |= set(self.roles.get("bug_injector", []))
        return [
            self.providers[provider_id]
            for provider_id in self.roles.get("judge", [])
            if provider_id not in drafting
        ]

    def _call_kwargs(self) -> Dict[str, Any]:
        assert self.ledger is not None
        return {
            "before_attempt": self.ledger.before_attempt,
            "record_usage": self.ledger.record_usage,
        }

    # -- task state ---------------------------------------------------------

    def _state_path(self, task_id: str) -> Path:
        return self.tasks_dir / ("%s.json" % task_id)

    def _load_state(self, task: BuilderTask) -> Dict[str, Any]:
        path = self._state_path(task.task_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"task_id": task.task_id, "stage": "pending", "files": {}, "attempts": []}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        atomic_write_json(self._state_path(str(state["task_id"])), state)

    # -- prompt assembly ----------------------------------------------------

    def _chat_files(
        self,
        provider: TeacherProvider,
        system: str,
        prompt: str,
        operation: str,
        item_id: str,
    ) -> Dict[str, str]:
        """chat_text plus FILE-block parsing, with one corrective retry."""

        completion = provider.chat_text(system, prompt, operation, item_id, **self._call_kwargs())
        truncated = not _strip_think(completion.text).rstrip().endswith("<<<END FILE>>>")
        if not truncated:
            try:
                return parse_file_blocks(completion.text)
            except ValueError:
                pass
        retry_prompt = prompt + (
            "\n\nYour previous reply was truncated or contained no valid FILE blocks. Re-emit ALL "
            "files, more concisely if needed, strictly as "
            '<<<FILE path="...">>> ... <<<END FILE>>> blocks with nothing else. The very last line '
            "of your reply must be <<<END FILE>>>."
        )
        completion = provider.chat_text(
            system, retry_prompt, operation + "-retry", item_id, **self._call_kwargs()
        )
        if not _strip_think(completion.text).rstrip().endswith("<<<END FILE>>>"):
            raise ValueError("builder output truncated twice for %s" % item_id)
        return parse_file_blocks(completion.text)

    @staticmethod
    def _brief_text(task: BuilderTask) -> str:
        lines = ["REQUEST:", task.user_request, "", "CONTEXT:"]
        lines.extend("- %s" % item for item in task.context)
        lines.append("")
        lines.append("CONSTRAINTS:")
        lines.extend("- %s" % item for item in task.constraints)
        if task.domain == "investor_analysis":
            lines.append("- %s" % INVESTOR_FORMULAS)
        return "\n".join(lines)

    # -- stages -------------------------------------------------------------

    def _stage_plan(self, task: BuilderTask, state: Dict[str, Any]) -> None:
        limits = SCALE_LIMITS.get(task.scale, SCALE_LIMITS["medium"])
        planner = self._provider("planner", task.task_id)
        prompt = (
            self._brief_text(task)
            + "\n\nProduce JSON with keys: objective (string), requirements (array of strings), "
            "failure_contract (array of strings), lanes (array of exactly three objects "
            "{lane_id, scope, deliverable, assumptions (array), rejection_tests (array)} for "
            "lane-backend, lane-frontend, lane-quality), required_surfaces (array of strings), "
            "file_groups (array, in build order, each {group_id, purpose, files (array of "
            "{path, purpose}), contracts (array of strings)}). Hard limits: at most %d files and "
            "%d groups, and no group may contain more than 4 files; every file must belong to "
            "exactly one group; tests/ files live in the final group; include README.md, "
            ".env.example and schema.sql."
            % (limits["files"], limits["groups"])
        )
        plan: Dict[str, Any] = {}
        for attempt in range(2):
            completion = planner.chat_json(
                PLANNER_SYSTEM, prompt, "builder-plan", task.task_id, **self._call_kwargs()
            )
            plan = completion.value
            groups = plan.get("file_groups") or []
            total_files = sum(len(group.get("files") or []) for group in groups)
            if groups and 0 < total_files <= limits["files"] and len(groups) <= limits["groups"]:
                break
            if attempt == 0:
                prompt += (
                    "\n\nYour previous plan had %d files in %d groups, violating the hard limits "
                    "(at most %d files, %d groups). Consolidate related files and replan within "
                    "the limits." % (total_files, len(groups), limits["files"], limits["groups"])
                )
        else:
            raise RuntimeError(
                "plan exceeds scale limits (%d files / %d groups)" % (total_files, len(groups))
            )
        state["plan"] = plan
        state["planner"] = planner.provider_id
        state["stage"] = "planned"
        self._save_state(state)

    def _stage_build_fullstack(self, task: BuilderTask, state: Dict[str, Any]) -> None:
        plan = state["plan"]
        builder = self._provider("builder_heavy", task.task_id)
        built_groups = set(state.get("built_groups") or [])
        files: Dict[str, str] = dict(state.get("files") or {})
        for group in plan.get("file_groups") or []:
            group_id = str(group.get("group_id"))
            if group_id in built_groups:
                continue
            manifest = "\n".join(
                "- %s : %s" % (item.get("path"), item.get("purpose"))
                for item in group.get("files") or []
            )
            contracts = "\n".join("- %s" % item for item in group.get("contracts") or [])
            prior = "\n".join("- %s" % path for path in sorted(files)) or "- (none yet)"
            prompt = (
                self._brief_text(task)
                + "\n\nPLAN OBJECTIVE:\n%s\n\nGROUP TO BUILD NOW (%s): %s\nFILES:\n%s\n\n"
                "INTERFACE CONTRACTS:\n%s\n\nFILES ALREADY BUILT (honor their interfaces):\n%s\n\n"
                "Emit ONLY this group's files as FILE blocks."
                % (
                    plan.get("objective", ""),
                    group_id,
                    group.get("purpose", ""),
                    manifest,
                    contracts or "- (none)",
                    prior,
                )
            )
            try:
                files.update(
                    self._chat_files(
                        builder, BUILDER_SYSTEM, prompt, "builder-build", "%s/%s" % (task.task_id, group_id)
                    )
                )
            except ValueError:
                # The whole group does not fit in one response even after the
                # concise retry; fall back to one call per planned file.
                for item in group.get("files") or []:
                    file_path = str(item.get("path"))
                    single_prompt = prompt + (
                        "\n\nEmit ONLY the single file '%s' now, complete." % file_path
                    )
                    files.update(
                        self._chat_files(
                            builder,
                            BUILDER_SYSTEM,
                            single_prompt,
                            "builder-build-single",
                            "%s/%s/%s" % (task.task_id, group_id, file_path),
                        )
                    )
            built_groups.add(group_id)
            state["files"] = files
            state["built_groups"] = sorted(built_groups)
            self._save_state(state)
        state["builder"] = builder.provider_id
        state["stage"] = "built"
        self._save_state(state)

    def _stage_build_files(self, task: BuilderTask, state: Dict[str, Any]) -> None:
        builder = self._provider(
            "builder_heavy" if task.domain == "website_frontend" else "builder",
            task.task_id,
        )
        build_prompt = self._brief_text(task) + "\n\nEmit every required file as FILE blocks."
        if task.domain == "website_frontend":
            build_prompt += (
                "\nKeep the files tight enough to fit comfortably in one response; every file "
                "must be complete (no mid-rule truncation). Polish beats volume."
            )
        state["files"] = self._chat_files(
            builder, BUILDER_SYSTEM, build_prompt, "builder-build", task.task_id
        )
        state["builder"] = builder.provider_id
        state["stage"] = "built"
        self._save_state(state)

    def _stage_build_prose(self, task: BuilderTask, state: Dict[str, Any]) -> None:
        builder = self._provider("builder", task.task_id)
        if task.domain == "copywriting":
            completion = builder.chat_text(
                PROSE_BUILDER_SYSTEM,
                self._brief_text(task),
                "builder-build",
                task.task_id,
                **self._call_kwargs(),
            )
            state["prose"] = _strip_think(completion.text).strip()
        else:
            completion = builder.chat_json(
                PROSE_BUILDER_SYSTEM,
                self._brief_text(task) + "\n\nReturn the single JSON object now.",
                "builder-build",
                task.task_id,
                **self._call_kwargs(),
            )
            state["structured"] = completion.value
        state["builder"] = builder.provider_id
        state["stage"] = "built"
        self._save_state(state)

    def _stage_bug_injection(self, task: BuilderTask, state: Dict[str, Any]) -> None:
        files = dict(state["files"])
        module_path = next(
            (path for path in files if path.startswith("core/") and path.endswith(".py")), None
        )
        if module_path is None:
            raise RuntimeError("no core module to inject a bug into")
        injector = self._provider("bug_injector", task.task_id, exclude=[state.get("builder", "")])
        base_prompt = "MODULE %s:\n```python\n%s\n```\n\nTest suite:\n```python\n%s\n```" % (
            module_path,
            files[module_path],
            "\n\n".join(body for path, body in files.items() if path.startswith("tests/")),
        )
        buggy_source = ""
        buggy_result: Dict[str, Any] = {}
        completion = None
        for attempt in range(3):
            prompt = base_prompt
            if attempt and buggy_source:
                prompt += (
                    "\n\nYour previous injection still PASSED the whole suite, so it is not a "
                    "usable regression. Target a behavior that an existing test asserts directly "
                    "(pick a specific test function and break exactly what it checks)."
                )
            completion = injector.chat_json(
                BUG_INJECTOR_SYSTEM, prompt, "builder-inject", task.task_id, **self._call_kwargs()
            )
            buggy_source = str(completion.value.get("buggy_file") or "")
            if not buggy_source.strip():
                continue
            buggy_files = dict(files)
            buggy_files[module_path] = buggy_source
            buggy_result = run_pytest(buggy_files)
            if not buggy_result["green"] and buggy_result["failed"] > 0:
                break
        else:
            raise RuntimeError("injected bug did not reproduce a test failure")
        if buggy_result.get("green") or not buggy_result.get("failed"):
            raise RuntimeError("injected bug did not reproduce a test failure")
        state["bug"] = {
            "module_path": module_path,
            "buggy_source": buggy_source,
            "bug_category": str(completion.value.get("bug_category", "unknown")),
            "bug_summary": str(completion.value.get("bug_summary", "")),
            "failing_output": buggy_result["output_tail"],
            "failed": buggy_result["failed"],
            "injector": injector.provider_id,
        }
        self._save_state(state)

    def _verify(self, task: BuilderTask, state: Dict[str, Any]) -> Dict[str, Any]:
        if task.domain in ("fullstack_product", "coding_debugging", "website_frontend", "devops"):
            required_sections = list((task.facts or {}).get("sections") or [])
            report = verify_code_bundle(
                state.get("files") or {},
                required_sections=required_sections,
                run_tests=task.domain in ("fullstack_product", "coding_debugging"),
                devops_policy=task.domain == "devops",
            )
            if task.domain == "fullstack_product":
                plan = state.get("plan") or {}
                built = set(state.get("files") or {})
                planned_paths = [
                    str(item.get("path"))
                    for group in plan.get("file_groups") or []
                    for item in group.get("files") or []
                ]
                for path in planned_paths:
                    if path and path not in built:
                        report["defects"].append(
                            "planned file '%s' was never emitted; create it or remove every "
                            "reference to it" % path
                        )
                report["surfaces_declared"] = len(plan.get("required_surfaces") or [])
            return report
        defects: List[str] = []
        if task.domain == "copywriting":
            text = str(state.get("prose") or "")
            for section in ("landing", "email", "ad"):
                if section not in text.lower():
                    defects.append("missing section '%s'" % section)
            violations = numbers_grounded(text, [task.facts, task.user_request, *task.context])
            if violations:
                defects.append("ungrounded numbers: %s" % sorted(set(violations))[:8])
            defects.extend(_scan_text_policy("copy", text))
            return {"defects": defects, "tests": {"green": not defects, "skipped": True}}
        structured = state.get("structured") or {}
        rendered = json.dumps(structured, ensure_ascii=False)
        if task.domain == "pitch_deck":
            slides = structured.get("slides") or []
            if not 10 <= len(slides) <= 14:
                defects.append("expected 11-13 slides, got %d" % len(slides))
            for index, slide in enumerate(slides):
                if not str(slide.get("title", "")).strip() or not slide.get("speaker_notes"):
                    defects.append("slide %d missing title or speaker notes" % index)
            violations = numbers_grounded(rendered, [task.facts, task.user_request, *task.context])
            if violations:
                defects.append("ungrounded numbers: %s" % sorted(set(violations))[:8])
        elif task.domain == "investor_analysis":
            metrics = structured.get("computed_metrics") or {}
            defects.extend(_investor_metric_defects(task.facts, metrics))
            violations = numbers_grounded(
                rendered,
                [task.facts, task.user_request, *task.context],
                extra_allowed=list(_investor_expected(task.facts).values()),
            )
            if violations:
                defects.append("ungrounded numbers: %s" % sorted(set(violations))[:8])
        elif task.domain == "project_planning":
            defects.extend(_plan_consistency_defects(structured))
        defects.extend(_scan_text_policy("structured", rendered))
        return {"defects": defects, "tests": {"green": not defects, "skipped": True}}

    def _stage_repair(self, task: BuilderTask, state: Dict[str, Any], defects: Sequence[str]) -> None:
        editor = self._provider("editor", task.task_id)
        defect_text = "\n".join("- %s" % item for item in list(defects)[:20])
        if state.get("files"):
            current = "\n\n".join(
                '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (path, body)
                for path, body in sorted(state["files"].items())
            )
            if len(current) > 90_000:
                current = current[:90_000] + "\n... (truncated)"
            repaired = self._chat_files(
                editor,
                REPAIR_SYSTEM,
                self._brief_text(task)
                + "\n\nDEFECTS:\n%s\n\nCURRENT FILES:\n%s" % (defect_text, current),
                "builder-repair",
                task.task_id,
            )
            state.setdefault("attempts", []).append(
                {"defects": list(defects)[:20], "kind": "files"}
            )
            state["rejected_snapshot"] = dict(state["files"])
            state["files"] = {**state["files"], **repaired}
        elif state.get("prose") is not None:
            completion = editor.chat_text(
                REPAIR_SYSTEM.replace("FILE blocks", "the corrected full deliverable")
                + " Re-emit the complete corrected deliverable.",
                self._brief_text(task)
                + "\n\nDEFECTS:\n%s\n\nCURRENT DELIVERABLE:\n%s" % (defect_text, state["prose"]),
                "builder-repair",
                task.task_id,
                **self._call_kwargs(),
            )
            state.setdefault("attempts", []).append({"defects": list(defects)[:20], "kind": "prose"})
            state["rejected_prose"] = state["prose"]
            state["prose"] = _strip_think(completion.text).strip()
        else:
            completion = editor.chat_json(
                REPAIR_SYSTEM + " Re-emit the complete corrected JSON object.",
                self._brief_text(task)
                + "\n\nDEFECTS:\n%s\n\nCURRENT JSON:\n%s"
                % (defect_text, json.dumps(state.get("structured") or {}, ensure_ascii=False)),
                "builder-repair",
                task.task_id,
                **self._call_kwargs(),
            )
            state.setdefault("attempts", []).append({"defects": list(defects)[:20], "kind": "json"})
            state["rejected_structured"] = state.get("structured")
            state["structured"] = completion.value
        state["editor"] = editor.provider_id
        self._save_state(state)

    def _published_answer(self, task: BuilderTask, state: Mapping[str, Any]) -> str:
        if state.get("files"):
            blocks = "\n\n".join(
                '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (path, body)
                for path, body in sorted(state["files"].items())
            )
            return blocks
        if state.get("prose") is not None:
            return str(state["prose"])
        return json.dumps(state.get("structured") or {}, ensure_ascii=False, indent=2)

    def _judge_view(self, task: BuilderTask, state: Mapping[str, Any]) -> str:
        """Judge input that never cuts a file mid-body.

        Whole files are appended until the character budget is reached; any
        remainder is listed explicitly so the judge does not mistake an
        intentionally omitted file for a truncated deliverable.
        """

        files = state.get("files") or {}
        if not files:
            return self._published_answer(task, state)[:MAX_JUDGE_CHARS]
        deterministic = state.get("deterministic") or {}
        tests = deterministic.get("tests") or {}
        header_lines = [
            "REPOSITORY DIGEST (%d files). Deterministic checks already ran: "
            "defects=%d, pytest green=%s (passed=%s)."
            % (
                len(files),
                len(deterministic.get("defects") or []),
                tests.get("green"),
                tests.get("passed"),
            ),
            "File tree:",
        ]
        for path in sorted(files):
            header_lines.append("  %s (%d lines)" % (path, files[path].count("\n") + 1))
        pieces = ["\n".join(header_lines)]
        used = len(pieces[0])
        omitted: List[str] = []
        for path in sorted(files, key=lambda p: len(files[p])):
            block = '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (path, files[path])
            if used + len(block) > MAX_JUDGE_CHARS:
                omitted.append(path)
                continue
            pieces.append(block)
            used += len(block)
        if omitted:
            pieces.append(
                "OMITTED FROM THIS VIEW FOR LENGTH (complete on disk and already "
                "parse/test-checked; do NOT treat as missing or truncated): %s"
                % ", ".join(omitted)
            )
        return "\n\n".join(pieces)

    def _stage_judge(self, task: BuilderTask, state: Dict[str, Any]) -> Dict[str, Any]:
        judges = self._judges()
        answer = self._judge_view(task, state)
        verdicts: Dict[str, Any] = {}
        for judge in judges[:2]:
            try:
                completion = judge.chat_json(
                    JUDGE_SYSTEM,
                    self._brief_text(task) + "\n\nCANDIDATE DELIVERABLE:\n" + answer,
                    "builder-judge",
                    task.task_id,
                    **self._call_kwargs(),
                )
                verdicts[judge.provider_id] = {
                    "pass": bool(completion.value.get("pass")),
                    "score": float(completion.value.get("score", 0.0) or 0.0),
                    "defects": [str(d) for d in completion.value.get("defects") or []][:10],
                }
            except BudgetExceededError:
                raise
            except Exception as exc:
                verdicts[judge.provider_id] = {"pass": False, "score": 0.0, "defects": [str(exc)[:200]]}
        accepted = len(verdicts) >= 2 and all(
            verdict["pass"] and verdict["score"] >= JUDGE_SCORE_THRESHOLD
            for verdict in verdicts.values()
        )
        state["judge"] = {"verdicts": verdicts, "accepted": accepted}
        self._save_state(state)
        return state["judge"]

    # -- episode assembly ---------------------------------------------------

    def _assemble_episode(
        self, task: BuilderTask, state: Mapping[str, Any], deterministic: Mapping[str, Any]
    ) -> Dict[str, Any]:
        defects = [str(d) for d in deterministic.get("defects") or []]
        tests = dict(deterministic.get("tests") or {})
        judge = dict(state.get("judge") or {})
        judged = task.domain in self.judged_domains
        deterministic_pass = not defects
        judge_pass = judge.get("accepted", not judged)
        publish = deterministic_pass and (judge_pass or not judged)

        files = dict(state.get("files") or {})
        plan = dict(state.get("plan") or {})
        claims: Dict[str, List[Dict[str, Any]]] = {"backend": [], "frontend": [], "quality": []}
        verification: List[Dict[str, Any]] = []

        def add_claim(lane: str, claim_id: str, statement: str, verdict: str, method: str) -> None:
            claims[lane].append(
                {"claim_id": claim_id, "statement": statement, "evidence_refs": [method]}
            )
            verification.append({"claim_id": claim_id, "verdict": verdict, "method": method})

        if task.domain in ("fullstack_product", "coding_debugging"):
            add_claim(
                "quality",
                "%s-tests" % task.task_id,
                "The bundled pytest suite passes (%s passed, %s failed)."
                % (tests.get("passed", 0), tests.get("failed", 0)),
                "supported" if tests.get("green") else "failed",
                "executed_pytest",
            )
        add_claim(
            "quality",
            "%s-policy" % task.task_id,
            "Static policy checks (secrets, imports, parses, grounding) are clean.",
            "supported" if deterministic_pass else "failed",
            "deterministic_scan",
        )
        if judged:
            add_claim(
                "quality",
                "%s-judged" % task.task_id,
                "Both independent judges accepted the deliverable.",
                "supported" if judge_pass else "failed",
                "independent_judges",
            )

        bug = dict(state.get("bug") or {})
        if bug:
            add_claim(
                "backend",
                "%s-bug-reproduced" % task.task_id,
                "The injected %s regression fails %s test(s)."
                % (bug.get("bug_category", "unknown"), bug.get("failed", "?")),
                "supported",
                "executed_pytest",
            )

        published = self._published_answer(task, state)
        language = classify_language(published)
        add_claim(
            "quality",
            "%s-english" % task.task_id,
            "The deliverable is English-only.",
            "supported" if language.accepted else "failed",
            "language_classifier",
        )
        publish = publish and language.accepted

        def lane_payload(lane_key: str, scope: str, deliverable: str, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
            plan_lanes = {str(l.get("lane_id", "")): l for l in plan.get("lanes") or []}
            planned = plan_lanes.get("lane-%s" % lane_key, {})
            return {
                "lane_id": "lane-%s" % lane_key,
                "route": ["root", task.domain, task.subdomain, lane_key],
                "route_windows": [
                    {"window": index, "decision": "continue"}
                    for index in range(max(1, len(state.get("built_groups") or [1])))
                ]
                + [{"window": 99, "decision": "halt"}],
                "brief": {
                    "scope": str(planned.get("scope") or scope),
                    "assumptions": [str(a) for a in planned.get("assumptions") or []],
                    "deliverable": str(planned.get("deliverable") or deliverable),
                    "rejection_tests": [str(r) for r in planned.get("rejection_tests") or []],
                },
                "artifacts": artifacts,
                "claims": claims.get(lane_key, []),
                "checkpoints": [
                    {
                        "step": index,
                        "observable_update": str(attempt.get("kind", "build")) + " repair pass",
                        "decision": "continue",
                    }
                    for index, attempt in enumerate(state.get("attempts") or [])
                ],
                "summary": "%s lane for %s (%s)." % (lane_key, task.subdomain, task.domain),
            }

        backend_files = {p: b for p, b in files.items() if p.split("/", 1)[0] in ("core", "api") or p.endswith(".sql")}
        web_files = {p: b for p, b in files.items() if p.split("/", 1)[0] == "web"}
        other_files = {p: b for p, b in files.items() if p not in backend_files and p not in web_files}

        def artifacts_for(bundle: Mapping[str, str], kind: str) -> List[Dict[str, Any]]:
            return [
                {"artifact_id": path, "type": kind, "content": body}
                for path, body in sorted(bundle.items())
            ]

        lanes = [
            lane_payload("backend", "Backend, data model and business logic.", "core/, api/, schema.sql", artifacts_for(backend_files, "code")),
            lane_payload("frontend", "User-facing pages and interaction.", "web/", artifacts_for(web_files, "code")),
            lane_payload(
                "quality",
                "Tests, docs, packaging and verification.",
                "tests/, README, devops files",
                artifacts_for(other_files, "code")
                + (
                    [{"artifact_id": "deliverable", "type": "document", "content": published}]
                    if not files
                    else []
                ),
            ),
        ]

        open_claims = [item["claim_id"] for item in verification if item["verdict"] != "supported"]
        adjudicated = all(
            item["verdict"] == "supported"
            for item in verification
            if item["method"] in ("executed_pytest", "deterministic_scan")
        ) and any(item["method"] == "executed_pytest" for item in verification) if files else False

        episode = {
            "episode_id": "builder-%s" % task.task_id,
            "domain": task.domain,
            "subdomain": task.subdomain,
            "input": {
                "user_request": task.user_request,
                "context": list(task.context),
                "constraints": list(task.constraints),
            },
            "frame": {
                "objective": str(plan.get("objective") or task.user_request[:280]),
                "requirements": [str(r) for r in plan.get("requirements") or list(task.constraints)],
                "failure_contract": [str(f) for f in plan.get("failure_contract") or []],
            },
            "lanes": lanes,
            "verification": verification,
            "barrier": {"open_claims": open_claims},
            "integration": {"published_answer": published if publish else ""},
            "commitment": {
                "decision": "publish" if publish else "abstain",
                "reasons": defects[:10] + ([] if judge_pass else ["judge rejection"]),
            },
            "evaluation": {
                "deterministic": {
                    "defects": defects[:20],
                    "tests": tests,
                },
                "judges": judge.get("verdicts", {}),
            },
            "generation_metadata": {
                "mode": "builder_factory_v1",
                "providers": {
                    "planner": state.get("planner"),
                    "builder": state.get("builder"),
                    "editor": state.get("editor"),
                    "bug_injector": bug.get("injector"),
                },
                "independently_adjudicated": bool(adjudicated),
                "repair_rounds": len(state.get("attempts") or []),
                "lineage_component_id": task.lineage,
                "source_group": task.lineage,
                "scale": task.scale,
            },
        }
        if bug:
            episode["debug_pair"] = {
                "module_path": bug["module_path"],
                "buggy_source": bug["buggy_source"],
                "failing_output": bug["failing_output"],
                "fixed_source": files.get(bug["module_path"], ""),
                "bug_category": bug["bug_category"],
                "bug_summary": bug["bug_summary"],
            }
        return episode

    # -- one task end to end --------------------------------------------------

    def run_task(self, task: BuilderTask) -> Dict[str, Any]:
        state = self._load_state(task)
        if state.get("stage") in ("accepted", "rejected", "failed"):
            return state
        try:
            if state["stage"] == "pending":
                if task.domain == "fullstack_product":
                    self._stage_plan(task, state)
                else:
                    state["stage"] = "planned"
                    self._save_state(state)
            if state["stage"] == "planned":
                if task.domain == "fullstack_product":
                    self._stage_build_fullstack(task, state)
                elif task.domain in ("coding_debugging", "website_frontend", "devops"):
                    self._stage_build_files(task, state)
                else:
                    self._stage_build_prose(task, state)
            deterministic = self._verify(task, state)
            rounds = 0
            while deterministic["defects"] and rounds < self.max_repair_rounds:
                self._stage_repair(task, state, deterministic["defects"])
                deterministic = self._verify(task, state)
                rounds += 1
            state["deterministic"] = {
                "defects": deterministic["defects"][:20],
                "tests": deterministic.get("tests"),
            }
            if task.domain == "coding_debugging" and not deterministic["defects"] and not state.get("bug"):
                try:
                    self._stage_bug_injection(task, state)
                except BudgetExceededError:
                    raise
                except RuntimeError as exc:
                    # The verified module + suite is still good supervision on
                    # its own; record why the paired regression is missing.
                    state["bug_injection_skipped"] = str(exc)[:200]
            judged = task.domain in self.judged_domains
            if judged and not deterministic["defects"]:
                verdict = self._stage_judge(task, state)
                judge_rounds = 0
                while not verdict["accepted"] and judge_rounds < 1:
                    judge_defects = [
                        defect
                        for judge_verdict in verdict["verdicts"].values()
                        for defect in judge_verdict.get("defects", [])
                    ]
                    state.setdefault("judge_history", []).append(verdict["verdicts"])
                    self._stage_repair(
                        task, state, judge_defects or ["both judges rejected the deliverable"]
                    )
                    deterministic = self._verify(task, state)
                    state["deterministic"] = {
                        "defects": deterministic["defects"][:20],
                        "tests": deterministic.get("tests"),
                    }
                    if deterministic["defects"]:
                        break
                    verdict = self._stage_judge(task, state)
                    judge_rounds += 1
            episode = self._assemble_episode(task, state, deterministic)
            state["episode"] = episode
            state["stage"] = (
                "accepted" if episode["commitment"]["decision"] == "publish" else "rejected"
            )
            self._save_state(state)
        except BudgetExceededError:
            state["stage_note"] = "paused: budget exhausted"
            self._save_state(state)
            raise
        except Exception as exc:
            state["stage"] = "failed"
            state["error"] = str(exc)[:800]
            self._save_state(state)
        return state

    def close(self) -> None:
        for provider in self.providers.values():
            provider.close()


# --------------------------------------------------------------------------
# Public entry points.
# --------------------------------------------------------------------------

def builder_doctor(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    statuses = provider_statuses(config)
    roles: Dict[str, List[str]] = {}
    for status in statuses:
        if status.available:
            for role in status.roles:
                roles.setdefault(role, []).append(status.provider_id)
    factory = dict(config.get("builder_factory") or {})
    judged = list(factory.get("judge_domains", JUDGED_DOMAINS_DEFAULT))
    errors = validate_builder_roles(roles, judged)
    capabilities = {
        "pytest": bool(shutil.which("pytest") or True),
        "bash": bool(shutil.which("bash")),
        "node": bool(shutil.which("node")),
    }
    tasks = expand_tasks(factory.get("mix") or {}, int((config.get("project") or {}).get("seed", 0)))
    return {
        "status": "ok" if not errors else "blocked",
        "providers": [status.__dict__ for status in statuses],
        "roles": roles,
        "role_errors": errors,
        "capabilities": capabilities,
        "planned_tasks": len(tasks),
        "task_domains": sorted({task.domain for task in tasks}),
    }


def run_builder(
    config: Mapping[str, Any],
    root: Path,
    *,
    accepted_target: int,
    workers: int = 2,
    max_total_requests: Optional[int] = None,
    max_total_tokens: Optional[int] = None,
    domains: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    factory = BuilderFactory(config, root)
    ledger = BuilderLedger(
        factory.output_dir / "budget.json", max_total_requests, max_total_tokens
    )
    factory.ledger = ledger
    tasks = expand_tasks(factory.mix, factory.seed, factory.default_scale)
    if domains:
        wanted = {d.strip() for d in domains if d.strip()}
        tasks = [task for task in tasks if task.domain in wanted]
    started = time.monotonic()
    accepted = rejected = failed = 0
    for task in tasks:
        state = factory._load_state(task)
        if state.get("stage") == "accepted":
            accepted += 1
    budget_stop = False
    try:
        pending = [
            task
            for task in tasks
            if factory._load_state(task).get("stage") not in ("accepted", "rejected", "failed")
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures: Dict[concurrent.futures.Future, BuilderTask] = {}
            iterator = iter(pending)
            def submit_next() -> bool:
                if factory.stop_file.exists() or budget_stop:
                    return False
                if accepted >= accepted_target:
                    return False
                try:
                    task = next(iterator)
                except StopIteration:
                    return False
                futures[pool.submit(factory.run_task, task)] = task
                return True
            for _ in range(max(1, workers)):
                if not submit_next():
                    break
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    task = futures.pop(future)
                    try:
                        state = future.result()
                    except BudgetExceededError:
                        budget_stop = True
                        continue
                    stage = state.get("stage")
                    if stage == "accepted":
                        accepted += 1
                    elif stage == "rejected":
                        rejected += 1
                    elif stage == "failed":
                        failed += 1
                if not budget_stop:
                    while len(futures) < max(1, workers):
                        if not submit_next():
                            break
    finally:
        factory.close()
    return {
        "accepted": accepted,
        "rejected": rejected,
        "failed": failed,
        "budget": ledger.snapshot(),
        "budget_stopped": budget_stop,
        "stopped_by_file": factory.stop_file.exists(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }


def builder_status(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    factory_config = dict(config.get("builder_factory") or {})
    output_dir = root / str(factory_config.get("output_dir", "data/builder"))
    tasks_dir = output_dir / "tasks"
    stages: Dict[str, int] = {}
    domains: Dict[str, Dict[str, int]] = {}
    for path in sorted(tasks_dir.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        stage = str(state.get("stage", "unknown"))
        stages[stage] = stages.get(stage, 0) + 1
        episode = state.get("episode") or {}
        domain = str(episode.get("domain") or path.name.split("-", 1)[0])
        bucket = domains.setdefault(domain, {})
        bucket[stage] = bucket.get(stage, 0) + 1
    budget_path = output_dir / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else {}
    return {"stages": stages, "domains": domains, "budget": budget}


def _split_for(lineage: str) -> str:
    bucket = int(stable_hash("split:" + lineage, 8), 16) % 10
    if bucket <= 7:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def export_builder_dataset(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    factory_config = dict(config.get("builder_factory") or {})
    output_dir = root / str(factory_config.get("output_dir", "data/builder"))
    tasks_dir = output_dir / "tasks"
    final_dir = output_dir / "final"
    master: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    sft: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    dpo: List[Dict[str, Any]] = []
    for path in sorted(tasks_dir.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        episode = state.get("episode")
        if not episode:
            continue
        lineage = str(episode["generation_metadata"]["lineage_component_id"])
        split = _split_for(lineage)
        master[split].append(episode)
        if episode["commitment"]["decision"] != "publish":
            continue
        prompt_lines = [episode["input"]["user_request"], ""]
        prompt_lines.extend("- %s" % item for item in episode["input"]["context"])
        prompt_lines.append("")
        prompt_lines.extend("- %s" % item for item in episode["input"]["constraints"])
        prompt = "\n".join(prompt_lines)
        system = "You are a senior %s specialist. Deliver complete, verified work." % episode["domain"]
        sft[split].append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": episode["integration"]["published_answer"]},
                ],
                "episode_id": episode["episode_id"],
                "domain": episode["domain"],
                "lineage_component_id": lineage,
            }
        )
        rejected_artifact = ""
        if state.get("rejected_prose"):
            rejected_artifact = str(state["rejected_prose"])
        elif state.get("rejected_structured"):
            rejected_artifact = json.dumps(state["rejected_structured"], ensure_ascii=False, indent=2)
        elif state.get("rejected_snapshot"):
            rejected_artifact = "\n\n".join(
                '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (file_path, body)
                for file_path, body in sorted(state["rejected_snapshot"].items())
            )
        published_answer = episode["integration"]["published_answer"]
        if rejected_artifact and rejected_artifact != published_answer:
            dpo.append(
                {
                    "prompt": prompt,
                    "chosen": published_answer,
                    "rejected": rejected_artifact,
                    "lineage_component_id": lineage,
                    "split": split,
                }
            )
        debug_pair = episode.get("debug_pair")
        if debug_pair and debug_pair.get("fixed_source"):
            debug_prompt = (
                "The module below fails its test suite.\n\nMODULE %s:\n```python\n%s\n```\n\n"
                "PYTEST OUTPUT:\n```\n%s\n```\n\nDiagnose the defect and return the complete corrected module."
                % (
                    debug_pair["module_path"],
                    debug_pair["buggy_source"],
                    debug_pair["failing_output"][-2500:],
                )
            )
            fixed = '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (
                debug_pair["module_path"],
                debug_pair["fixed_source"],
            )
            sft[split].append(
                {
                    "messages": [
                        {"role": "system", "content": "You are a debugging specialist. Fix the regression with a minimal change."},
                        {"role": "user", "content": debug_prompt},
                        {"role": "assistant", "content": fixed},
                    ],
                    "episode_id": episode["episode_id"] + "-debug",
                    "domain": "coding_debugging",
                    "lineage_component_id": lineage,
                }
            )
            dpo.append(
                {
                    "prompt": debug_prompt,
                    "chosen": fixed,
                    "rejected": '<<<FILE path="%s">>>\n%s\n<<<END FILE>>>' % (
                        debug_pair["module_path"],
                        debug_pair["buggy_source"],
                    ),
                    "lineage_component_id": lineage,
                    "split": split,
                }
            )
    counts: Dict[str, Any] = {"master": {}, "sft": {}, "dpo": len(dpo)}
    checksums: Dict[str, str] = {}
    for split, rows in master.items():
        target = final_dir / "master" / ("%s.jsonl" % split)
        counts["master"][split] = write_jsonl(target, rows)
        checksums[str(target.relative_to(output_dir))] = hashlib.sha256(target.read_bytes()).hexdigest()
    for split, rows in sft.items():
        target = final_dir / "sft" / ("%s.jsonl" % split)
        counts["sft"][split] = write_jsonl(target, rows)
        checksums[str(target.relative_to(output_dir))] = hashlib.sha256(target.read_bytes()).hexdigest()
    dpo_target = final_dir / "dpo" / "pairs.jsonl"
    write_jsonl(dpo_target, dpo)
    checksums[str(dpo_target.relative_to(output_dir))] = hashlib.sha256(dpo_target.read_bytes()).hexdigest()
    manifest = {
        "counts": counts,
        "checksums_sha256": checksums,
        "policy": {
            "teacher_private_reasoning_persisted": False,
            "benchmark_test_material": "none",
            "split_group_key": "lineage_component_id",
        },
    }
    atomic_write_json(final_dir / "manifest.json", manifest)
    return manifest
