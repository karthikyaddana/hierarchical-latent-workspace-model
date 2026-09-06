from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


PROMPT_KEYS = ("question", "prompt", "text", "problem", "task", "description")


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", value.lower()))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if len(value.strip()) >= 40:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _shingles(value: str, size: int = 8) -> Set[str]:
    tokens = _normalize(value).split()
    if len(tokens) < size:
        return set()
    return {
        hashlib.sha256(" ".join(tokens[index : index + size]).encode("utf-8")).hexdigest()[:16]
        for index in range(len(tokens) - size + 1)
    }


def load_benchmark_prompts(root: Path) -> List[Tuple[str, str]]:
    prompts: List[Tuple[str, str]] = []
    if not root.exists():
        return prompts
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return prompts
    for path in sorted(root.rglob("*.parquet")):
        table = parquet.read_table(str(path))
        for index, row in enumerate(table.to_pylist()):
            values = [row.get(key) for key in PROMPT_KEYS if isinstance(row.get(key), str)]
            if not values:
                values = list(_strings(row))[:1]
            for value in values:
                normalized = _normalize(value)
                if len(normalized.split()) >= 8:
                    prompts.append(("%s:%d" % (path.relative_to(root), index), normalized))
    return prompts


def contamination_report(
    episodes: Sequence[Dict[str, Any]], benchmark_root: Path, coverage_threshold: float = 0.45
) -> Dict[str, Any]:
    prompts = load_benchmark_prompts(benchmark_root)
    prompt_shingles: List[Set[str]] = []
    inverted: Dict[str, List[int]] = defaultdict(list)
    for index, (_, prompt) in enumerate(prompts):
        shingles = _shingles(prompt)
        prompt_shingles.append(shingles)
        for shingle in shingles:
            inverted[shingle].append(index)
    flags = []
    for episode in episodes:
        text = "\n".join(_strings({"input": episode.get("input"), "answer": episode.get("integration", {})}))
        normalized = _normalize(text)
        episode_shingles = _shingles(normalized)
        candidates: Counter = Counter()
        for shingle in episode_shingles:
            for prompt_index in inverted.get(shingle, []):
                candidates[prompt_index] += 1
        for prompt_index, overlap in candidates.most_common(10):
            prompt = prompts[prompt_index][1]
            denominator = max(1, len(prompt_shingles[prompt_index]))
            coverage = overlap / denominator
            exact = prompt in normalized
            if exact or (overlap >= 4 and coverage >= coverage_threshold):
                flags.append({
                    "episode_id": episode.get("episode_id"),
                    "benchmark_item": prompts[prompt_index][0],
                    "exact": exact,
                    "matching_shingles": overlap,
                    "benchmark_coverage": round(coverage, 4),
                })
    return {
        "benchmark_root": str(benchmark_root),
        "benchmark_prompts": len(prompts),
        "episodes_checked": len(episodes),
        "flags": flags,
        "clean": not flags,
    }
