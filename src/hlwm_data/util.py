from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def sanitize_unicode(text: str) -> str:
    """Return valid Unicode, preserving pairs and replacing lone surrogates."""
    value = str(text)
    if not any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        return value
    cleaned: List[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF and index + 1 < len(value):
            low = ord(value[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                cleaned.append(chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)))
                index += 2
                continue
        if 0xD800 <= codepoint <= 0xDFFF:
            cleaned.append("\N{REPLACEMENT CHARACTER}")
        else:
            cleaned.append(value[index])
        index += 1
    return "".join(cleaned)


def sanitize_json_value(value: Any) -> Any:
    """Recursively remove invalid Unicode from values before hashing or JSON output."""
    if isinstance(value, str):
        return sanitize_unicode(value)
    if isinstance(value, dict):
        return {sanitize_unicode(str(key)): sanitize_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(child) for child in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(child) for child in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(sanitize_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, length: int = 24) -> str:
    payload = sanitize_unicode(value) if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", sanitize_unicode(text)).strip()


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_json_value(value), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_number, exc))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(sanitize_json_value(value), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(sanitize_json_value(row), ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count


def resolve_env(value: Any, env_name: str, cast=None) -> Any:
    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return value
    return cast(raw) if cast else raw
