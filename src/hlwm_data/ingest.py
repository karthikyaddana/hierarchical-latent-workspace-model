from __future__ import annotations

import csv
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from .config import load_yaml
from .language import classify_language, contains_blocked_script
from .util import iter_jsonl, normalize_text, sanitize_unicode, stable_hash, write_jsonl


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".sql",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".tf", ".css", ".scss",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".epub", ".pdf", ".docx", ".html", ".htm", ".json", ".jsonl", ".csv", ".ipynb", ".parquet"
}
DEFAULT_EXCLUDED_PATH_PARTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build",
    "translations", "translated_images",
}


@dataclass
class SourceDocument:
    text: str
    path: Path
    part: str
    metadata: Dict[str, Any]


def _clean_markup(value: bytes) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return "\n\n".join(
        normalize_text(part) for part in soup.get_text("\n").splitlines() if normalize_text(part)
    )


def _extract_epub(path: Path) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(str(path)) as archive:
        names = set(archive.namelist())
        ordered_names: List[str] = []
        try:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                element.attrib["full-path"]
                for element in container.iter()
                if element.tag.endswith("rootfile")
            )
            package = ElementTree.fromstring(archive.read(rootfile))
            base = posixpath.dirname(rootfile)
            manifest: Dict[str, str] = {}
            for element in package.iter():
                if element.tag.endswith("item") and "id" in element.attrib and "href" in element.attrib:
                    manifest[element.attrib["id"]] = posixpath.normpath(
                        posixpath.join(base, element.attrib["href"])
                    )
            for element in package.iter():
                if element.tag.endswith("itemref") and element.attrib.get("idref") in manifest:
                    ordered_names.append(manifest[element.attrib["idref"]])
        except (KeyError, StopIteration, ElementTree.ParseError):
            ordered_names = []
        if not ordered_names:
            ordered_names = sorted(
                name for name in names if Path(name).suffix.lower() in {".html", ".htm", ".xhtml"}
            )
        parts = []
        for name in ordered_names:
            if name not in names:
                continue
            text = _clean_markup(archive.read(name))
            if text:
                parts.append((name, text))
        return parts


def _extract_docx(path: Path) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(str(path)) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in document.iter():
        if not paragraph.tag.endswith("}p"):
            continue
        text = "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))
        if normalize_text(text):
            paragraphs.append(normalize_text(text))
    return [("document", "\n\n".join(paragraphs))]


def _extract_pdf(path: Path) -> List[Tuple[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Run: pip install -e .") from exc
    reader = PdfReader(str(path))
    parts = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if normalize_text(text):
            parts.append(("page-%d" % (index + 1), text))
    return parts


def _json_strings(value: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(value, str) and normalize_text(value):
        yield prefix or "value", value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = "%s.%s" % (prefix, key) if prefix else str(key)
            yield from _json_strings(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _json_strings(child, "%s[%d]" % (prefix, index))


def extract_path(path: Path, metadata: Dict[str, Any]) -> List[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []
    parts: List[Tuple[str, str]]
    if suffix in TEXT_EXTENSIONS:
        parts = [("document", path.read_text(encoding="utf-8", errors="replace"))]
    elif suffix in {".html", ".htm"}:
        parts = [("document", _clean_markup(path.read_bytes()))]
    elif suffix == ".epub":
        parts = _extract_epub(path)
    elif suffix == ".docx":
        parts = _extract_docx(path)
    elif suffix == ".pdf":
        parts = _extract_pdf(path)
    elif suffix == ".jsonl":
        parts = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    parts.append(("row-%d" % index, json.dumps(value, ensure_ascii=False, indent=2)))
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        extracted = list(_json_strings(value))
        parts = extracted or [("document", json.dumps(value, ensure_ascii=False, indent=2))]
    elif suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        for index, cell in enumerate(notebook.get("cells", [])):
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            if normalize_text(text):
                parts.append(("%s-%d" % (cell.get("cell_type", "cell"), index), text))
    elif suffix == ".csv":
        parts = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            for index, row in enumerate(csv.DictReader(handle, dialect=dialect), 1):
                parts.append(("row-%d" % index, json.dumps(row, ensure_ascii=False)))
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet support requires pyarrow. Run: pip install pyarrow") from exc
        parts = [
            ("row-%d" % index, json.dumps(row, ensure_ascii=False, default=str))
            for index, row in enumerate(parquet.read_table(str(path)).to_pylist(), 1)
        ]
    else:
        parts = []
    return [
        SourceDocument(text=text, path=path, part=part, metadata=dict(metadata))
        for part, text in parts
        if normalize_text(text)
    ]


def _expand_entry(
    path: Path, excluded_extensions: Sequence[str] = (), max_file_bytes: Optional[int] = None
) -> Iterable[Path]:
    excluded = {str(item).lower() for item in excluded_extensions}
    if path.is_file():
        if path.suffix.lower() not in excluded and (max_file_bytes is None or path.stat().st_size <= max_file_bytes):
            yield path
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            relative_parts = set(child.relative_to(path).parts)
            if relative_parts & DEFAULT_EXCLUDED_PATH_PARTS:
                continue
            if (
                child.is_file()
                and child.suffix.lower() in SUPPORTED_EXTENSIONS
                and child.suffix.lower() not in excluded
                and (max_file_bytes is None or child.stat().st_size <= max_file_bytes)
            ):
                yield child
    else:
        raise FileNotFoundError(path)


def _paragraph_chunks(text: str, target_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: List[Tuple[int, int, str]] = []
    current: List[str] = []
    current_length = 0
    cursor = 0
    start = 0
    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 2 > target_chars:
            value = "\n\n".join(current)
            end = start + len(value)
            chunks.append((start, end, value))
            tail = value[-overlap_chars:] if overlap_chars else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_length = sum(len(item) for item in current) + 2 * max(0, len(current) - 1)
            start = max(cursor - len(tail), 0)
        else:
            if not current:
                start = cursor
            current.append(paragraph)
            current_length += len(paragraph) + (2 if len(current) > 1 else 0)
        cursor += len(paragraph) + 2
    if current:
        value = "\n\n".join(current)
        chunks.append((start, start + len(value), value))
    return chunks


def ingest_manifest(
    manifest_path: Path,
    output_path: Path,
    target_chars: int = 4200,
    overlap_chars: int = 400,
    min_chunk_chars: int = 300,
    max_chunks_per_source_group: int = 0,
) -> Dict[str, Any]:
    manifest = load_yaml(manifest_path)
    entries = manifest.get("sources", [])
    if not isinstance(entries, list):
        raise ValueError("sources must be a list in %s" % manifest_path)
    language_policy = manifest.get("language_policy", {}) or {}
    allowed_languages = tuple(str(item).lower() for item in language_policy.get("allowed_languages", []))
    language_required = bool(language_policy.get("required", bool(allowed_languages)))
    minimum_language_confidence = float(language_policy.get("minimum_confidence", 0.78))
    if language_required and allowed_languages != ("en",):
        raise ValueError("This pipeline currently supports strict language filtering only for allowed_languages: [en]")
    rows: List[Dict[str, Any]] = []
    seen_chunk_hashes: Dict[str, str] = {}
    duplicates_removed = 0
    source_count = 0
    document_count = 0
    extraction_errors = 0
    extraction_error_samples: List[Dict[str, str]] = []
    non_english_sources_removed = 0
    non_english_chunks_removed = 0
    corrupted_unicode_chunks_removed = 0
    language_rejection_samples: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every source entry must be a mapping")
        if not entry.get("allowed_for_training", False):
            continue
        declared_language = entry.get("language")
        declared_languages = (
            {str(item).lower() for item in declared_language}
            if isinstance(declared_language, list)
            else {str(declared_language).lower()} if declared_language else set()
        )
        if language_required and declared_languages and "en" not in declared_languages:
            non_english_sources_removed += 1
            continue
        if not entry.get("license"):
            raise ValueError("Source %r is missing a license or usage-rights label" % entry.get("path"))
        required_metadata = ["title", "domain", "source_group", "version", "license_evidence"]
        missing_metadata = [key for key in required_metadata if not entry.get(key)]
        if missing_metadata:
            raise ValueError(
                "Source %r is missing required metadata: %s"
                % (entry.get("path"), ", ".join(missing_metadata))
            )
        path = (manifest_path.parent / str(entry["path"])).resolve()
        metadata = {key: value for key, value in entry.items() if key != "path"}
        metadata["source_path"] = str(path)
        metadata.setdefault("source_group", stable_hash(str(path), 16))
        metadata.setdefault("lineage_component_id", metadata["source_group"])
        is_github_repository = str(entry.get("url", "")).startswith("https://github.com/")
        default_repo_exclusions = [".csv", ".json", ".jsonl", ".parquet"] if is_github_repository else []
        excluded_extensions = entry.get("exclude_extensions", default_repo_exclusions)
        max_file_bytes = entry.get("max_file_bytes", 8 * 1024 * 1024 if is_github_repository else None)
        for concrete_path in _expand_entry(path, excluded_extensions, max_file_bytes):
            source_count += 1
            try:
                extracted_documents = extract_path(concrete_path, metadata)
            except Exception as exc:
                extraction_errors += 1
                if len(extraction_error_samples) < 20:
                    extraction_error_samples.append(
                        {"path": str(concrete_path), "error_type": type(exc).__name__, "error": str(exc)[:300]}
                    )
                continue
            for document in extracted_documents:
                document_count += 1
                lineage_component_id = str(metadata["lineage_component_id"])
                lineage_mode = str(metadata.get("lineage_mode", "source"))
                if lineage_mode == "part":
                    lineage_component_id = "%s-part-%s" % (
                        lineage_component_id,
                        stable_hash(document.part, 12),
                    )
                elif lineage_mode == "page_block":
                    match = re.match(r"page-(\d+)$", document.part)
                    if match:
                        block_size = max(1, int(metadata.get("lineage_block_size", 16)))
                        block = (int(match.group(1)) - 1) // block_size
                        lineage_component_id = "%s-pages-%04d" % (lineage_component_id, block)
                content_hash = stable_hash(document.text, 32)
                source_id = stable_hash(
                    {"path": str(concrete_path), "part": document.part, "content": content_hash}, 24
                )
                for index, (start, end, text) in enumerate(
                    _paragraph_chunks(document.text, target_chars, overlap_chars)
                ):
                    text = sanitize_unicode(text)
                    if "\N{REPLACEMENT CHARACTER}" in text:
                        corrupted_unicode_chunks_removed += 1
                        continue
                    if len(normalize_text(text)) < min_chunk_chars:
                        continue
                    language = None
                    language_confidence = None
                    language_reason = None
                    if language_required:
                        decision = classify_language(
                            text,
                            expected="en",
                            minimum_confidence=minimum_language_confidence,
                        )
                        if not decision.accepted:
                            non_english_chunks_removed += 1
                            if len(language_rejection_samples) < 20:
                                language_rejection_samples.append(
                                    {
                                        "path": str(concrete_path),
                                        "part": document.part,
                                        "detected_language": decision.language,
                                        "confidence": round(decision.confidence, 6),
                                        "reason": decision.reason,
                                    }
                                )
                            continue
                        language = decision.language
                        language_confidence = round(decision.confidence, 6)
                        language_reason = decision.reason
                    chunk_content_hash = stable_hash(normalize_text(text).lower(), 32)
                    if chunk_content_hash in seen_chunk_hashes:
                        duplicates_removed += 1
                        continue
                    seen_chunk_hashes[chunk_content_hash] = str(concrete_path)
                    rows.append(
                        {
                            "schema_version": "1.0",
                            "source_id": source_id,
                            "chunk_id": "%s-%04d" % (source_id, index),
                            "source_group": metadata["source_group"],
                            "lineage_component_id": lineage_component_id,
                            "domain": metadata.get("domain", "unassigned"),
                            "title": metadata.get("title") or concrete_path.name,
                            "author": metadata.get("author"),
                            "version": metadata.get("version"),
                            "license": metadata["license"],
                            "license_evidence": metadata["license_evidence"],
                            "data_role": metadata.get("data_role", "visible_context"),
                            "transform_version": "ingest-v1",
                            "language": language or metadata.get("language"),
                            "language_confidence": language_confidence,
                            "language_decision": language_reason,
                            "url": metadata.get("url"),
                            "path": str(concrete_path),
                            "part": document.part,
                            "char_start": start,
                            "char_end": end,
                            "content_hash": content_hash,
                            "chunk_content_hash": chunk_content_hash,
                            "text": text,
                        }
                    )
    chunks_removed_by_source_cap = 0
    if max_chunks_per_source_group > 0:
        by_group: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(str(row["source_group"]), []).append(row)
        capped_rows: List[Dict[str, Any]] = []
        for group in sorted(by_group):
            group_rows = by_group[group]
            if len(group_rows) <= max_chunks_per_source_group:
                capped_rows.extend(group_rows)
                continue
            if max_chunks_per_source_group == 1:
                selected_indexes = [len(group_rows) // 2]
            else:
                selected_indexes = sorted({
                    round(index * (len(group_rows) - 1) / (max_chunks_per_source_group - 1))
                    for index in range(max_chunks_per_source_group)
                })
            capped_rows.extend(group_rows[index] for index in selected_indexes)
            chunks_removed_by_source_cap += len(group_rows) - len(selected_indexes)
        rows = capped_rows
    write_jsonl(output_path, rows)
    return {
        "sources": source_count,
        "documents": document_count,
        "chunks": len(rows),
        "exact_duplicate_chunks_removed": duplicates_removed,
        "extraction_errors": extraction_errors,
        "extraction_error_samples": extraction_error_samples,
        "chunks_removed_by_source_cap": chunks_removed_by_source_cap,
        "non_english_sources_removed": non_english_sources_removed,
        "non_english_chunks_removed": non_english_chunks_removed,
        "corrupted_unicode_chunks_removed": corrupted_unicode_chunks_removed,
        "language_rejection_samples": language_rejection_samples,
    }


def append_ingested_manifest(
    manifest_path: Path,
    output_path: Path,
    target_chars: int = 4200,
    overlap_chars: int = 400,
    min_chunk_chars: int = 300,
    max_chunks_per_source_group: int = 0,
) -> Dict[str, Any]:
    """Atomically add a small manifest to an existing chunk corpus.

    Existing rows win exact-content collisions so active episode IDs and source
    references remain stable. The caller must still avoid running this while an
    ingest process is writing the same output file.
    """
    temporary = output_path.with_name(output_path.name + ".append-input")
    try:
        stats = ingest_manifest(
            manifest_path,
            temporary,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
            max_chunks_per_source_group=max_chunks_per_source_group,
        )
        additions = list(iter_jsonl(temporary))
        manifest = load_yaml(manifest_path)
        language_policy = manifest.get("language_policy", {}) or {}
        allowed_languages = tuple(str(item).lower() for item in language_policy.get("allowed_languages", []))
        language_required = bool(language_policy.get("required", bool(allowed_languages)))
        minimum_confidence = float(language_policy.get("minimum_confidence", 0.78))
        if language_required and allowed_languages != ("en",):
            raise ValueError("Append filtering currently supports only allowed_languages: [en]")

        seen: set = set()
        counters = {
            "existing_before": 0,
            "existing_kept": 0,
            "existing_non_english_removed": 0,
            "duplicates_removed": 0,
            "appended": 0,
        }

        def combined_rows() -> Iterable[Dict[str, Any]]:
            if output_path.exists():
                for original in iter_jsonl(output_path):
                    counters["existing_before"] += 1
                    row = original
                    if language_required:
                        if contains_blocked_script(str(row.get("text", ""))):
                            counters["existing_non_english_removed"] += 1
                            continue
                        language = str(row.get("language") or "").lower()
                        confidence = float(row.get("language_confidence") or 0.0)
                        if language != "en" or confidence < minimum_confidence:
                            decision = classify_language(
                                str(row.get("text", "")),
                                expected="en",
                                minimum_confidence=minimum_confidence,
                            )
                            if not decision.accepted:
                                counters["existing_non_english_removed"] += 1
                                continue
                            row = dict(row)
                            row["language"] = "en"
                            row["language_confidence"] = round(decision.confidence, 6)
                            row["language_decision"] = decision.reason
                    content_hash = str(
                        row.get("chunk_content_hash")
                        or stable_hash(normalize_text(str(row.get("text", ""))).lower(), 32)
                    )
                    if content_hash in seen:
                        counters["duplicates_removed"] += 1
                        continue
                    seen.add(content_hash)
                    counters["existing_kept"] += 1
                    yield row
            for row in additions:
                content_hash = str(
                    row.get("chunk_content_hash")
                    or stable_hash(normalize_text(str(row.get("text", ""))).lower(), 32)
                )
                if content_hash in seen:
                    counters["duplicates_removed"] += 1
                    continue
                seen.add(content_hash)
                counters["appended"] += 1
                yield row

        combined_count = write_jsonl(output_path, combined_rows())
        return {
            **stats,
            "existing_chunks_before_filter": counters["existing_before"],
            "existing_chunks": counters["existing_kept"],
            "existing_non_english_chunks_removed": counters["existing_non_english_removed"],
            "appended_chunks": counters["appended"],
            "combined_chunks": combined_count,
            "cross_manifest_duplicates_removed": counters["duplicates_removed"],
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def filter_chunk_file_by_language(
    output_path: Path,
    expected: str = "en",
    minimum_confidence: float = 0.78,
) -> Dict[str, int]:
    """Atomically enforce a strict language boundary on an existing chunk file."""
    if expected != "en":
        raise ValueError("Chunk filtering currently supports only expected='en'")
    counters = {
        "input_chunks": 0,
        "kept_chunks": 0,
        "removed_chunks": 0,
        "blocked_script_chunks": 0,
        "corrupted_unicode_chunks": 0,
    }

    def accepted_rows() -> Iterable[Dict[str, Any]]:
        for original in iter_jsonl(output_path):
            counters["input_chunks"] += 1
            text = sanitize_unicode(str(original.get("text", "")))
            if "\N{REPLACEMENT CHARACTER}" in text:
                counters["removed_chunks"] += 1
                counters["corrupted_unicode_chunks"] += 1
                continue
            if contains_blocked_script(text):
                counters["removed_chunks"] += 1
                counters["blocked_script_chunks"] += 1
                continue
            decision = classify_language(text, expected=expected, minimum_confidence=minimum_confidence)
            if not decision.accepted:
                counters["removed_chunks"] += 1
                continue
            row = dict(original)
            row["language"] = expected
            row["language_confidence"] = round(decision.confidence, 6)
            row["language_decision"] = decision.reason
            counters["kept_chunks"] += 1
            yield row

    if not output_path.exists():
        raise FileNotFoundError(output_path)
    write_jsonl(output_path, accepted_rows())
    return counters
