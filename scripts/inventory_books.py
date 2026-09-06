#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from xml.etree import ElementTree

import yaml
from pypdf import PdfReader


DOMAIN_RULES = [
    ("edge_ai", ("edge ai", "tinyml", "embedded ai")),
    ("llm_engineering", ("llm", "language model", "prompt engineering", "rag ", " ai agent", "agentic", "langchain", "mcp ", "generative ai", "transformers and diffusion")),
    ("machine_learning_systems", ("machine learning system", "ml system", "applied machine learning", "ai engineering", "machine learning design", "artificial intelligence in finance")),
    ("framework_engineering", ("pytorch", "tensorflow", "fastai")),
    ("deep_learning_fundamentals", ("deep learning", "neural network", "unsupervised learning", "strengthening deep neural")),
    ("mathematics_and_optimization", ("math for ai", "mathematics", "algorithm", "data structures")),
    ("distributed_data_systems", ("distributed system", "data-intensive", "data engineering", "microservice", "mongodb", "postgresql", "sql ", "duckdb", "consul", "tcpip", "network administration", "operating system")),
    ("cloud_and_devops", ("cloud platform", "google cloud", "serverless", "high availability", "gitnotes", "progit")),
    ("software_architecture", ("software architecture", "architecture pattern", "evolutionary architecture", "api design", "web api", "design pattern", "robust python", "pragmatic programmer", "communication patterns")),
    ("programming_and_web", ("python", "javascript", "typescript", "node.js", "nodejs", "react", "angular", "css", "html", "web development", "full-stack", "full stack", "dotnet", "java ", "android", "progressive web")),
    ("sales_and_copywriting", ("sales", "selling", "closing", "sales letter", "what to say", "straight line", "customers", "offers")),
    ("marketing_and_brand", ("marketing", "brand", "contagious", "chasm", "blue ocean", "category", "positioning", "storytelling in design")),
    ("product_strategy", ("lean startup", "lean ux", "mom test", "product", "research", "innovation", "innovator", "play bigger", "start at the end", "lean analytics")),
    ("entrepreneurship_and_operations", ("startup", "start-up", "entrepreneur", "business", "ceo", "million dollar", "built to", "scaling up", "personal mba", "buy back your time", "measure what matters", "street smarts", "hospitality")),
    ("communication_and_writing", ("communicat", "talk smarter", "speak", "pitch", "small talk", "strategic writing", "words that change", "first minute", "content guide", "writing")),
    ("research_and_decision_making", ("decision", "thinking", "mental model", "clear thinking", "systems thinking", "rapid idea", "strategic thinking", "psychology of money", "hour between")),
    ("psychology_and_behavior", ("psychology", "emotional intelligence", "human nature", "nudge", "people", "psychopath", "idiot", "seduction", "manipulation", "dark psychology")),
    ("general_reasoning_and_learning", ("learn", "focus", "productivity", "habit", "mastery", "essentialism", "deep work", "make time", "indistractable", "teaching", "skill acquisition", "hidden potential", "art of impossible")),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def epub_metadata(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        with zipfile.ZipFile(str(path)) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                node.attrib["full-path"] for node in container.iter() if local_name(node.tag) == "rootfile"
            )
            package = ElementTree.fromstring(archive.read(rootfile))
            title = next((node.text for node in package.iter() if local_name(node.tag) == "title" and node.text), None)
            author = next((node.text for node in package.iter() if local_name(node.tag) == "creator" and node.text), None)
            return title, author
    except Exception:
        return None, None


def pdf_metadata(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        metadata = PdfReader(str(path)).metadata or {}
        title = metadata.get("/Title")
        author = metadata.get("/Author")
        return (str(title) if title else None), (str(author) if author else None)
    except Exception:
        return None, None


def clean_title(value: str) -> str:
    value = re.sub(r"\s*\((?:Z-Library|Z-Lib\.io).*?\)\s*", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" _-")
    return value


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized[:64] or "book").strip("-")


def classify(title: str, filename: str) -> str:
    searchable = (title + " " + filename).lower().replace("_", " ")
    for domain, terms in DOMAIN_RULES:
        if any(term in searchable for term in terms):
            return domain
    return "general_reasoning_and_learning"


def iter_books(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.lower() in {".epub", ".pdf"}:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory authorized local books into an ingestion manifest")
    parser.add_argument("--books-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    books_dir = Path(args.books_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    seen_hashes: Dict[str, str] = {}
    entries = []
    domain_counts: Dict[str, int] = {}
    duplicate_count = 0
    total_bytes = 0

    for path in iter_books(books_dir):
        file_hash = sha256_file(path)
        total_bytes += path.stat().st_size
        raw_title, author = epub_metadata(path) if path.suffix.lower() == ".epub" else pdf_metadata(path)
        title = clean_title(raw_title or path.stem)
        domain = classify(title, path.name)
        duplicate_of = seen_hashes.get(file_hash)
        if duplicate_of:
            duplicate_count += 1
        else:
            seen_hashes[file_hash] = str(path)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        source_group = "book-%s-%s" % (slug(title), file_hash[:10])
        entry = {
            "path": str(path),
            "title": title,
            "author": author or "metadata-unavailable",
            "domain": domain,
            "source_group": source_group,
            "lineage_component_id": source_group,
            "lineage_mode": "part" if path.suffix.lower() == ".epub" else "page_block",
            "version": "sha256:%s" % file_hash,
            "license": "user-asserted-training-rights",
            "license_evidence": "User explicitly confirmed training rights for the local Books corpus in this Codex task on 2026-08-17",
            "allowed_for_training": duplicate_of is None,
            "file_bytes": path.stat().st_size,
            "content_sha256": file_hash,
        }
        if path.suffix.lower() == ".pdf":
            entry["lineage_block_size"] = 16
        if duplicate_of:
            entry["duplicate_of"] = duplicate_of
            entry["quarantine_reason"] = "exact duplicate file"
        entries.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Generated from the user-authorized local Books corpus on 2026-08-17.\n"
        "# Exact duplicate files remain recorded but are not ingested.\n"
        + yaml.safe_dump({"sources": entries}, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    report = {
        "books_directory": str(books_dir),
        "files_found": len(entries),
        "unique_files_enabled": len(entries) - duplicate_count,
        "exact_duplicate_files_quarantined": duplicate_count,
        "total_bytes": total_bytes,
        "domain_counts": dict(sorted(domain_counts.items())),
        "manifest": str(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
