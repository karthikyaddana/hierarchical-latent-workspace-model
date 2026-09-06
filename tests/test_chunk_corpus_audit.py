import json

from scripts.audit_chunk_corpus import audit


def chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "source_id": "source-" + chunk_id,
        "source_group": "source-group",
        "lineage_component_id": "lineage",
        "domain": "programming_and_web",
        "license": "MIT",
        "license_evidence": "test fixture",
        "language": "en",
        "language_confidence": 0.99,
        "chunk_content_hash": "hash-" + chunk_id,
        "text": text,
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_chunk_corpus_audit_accepts_clean_english(tmp_path):
    path = tmp_path / "chunks.jsonl"
    write_rows(
        path,
        [chunk("clean", "The verifier checks every claim against evidence and rejects unsupported conclusions. " * 6)],
    )

    report = audit(path)

    assert report["valid"]
    assert report["chunks"] == 1
    assert not any(report["violations"].values())


def test_chunk_corpus_audit_rejects_unsafe_content_and_duplicates(tmp_path):
    path = tmp_path / "chunks.jsonl"
    unsafe = chunk(
        "unsafe",
        "Internal reasoning: Проверка скрыта, and this corrupted marker \N{REPLACEMENT CHARACTER} must not be trained. " * 4,
    )
    duplicate = dict(unsafe)
    write_rows(path, [unsafe, duplicate])

    report = audit(path)

    assert not report["valid"]
    assert report["violations"]["duplicate_chunk_ids"] == 1
    assert report["violations"]["duplicate_content_hashes"] == 1
    assert report["violations"]["blocked_script_chunks"] == 2
    assert report["violations"]["corrupted_unicode_chunks"] == 2
    assert report["violations"]["private_reasoning_marker_chunks"] == 2
