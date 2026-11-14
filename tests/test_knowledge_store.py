import hashlib
import json
import threading

import pytest

from src.knowledge import extractors
from src.knowledge.jobs import shutdown_jobs, submit_ingestion
from src.knowledge.store import KnowledgeConflict, KnowledgeStore, UploadTooLarge


@pytest.fixture
def store(tmp_path, monkeypatch):
    def extract(path, filename):
        return {
            "blocks": [
                {
                    "block_id": "page-1",
                    "text": path.read_text(),
                    "page": 1,
                    "locator": {"page": 1},
                    "method": "native",
                }
            ],
            "warnings": [],
            "parser_version": "test-v1",
        }

    monkeypatch.setattr(extractors, "extract", extract)
    return KnowledgeStore(tmp_path / "knowledge")


def ready(store, content=b"density 35.6 API", **kwargs):
    document = store.upload("report.pdf", content, **kwargs)
    return store.process(document["current_version_id"])


def test_upload_is_persistent_and_ready_means_extracted_not_indexed(store):
    document = store.upload(
        "report.pdf",
        b"density 35.6 API",
        "Specifications",
        "2026-11-10",
        "https://example.org/report.pdf",
        "Publisher",
    )
    assert document["status"] == "queued"
    assert document["active_version_id"] is None
    assert document["index_status"] == "unavailable"
    assert store.ready_chunks() == []
    document = store.process(document["current_version_id"])
    assert document["status"] == "ready"
    assert document["index_status"] == "pending"
    assert document["active_version_id"] == document["current_version_id"]
    chunk = store.ready_chunks()[0]
    assert chunk["kind"] == "report"
    assert chunk["document_id"] == document["id"] == chunk["report"]
    assert chunk["version_id"] == document["current_version_id"]
    assert chunk["title"] == "Specifications"
    assert chunk["date"] == "2026-11-10"
    assert chunk["url"] == "https://example.org/report.pdf"
    assert chunk["local_url"].endswith("/file")
    assert chunk["locator"] == {"page": 1}
    reopened = KnowledgeStore(store.root)
    assert reopened.ready_chunks() == [chunk]
    assert reopened.document(document["id"])["status"] == "ready"


def test_checksum_deduplicates_and_does_not_replace_original(store):
    first = ready(store)
    original = store.version_file(first["id"], first["current_version_id"])
    repeated = store.upload("renamed.pdf", b"density 35.6 API", title="Changed metadata")
    assert repeated["id"] == first["id"]
    assert repeated["deduplicated"] is True
    assert repeated["title"] == "report"
    assert len(store.versions(first["id"])) == 1
    assert original.read_bytes() == b"density 35.6 API"


def test_new_version_is_atomic_and_old_citations_survive(store):
    first = ready(store, title="Old edition")
    first_version = first["current_version_id"]
    old_revision = store.corpus_revision()
    second = store.upload(
        "new.pdf", b"density 36 API", title="New edition", document_id=first["id"]
    )
    assert second["active_version_id"] == first_version
    assert store.corpus_revision() == old_revision
    assert store.ready_chunks()[0]["title"] == "Old edition"
    updated = store.process(second["current_version_id"])
    assert updated["versions_count"] == 2
    assert updated["active_version_id"] != first_version
    assert store.corpus_revision() != old_revision
    assert store.ready_chunks()[0]["text"] == "density 36 API"
    assert store.blocks(first["id"], first_version)[0]["text"] == "density 35.6 API"
    assert store.version_file(first["id"], first_version).read_bytes() == b"density 35.6 API"
    assert store.version(first["id"], first_version)["title"] == "Old edition"


def test_failed_replacement_keeps_previous_searchable_and_retry_is_explicit(store, monkeypatch):
    original = ready(store)
    replacement = store.upload("report.pdf", b"broken", document_id=original["id"])
    version = replacement["current_version_id"]

    def fail(*args):
        raise ValueError("Файл повреждён.")

    monkeypatch.setattr(extractors, "extract", fail)
    failed = store.process(version)
    assert failed["status"] == "error"
    assert failed["error"] == "Файл повреждён."
    assert failed["active_version_id"] == original["current_version_id"]
    assert store.ready_chunks()[0]["text"] == "density 35.6 API"
    queued = store.retry(original["id"], version)
    assert queued["status"] == "queued"
    assert queued["error"] is None
    with pytest.raises(KnowledgeConflict):
        store.retry(original["id"], original["current_version_id"])


def test_partial_extraction_is_searchable_and_preserves_warning(store, monkeypatch):
    monkeypatch.setattr(
        extractors,
        "extract",
        lambda *args: {
            "blocks": [
                {
                    "block_id": "cell",
                    "text": "Urals 1.4",
                    "page": None,
                    "locator": {"sheet": "Quality", "cells": "A1:B2"},
                    "method": "native",
                }
            ],
            "warnings": ["Один лист пропущен."],
            "parser_version": "test-v2",
        },
    )
    document = ready(store)
    assert document["status"] == "partial"
    assert document["warnings"] == ["Один лист пропущен."]
    chunk = store.ready_chunks()[0]
    assert chunk["page"] is None
    assert chunk["locator"]["cells"] == "A1:B2"
    assert chunk["url"] == chunk["local_url"]


def test_archive_hides_search_without_breaking_version_urls(store):
    document = ready(store)
    doc_id, version = document["id"], document["current_version_id"]
    old_revision = store.corpus_revision()
    archived = store.archive(doc_id)
    assert archived["status"] == "archived"
    assert archived["active_version_id"] is None
    assert store.list_documents() == []
    assert len(store.list_documents(include_archived=True)) == 1
    assert store.corpus_revision() != old_revision
    assert store.ready_chunks() == []
    assert store.blocks(doc_id, version)
    assert store.version_file(doc_id, version).is_file()
    with pytest.raises(KnowledgeConflict):
        store.upload("update.pdf", b"new", document_id=doc_id)


def test_restart_marks_queued_and_processing_as_retryable_errors(store):
    queued = store.upload("queued.pdf", b"queued")
    processing = store.upload("processing.pdf", b"processing")
    with store._connection() as conn:
        conn.execute(
            "UPDATE versions SET status='processing' WHERE id=?",
            (processing["current_version_id"],),
        )
    reopened = KnowledgeStore(store.root)
    for document in (queued, processing):
        recovered = reopened.document(document["id"])
        assert recovered["status"] == "error"
        assert "перезапуск" in recovered["error"]
        assert reopened.retry(document["id"], document["current_version_id"])["status"] == "queued"


def test_index_status_acknowledges_only_matching_snapshot(store):
    first = ready(store)
    snapshot = store.corpus_snapshot()
    assert snapshot["active_document_ids"] == [first["id"]]
    assert snapshot["baseline_document_ids"] == []
    assert store.mark_indexed(snapshot["revision"])
    assert store.document(first["id"])["index_status"] == "indexed"
    ready(store, b"another")
    assert not store.mark_indexed(snapshot["revision"])
    assert store.document(first["id"])["index_status"] == "pending"


def test_index_readiness_is_separate_for_each_embedding_model(store, monkeypatch):
    from src.config import settings

    first = ready(store)
    revision = store.corpus_revision()
    assert store.mark_indexed(revision, model="minilm-multi")
    monkeypatch.setenv("RAG_EMBEDDING", "e5-small")
    settings.cache_clear()
    assert store.document(first["id"])["index_status"] == "pending"
    assert store.mark_indexed(revision, model="e5-small")
    assert store.document(first["id"])["index_status"] == "indexed"
    monkeypatch.setenv("RAG_EMBEDDING", "minilm-multi")
    settings.cache_clear()
    assert store.document(first["id"])["index_status"] == "indexed"


@pytest.mark.parametrize(
    "filename", ["../report.pdf", "a/b.pdf", "a\\b.pdf", "file\x00.pdf", "x.exe"]
)
def test_invalid_filename_is_rejected_before_disk_write(store, filename):
    with pytest.raises(ValueError):
        store.upload(filename, b"content")
    assert store.list_documents() == []
    assert list((store.root / "originals").iterdir()) == []


@pytest.mark.parametrize(
    "source_url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://user:password@example.org",
        "https://bad url",
    ],
)
def test_source_urls_are_only_metadata_but_validated(store, source_url):
    with pytest.raises(ValueError):
        store.upload("file.pdf", b"content", source_url=source_url)


def test_empty_large_unknown_document_and_invalid_date_are_rejected(store, monkeypatch):
    with pytest.raises(ValueError, match="пуст"):
        store.upload("empty.pdf", b"")
    monkeypatch.setattr("src.knowledge.store.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(UploadTooLarge):
        store.upload("large.pdf", b"a" * 11)
    with pytest.raises(KeyError):
        store.upload("file.pdf", b"a", document_id="unknown")
    with pytest.raises(ValueError, match="ГГГГ"):
        store.upload("file.pdf", b"a", date="yesterday")


def test_lookups_do_not_turn_untrusted_ids_into_paths(store):
    document = ready(store)
    assert store.document("missing") is None
    assert store.version_file(document["id"], "../../outside") is None
    assert store.blocks(document["id"], "../../outside") == []
    assert store.version("other-document", document["current_version_id"]) is None


def test_unexpected_parser_error_is_not_returned_verbatim(store, monkeypatch):
    def fail(*args):
        raise RuntimeError("secret contents /private/path token=123")

    monkeypatch.setattr(extractors, "extract", fail)
    result = ready(store)
    assert result["status"] == "error"
    assert "secret" not in result["error"]
    assert store.ready_chunks() == []


def test_empty_extraction_and_duplicate_block_ids_are_errors(store, monkeypatch):
    for blocks in ([], [{"block_id": "x", "text": "a"}, {"block_id": "x", "text": "b"}]):
        monkeypatch.setattr(
            extractors,
            "extract",
            lambda *args: {
                "blocks": blocks,
                "warnings": [],
                "parser_version": "test",
            },
        )
        result = ready(store, str(blocks).encode())
        assert result["status"] == "error"


def test_original_checksum_detects_changed_file(store):
    document = store.upload("report.pdf", b"original")
    store.version_file(document["id"], document["current_version_id"]).write_bytes(b"changed")
    result = store.process(document["current_version_id"])
    assert result["status"] == "error"
    assert "Контрольная сумма" in result["error"]


def test_builtin_catalog_no_duplicate_chunks_and_user_replacement_wins(store, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    content = b"original baseline"
    (reports / "baseline.pdf").write_bytes(content)
    manifest = tmp_path / "reports.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "eia-baseline",
                    "file": "baseline.pdf",
                    "title": "EIA",
                    "date": "2025-07-24",
                    "url": "https://eia.gov/baseline.pdf",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ]
        )
    )
    store.register_baseline(manifest)
    assert store.baseline_document_ids() == ["eia-baseline"]
    assert store.ready_chunks() == []
    assert store.blocks("eia-baseline")[0]["text"] == "original baseline"
    document = store.document("eia-baseline")
    assert document["builtin"] is True
    assert document["source_url"] == "https://eia.gov/baseline.pdf"
    store.register_baseline(manifest)
    assert len(store.versions("eia-baseline")) == 1
    updated = ready(store, b"new baseline edition", document_id="eia-baseline")
    assert updated["builtin"] is True
    assert store.baseline_document_ids() == []
    assert store.ready_chunks()[0]["text"] == "new baseline edition"
    store.archive("eia-baseline")
    store.register_baseline(manifest)
    assert store.list_documents() == []


def test_background_submission_deduplicates_running_job(store, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []

    def process(version):
        calls.append(version)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(store, "process", process)
    try:
        one = submit_ingestion(store, "version")
        assert started.wait(2)
        two = submit_ingestion(store, "version")
        assert one is two
        assert calls == ["version"]
        release.set()
        one.result(timeout=2)
    finally:
        release.set()
        shutdown_jobs(wait=True)
