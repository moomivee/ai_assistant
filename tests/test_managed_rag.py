"""Retrieval mechanics, not embedding quality: deterministic vectors + real FAISS."""

from copy import deepcopy
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src.config import settings
from src.rag import managed


class FakeEncoder:
    def __init__(self, model="minilm-multi", max_length=128):
        self.model = model
        self.spec = {"max_length": max_length, "passage_prefix": "passage: "}
        self.tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "passage": 3, ":": 4},
                unk_token="[UNK]",
            )
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        self.calls = []

    def encode(self, texts, role):
        self.calls.append((list(texts), role))
        values = []
        for text in texts:
            if "target" in text:
                values.append([1.0, 0.0])
            elif "baseline" in text and "irrelevant" not in text:
                values.append([0.8, 0.6])
            else:
                values.append([0.0, 1.0])
        return np.array(values, dtype="float32")


class FakeStore:
    def __init__(self):
        self.state = {
            "revision": 1,
            "active_document_ids": ["builtin-a", "builtin-b"],
            "baseline_document_ids": ["builtin-a", "builtin-b"],
            "chunks": [],
            "builtin_documents": [],
        }
        self.indexed = None

    def corpus_snapshot(self):
        return deepcopy(self.state)

    def corpus_revision(self):
        return self.state["revision"]

    def document(self, document_id):
        if document_id not in self.state["active_document_ids"]:
            return None
        return {"archived": False, "active_version_id": "v1"}

    def mark_indexed(self, revision, model=None):
        if revision != self.corpus_revision():
            return False
        self.indexed = revision
        return True

    def publish(self, document="uploaded", version="v1", text="target uploaded"):
        state = self.state
        if document not in state["active_document_ids"]:
            state["active_document_ids"].append(document)
        if document in state["baseline_document_ids"]:
            state["baseline_document_ids"].remove(document)
        state["chunks"] = [c for c in state["chunks"] if c["report"] != document]
        state["chunks"].append(
            {
                "report": document,
                "document_id": document,
                "version_id": version,
                "block_id": "block-1",
                "kind": "report",
                "text": text,
                "title": "Uploaded report",
                "date": "2026-11-14",
                "page": 7,
                "url": "",
                "local_url": f"/api/knowledge/{document}/versions/{version}/file",
                "locator": {"page": 7},
                "method": "native",
            }
        )
        state["revision"] += 1

    def archive(self, document):
        self.state["active_document_ids"].remove(document)
        self.state["baseline_document_ids"] = [
            d for d in self.state["baseline_document_ids"] if d != document
        ]
        self.state["chunks"] = [c for c in self.state["chunks"] if c["report"] != document]
        self.state["revision"] += 1


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings.cache_clear()
    managed._cache.clear()
    encoder = FakeEncoder()
    chunks = [
        {"report": "builtin-a", "chunk_id": "builtin-a:p1:w0", "text": "baseline report"},
        {"report": "builtin-b", "chunk_id": "builtin-b:p2:w0", "text": "irrelevant report"},
    ]
    vectors = encoder.encode([c["text"] for c in chunks], "passage")
    dense = faiss.IndexFlatIP(2)
    dense.add(vectors)
    baseline = SimpleNamespace(
        chunks=chunks, encoder=encoder, index=dense, fingerprint="baseline-frozen"
    )
    store = FakeStore()
    monkeypatch.setattr(managed, "index", lambda model: baseline)
    monkeypatch.setattr(managed, "get_store", lambda: store)
    yield store, encoder, baseline
    managed._cache.clear()


def test_uploads_and_builtins_share_global_ranking(corpus):
    store, _, baseline = corpus
    store.publish()
    results = managed.managed_search("target", 2)
    assert [r["report"] for r in results] == ["uploaded", "builtin-a"]
    assert [r["id"] for r in results] == ["R1", "R2"]
    assert results[0]["version_id"] == "v1"
    assert results[0]["block_id"] == "block-1"
    assert results[0]["page"] == 7
    assert results[0]["local_url"].endswith("/v1/file#page=7")
    assert results[0]["locator"] == {"page": 7}
    assert results[1]["chunk_id"] == "builtin-a:p1:w0"
    assert "uploaded" not in {c["report"] for c in baseline.chunks}


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_scope_is_applied_before_top_k(corpus, mode):
    store, _, _ = corpus
    store.publish()
    result = managed.managed_search("target", 1, ["builtin-b"], mode=mode)
    assert [row["report"] for row in result] == ["builtin-b"]


def test_empty_and_invalid_scope_never_fall_back_to_global(corpus):
    store, encoder, _ = corpus
    before = len(encoder.calls)
    assert managed.managed_search("target", 1, []) == []
    with pytest.raises(managed.DocumentScopeError, match="отсутствуют или архивированы"):
        managed.managed_search("target", 1, ["not-found"])
    store.state["active_document_ids"].append("pending-document")
    with pytest.raises(managed.DocumentScopeError, match="доступного для поиска текста"):
        managed.managed_search("target", 1, ["pending-document"])
    assert len(encoder.calls) == before


def test_archive_invalidates_snapshot_without_mutating_old_quote(corpus):
    store, _, _ = corpus
    store.publish()
    old = managed.managed_search("target", 1)[0]
    old_fingerprint = managed.managed_status()["fingerprint"]
    store.archive("uploaded")
    assert managed.managed_status()["status"] == "pending"
    assert all(row["report"] != "uploaded" for row in managed.managed_search("target", 10))
    assert managed.managed_status()["fingerprint"] != old_fingerprint
    assert old["version_id"] == "v1" and old["text"] == "target uploaded"
    with pytest.raises(managed.DocumentScopeError):
        managed.managed_search("target", 1, ["uploaded"])


def test_new_version_replaces_old_and_builtin_update_hides_baseline(corpus):
    store, _, _ = corpus
    store.publish(document="builtin-a", version="v2", text="target revised")
    first = managed.managed_search("target", 10, ["builtin-a"])
    assert len(first) == 1 and first[0]["version_id"] == "v2"
    store.publish(document="builtin-a", version="v3", text="irrelevant changed")
    second = managed.managed_search("target", 10, ["builtin-a"])
    assert len(second) == 1 and second[0]["version_id"] == "v3"
    assert second[0]["text"] == "irrelevant changed"
    assert first[0]["text"] == "target revised"


def test_builtin_results_point_to_immutable_version(corpus):
    store, _, baseline = corpus
    baseline.chunks[0]["page"] = 6
    store.state["builtin_documents"] = [
        {
            "document_id": "builtin-a",
            "id": "original-version",
            "local_url": "/api/knowledge/documents/builtin-a/versions/original-version/file",
        }
    ]
    hit = managed.managed_search("target", 1, ["builtin-a"])[0]
    assert hit["version_id"] == "original-version"
    assert hit["document_id"] == "builtin-a"
    assert hit["local_url"].endswith("/original-version/file#page=6")
    assert hit["locator"] == {"page": 6}
    assert hit["chunk_id"] == "builtin-a:p1:w0"


def test_changed_repository_pdf_cannot_be_cited_as_old_catalog_version(corpus, monkeypatch):
    store, _, _ = corpus
    store.state["builtin_documents"] = [
        {
            "document_id": "builtin-a",
            "id": "original-version",
            "sha256": "old-digest",
        }
    ]
    monkeypatch.setattr(managed, "manifest", lambda: [{"id": "builtin-a", "file": "new.pdf"}])
    monkeypatch.setattr(managed, "digest", lambda path: "new-digest")
    with pytest.raises(ValueError, match="отличается от версии в базе знаний"):
        managed.managed_search("target", 1)


def test_same_revision_reuses_embeddings_and_disk_snapshot(corpus):
    store, encoder, _ = corpus
    store.publish()
    managed.managed_search("target", 2)
    passage_calls = sum(role == "passage" for _, role in encoder.calls)
    managed.managed_search("target", 2)
    managed._cache.clear()  # A fresh process can reuse the atomic disk snapshot.
    managed.managed_search("target", 2)
    assert sum(role == "passage" for _, role in encoder.calls) == passage_calls


def test_archive_racing_embedding_never_returns_archived_chunks(corpus, monkeypatch):
    store, encoder, _ = corpus
    store.publish()
    original = encoder.encode

    def encode_and_archive(texts, role):
        values = original(texts, role)
        if role == "passage" and "uploaded" in store.state["active_document_ids"]:
            store.archive("uploaded")
        return values

    monkeypatch.setattr(encoder, "encode", encode_and_archive)
    result = managed.managed_search("target", 10)
    assert all(row["report"] != "uploaded" for row in result)
    assert store.indexed == store.corpus_revision()


def test_status_and_empty_corpus_do_not_load_encoder(corpus, monkeypatch):
    store, _, _ = corpus

    def forbidden(*args, **kwargs):
        pytest.fail("Status or an empty selection must not load an embedding model")

    monkeypatch.setattr(managed, "index", forbidden)
    assert managed.managed_status()["status"] == "pending"
    assert managed.managed_status()["dimensions"] is None
    store.archive("builtin-a")
    store.archive("builtin-b")
    assert managed.managed_search("target", 5) == []


def test_token_windows_preserve_exact_offsets_and_fit_with_prefix_and_header():
    encoder = FakeEncoder(max_length=18)
    text = "Grade | Sulfur\n" + " ".join(f"word{i}" for i in range(35))
    block = {
        "report": "table",
        "version_id": "v9",
        "block_id": "b1",
        "text": text,
        "table_header": "Grade | Sulfur",
        "locator": {"sheet": "Oil", "rows": [1, 10]},
    }
    chunks = managed.token_chunks([block], encoder)
    assert len(chunks) > 1
    covered = set()
    tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
    tokenizer.no_truncation()
    for row in chunks:
        assert row["text"] == text[row["char_start"] : row["char_end"]]
        assert len(tokenizer.encode("passage: " + row["_embedding_text"]).ids) <= 18
        assert row["locator"] == block["locator"]
        covered.update(range(row["char_start"], row["char_end"]))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert chunks[1]["token_start"] < chunks[0]["token_end"]
    assert encoder.tokenizer.truncation["max_length"] == 18


def test_real_catalog_upload_replace_archive_retains_immutable_citations(
    corpus, monkeypatch, tmp_path
):
    from src.knowledge.store import KnowledgeStore

    store = KnowledgeStore(tmp_path / "knowledge")
    monkeypatch.setattr(managed, "get_store", lambda: store)
    document = store.upload("oil.csv", b"Grade,Sulfur\ntarget,0.5\n", title="Oil grades")
    doc_id = document["id"]
    first_version = document["current_version_id"]
    with pytest.raises(managed.DocumentScopeError, match="доступного для поиска текста"):
        managed.managed_search("target", 2, [doc_id])
    store.process(first_version)
    first = managed.managed_search("target", 2, [doc_id])
    assert first and first[0]["version_id"] == first_version
    assert first[0]["locator"]["type"] == "csv"
    assert store.document(doc_id)["index_status"] == "indexed"

    replacement = store.upload(
        "oil.csv", b"Grade,Sulfur\nnew,0.8\n", title="Oil revised", document_id=doc_id
    )
    # An uploaded, unprocessed version must not evict the previous ready version.
    assert managed.managed_search("target", 2, [doc_id])[0]["version_id"] == first_version
    second_version = replacement["current_version_id"]
    store.process(second_version)
    second = managed.managed_search("target", 2, [doc_id])
    assert all(row["version_id"] == second_version for row in second)
    assert not any("target" in row["text"] for row in second)

    store.archive(doc_id)
    assert managed.managed_search("target", 5) == []
    with pytest.raises(managed.DocumentScopeError):
        managed.managed_search("target", 2, [doc_id])
    assert b"target" in store.version_file(doc_id, first_version).read_bytes()
    assert "target" in store.blocks(doc_id, first_version)[0]["text"]
