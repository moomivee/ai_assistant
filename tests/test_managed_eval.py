"""Frozen-corpus evaluation contracts, using deterministic vectors, not quality claims."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eval import managed_retrieval as evaluator
from src.rag.index import digest
from src.rag.managed import ManagedIndex


class TestEncoder:
    __test__ = False

    def __init__(self, *args):
        self.model = "minilm-multi"
        self.spec = {"passage_prefix": "", "query_prefix": "", "max_length": 16}
        self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def encode(self, texts, role):
        return np.array(
            [[1.0, 0.0] if "density" in text.lower() else [0.0, 1.0] for text in texts],
            dtype="float32",
        )


@pytest.fixture
def frozen(tmp_path):
    chunks = [
        {
            "document_id": "oil",
            "report": "oil",
            "version_id": "v1",
            "block_id": "b1",
            "chunk_id": "oil:v1:b1:t0-4",
            "text": "Density equals 33 API",
            "page": 2,
            "locator": {"page": 2},
            "embedding_input": "Density equals 33 API",
        },
        {
            "document_id": "other",
            "report": "other",
            "version_id": "v8",
            "block_id": "b9",
            "chunk_id": "other:v8:b9:t0-4",
            "text": "Sulfur equals 0.5 percent",
            "page": None,
            "locator": {"sheet": "Oil", "cell_range": "A1:B2"},
            "embedding_input": "Sulfur equals 0.5 percent",
        },
    ]
    payload = {
        "schema_version": 1,
        "corpus_revision": "fixed-r1",
        "reference_configuration": {"embedding": "minilm-multi", "chunking_version": "test"},
        "chunks": chunks,
        "chunks_sha256": evaluator._hash(chunks),
        "documents": [
            {"document_id": "oil", "version_id": "v1", "original_sha256": "a" * 64},
            {"document_id": "other", "version_id": "v8", "original_sha256": "b" * 64},
        ],
    }
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    cases = [
        {
            "id": "q1",
            "question": "What is the density?",
            "split": "dev",
            "group": "density",
            "snapshot_sha256": digest(snapshot),
            "document_ids": ["oil"],
            "gold": [
                {
                    "document_id": "oil",
                    "version_id": "v1",
                    "block_id": "b1",
                    "locator": {"page": 2},
                    "quote": "33 API",
                }
            ],
        }
    ]
    labels.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    return snapshot, labels, payload, cases


def test_snapshot_and_manual_labels_are_checked_before_model_loading(frozen, monkeypatch, tmp_path):
    snapshot, labels, _, cases = frozen
    cases[0]["gold"][0]["version_id"] = "old-version"
    labels.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluator, "Encoder", lambda *args: pytest.fail("Do not embed invalid labels")
    )
    with pytest.raises(ValueError, match="stale or nonexistent"):
        evaluator.run(snapshot, labels, tmp_path / "results.json", allow_model_download=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_id", "wrong-block"),
        ("locator", {"page": 3}),
        ("quote", "a made-up quote"),
    ],
)
def test_gold_requires_exact_source_block_locator_and_quote(frozen, field, value):
    snapshot, labels, payload, cases = frozen
    cases[0]["gold"][0][field] = value
    labels.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale or nonexistent"):
        evaluator.load_labels(labels, snapshot, payload)


def test_changed_snapshot_or_chunk_content_rejected(frozen):
    snapshot, labels, payload, _ = frozen
    payload["chunks"][0]["text"] = "Changed evidence"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity hash"):
        evaluator.load_snapshot(snapshot)
    payload["chunks_sha256"] = evaluator._hash(payload["chunks"])
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different or changed snapshot"):
        evaluator.load_labels(labels, snapshot, payload)


def test_metrics_distinguish_any_hit_from_complete_evidence(frozen):
    _, _, snapshot, cases = frozen
    first, second = snapshot["chunks"]
    golds = cases[0]["gold"] + [
        {
            "document_id": "other",
            "version_id": "v8",
            "block_id": "b9",
            "locator": second["locator"],
            "quote": "0.5 percent",
        }
    ]
    measured = evaluator.metrics([first], golds)
    assert measured == {
        "hit@1": 1.0,
        "hit@5": 1.0,
        "mrr@10": 1.0,
        "evidence_recall@5": 0.5,
        "all_evidence@5": 0.0,
    }
    assert evaluator.metrics([second, first], cases[0]["gold"])["mrr@10"] == 0.5


def test_run_uses_frozen_chunks_and_never_live_catalog(frozen, monkeypatch, tmp_path):
    snapshot, labels, _, _ = frozen
    monkeypatch.setattr(evaluator, "Encoder", TestEncoder)
    monkeypatch.setattr(
        evaluator, "get_store", lambda: pytest.fail("Offline run read live catalog")
    )
    result = evaluator.run(
        snapshot,
        labels,
        tmp_path / "results.json",
        model="minilm-multi",
        allow_model_download=True,
    )
    assert result["summary"]["dev"]["hit@1"] == 1.0
    assert result["rows"][0]["ranking"] == ["oil:v1:b1:t0-4"]
    assert result["snapshot_sha256"] == digest(snapshot)
    assert result["dataset_sha256"] == digest(labels)
    assert result["configuration"]["fixed_chunks_across_models"] is True
    assert result["configuration"]["passages_over_model_token_limit"] == 0


def test_offline_default_rejects_missing_model_before_download(frozen, monkeypatch, tmp_path):
    snapshot, labels, _, _ = frozen
    monkeypatch.setattr(evaluator, "Encoder", lambda *args: pytest.fail("Offline model download"))
    with pytest.raises(ValueError, match="--allow-model-download"):
        evaluator.run(snapshot, labels, tmp_path / "result.json", model="minilm-multi")


def test_template_is_blank_and_cannot_be_reported_as_a_dataset(frozen, monkeypatch, tmp_path):
    _, _, snapshot, _ = frozen
    encoder = TestEncoder()
    chunks = snapshot["chunks"]
    corpus = ManagedIndex(
        encoder,
        chunks,
        encoder.encode([c["text"] for c in chunks], "passage"),
        "index-fingerprint",
        "fixed-r1",
    )
    versions = {
        (doc["document_id"], doc["version_id"]): {
            "sha256": doc["original_sha256"],
            "parser_version": "test-parser-1",
            "filename": "report.pdf",
            "title": "Test report",
            "source_url": None,
        }
        for doc in snapshot["documents"]
    }
    store = SimpleNamespace(
        corpus_snapshot=lambda: {"revision": "fixed-r1"},
        corpus_revision=lambda: "fixed-r1",
        version=lambda doc, version: versions[(doc, version)],
    )
    monkeypatch.setattr(evaluator, "get_store", lambda: store)
    monkeypatch.setattr(evaluator, "managed_index", lambda model: corpus)
    output, labels = tmp_path / "export.json", tmp_path / "template.jsonl"
    result = evaluator.export_snapshot(output, labels, allow_model_download=True)
    assert result["unlabeled_templates"] == 2
    exported = evaluator.load_snapshot(output)
    assert exported["documents"][0]["original_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="finish manual annotation"):
        evaluator.load_labels(labels, output, exported)


def test_group_leakage_and_out_of_scope_gold_rejected(frozen):
    snapshot, labels, payload, cases = frozen
    second = deepcopy(cases[0])
    second.update(id="q2", question="Density in API?", split="test")
    labels.write_text("\n".join(json.dumps(row) for row in [cases[0], second]), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses development/test"):
        evaluator.load_labels(labels, snapshot, payload)
    cases[0]["document_ids"] = ["other"]
    labels.write_text(json.dumps(cases[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or nonexistent"):
        evaluator.load_labels(labels, snapshot, payload)
