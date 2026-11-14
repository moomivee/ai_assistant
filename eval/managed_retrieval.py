"""Freeze a managed corpus, hand-label evidence, evaluate it without an LLM.

No live catalog is consulted by ``run``. Model downloads require an explicit flag;
the default evaluates with locally cached, checksum-verified embedding models.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer

from src.config import settings
from src.knowledge.store import get_store
from src.rag.index import MODEL_SPECS, Encoder, digest
from src.rag.managed import MANAGED_VERSION, ManagedIndex, managed_index
from src.validation import normalized

from .metrics import mean_metrics

SCHEMA_VERSION = 1


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def _check_model_files(model: str, allow_download: bool) -> None:
    if allow_download:
        return
    folder = settings().cache / "models" / model
    for name, metadata in MODEL_SPECS[model]["files"].items():
        path = folder / name
        if not path.is_file() or digest(path) != metadata["sha256"]:
            raise ValueError(
                f"Verified model file is not cached: {path}. "
                "Use --allow-model-download once, then rerun offline."
            )


def _full_tokenizer(encoder):
    tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _embedding_input(chunk: dict, encoder, tokenizer) -> str:
    header = str(chunk.get("table_header") or "")
    prefix, limit = encoder.spec["passage_prefix"], encoder.spec["max_length"]
    if len(tokenizer.encode(prefix + header).ids) > limit // 2:
        header = ""
    text = chunk["text"]
    return header + "\n" + text if header and not text.startswith(header) else text


def export_snapshot(
    output: Path,
    labels_output: Path,
    model: str | None = None,
    document_ids: list[str] | None = None,
    allow_model_download: bool = False,
) -> dict:
    """Write exact retrieval inputs plus an intentionally unfinished label template."""
    if output.exists() or labels_output.exists():
        raise FileExistsError(
            "Choose new snapshot and label paths; existing versions are preserved."
        )
    if output.resolve() == labels_output.resolve():
        raise ValueError("Snapshot and labels require different paths")
    model = model or settings().embedding
    _check_model_files(model, allow_model_download)
    store = get_store()
    for _ in range(3):
        catalog = store.corpus_snapshot()
        corpus = managed_index(model)
        if corpus.revision != catalog["revision"]:
            continue
        available = {row["document_id"] for row in corpus.chunks}
        selected = available if document_ids is None else set(document_ids)
        if not selected or selected - available:
            raise ValueError("Select existing documents with extracted, indexed text")
        selected_corpus = corpus.scoped(selected)
        tokenizer = _full_tokenizer(corpus.encoder)
        chunks = [
            dict(row, embedding_input=_embedding_input(row, corpus.encoder, tokenizer))
            for row in selected_corpus.chunks
        ]
        documents = []
        for document_id, version_id in sorted(
            {(row["document_id"], row["version_id"]) for row in chunks}
        ):
            version = store.version(document_id, version_id)
            if not version:
                raise ValueError("A source version disappeared while exporting the corpus")
            documents.append(
                {
                    "document_id": document_id,
                    "version_id": version_id,
                    "original_sha256": version["sha256"],
                    "parser_version": version.get("parser_version"),
                    "filename": version["filename"],
                    "title": version["title"],
                    "source_url": version.get("source_url"),
                }
            )
        if store.corpus_revision() != catalog["revision"]:
            continue
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "immutable_managed_retrieval_corpus",
            "corpus_revision": catalog["revision"],
            "reference_configuration": {
                "embedding": model,
                "model_spec": MODEL_SPECS[model],
                "chunking_version": MANAGED_VERSION,
                "index_fingerprint": corpus.fingerprint,
                "dimensions": corpus.index.d,
                "legacy_baseline_chunks_preserved": True,
            },
            "documents": documents,
            "chunks_sha256": _hash(chunks),
            "chunks": chunks,
        }
        _write_new(output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        snapshot_hash = digest(output)
        templates = []
        for i, doc in enumerate(documents, 1):
            example = next(row for row in chunks if row["document_id"] == doc["document_id"])
            templates.append(
                {
                    "id": f"manual-{i:03}",
                    "split": "dev",
                    "group": doc["document_id"],
                    "snapshot_sha256": snapshot_hash,
                    "question": "",
                    "document_ids": [doc["document_id"]],
                    "gold": [
                        {
                            "document_id": doc["document_id"],
                            "version_id": doc["version_id"],
                            "block_id": example.get("block_id"),
                            "locator": example.get("locator", {}),
                            "quote": "",
                        }
                    ],
                    "annotation_note": "Write a question and exact supporting quote manually. "
                    "The blank template is not an evaluation dataset. Add independent test groups.",
                    "candidate_context": example["text"],
                }
            )
        _write_new(labels_output, "".join(_canonical(row) + "\n" for row in templates))
        return {
            "snapshot_sha256": snapshot_hash,
            "documents": len(documents),
            "chunks": len(chunks),
            "unlabeled_templates": len(templates),
        }
    raise RuntimeError("Catalog changed repeatedly; retry after ingestion finishes")


def load_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported managed-corpus snapshot schema")
    chunks = snapshot.get("chunks") or []
    if not chunks or snapshot.get("chunks_sha256") != _hash(chunks):
        raise ValueError("Snapshot chunks are empty or their integrity hash does not match")
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk identifiers in snapshot")
    if any(
        not isinstance(row.get("original_sha256"), str)
        or len(row["original_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in row["original_sha256"])
        for row in snapshot["documents"]
    ):
        raise ValueError("Snapshot manifest requires SHA-256 hashes of original documents")
    versions = {(row["document_id"], row["version_id"]) for row in snapshot["documents"]}
    if any((row["document_id"], row["version_id"]) not in versions for row in chunks):
        raise ValueError("Chunk refers to a source version missing from the snapshot manifest")
    if any(not row.get("embedding_input") or not row.get("text") for row in chunks):
        raise ValueError("Snapshot must retain exact source text and embedding inputs")
    return snapshot


def relevant(chunk: dict, gold: dict) -> bool:
    return (
        chunk["document_id"] == gold["document_id"]
        and chunk["version_id"] == gold["version_id"]
        and chunk.get("block_id") == gold.get("block_id")
        and chunk.get("locator", {}) == gold["locator"]
        and normalized(gold["quote"]) in normalized(chunk["text"])
    )


def load_labels(path: Path, snapshot_path: Path, snapshot: dict) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Labels are empty or case identifiers are duplicated")
    snapshot_hash = digest(snapshot_path)
    available = {row["document_id"] for row in snapshot["chunks"]}
    groups, questions = {}, set()
    for row in rows:
        name = row["id"]
        if row.get("snapshot_sha256") != snapshot_hash:
            raise ValueError(f"{name}: labels belong to a different or changed snapshot")
        if not row.get("question", "").strip() or not row.get("gold"):
            raise ValueError(f"{name}: finish manual annotation (question and gold evidence)")
        if row.get("split") not in {"dev", "test"}:
            raise ValueError(f"{name}: split must be dev or test")
        group = row.get("group") or name
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError(f"{name}: one question group crosses development/test splits")
        selected = set(row["document_ids"]) if row.get("document_ids") is not None else available
        if not selected or selected - available:
            raise ValueError(f"{name}: document scope is empty or absent from the frozen corpus")
        question_key = (normalized(row["question"]), tuple(sorted(selected)))
        if question_key in questions:
            raise ValueError(f"{name}: duplicate question and document scope")
        questions.add(question_key)
        if len({_canonical(gold) for gold in row["gold"]}) != len(row["gold"]):
            raise ValueError(f"{name}: duplicate gold evidence")
        for gold in row["gold"]:
            if not gold.get("document_id") or not gold.get("version_id"):
                raise ValueError(
                    f"{name}: gold requires document and immutable version identifiers"
                )
            if not gold.get("quote", "").strip() or not isinstance(gold.get("locator"), dict):
                raise ValueError(f"{name}: gold requires an exact quote and a source locator")
            if gold.get("document_id") not in selected or not any(
                relevant(chunk, gold) for chunk in snapshot["chunks"]
            ):
                raise ValueError(f"{name}: stale or nonexistent gold version/block/locator/quote")
    return rows


def metrics(ranked: list[dict], golds: list[dict]) -> dict:
    positions = [
        i for i, chunk in enumerate(ranked[:10], 1) if any(relevant(chunk, gold) for gold in golds)
    ]
    recall = sum(any(relevant(chunk, gold) for chunk in ranked[:5]) for gold in golds) / len(golds)
    return {
        "hit@1": float(1 in positions),
        "hit@5": float(any(i <= 5 for i in positions)),
        "mrr@10": 1 / min(positions) if positions else 0.0,
        "evidence_recall@5": recall,
        "all_evidence@5": float(recall == 1.0),
    }


def run(
    snapshot_path: Path,
    labels_path: Path,
    output: Path,
    model: str | None = None,
    mode: str = "dense",
    split: str = "all",
    allow_model_download: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new result path to preserve previous measurements")
    if mode not in {"dense", "hybrid"} or split not in {"all", "dev", "test"}:
        raise ValueError("Invalid retrieval mode or data split")
    snapshot = load_snapshot(snapshot_path)
    cases = load_labels(labels_path, snapshot_path, snapshot)
    cases = [row for row in cases if split == "all" or row["split"] == split]
    if not cases:
        raise ValueError("The selected split has no labeled questions")
    model = model or settings().embedding
    _check_model_files(model, allow_model_download)
    encoder = Encoder(settings().cache / "models" / model, model)
    chunks = snapshot["chunks"]
    inputs = [row["embedding_input"] for row in chunks]
    tokenizer = _full_tokenizer(encoder)
    prefix = encoder.spec["passage_prefix"]
    truncated = sum(
        len(tokenizer.encode(prefix + text).ids) > encoder.spec["max_length"] for text in inputs
    )
    vectors = encoder.encode(inputs, "passage")
    corpus = ManagedIndex(
        encoder, chunks, vectors, snapshot["chunks_sha256"], snapshot["corpus_revision"]
    )
    rows = []
    for case in cases:
        selected = (
            corpus.scoped(set(case["document_ids"]))
            if case.get("document_ids") is not None
            else corpus
        )
        ranking = selected.search(case["question"], 10, mode=mode)
        rows.append(
            {
                "case_id": case["id"],
                "split": case["split"],
                "document_ids": case.get("document_ids"),
                "metrics": metrics(ranking, case["gold"]),
                "ranking": [row["chunk_id"] for row in ranking],
            }
        )
    result = {
        "scope": "fixed_managed_corpus_retrieval_only",
        "time": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": digest(snapshot_path),
        "dataset_sha256": digest(labels_path),
        "corpus_revision": snapshot["corpus_revision"],
        "chunks_sha256": snapshot["chunks_sha256"],
        "configuration": {
            "embedding": model,
            "model_spec": MODEL_SPECS[model],
            "mode": mode,
            "split": split,
            "reference_chunking": snapshot["reference_configuration"],
            "chunks": len(chunks),
            "dimensions": vectors.shape[1],
            "passages_over_model_token_limit": truncated,
            "queries_over_model_token_limit": sum(
                len(tokenizer.encode(encoder.spec["query_prefix"] + case["question"]).ids)
                > encoder.spec["max_length"]
                for case in cases
            ),
            "fixed_chunks_across_models": True,
        },
        "limitations": [
            "Measures retrieval against manual labels; not answer correctness, OCR accuracy or agent routing.",
            "Chunks/embedding inputs remain fixed across models; tokenization and model limits can differ.",
            "Report development and held-out test separately; do not tune on test results.",
        ],
        "summary": {
            part: {"n": len(selected), **mean_metrics([row["metrics"] for row in selected])}
            for part in ("dev", "test", "all")
            if (selected := [row for row in rows if part == "all" or row["split"] == part])
        },
        "rows": rows,
    }
    _write_new(output, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_command = commands.add_parser(
        "snapshot", help="Freeze exact chunks and create blank manual labels"
    )
    snapshot_command.add_argument("--output", type=Path, required=True)
    snapshot_command.add_argument("--labels", type=Path, required=True)
    snapshot_command.add_argument("--documents", nargs="+")
    snapshot_command.add_argument("--model", choices=MODEL_SPECS)
    snapshot_command.add_argument("--allow-model-download", action="store_true")
    for command in ("validate", "run"):
        item = commands.add_parser(command)
        item.add_argument("--snapshot", type=Path, required=True)
        item.add_argument("--labels", type=Path, required=True)
        if command == "run":
            item.add_argument("--output", type=Path, required=True)
            item.add_argument("--model", choices=MODEL_SPECS)
            item.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
            item.add_argument("--split", choices=["all", "dev", "test"], default="all")
            item.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        result = export_snapshot(
            args.output, args.labels, args.model, args.documents, args.allow_model_download
        )
    elif args.command == "validate":
        result = {
            "cases": len(load_labels(args.labels, args.snapshot, load_snapshot(args.snapshot)))
        }
    else:
        result = run(
            args.snapshot,
            args.labels,
            args.output,
            args.model,
            args.mode,
            args.split,
            args.allow_model_download,
        )
        result = {"summary": result["summary"], "configuration": result["configuration"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
