"""Versioned retrieval over the built-in reports and the editable knowledge base.

``index.index()`` deliberately remains the frozen evaluation corpus. Production
retrieval uses this module: one encoder, one ranking, and a catalog snapshot whose
revision is checked again before results can leave the retriever.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections import Counter, OrderedDict
from pathlib import Path

import faiss
import numpy as np
from filelock import FileLock
from tokenizers import Tokenizer

from ..config import manifest, settings
from ..knowledge.store import get_store
from ..logging import log
from .index import MODEL_SPECS, ReportIndex, digest, index

MANAGED_VERSION = "managed-token-windows-v3"
_lock = threading.RLock()
_cache: OrderedDict[tuple, "ManagedIndex"] = OrderedDict()


class DocumentScopeError(ValueError):
    """An explicit document selection cannot currently be searched."""


def _document_id(chunk: dict) -> str:
    return chunk.get("document_id") or chunk["report"]


def _ready_ids(snapshot: dict) -> set[str]:
    return set(snapshot["baseline_document_ids"]) | {
        _document_id(chunk) for chunk in snapshot["chunks"] if chunk.get("text", "").strip()
    }


def _baseline_chunk(row: dict, versions: dict[str, dict]) -> dict:
    """Attach the immutable catalog version without changing legacy chunk IDs."""
    document_id = _document_id(row)
    result = dict(row, document_id=document_id, locator={"page": row.get("page")})
    version = versions.get(document_id)
    if version:
        version_id = version.get("version_id") or version["id"]
        result["version_id"] = version_id
        path = version.get("local_url") or (
            f"/api/knowledge/documents/{document_id}/versions/{version_id}/file"
        )
        result["local_url"] = path.split("#", 1)[0] + (
            f"#page={row['page']}" if row.get("page") else ""
        )
        result["method"] = "native"
    return result


def _scope(snapshot: dict, document_ids: list[str] | None, store) -> set[str]:
    ready = _ready_ids(snapshot)
    if document_ids is None:
        return ready
    selected = set(document_ids)
    missing = selected - set(snapshot["active_document_ids"])
    unavailable = selected - ready - missing
    for document_id in tuple(missing):
        document = store.document(document_id)
        if document and not document["archived"] and not document["active_version_id"]:
            unavailable.add(document_id)
            missing.remove(document_id)
    if missing:
        raise DocumentScopeError(
            "Документы отсутствуют или архивированы: " + ", ".join(sorted(missing))
        )
    if unavailable:
        raise DocumentScopeError(
            "Документы ещё не содержат доступного для поиска текста: "
            + ", ".join(sorted(unavailable))
        )
    return selected


def token_chunks(blocks: list[dict], encoder) -> list[dict]:
    """Keep exact source substrings; size windows using the encoder's tokenizer.

    A separate tokenizer avoids mutating truncation/padding on the shared encoder.
    Prefixes, special tokens, and a repeated table header count toward the limit.
    Table headers are embedding context; ``text`` always remains an exact excerpt
    of the immutable source block, identified by its character offsets.
    """
    tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    maximum = encoder.spec["max_length"]
    prefix = encoder.spec["passage_prefix"]
    budget = maximum - len(tokenizer.encode(prefix).ids)
    if budget < 2:
        raise ValueError("Embedding token limit leaves no room for source text")
    overlap = min(32, max(1, budget // 5))
    chunks = []
    for block in blocks:
        text = block.get("text", "")
        offsets = [
            pair
            for pair in tokenizer.encode(text, add_special_tokens=False).offsets
            if pair[1] > pair[0]
        ]
        if not offsets or not text.strip():
            continue
        header = str(block.get("table_header") or "")
        # An exceptionally wide header must not consume the entire chunk budget.
        if len(tokenizer.encode(prefix + header).ids) > maximum // 2:
            header = ""
        start = 0
        while start < len(offsets):
            end = min(len(offsets), start + budget)
            while end > start:
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                excerpt = text[char_start:char_end]
                repeated = header + "\n" if header and not excerpt.startswith(header) else ""
                embedding_text = repeated + excerpt
                token_count = len(tokenizer.encode(prefix + embedding_text).ids)
                if token_count <= maximum:
                    break
                end -= 1
            if end == start:
                raise ValueError("Source token cannot fit in the embedding model context")
            doc = _document_id(block)
            local_url = block.get("local_url")
            if local_url and block.get("page"):
                local_url = local_url.split("#", 1)[0] + f"#page={block['page']}"
            chunks.append(
                dict(
                    block,
                    report=doc,
                    document_id=doc,
                    kind="report",
                    local_url=local_url,
                    chunk_id=f"{doc}:{block['version_id']}:{block['block_id']}:t{start}-{end}",
                    text=excerpt,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=start,
                    token_end=end,
                    embedding_tokens=token_count,
                    _embedding_text=embedding_text,
                )
            )
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
    return chunks


class ManagedIndex(ReportIndex):
    """Immutable merged corpus; shares ReportIndex's dense/hybrid ranking code."""

    def __init__(self, encoder, chunks, vectors, fingerprint, revision):
        self.encoder = encoder
        self.model = encoder.model
        self.chunks = chunks
        self.fingerprint = fingerprint
        self.revision = revision
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.terms = [Counter(re.findall(r"\w+", c["text"].casefold())) for c in chunks]
        self.document_frequency = Counter(term for row in self.terms for term in row)
        self.average_length = (
            sum(sum(row.values()) for row in self.terms) / len(self.terms) if self.terms else 0
        )

    def scoped(self, document_ids: set[str]) -> "ManagedIndex":
        positions = [i for i, row in enumerate(self.chunks) if _document_id(row) in document_ids]
        if len(positions) == len(self.chunks):
            return self
        vectors = np.array([self.index.reconstruct(i) for i in positions], dtype="float32").reshape(
            len(positions), self.index.d
        )
        return ManagedIndex(
            self.encoder,
            [self.chunks[i] for i in positions],
            vectors,
            self.fingerprint,
            self.revision,
        )

    def search(self, question: str, k: int, mode: str | None = None) -> list[dict]:
        if k < 1 or not self.chunks:
            return []
        return super().search(question, k, mode)


def _status_path(model: str) -> Path:
    name = hashlib.sha256(model.encode()).hexdigest()[:16]
    return settings().cache / f"managed-status-{name}.json"


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        tmp = Path(f.name)
        json.dump(value, f, ensure_ascii=False)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _build(snapshot: dict, model: str) -> ManagedIndex:
    cfg = settings()
    catalog_versions = {
        row["document_id"]: row for row in snapshot["builtin_documents"] if row.get("sha256")
    }
    if catalog_versions:
        current_reports = {row["id"]: row for row in manifest()}
        for document_id, version in catalog_versions.items():
            report = current_reports.get(document_id)
            if report is None or digest(cfg.data / "reports" / report["file"]) != version["sha256"]:
                raise ValueError(
                    f"Встроенный отчёт {document_id} отличается от версии в базе знаний. "
                    "Загрузите изменённый файл как новый выпуск документа через базу знаний."
                )
    baseline = index(model)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "version": MANAGED_VERSION,
                "revision": snapshot["revision"],
                "baseline": baseline.fingerprint,
                "baseline_documents": sorted(snapshot["baseline_document_ids"]),
                "blocks": snapshot["chunks"],
                "model": model,
                "model_spec": MODEL_SPECS[model],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    folder = cfg.cache / ("managed-index-" + fingerprint[:24])
    cfg.cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder) + ".lock"):
        if not folder.is_dir():
            active_baseline = set(snapshot["baseline_document_ids"])
            positions = [
                i for i, row in enumerate(baseline.chunks) if _document_id(row) in active_baseline
            ]
            versions = {v["document_id"]: v for v in snapshot["builtin_documents"]}
            chunks = [_baseline_chunk(baseline.chunks[i], versions) for i in positions]
            vectors = np.array(
                [baseline.index.reconstruct(i) for i in positions], dtype="float32"
            ).reshape(len(positions), baseline.index.d)
            uploaded = token_chunks(snapshot["chunks"], baseline.encoder)
            if uploaded:
                new_vectors = baseline.encoder.encode(
                    [row.pop("_embedding_text") for row in uploaded], "passage"
                )
                vectors = np.vstack([vectors, new_vectors])
                chunks.extend(uploaded)
            merged = faiss.IndexFlatIP(baseline.index.d)
            merged.add(np.ascontiguousarray(vectors, dtype="float32"))
            temporary = Path(tempfile.mkdtemp(dir=cfg.cache))
            try:
                faiss.write_index(merged, str(temporary / "index.faiss"))
                (temporary / "chunks.json").write_text(
                    json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
                )
                temporary.rename(folder)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            log(
                "rag.managed_index_built",
                embedding=model,
                revision=snapshot["revision"],
                chunks=len(chunks),
                fingerprint=fingerprint,
            )
        dense = faiss.read_index(str(folder / "index.faiss"))
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    vectors = dense.reconstruct_n(0, dense.ntotal)
    return ManagedIndex(baseline.encoder, chunks, vectors, fingerprint, snapshot["revision"])


def _snapshot_index(snapshot: dict, model: str) -> ManagedIndex:
    cfg = settings()
    key = (str(cfg.data), str(cfg.cache), model, snapshot["revision"])
    with _lock:
        result = _cache.get(key)
        if result is None:
            result = _build(snapshot, model)
            _cache[key] = result
            while len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(key)
        return result


def _record_ready(store, result: ManagedIndex) -> bool:
    if not store.mark_indexed(result.revision, model=result.model):
        return False
    _write_json_atomic(
        _status_path(result.model),
        {
            "revision": result.revision,
            "fingerprint": result.fingerprint,
            "dimensions": result.index.d,
            "chunks": len(result.chunks),
            "documents": len({_document_id(row) for row in result.chunks}),
            "data_dir": str(settings().data),
            "chunking_version": MANAGED_VERSION,
        },
    )
    return True


def managed_index(model: str | None = None) -> ManagedIndex:
    """Build the current corpus, retrying if publication/archive raced the build."""
    model = model or settings().embedding
    store = get_store()
    for _ in range(3):
        result = _snapshot_index(store.corpus_snapshot(), model)
        if _record_ready(store, result):
            return result
    raise RuntimeError("База знаний изменяется. Повторите поиск после завершения обработки.")


def managed_search(
    question: str,
    k: int,
    document_ids: list[str] | None = None,
    model: str | None = None,
    mode: str | None = None,
) -> list[dict]:
    """Rank only the selected, active versions; None means all ready documents."""
    model = model or settings().embedding
    store = get_store()
    for _ in range(3):
        snapshot = store.corpus_snapshot()
        selected = _scope(snapshot, document_ids, store)
        if not selected or k < 1:
            return []
        corpus = _snapshot_index(snapshot, model)
        result = corpus.scoped(selected).search(question, k, mode)
        # Final revision check is the read's linearization point. A corpus which
        # changed during embedding or ranking never produces a stale answer.
        if _record_ready(store, corpus):
            return result
    raise RuntimeError("База знаний изменяется. Повторите поиск после завершения обработки.")


def managed_status(model: str | None = None) -> dict:
    """Catalog/index metadata only: never instantiate or download an encoder."""
    model = model or settings().embedding
    spec = MODEL_SPECS[model]
    revision = get_store().corpus_revision()
    saved = {}
    try:
        saved = json.loads(_status_path(model).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if saved.get("data_dir") != str(settings().data):
        saved = {}
    current = saved.get("revision") == revision and bool(saved.get("fingerprint"))
    return {
        "model": model,
        "model_repo": spec["repo"],
        "model_revision": spec["revision"],
        "max_tokens": spec["max_length"],
        "search_mode": settings().rag_search,
        "revision": revision,
        "indexed_revision": saved.get("revision"),
        "status": "ready" if current else "pending",
        "fingerprint": saved.get("fingerprint") if current else None,
        "dimensions": saved.get("dimensions"),
        "chunks": saved.get("chunks") if current else None,
        "documents": saved.get("documents") if current else None,
        "chunking_version": MANAGED_VERSION,
        "baseline_chunking": "legacy evaluation chunks retained",
    }
