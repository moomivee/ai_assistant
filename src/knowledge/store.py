"""SQLite catalog with immutable originals and extracted document versions.

``ready`` means extraction succeeded. Embeddings are built lazily by the RAG
layer, which acknowledges the exact corpus revision with ``mark_indexed``.
One application process owns the ingestion worker; this is a local workbench,
not a distributed job queue. Archival never removes old citation targets.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date as Date
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from ..config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MIB, settings
from ..logging import log

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}
SUCCESS = ("ready", "partial")


class UploadTooLarge(ValueError):
    pass


class KnowledgeConflict(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _metadata(filename, title, date, source_url, publisher) -> dict:
    if (
        not filename
        or len(filename) > 200
        or any(char in filename for char in ("/", "\\", "\x00", "\r", "\n"))
        or filename in {".", ".."}
    ):
        raise ValueError("Некорректное имя файла: укажите имя без пути.")
    suffix = Path(filename).suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError("Поддерживаются PDF, DOCX, XLSX, CSV, PNG и JPEG.")
    title = (title or Path(filename).stem).strip()
    publisher = (publisher or "").strip()
    if not title or len(title) > 500 or len(publisher) > 200:
        raise ValueError("Название обязательно: до 500 символов; издатель — до 200.")
    if any("\x00" in value for value in (title, publisher)):
        raise ValueError("Метаданные содержат недопустимые символы.")
    date = (date or "").strip()
    if date:
        try:
            parsed_date = Date.fromisoformat(date)
            if parsed_date.isoformat() != date:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Дата публикации должна иметь формат ГГГГ-ММ-ДД.") from exc
    source_url = (source_url or "").strip()
    if source_url:
        try:
            parsed = urlsplit(source_url)
            if (
                len(source_url) > 2000
                or parsed.scheme not in {"https", "http"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or any(char.isspace() for char in source_url)
            ):
                raise ValueError
        except ValueError as exc:
            raise ValueError("Источник должен быть HTTP(S)-ссылкой без логина и пароля.") from exc
    return {
        "title": title,
        "published_date": date or None,
        "source_url": source_url or None,
        "publisher": publisher or None,
        "media_type": MEDIA_TYPES[suffix],
    }


class KnowledgeStore:
    def __init__(self, root: Path, *, recover: bool = True):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "originals").mkdir(exist_ok=True)
        (self.root / "blocks").mkdir(exist_ok=True)
        self.database = self.root / "catalog.sqlite3"
        self._lock = threading.RLock()
        with self._connection() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    builtin INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS versions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    sha256 TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    original_path TEXT NOT NULL,
                    builtin INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    warnings TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    parser_version TEXT,
                    block_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(document_id, sha256)
                );
                CREATE INDEX IF NOT EXISTS versions_document ON versions(document_id, sequence);
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            if recover:
                interrupted = conn.execute(
                    "UPDATE versions SET status='error', error=?, processed_at=? "
                    "WHERE status IN ('queued', 'processing')",
                    ("Обработка прервана перезапуском. Повторите загрузку в обработку.", _now()),
                ).rowcount
                if interrupted:
                    log("knowledge.jobs_recovered", count=interrupted)

    @contextmanager
    def _connection(self):
        # The lock also makes catalog + immutable block reads an atomic snapshot
        # within the supported single-process application deployment.
        with self._lock:
            conn = sqlite3.connect(self.database, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _path(self, relative: str) -> Path:
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Недопустимый путь файла.")
        return result

    def _version(self, row) -> dict:
        value = dict(row)
        value.update(json.loads(value.pop("metadata")))
        value["warnings"] = json.loads(value["warnings"])
        value["builtin"] = bool(value["builtin"])
        value.pop("original_path")
        value.pop("sequence")
        value["version_id"] = value["id"]
        value["date"] = value["published_date"]
        value["local_url"] = (
            f"/api/knowledge/documents/{value['document_id']}/versions/{value['id']}/file"
        )
        return value

    def _document(self, conn, row, *, indexed: bool | None = None) -> dict:
        versions = conn.execute(
            "SELECT * FROM versions WHERE document_id=? ORDER BY sequence DESC", (row["id"],)
        ).fetchall()
        latest = self._version(versions[0])
        active = next((v for v in versions if v["status"] in SUCCESS), None)
        if indexed is None:
            indexed = self._is_indexed(conn)
        return {
            **latest,
            "id": row["id"],
            "document_id": row["id"],
            "builtin": bool(row["builtin"]),
            "archived": bool(row["archived"]),
            "status": "archived" if row["archived"] else latest["status"],
            "ingestion_status": latest["status"],
            "created_at": row["created_at"],
            "updated_at": latest["created_at"],
            "current_version_id": latest["id"],
            "active_version_id": active["id"] if active and not row["archived"] else None,
            "versions_count": len(versions),
            "index_status": ("indexed" if indexed else "pending")
            if active and not row["archived"]
            else "unavailable",
            "update_policy": "manual_upload",
            "update_note": "Источник указан для проверки. Новый выпуск загружается вручную.",
        }

    def list_documents(self, include_archived: bool = False) -> list[dict]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE archived=0 OR ? ORDER BY created_at DESC, id",
                (include_archived,),
            ).fetchall()
            indexed = self._is_indexed(conn)
            return [self._document(conn, row, indexed=indexed) for row in rows]

    def document(self, document_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return self._document(conn, row) if row else None

    def versions(self, document_id: str) -> list[dict]:
        with self._connection() as conn:
            return [
                self._version(row)
                for row in conn.execute(
                    "SELECT * FROM versions WHERE document_id=? ORDER BY sequence DESC",
                    (document_id,),
                )
            ]

    def version(self, document_id: str, version_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM versions WHERE document_id=? AND id=?", (document_id, version_id)
            ).fetchone()
            return self._version(row) if row else None

    def version_file(self, document_id: str, version_id: str) -> Path | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT original_path FROM versions WHERE document_id=? AND id=?",
                (document_id, version_id),
            ).fetchone()
            return self._path(row["original_path"]) if row else None

    def _blocks(self, version_id: str) -> list[dict]:
        # Version identifiers come from SQLite, never from a file-system URL.
        path = self._path(f"blocks/{version_id}.json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def blocks(self, document_id: str, version_id: str | None = None) -> list[dict]:
        with self._connection() as conn:
            if version_id:
                row = conn.execute(
                    "SELECT id FROM versions WHERE document_id=? AND id=?",
                    (document_id, version_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id FROM versions WHERE document_id=? AND status IN ('ready','partial') "
                    "ORDER BY sequence DESC LIMIT 1",
                    (document_id,),
                ).fetchone()
            return self._blocks(row["id"]) if row else []

    def upload(
        self,
        filename: str,
        content: bytes,
        title: str = "",
        date: str = "",
        source_url: str = "",
        publisher: str = "",
        document_id: str | None = None,
        *,
        _builtin: bool = False,
    ) -> dict:
        if len(content) > MAX_UPLOAD_BYTES:
            raise UploadTooLarge(f"Размер документа превышает {MAX_UPLOAD_MIB} МБ.")
        if not content:
            raise ValueError("Файл пуст.")
        metadata = _metadata(filename, title, date, source_url, publisher)
        checksum = hashlib.sha256(content).hexdigest()
        with self._connection() as conn:
            existing = None
            if document_id:
                document = conn.execute(
                    "SELECT * FROM documents WHERE id=?", (document_id,)
                ).fetchone()
                if document is None and not _builtin:
                    raise KeyError(document_id)
                if document is not None and document["archived"]:
                    raise KnowledgeConflict("Документ архивирован. Создайте новый документ.")
                existing = conn.execute(
                    "SELECT document_id FROM versions WHERE document_id=? AND sha256=?",
                    (document_id, checksum),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT v.document_id FROM versions v JOIN documents d ON d.id=v.document_id "
                    "WHERE v.sha256=? AND d.archived=0 ORDER BY v.sequence DESC LIMIT 1",
                    (checksum,),
                ).fetchone()
            if existing:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id=?", (existing["document_id"],)
                ).fetchone()
                result = self._document(conn, row)
                result["deduplicated"] = True
                return result
            document_id = document_id or uuid4().hex
            version_id = uuid4().hex
            relative = f"originals/{checksum}"
            path = self._path(relative)
            if not path.exists():
                # Atomic rename publishes a fully written immutable object.
                temporary = self._path(f"originals/.{uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(content)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Контрольная сумма сохранённого оригинала не совпала.")
            conn.execute(
                "INSERT OR IGNORE INTO documents(id,builtin,created_at) VALUES (?,?,?)",
                (document_id, int(_builtin), _now()),
            )
            conn.execute(
                "INSERT INTO versions(id,document_id,sha256,filename,size_bytes,original_path,"
                "builtin,status,metadata,created_at) VALUES (?,?,?,?,?,?,?,'queued',?,?)",
                (
                    version_id,
                    document_id,
                    checksum,
                    filename,
                    len(content),
                    relative,
                    int(_builtin),
                    _json(metadata),
                    _now(),
                ),
            )
            row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            result = self._document(conn, row)
        log(
            "knowledge.uploaded", document_id=document_id, version_id=version_id, bytes=len(content)
        )
        return result

    def process(self, version_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
            if row is None:
                raise KeyError(version_id)
            if row["status"] != "queued":
                return self.document(row["document_id"])
            conn.execute("UPDATE versions SET status='processing' WHERE id=?", (version_id,))
            original = self._path(row["original_path"])
        log("knowledge.processing", version_id=version_id)
        try:
            from .extractors import extract

            if hashlib.sha256(original.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Контрольная сумма оригинала не совпала; загрузите файл заново.")
            result = extract(original, row["filename"])
            blocks = []
            identifiers = set()
            for i, block in enumerate(result["blocks"], 1):
                text = str(block.get("text", "")).strip()
                if not text:
                    continue
                block_id = str(block.get("block_id") or block.get("id") or f"b{i}")
                if block_id in identifiers:
                    raise ValueError("Извлекатель вернул повторяющиеся идентификаторы фрагментов.")
                identifiers.add(block_id)
                blocks.append(
                    {
                        **block,
                        "block_id": block_id,
                        "text": text,
                        "page": block.get("page"),
                        "locator": block.get("locator") or {},
                        "method": block.get("method", "native"),
                    }
                )
            if not blocks:
                raise ValueError("В документе не удалось извлечь текст.")
            warnings = [str(item) for item in result.get("warnings", [])]
            status = "partial" if warnings else "ready"
            temporary = self._path(f"blocks/.{version_id}.tmp")
            try:
                temporary.write_text(_json(blocks), encoding="utf-8")
                temporary.replace(self._path(f"blocks/{version_id}.json"))
            finally:
                temporary.unlink(missing_ok=True)
            with self._connection() as conn:
                conn.execute(
                    "UPDATE versions SET status=?,warnings=?,error=NULL,processed_at=?,"
                    "parser_version=?,block_count=? WHERE id=? AND status='processing'",
                    (
                        status,
                        _json(warnings),
                        _now(),
                        result.get("parser_version", "unknown"),
                        len(blocks),
                        version_id,
                    ),
                )
            log("knowledge.processed", version_id=version_id, status=status, blocks=len(blocks))
        except Exception as exc:
            # Extractors raise curated ValueErrors; unexpected errors never echo
            # raw file content, host paths, tracebacks, or external credentials.
            error = (
                str(exc)[:1000]
                if isinstance(exc, ValueError)
                else ("Не удалось обработать документ. Проверьте файл и повторите попытку.")
            )
            with self._connection() as conn:
                conn.execute(
                    "UPDATE versions SET status='error',error=?,processed_at=? "
                    "WHERE id=? AND status='processing'",
                    (error, _now(), version_id),
                )
            log(
                "knowledge.processing_failed",
                logging.WARNING,
                version_id=version_id,
                error_type=type(exc).__name__,
            )
        return self.document(row["document_id"])

    def retry(self, document_id: str, version_id: str) -> dict:
        with self._connection() as conn:
            document = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            row = conn.execute(
                "SELECT status FROM versions WHERE id=? AND document_id=?",
                (version_id, document_id),
            ).fetchone()
            if not row or not document:
                raise KeyError(version_id)
            if document["archived"]:
                raise KnowledgeConflict("Архивированный документ нельзя отправить в обработку.")
            if row["status"] != "error":
                raise KnowledgeConflict("Повторная обработка доступна только для ошибок.")
            conn.execute(
                "UPDATE versions SET status='queued',error=NULL,processed_at=NULL WHERE id=?",
                (version_id,),
            )
            return self._document(conn, document)

    def archive(self, document_id: str) -> dict:
        with self._connection() as conn:
            if not conn.execute("SELECT id FROM documents WHERE id=?", (document_id,)).fetchone():
                raise KeyError(document_id)
            conn.execute("UPDATE documents SET archived=1 WHERE id=?", (document_id,))
        log("knowledge.archived", document_id=document_id)
        return self.document(document_id)

    def _active_versions(self, conn):
        return conn.execute(
            "SELECT v.* FROM versions v JOIN documents d ON d.id=v.document_id "
            "WHERE d.archived=0 AND v.sequence=(SELECT MAX(v2.sequence) FROM versions v2 "
            "WHERE v2.document_id=d.id AND v2.status IN ('ready','partial')) ORDER BY d.id"
        ).fetchall()

    def _revision(self, conn) -> str:
        rows = self._active_versions(conn)
        # Archive and successful version replacement alter this fingerprint;
        # pending uploads cannot evict the last usable version from retrieval.
        content = [
            (row["document_id"], row["id"], row["sha256"], row["parser_version"]) for row in rows
        ]
        return hashlib.sha256(_json(content).encode()).hexdigest()

    def corpus_revision(self) -> str:
        with self._connection() as conn:
            return self._revision(conn)

    def _is_indexed(self, conn) -> bool:
        row = conn.execute(
            "SELECT value FROM state WHERE key=?", (f"indexed_revision:{settings().embedding}",)
        ).fetchone()
        return bool(row and row["value"] == self._revision(conn))

    def mark_indexed(self, revision: str, model: str | None = None) -> bool:
        with self._connection() as conn:
            if revision != self._revision(conn):
                return False
            conn.execute(
                "INSERT OR REPLACE INTO state(key,value) VALUES (?,?)",
                (f"indexed_revision:{model or settings().embedding}", revision),
            )
            return True

    def corpus_snapshot(self) -> dict:
        with self._connection() as conn:
            conn.execute("BEGIN")
            rows = self._active_versions(conn)
            chunks, baseline = [], []
            for row in rows:
                version = self._version(row)
                if row["builtin"]:
                    baseline.append(version)
                    continue
                for block in self._blocks(row["id"]):
                    chunks.append(
                        {
                            **block,
                            "kind": "report",
                            "report": row["document_id"],
                            "document_id": row["document_id"],
                            "version_id": row["id"],
                            "title": version["title"],
                            "date": version["published_date"],
                            "url": version["source_url"] or version["local_url"],
                            "local_url": version["local_url"],
                            "publisher": version["publisher"],
                            "filename": version["filename"],
                        }
                    )
            return {
                "revision": self._revision(conn),
                "active_document_ids": [row["document_id"] for row in rows],
                "baseline_document_ids": [v["document_id"] for v in baseline],
                "builtin_documents": baseline,
                "chunks": chunks,
            }

    def ready_chunks(self) -> list[dict]:
        return self.corpus_snapshot()["chunks"]

    def active_document_ids(self) -> list[str]:
        return self.corpus_snapshot()["active_document_ids"]

    def baseline_document_ids(self) -> list[str]:
        return self.corpus_snapshot()["baseline_document_ids"]

    def register_baseline(self, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            # An existing catalog is authoritative, including archival choices.
            if self.document(entry["id"]) is not None:
                continue
            path = (manifest_path.parent / "reports" / entry["file"]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()) or not path.is_file():
                log("knowledge.baseline_missing", logging.WARNING, document_id=entry["id"])
                continue
            content = path.read_bytes()
            if entry.get("sha256") and hashlib.sha256(content).hexdigest() != entry["sha256"]:
                log("knowledge.baseline_checksum_failed", logging.WARNING, document_id=entry["id"])
                continue
            document = self.upload(
                entry["file"],
                content,
                entry["title"],
                entry.get("date", ""),
                entry.get("url", ""),
                entry.get("publisher", ""),
                entry["id"],
                _builtin=True,
            )
            self.process(document["current_version_id"])


@lru_cache(maxsize=8)
def _store_for_root(root: str) -> KnowledgeStore:
    store = KnowledgeStore(Path(root))
    store.register_baseline(Path(root).parent / "reports.json")
    return store


_initialization_lock = threading.Lock()


def get_store() -> KnowledgeStore:
    # lru_cache alone can execute its initializer twice on concurrent misses;
    # a second initializer must not recover the first worker's active jobs.
    with _initialization_lock:
        return _store_for_root(str(settings().data / "knowledge"))
