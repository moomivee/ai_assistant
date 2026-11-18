"""Explicit revision checks for the two registered baseline PDF URLs.

This does not discover new issues, scrape publisher sites, or run on a schedule.
A changed file becomes an immutable pending version; its publication date must
be reviewed instead of inherited from the old file.
"""

import hashlib
import ipaddress
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import urlsplit

import httpx

from ..budget import bounded_timeout, check_budget
from ..config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MIB, settings
from ..logging import log
from .jobs import submit_ingestion
from .store import KnowledgeConflict, KnowledgeStore, UploadTooLarge

# Exact URLs are intentional. Editing a document's source_url never authorizes
# arbitrary server-side downloads, even from another path on an allowed host.
REGISTERED_URLS = {
    "eia-russia-2025": (
        "https://www.eia.gov/international/content/analysis/countries_long/"
        "Russia/pdf/Russia%20CAB_2025.pdf"
    ),
    "cbr-oct-2025": "https://www.cbr.ru/Content/Document/File/184551/forecast_251024.pdf",
}
REGISTERED_HOSTS = {"www.eia.gov", "www.cbr.ru"}
MAX_DOWNLOAD_SECONDS = 30
_dns_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-dns")
_locks = {document_id: threading.Lock() for document_id in REGISTERED_URLS}
POLICY_NOTE = (
    "Проверяется изменение PDF по прежней зарегистрированной ссылке. "
    "Новые выпуски по другим адресам автоматически не ищутся. "
    "Проверка запускается только вручную."
)


class UpdateUnavailable(ValueError):
    pass


class UpdateFetchError(ValueError):
    pass


def _registered_source(document_id: str) -> dict | None:
    expected = REGISTERED_URLS.get(document_id)
    if not expected:
        return None
    try:
        entries = json.loads((settings().data / "reports.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") != document_id:
            continue
        if entry.get("url") == expected:
            return entry
    return None


def update_capability(store: KnowledgeStore, document_id: str) -> dict:
    document = store.document(document_id)
    if document is None:
        raise KeyError(document_id)
    source = _registered_source(document_id) if document["builtin"] else None
    supported = bool(source and not document["archived"])
    return {
        "document_id": document_id,
        "supported": supported,
        "policy": "same_url_revision_check" if source else "manual_upload",
        "source_url": source["url"] if source else document.get("source_url"),
        "discovers_new_releases": False,
        "automatic_schedule": False,
        "note": POLICY_NOTE if source else "Новая версия этого документа загружается вручную.",
        "unavailable_reason": "Документ архивирован." if document["archived"] else None,
    }


def _validate_resolved_source(url: str) -> None:
    """Validate all resolved addresses before requesting a fixed publisher URL."""
    if url not in REGISTERED_URLS.values():
        raise UpdateUnavailable("Источник отсутствует в списке разрешённых PDF.")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in REGISTERED_HOSTS
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise UpdateUnavailable("Небезопасный адрес исходного PDF.")
    try:
        lookup = _dns_pool.submit(socket.getaddrinfo, parsed.hostname, 443, type=socket.SOCK_STREAM)
        try:
            addresses = lookup.result(timeout=bounded_timeout(5))
        except FutureTimeout:
            lookup.cancel()
            raise UpdateFetchError("Превышено время определения адреса сайта издателя.") from None
        if not addresses:
            raise ValueError
        for _, _, _, _, address in addresses:
            if not ipaddress.ip_address(address[0]).is_global:
                raise ValueError
    except UpdateFetchError:
        raise
    except OSError:
        raise UpdateFetchError("Не удалось определить адрес сайта издателя.") from None
    except ValueError:
        raise UpdateFetchError(
            "Сайт издателя разрешается в непубличный IP; загрузка запрещена."
        ) from None


def _download_pdf(url: str) -> bytes:
    check_budget()
    started = monotonic()
    _validate_resolved_source(url)
    check_budget()
    if monotonic() - started > MAX_DOWNLOAD_SECONDS:
        raise UpdateFetchError("Превышено время загрузки исходного PDF.")
    timeout = min(settings().timeout, 15, MAX_DOWNLOAD_SECONDS)
    try:
        # Ignore proxy-related environment variables for this narrowly scoped
        # server-side fetch. HTTPS certificate verification remains enabled.
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=bounded_timeout(timeout),
        ) as client:
            with client.stream(
                "GET", url, headers={"Accept": "application/pdf", "Accept-Encoding": "identity"}
            ) as response:
                if 300 <= response.status_code < 400:
                    raise UpdateFetchError(
                        "Источник перенаправляет на другой адрес. Проверьте новую ссылку вручную; "
                        "автоматический переход запрещён."
                    )
                response.raise_for_status()
                if str(response.url) != url:
                    raise UpdateFetchError(
                        "Адрес ответа не совпал с зарегистрированным источником."
                    )
                encoding = response.headers.get("content-encoding", "identity").lower()
                if encoding not in {"", "identity"}:
                    raise UpdateFetchError("Источник вернул сжатую передачу вместо исходного PDF.")
                length = response.headers.get("content-length")
                if length is not None:
                    try:
                        expected = int(length)
                        if expected < 0:
                            raise ValueError
                    except ValueError:
                        raise UpdateFetchError(
                            "Источник вернул некорректный размер файла."
                        ) from None
                    if expected > MAX_UPLOAD_BYTES:
                        raise UploadTooLarge(f"Размер исходного PDF превышает {MAX_UPLOAD_MIB} МБ.")
                content = bytearray()
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    check_budget()
                    if monotonic() - started > MAX_DOWNLOAD_SECONDS:
                        raise UpdateFetchError("Превышено время загрузки исходного PDF.")
                    if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
                        raise UploadTooLarge(f"Размер исходного PDF превышает {MAX_UPLOAD_MIB} МБ.")
                    content.extend(chunk)
                if not content.startswith(b"%PDF-"):
                    raise UpdateFetchError("По исходной ссылке получен не PDF-документ.")
                return bytes(content)
    except httpx.HTTPStatusError as exc:
        raise UpdateFetchError(
            f"Сайт издателя вернул HTTP {exc.response.status_code}. Проверьте ссылку вручную."
        ) from None
    except httpx.HTTPError:
        raise UpdateFetchError(
            "Не удалось скачать исходный PDF. Проверьте доступность сайта издателя."
        ) from None


def check_update(store: KnowledgeStore, document_id: str) -> dict:
    capability = update_capability(store, document_id)
    if not capability["supported"]:
        raise UpdateUnavailable(capability.get("unavailable_reason") or capability["note"])
    source = _registered_source(document_id)
    if source is None:
        raise UpdateUnavailable("Источник не зарегистрирован для проверки обновления.")
    lock = _locks[document_id]
    if not lock.acquire(blocking=False):
        raise KnowledgeConflict("Проверка этого источника уже выполняется.")
    try:
        initial = store.document(document_id)
        log("knowledge.update_check_started", document_id=document_id)
        content = _download_pdf(source["url"])
        checksum = hashlib.sha256(content).hexdigest()
        current = store.document(document_id)
        if current["current_version_id"] != initial["current_version_id"] or current["archived"]:
            raise KnowledgeConflict(
                "Документ изменился во время проверки. Обновите список и повторите."
            )
        known = next(
            (version for version in store.versions(document_id) if version["sha256"] == checksum),
            None,
        )
        if known:
            status = "unchanged" if current["sha256"] == checksum else "known_version"
            result = {
                "status": status,
                "document": current,
                "sha256": checksum,
                "known_version_id": known["version_id"],
                "checked_at": datetime.now(UTC).isoformat(),
                "date_review_required": current.get("published_date") is None,
                "notices": [
                    "Исходный файл не изменился."
                    if status == "unchanged"
                    else "По ссылке находится уже сохранённая старая версия. Текущая версия не заменена.",
                    POLICY_NOTE,
                ],
            }
        else:
            document = store.upload(
                filename=source["file"],
                content=content,
                title=current["title"],
                date="",
                source_url=source["url"],
                publisher=current.get("publisher") or "",
                document_id=document_id,
            )
            if document["ingestion_status"] == "queued":
                submit_ingestion(store, document["current_version_id"])
            result = {
                "status": "queued",
                "document": document,
                "sha256": checksum,
                "checked_at": datetime.now(UTC).isoformat(),
                "date_review_required": True,
                "notices": [
                    "Сохранена новая версия файла и отправлена на извлечение текста. "
                    "Предыдущая версия сохранена и используется до успешной обработки новой.",
                    "Дата публикации изменившегося файла неизвестна и требует ручной проверки; "
                    "дата старого отчёта не перенесена.",
                    POLICY_NOTE,
                ],
            }
        log("knowledge.update_check_completed", document_id=document_id, status=result["status"])
        return result
    finally:
        lock.release()
