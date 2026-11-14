import hashlib
import json
import socket

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import settings
from src.knowledge import extractors, updates, updates_api
from src.knowledge.store import KnowledgeStore

DOCUMENT = "eia-russia-2025"
URL = updates.REGISTERED_URLS[DOCUMENT]
OLD = b"%PDF-1.4\nOriginal fixed test data"
NEW = b"%PDF-1.4\nRevised fixed test data"


@pytest.fixture
def updater(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    manifest = [
        {
            "id": DOCUMENT,
            "title": "EIA Russia baseline",
            "date": "2025-07-24",
            "file": "Russia.pdf",
            "url": URL,
            "sha256": hashlib.sha256(OLD).hexdigest(),
        }
    ]
    (data / "reports.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("DATA_DIR", str(data))
    settings.cache_clear()
    store = KnowledgeStore(data / "knowledge")
    original = store.upload(
        "Russia.pdf",
        OLD,
        "EIA Russia baseline",
        "2025-07-24",
        URL,
        document_id=DOCUMENT,
        _builtin=True,
    )
    monkeypatch.setattr(
        extractors,
        "extract",
        lambda *args: {
            "blocks": [{"block_id": "b1", "page": 1, "text": "Test evidence"}],
            "warnings": [],
            "parser_version": "test",
        },
    )
    store.process(original["current_version_id"])
    scheduled = []
    monkeypatch.setattr(
        updates, "submit_ingestion", lambda store, version: scheduled.append(version)
    )
    monkeypatch.setattr(updates_api, "get_store", lambda: store)
    monkeypatch.setattr(
        updates.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        ],
    )
    app = FastAPI()
    app.include_router(updates_api.router)
    with TestClient(app) as client:
        yield client, store, scheduled, original


def mock_download(monkeypatch, content=NEW, status=200, headers=None, on_request=None):
    original_client = httpx.Client
    calls, configurations = [], []

    def handle(request):
        calls.append(request)
        if on_request:
            on_request(request)
        return httpx.Response(status, content=content, headers=headers)

    transport = httpx.MockTransport(handle)

    def client(**kwargs):
        configurations.append(kwargs)
        return original_client(transport=transport, **kwargs)

    monkeypatch.setattr(updates.httpx, "Client", client)
    return calls, configurations


def check(client, **kwargs):
    return client.post(f"/api/knowledge/documents/{DOCUMENT}/check-update", **kwargs)


def test_capability_honestly_describes_same_url_check(updater):
    client, store, _, _ = updater
    capability = client.get(f"/api/knowledge/documents/{DOCUMENT}/update-capability").json()
    assert capability["supported"] is True
    assert capability["policy"] == "same_url_revision_check"
    assert capability["source_url"] == URL
    assert capability["discovers_new_releases"] is False
    assert capability["automatic_schedule"] is False
    assert "другим адресам" in capability["note"]
    user_doc = store.upload("personal.pdf", OLD + b"User", "User", source_url=URL)
    path = f"/api/knowledge/documents/{user_doc['id']}"
    assert client.get(path + "/update-capability").json()["supported"] is False
    assert client.post(path + "/check-update").status_code == 422
    assert client.get("/api/knowledge/documents/unknown/update-capability").status_code == 404


def test_changed_source_creates_new_immutable_version_with_unknown_date(updater, monkeypatch):
    client, store, scheduled, old = updater
    calls, config = mock_download(monkeypatch)
    response = check(client, headers={"Origin": "http://testserver"})
    assert response.status_code == 202
    data = response.json()
    document = data["document"]
    assert data["status"] == "queued" and data["date_review_required"] is True
    assert document["published_date"] is None
    assert document["source_url"] == URL
    assert document["title"] == "EIA Russia baseline"
    assert document["active_version_id"] == old["current_version_id"]
    assert document["versions_count"] == 2
    assert scheduled == [document["current_version_id"]]
    assert store.version_file(DOCUMENT, old["current_version_id"]).read_bytes() == OLD
    assert store.version_file(DOCUMENT, document["current_version_id"]).read_bytes() == NEW
    assert len(calls) == 1 and str(calls[0].url) == URL
    assert config[0]["trust_env"] is False and config[0]["follow_redirects"] is False
    assert calls[0].headers["accept-encoding"] == "identity"
    store.process(document["current_version_id"])
    assert store.document(DOCUMENT)["active_version_id"] == document["current_version_id"]


def test_unchanged_file_does_not_create_version_or_schedule_ingestion(updater, monkeypatch):
    client, store, scheduled, old = updater
    calls, _ = mock_download(monkeypatch, OLD)
    response = check(client)
    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"
    assert response.json()["known_version_id"] == old["current_version_id"]
    assert scheduled == [] and len(store.versions(DOCUMENT)) == 1
    assert len(calls) == 1


def test_invalid_pdf_body_retains_old_ready_version_after_extraction_error(updater, monkeypatch):
    client, store, _, old = updater
    mock_download(monkeypatch, b"%PDF-this-is-not-a-valid-document")
    response = check(client)
    assert response.status_code == 202

    def fail(*args):
        raise ValueError("Невозможно прочитать PDF.")

    monkeypatch.setattr(extractors, "extract", fail)
    version_id = response.json()["document"]["current_version_id"]
    store.process(version_id)
    document = store.document(DOCUMENT)
    assert document["ingestion_status"] == "error"
    assert document["active_version_id"] == old["current_version_id"]
    assert len(store.versions(DOCUMENT)) == 2


def test_known_old_version_never_rolls_back_current_document(updater, monkeypatch):
    client, store, scheduled, old = updater
    latest = store.upload("Russia.pdf", NEW, "Updated", source_url=URL, document_id=DOCUMENT)
    store.process(latest["current_version_id"])
    mock_download(monkeypatch, OLD)
    response = check(client)
    assert response.status_code == 200 and response.json()["status"] == "known_version"
    assert response.json()["known_version_id"] == old["current_version_id"]
    assert response.json()["document"]["current_version_id"] == latest["current_version_id"]
    assert len(store.versions(DOCUMENT)) == 2 and scheduled == []


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://testserver:8080"])
def test_cross_origin_request_is_rejected_without_network(updater, monkeypatch, origin):
    client, store, _, _ = updater
    calls, _ = mock_download(monkeypatch)
    assert check(client, headers={"Origin": origin}).status_code == 403
    assert calls == [] and len(store.versions(DOCUMENT)) == 1


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.2.3.4", "169.254.169.254", "::1", "fc00::1"])
def test_private_dns_addresses_block_network(updater, monkeypatch, ip):
    client, _, _, _ = updater
    calls, _ = mock_download(monkeypatch)
    monkeypatch.setattr(
        updates.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)),
        ],
    )
    response = check(client)
    assert response.status_code == 502
    assert "непубличный IP" in response.json()["detail"]
    assert calls == []


def test_mixed_public_private_dns_is_also_rejected(updater, monkeypatch):
    client, _, _, _ = updater
    calls, _ = mock_download(monkeypatch)
    monkeypatch.setattr(
        updates.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    assert check(client).status_code == 502 and calls == []


@pytest.mark.parametrize(
    "location", ["https://evil.example/a.pdf", "https://www.eia.gov/new.pdf", URL]
)
def test_all_redirects_rejected_without_following(updater, monkeypatch, location):
    client, store, scheduled, _ = updater
    calls, _ = mock_download(monkeypatch, status=302, headers={"Location": location})
    response = check(client)
    assert response.status_code == 502
    assert "перенаправляет" in response.json()["detail"]
    assert len(calls) == 1 and scheduled == [] and len(store.versions(DOCUMENT)) == 1


@pytest.mark.parametrize(
    "content, headers, status",
    [
        (b"<html>Access denied</html>", {}, 502),
        (NEW, {"Content-Length": str(21 * 1024 * 1024)}, 413),
        (NEW, {"Content-Length": "invalid"}, 502),
        (NEW, {"Content-Encoding": "gzip"}, 502),
    ],
)
def test_bad_source_payload_never_enters_catalog(updater, monkeypatch, content, headers, status):
    client, store, scheduled, _ = updater
    mock_download(monkeypatch, content, headers=headers)
    assert check(client).status_code == status
    assert len(store.versions(DOCUMENT)) == 1 and scheduled == []


def test_stream_limit_applies_even_without_large_content_length(updater, monkeypatch):
    client, store, _, _ = updater
    monkeypatch.setattr(updates, "MAX_UPLOAD_BYTES", 12)
    mock_download(monkeypatch, NEW, headers={"Content-Length": "1"})
    assert check(client).status_code == 413
    assert len(store.versions(DOCUMENT)) == 1


def test_download_failure_does_not_expose_remote_response(updater, monkeypatch):
    client, _, _, _ = updater
    mock_download(monkeypatch, b"untrusted provider body", status=403)
    response = check(client)
    assert response.status_code == 502
    assert "HTTP 403" in response.json()["detail"]
    assert "untrusted provider body" not in response.text


def test_document_metadata_and_body_cannot_override_registered_url(updater, monkeypatch):
    client, store, _, _ = updater
    store.upload(
        "Russia.pdf",
        NEW + b"local",
        "Updated",
        source_url="http://127.0.0.1/internal",
        document_id=DOCUMENT,
    )
    calls, _ = mock_download(monkeypatch, OLD)
    response = check(client, json={"url": "file:///etc/passwd"})
    assert response.status_code == 200
    assert str(calls[0].url) == URL


def test_changed_manifest_url_disables_automatic_check(updater, monkeypatch):
    client, _, _, _ = updater
    calls, _ = mock_download(monkeypatch)
    path = settings().data / "reports.json"
    manifest = json.loads(path.read_text())
    manifest[0]["url"] = "https://www.eia.gov/arbitrary.pdf"
    path.write_text(json.dumps(manifest))
    assert check(client).status_code == 422 and calls == []


def test_archived_or_missing_document_does_not_make_requests(updater, monkeypatch):
    client, store, _, _ = updater
    calls, _ = mock_download(monkeypatch)
    store.archive(DOCUMENT)
    assert check(client).status_code == 422
    assert client.post("/api/knowledge/documents/unknown/check-update").status_code == 404
    assert calls == []


def test_concurrent_document_change_preserves_user_version(updater, monkeypatch):
    client, store, scheduled, _ = updater

    def change(request):
        store.upload("Russia.pdf", NEW + b"manual", "Manual", document_id=DOCUMENT)

    mock_download(monkeypatch, NEW, on_request=change)
    response = check(client)
    assert response.status_code == 409
    assert store.document(DOCUMENT)["title"] == "Manual"
    assert len(store.versions(DOCUMENT)) == 2 and scheduled == []
