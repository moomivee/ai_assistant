from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from test_extractors import text_pdf

from src.knowledge import api, extractors
from src.knowledge.store import KnowledgeStore


@pytest.fixture
def ingestion_api(tmp_path, monkeypatch):
    store = KnowledgeStore(tmp_path / "knowledge")
    scheduled = []
    monkeypatch.setattr(api, "get_store", lambda: store)
    monkeypatch.setattr(api, "submit_ingestion", lambda store, version: scheduled.append(version))
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        yield client, store, scheduled


@pytest.fixture
def knowledge_api(ingestion_api, monkeypatch):
    monkeypatch.setattr(
        extractors,
        "extract",
        lambda *args: {
            "blocks": [
                {
                    "block_id": "b1",
                    "text": "Density = 35 API",
                    "page": 2,
                    "locator": {"page": 2},
                    "method": "native",
                }
            ],
            "warnings": [],
            "parser_version": "test",
        },
    )
    return ingestion_api


def upload(client, **kwargs):
    return client.post(
        "/api/knowledge/documents",
        files={"file": ("test.pdf", b"test-pdf", "application/pdf")},
        **kwargs,
    )


def test_upload_returns_202_and_does_not_run_parser_in_request(knowledge_api):
    client, store, scheduled = knowledge_api
    response = upload(
        client,
        data={
            "title": "Quality report",
            "published_date": "2026-11-14",
            "source_url": "https://eia.gov/source.pdf",
            "publisher": "EIA",
        },
    )
    assert response.status_code == 202
    doc = response.json()["document"]
    assert doc["status"] == "queued"
    assert scheduled == [doc["current_version_id"]]
    assert store.blocks(doc["id"]) == []
    assert doc["published_date"] == "2026-11-14"
    assert doc["publisher"] == "EIA"
    assert client.get("/api/knowledge/documents").json()["documents"][0]["id"] == doc["id"]


def test_details_versions_and_exact_blocks(knowledge_api):
    client, store, _ = knowledge_api
    doc = upload(client).json()["document"]
    doc_id, version = doc["id"], doc["current_version_id"]
    store.process(version)
    detail = client.get(f"/api/knowledge/documents/{doc_id}").json()
    assert detail["versions"][0]["id"] == version
    blocks = client.get(f"/api/knowledge/documents/{doc_id}/versions/{version}/blocks")
    assert blocks.status_code == 200
    assert blocks.json()["blocks"][0]["text"] == "Density = 35 API"
    assert blocks.json()["blocks"][0]["page"] == 2
    assert blocks.json()["warnings"] == []


def test_original_file_survives_archive_and_cannot_be_read_through_other_id(knowledge_api):
    client, store, _ = knowledge_api
    doc = upload(client).json()["document"]
    store.process(doc["current_version_id"])
    url = doc["local_url"]
    first = client.get(url)
    assert first.status_code == 200
    assert first.content == b"test-pdf"
    assert first.headers["content-type"] == "application/pdf"
    assert first.headers["x-content-type-options"] == "nosniff"
    archived = client.post(f"/api/knowledge/documents/{doc['id']}/archive")
    assert archived.json()["document"]["status"] == "archived"
    assert client.get("/api/knowledge/documents").json()["documents"] == []
    assert client.get(url).content == b"test-pdf"
    assert (
        len(client.get("/api/knowledge/documents?include_archived=true").json()["documents"]) == 1
    )
    assert client.get(url.replace(doc["id"], "other-document")).status_code == 404


@pytest.mark.parametrize("origin", ["https://evil.example", "http://testserver:3000", "null"])
def test_cross_origin_upload_is_rejected_before_catalog_change(knowledge_api, origin):
    client, store, scheduled = knowledge_api
    response = upload(client, headers={"Origin": origin})
    assert response.status_code == 403
    assert store.list_documents() == []
    assert scheduled == []


def test_same_origin_and_cli_without_origin_work(knowledge_api):
    client, _, _ = knowledge_api
    assert upload(client, headers={"Origin": "http://testserver"}).status_code == 202
    assert upload(client).status_code == 202


def test_archive_and_retry_require_same_origin(knowledge_api):
    client, store, _ = knowledge_api
    doc = upload(client).json()["document"]
    prefix = f"/api/knowledge/documents/{doc['id']}"
    headers = {"Origin": "https://evil.example"}
    assert client.post(prefix + "/archive", headers=headers).status_code == 403
    assert (
        client.post(
            prefix + f"/versions/{doc['current_version_id']}/retry", headers=headers
        ).status_code
        == 403
    )
    assert not store.document(doc["id"])["archived"]


def test_error_can_be_retried_but_ready_version_is_immutable(knowledge_api, monkeypatch):
    client, store, scheduled = knowledge_api
    doc = upload(client).json()["document"]
    version = doc["current_version_id"]
    retry_url = f"/api/knowledge/documents/{doc['id']}/versions/{version}/retry"
    assert client.post(retry_url).status_code == 409

    def fail(*args):
        raise ValueError("Невозможно прочитать документ.")

    monkeypatch.setattr(extractors, "extract", fail)
    store.process(version)
    retried = client.post(retry_url)
    assert retried.status_code == 202
    assert retried.json()["document"]["status"] == "queued"
    assert scheduled == [version, version]


def test_retry_500_page_pdf_preserves_version_and_late_page_citations(
    ingestion_api, tmp_path, monkeypatch
):
    client, store, scheduled = ingestion_api
    writer = PdfWriter()
    markers = {101: "Oil revenue in the first quarter.", 500: "Annual report appendix."}
    for number in range(1, 501):
        if number in markers:
            path = text_pdf(tmp_path / f"page-{number}.pdf", markers[number])
            writer.add_page(PdfReader(path).pages[0])
        else:
            writer.add_blank_page(width=595, height=842)
    pdf = BytesIO()
    writer.write(pdf)
    uploaded = client.post(
        "/api/knowledge/documents",
        files={"file": ("annual.pdf", pdf.getvalue(), "application/pdf")},
    )
    assert uploaded.status_code == 202
    document = uploaded.json()["document"]
    version = document["current_version_id"]
    assert scheduled == [version]
    with monkeypatch.context() as previous_release:
        previous_release.setattr(extractors, "MAX_PAGES", 100)
        failed = store.process(version)
    assert failed["status"] == "error"
    assert "более 100 страниц" in failed["error"]

    url = f"/api/knowledge/documents/{document['id']}/versions/{version}"
    assert client.post(url + "/retry").status_code == 202
    assert scheduled == [version, version]
    ready = store.process(version)
    assert (ready["status"], ready["active_version_id"], ready["versions_count"]) == (
        "ready",
        version,
        1,
    )
    result = client.get(url + "/blocks").json()
    assert [(b["page"], b["locator"]["page"], b["text"]) for b in result["blocks"]] == [
        (page, page, text) for page, text in markers.items()
    ]


def test_501_page_pdf_is_rejected_before_publishing_blocks(tmp_path):
    writer = PdfWriter()
    for _ in range(501):
        writer.add_blank_page(width=595, height=842)
    pdf = BytesIO()
    writer.write(pdf)
    store = KnowledgeStore(tmp_path / "knowledge")
    document = store.upload("oversized.pdf", pdf.getvalue())
    failed = store.process(document["current_version_id"])
    assert failed["status"] == "error"
    assert "более 500 страниц" in failed["error"]
    assert failed["active_version_id"] is None
    assert store.blocks(document["id"], document["current_version_id"]) == []


def test_upload_limits_validation_and_unknown_ids(knowledge_api, monkeypatch):
    client, _, _ = knowledge_api
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 2)
    monkeypatch.setattr("src.knowledge.store.MAX_UPLOAD_BYTES", 2)
    assert upload(client).status_code == 413
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr("src.knowledge.store.MAX_UPLOAD_BYTES", 100)
    assert upload(client, data={"published_date": "invalid"}).status_code == 422
    assert upload(client, data={"document_id": "unknown"}).status_code == 404
    response = client.post("/api/knowledge/documents", files={"file": ("../evil.pdf", b"a")})
    assert response.status_code == 422
    assert client.get("/api/knowledge/documents/unknown").status_code == 404
    assert client.get("/api/knowledge/documents/unknown/versions/unknown/blocks").status_code == 404
    assert client.post("/api/knowledge/documents/unknown/archive").status_code == 404
