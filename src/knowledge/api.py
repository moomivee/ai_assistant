"""Knowledge-base endpoints; uploads are processed outside the HTTP request."""

from urllib.parse import urlsplit

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .jobs import submit_ingestion
from .store import MAX_UPLOAD_BYTES, KnowledgeConflict, UploadTooLarge, get_store

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
            expected = urlsplit(str(request.url))
            same = parsed.scheme == expected.scheme and parsed.netloc == expected.netloc
        except ValueError:
            same = False
        if not same:
            raise HTTPException(403, "Изменение базы знаний доступно только с этого сайта.")


def _document(document_id: str):
    document = get_store().document(document_id)
    if document is None:
        raise HTTPException(404, "Документ не найден.")
    return document


@router.get("/documents")
def documents(include_archived: bool = False):
    return {"documents": get_store().list_documents(include_archived=include_archived)}


@router.post("/documents", status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    published_date: str = Form(""),
    source_url: str = Form(""),
    publisher: str = Form(""),
    document_id: str | None = Form(None),
):
    _same_origin(request)
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    try:
        store = get_store()
        document = store.upload(
            file.filename or "",
            content,
            title,
            published_date,
            source_url,
            publisher,
            document_id=document_id or None,
        )
        if document["ingestion_status"] == "queued":
            submit_ingestion(store, document["current_version_id"])
        return {"document": document}
    except UploadTooLarge as exc:
        raise HTTPException(413, str(exc)) from None
    except KnowledgeConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "Документ для новой версии не найден.") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get("/documents/{document_id}")
def document_details(document_id: str):
    document = _document(document_id)
    return {**document, "versions": get_store().versions(document_id)}


@router.get("/documents/{document_id}/versions/{version_id}/blocks")
def document_blocks(document_id: str, version_id: str):
    version = get_store().version(document_id, version_id)
    if version is None:
        raise HTTPException(404, "Версия документа не найдена.")
    return {
        "blocks": get_store().blocks(document_id, version_id),
        "warnings": version["warnings"],
        "status": version["status"],
        "error": version["error"],
    }


@router.get("/documents/{document_id}/versions/{version_id}/file")
def document_file(document_id: str, version_id: str):
    store = get_store()
    version = store.version(document_id, version_id)
    path = store.version_file(document_id, version_id)
    if version is None or path is None or not path.is_file():
        raise HTTPException(404, "Оригинал документа не найден.")
    inline = version["media_type"] in {"application/pdf", "image/png", "image/jpeg"}
    return FileResponse(
        path,
        media_type=version["media_type"],
        filename=version["filename"],
        content_disposition_type="inline" if inline else "attachment",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post("/documents/{document_id}/archive")
def archive_document(request: Request, document_id: str):
    _same_origin(request)
    _document(document_id)
    return {"document": get_store().archive(document_id)}


@router.post("/documents/{document_id}/versions/{version_id}/retry", status_code=202)
def retry_document(request: Request, document_id: str, version_id: str):
    _same_origin(request)
    try:
        store = get_store()
        document = store.retry(document_id, version_id)
        submit_ingestion(store, version_id)
        return {"document": document}
    except KeyError:
        raise HTTPException(404, "Версия документа не найдена.") from None
    except KnowledgeConflict as exc:
        raise HTTPException(409, str(exc)) from None
