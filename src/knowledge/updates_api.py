"""Manual checks of registered source PDFs; no arbitrary URL input."""

from fastapi import APIRouter, HTTPException, Request, Response

from .api import _same_origin
from .store import KnowledgeConflict, UploadTooLarge, get_store
from .updates import UpdateFetchError, UpdateUnavailable, check_update, update_capability

router = APIRouter(prefix="/api/knowledge", tags=["knowledge-updates"])


@router.get("/documents/{document_id}/update-capability")
def capability(document_id: str):
    try:
        return update_capability(get_store(), document_id)
    except KeyError:
        raise HTTPException(404, "Документ не найден.") from None


@router.post("/documents/{document_id}/check-update")
def check_source_update(request: Request, response: Response, document_id: str):
    _same_origin(request)
    try:
        result = check_update(get_store(), document_id)
        if result["status"] == "queued":
            response.status_code = 202
        return result
    except KeyError:
        raise HTTPException(404, "Документ не найден.") from None
    except KnowledgeConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except UploadTooLarge as exc:
        raise HTTPException(413, str(exc)) from None
    except UpdateUnavailable as exc:
        raise HTTPException(422, str(exc)) from None
    except UpdateFetchError as exc:
        raise HTTPException(502, str(exc)) from None
    except ValueError:
        raise HTTPException(422, "Не удалось сохранить новую версию документа.") from None
