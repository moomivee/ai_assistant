"""Explicit save/continue/archive actions for unverified research notes."""

from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ..integrations.research_notes import (
    NoteConflict,
    NoteWriteError,
    archive_research_note,
    list_research_notes,
    read_research_note,
    research_note_history,
    restore_research_note,
    save_research_note,
)

router = APIRouter(prefix="/api/notes", tags=["research-notes"])


class NotePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=240)
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=60000)
    source_refs: list[dict] = Field(default_factory=list, max_length=30)
    trace_id: str = Field(default="", max_length=128)
    unresolved: list[str] = Field(default_factory=list, max_length=20)


class NoteUpdate(NotePayload):
    expected_revision: str = Field(pattern=r"^[a-f0-9]{64}$")


class NoteRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: str = Field(pattern=r"^[a-f0-9]{64}$")


class NoteArchive(NoteRevision):
    reason: str = Field(default="Архивировано пользователем", min_length=1, max_length=1000)


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin:
        expected, actual = urlsplit(str(request.url)), urlsplit(origin)
        if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
            raise HTTPException(403, "Изменение заметок доступно только с этого сайта.")


def _call(action, *args, **kwargs):
    try:
        return action(*args, **kwargs)
    except FileNotFoundError:
        raise HTTPException(404, "Заметка не найдена.") from None
    except NoteConflict as exc:
        raise HTTPException(
            409, {"message": str(exc), "current_revision": exc.current_revision}
        ) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except NoteWriteError:
        raise HTTPException(
            503, "Сохранение не завершено. Проверьте актуальную версию заметки."
        ) from None


@router.get("")
def list_notes(include_archived: bool = False):
    return {"notes": _call(list_research_notes, include_archived=include_archived)}


@router.post("", status_code=201)
def create_note(request: Request, body: NotePayload):
    _same_origin(request)
    return _call(save_research_note, **body.model_dump())


@router.get("/{note_id}")
def get_note(note_id: str):
    return _call(read_research_note, note_id)


@router.put("/{note_id}")
def update_note(request: Request, note_id: str, body: NoteUpdate):
    _same_origin(request)
    return _call(save_research_note, note_id=note_id, **body.model_dump())


@router.post("/{note_id}/archive")
def archive_note(request: Request, note_id: str, body: NoteArchive):
    _same_origin(request)
    return _call(archive_research_note, note_id, **body.model_dump())


@router.post("/{note_id}/restore")
def restore_note(request: Request, note_id: str, body: NoteRevision):
    _same_origin(request)
    return _call(restore_research_note, note_id, **body.model_dump())


@router.get("/{note_id}/history")
def note_history(note_id: str):
    return {"history": _call(research_note_history, note_id)}
