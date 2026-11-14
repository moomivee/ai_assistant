import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address

from src.config import settings
from src.integrations.research_notes import (
    NoteConflict,
    archive_research_note,
    list_research_notes,
    load_note_context,
    read_research_note,
    research_note_history,
    restore_research_note,
    save_research_note,
)
from src.knowledge.notes_api import router


@pytest.fixture
def payload():
    return {
        "title": "Urals и ESPO",
        "question": "Как различаются сорта нефти?",
        "answer": "Сравнение по найденным данным [R1].",
        "source_refs": [
            {
                "id": "R1",
                "kind": "report",
                "title": "Отчёт",
                "url": "https://www.eia.gov/report.pdf",
                "page": 6,
                "text": "Фрагмент исходного отчёта.",
                "date": "2025-01-01",
            }
        ],
        "trace_id": "a" * 32,
        "unresolved": ["Проверить дату таблицы"],
    }


def test_save_is_native_versioned_markdown_and_not_verified(tmp_path, payload):
    saved = save_research_note(**payload, runtime_root=tmp_path)
    address = resolve_knowledge_address(tmp_path, f"research/{saved['note_id']}", scope="global")
    native = read_knowledge_note(address)
    assert native.revision == saved["revision"]
    assert native.metadata["type"] == "analyst_research_note"
    assert native.metadata["origin"] == "explicit_user_save"
    assert native.metadata["verified"] is False
    assert native.metadata["source_refs"] == payload["source_refs"]
    assert (address.shelf / "index-full.md").exists()
    assert saved["notice"] and saved["state"] == "active"
    assert read_research_note(saved["note_id"], runtime_root=tmp_path) == saved
    listing = list_research_notes(runtime_root=tmp_path)
    assert len(listing) == 1 and "answer" not in listing[0]


def test_updates_require_revision_and_native_history_retains_versions(tmp_path, payload):
    first = save_research_note(**payload, runtime_root=tmp_path)
    with pytest.raises(NoteConflict):
        save_research_note(**payload, note_id=first["note_id"], runtime_root=tmp_path)
    changed = {**payload, "answer": "Проверенный заново ответ [R1]."}
    second = save_research_note(
        **changed,
        note_id=first["note_id"],
        expected_revision=first["revision"],
        runtime_root=tmp_path,
    )
    assert second["revision"] != first["revision"]
    with pytest.raises(NoteConflict) as conflict:
        save_research_note(
            **payload,
            note_id=first["note_id"],
            expected_revision=first["revision"],
            runtime_root=tmp_path,
        )
    assert conflict.value.current_revision == second["revision"]
    history = research_note_history(first["note_id"], runtime_root=tmp_path)
    assert [row["revision"] for row in history] == [first["revision"], second["revision"]]
    assert history[0]["publication"] == "captured_snapshot"
    assert not history[0]["current"] and history[1]["current"]
    raw_history = (tmp_path / "memory/knowledge_history.jsonl").read_text()
    last = json.loads(raw_history.splitlines()[-1])
    assert payload["answer"] in last["old_content"]
    assert changed["answer"] in last["new_content"]
    assert last["writer"] == "analyst.explicit_user_save"


def test_archive_restore_preserve_answer_and_require_current_revision(tmp_path, payload):
    original = save_research_note(**payload, runtime_root=tmp_path)
    archived = archive_research_note(
        original["note_id"],
        expected_revision=original["revision"],
        reason="Завершено",
        runtime_root=tmp_path,
    )
    assert archived["state"] == "archived" and archived["answer"] == original["answer"]
    assert not list_research_notes(runtime_root=tmp_path)
    assert len(list_research_notes(include_archived=True, runtime_root=tmp_path)) == 1
    with pytest.raises(ValueError, match="Архивная"):
        load_note_context(original["note_id"], runtime_root=tmp_path)
    with pytest.raises(NoteConflict):
        restore_research_note(
            original["note_id"], expected_revision=original["revision"], runtime_root=tmp_path
        )
    restored = restore_research_note(
        original["note_id"], expected_revision=archived["revision"], runtime_root=tmp_path
    )
    assert restored["state"] == "active" and restored["answer"] == original["answer"]


def test_context_is_explicit_untrusted_bounded_and_does_not_reuse_source_ids(tmp_path, payload):
    assert list_research_notes(runtime_root=tmp_path) == []
    saved = save_research_note(**{**payload, "answer": "a" * 10000}, runtime_root=tmp_path)
    context = load_note_context(saved["note_id"], runtime_root=tmp_path)
    assert context["verified"] is False
    assert context["truncated"]
    assert len(context["answer"]) == 8000
    assert "недоверенный контекст" in context["notice"]
    assert "id" not in context["prior_source_refs"][0]
    assert "text" not in context["prior_source_refs"][0]


def test_notes_cannot_escape_workspace(tmp_path, payload):
    for note_id in ["../secret", "research/../../secret", "other-project"]:
        with pytest.raises(ValueError):
            read_research_note(note_id, runtime_root=tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "memory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        save_research_note(**payload, runtime_root=runtime)
    assert not list(outside.iterdir())


def test_notes_api_save_conflict_and_cross_origin(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings.cache_clear()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        blocked = client.post("/api/notes", json=payload, headers={"Origin": "https://other.test"})
        assert blocked.status_code == 403
        saved_response = client.post("/api/notes", json=payload)
        assert saved_response.status_code == 201
        saved = saved_response.json()
        path = f"/api/notes/{saved['note_id']}"
        assert client.get(path).json()["answer"] == payload["answer"]
        assert len(client.get("/api/notes").json()["notes"]) == 1
        body = {**payload, "answer": "Другой ответ", "expected_revision": saved["revision"]}
        assert client.put(path, json=body).status_code == 200
        assert client.put(path, json=body).status_code == 409
        assert len(client.get(path + "/history").json()["history"]) == 2
