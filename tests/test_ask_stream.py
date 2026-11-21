import asyncio
import json
from threading import Event

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_agent import plan
from test_agent_loop import Scripted, action

from src import api
from src.agent import ask
from src.integrations import web
from src.models.schemas import Question


def test_quote_fallback_is_delivered_while_search_is_running(monkeypatch):
    release_search = Event()

    def unavailable(grade):
        raise httpx.ReadTimeout("FRED")

    def publications(query, fresh, **options):
        assert fresh and options["quote_fallback"]
        assert release_search.wait(timeout=5)
        return []

    def run_question(question, previous):
        return ask(
            question,
            previous,
            client=Scripted([action("web_search", "Brent spot")]),
            planner=lambda *_: plan(quotes=["Brent"], fresh=True, wants_context=False),
        )

    monkeypatch.setattr(web, "latest_spot", unavailable)
    monkeypatch.setattr(web, "search_web", publications)
    monkeypatch.setattr(api, "ask", run_question)

    async def read():
        events = api._answer_events(Question(question="Цена Brent?"))
        try:
            progress = await asyncio.wait_for(anext(events), timeout=2)
            assert json.loads(progress) == {"type": "progress", "stage": "quote_fallback"}
        finally:
            release_search.set()
        remaining = [json.loads(event) async for event in events]
        assert len(remaining) == 1
        assert remaining[0]["type"] == "result"
        assert remaining[0]["result"]["status"] == "partial"

    asyncio.run(read())


def test_stream_has_same_response_as_json_without_spurious_progress():
    with TestClient(api.app) as client:
        body = {"question": "Цена Brent?"}
        plain = client.post("/api/ask", json=body).json()
        stream = client.post("/api/ask/stream", json=body)
        assert stream.headers["content-type"] == "application/x-ndjson"
        (event,) = [json.loads(line) for line in stream.iter_lines()]
        assert event["type"] == "result"
        # Two real requests have separate traces.
        assert set(plain) == set(event["result"])
        assert plain["status"] == event["result"]["status"]
        assert plain["answer"] == event["result"]["answer"]
        assert client.post("/api/ask/stream", json={"question": ""}).status_code == 422


def test_stream_preserves_error_details(monkeypatch):
    detail = {"code": "dialogue_busy", "message": "Ещё обрабатывается."}

    def fail(body):
        raise HTTPException(409, detail)

    monkeypatch.setattr(api, "query", fail)
    with TestClient(api.app) as client:
        response = client.post("/api/ask/stream", json={"question": "Brent"})
    assert response.json() == {"type": "error", "status": 409, "detail": detail}
