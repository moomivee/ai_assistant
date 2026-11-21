"""Replay a bad EventChronology item at the provider boundary of the running graph."""

import json
from datetime import UTC, datetime

import pytest
from test_agent import plan
from test_agent_loop import action, event

from src.agent import ask
from src.integrations.gigachat import GigaChatError, GigaChatResponse
from src.integrations.llm import LLMClient

QUESTION = "Когда прошло последнее заседание совета газового рынка?"
SCHEDULE = "На 8 октября 2026 года назначено заседание совета."
HELD = "Совет газового рынка провёл заседание 17 ноября 2026 года."


def publication(slug, text):
    return dict(
        kind="web",
        title="Сообщение совета",
        date=None,
        url=f"https://example.org/{slug}",
        text=text,
    )


class CouncilReplay:
    """Emit provider JSON, leaving schema parsing, evidence validation, and routing real."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.schemas = []
        self.chronology_inputs = []

    def generate(self, *, messages, schema, **kwargs):
        name = schema["title"]
        payload = json.loads(messages[-1]["content"])
        self.schemas.append(name)
        if name == "Plan":
            body = plan(fresh=True, latest_completed_event=True, rag_query=QUESTION).model_dump()
        elif "allowed_actions" in payload:
            query = "" if not payload["observations"] else "заседание совета сентябрь 2026"
            body = action("web_search", query)
        elif name == "EventChronology":
            body = self.classify(payload)
        elif name == "CitedAnswer":
            body = dict(claims=[dict(text=HELD, evidence_ids=["W2:B1"])], complete=True)
        elif name == "CitationReview":
            body = dict(supported=[0], coverage="sufficient", complete=True, feedback="")
        else:
            raise AssertionError(name)
        return GigaChatResponse(
            content=json.dumps(body, ensure_ascii=False), usage={}, model="scripted"
        )

    def classify(self, payload):
        evidence = payload["evidence"]
        self.chronology_inputs.append(set(evidence))
        has_outcome = any(row["text"] == HELD for row in evidence.values())
        if has_outcome and self.outcome == "offline":
            raise GigaChatError("connection", "Тестовый сбой соединения с провайдером.")
        if has_outcome and (self.outcome == "malformed" or not payload.get("format_repair")):
            # A string in place of the object gives blocks.0:model_type, like
            # the trace; the actual malformed value wasn't recorded there.
            return {"blocks": [next(iter(evidence))]}
        blocks = []
        for key, row in evidence.items():
            is_held = row["text"] == HELD
            candidate = dict(
                kind="held" if is_held else "scheduled",
                date="2026-11-17" if is_held else "2026-10-08",
                quote=row["text"],
                scope="совет газового рынка",
                match="in_scope",
            )
            blocks.append(dict(evidence_id=key, status="assessed", events=[candidate]))
        return {"blocks": blocks}


def run_replay(monkeypatch, outcome):
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setattr("src.clock._utc_now", lambda: datetime(2026, 11, 21, tzinfo=UTC))
    provider = CouncilReplay(outcome)
    calls = []

    def search(registry, name, **arguments):
        assert name == "web_search"
        calls.append(arguments["query"])
        if outcome == "no_search":
            return None, "Сбой соединения с поиском.", {**event(name), "status": "error"}
        item = publication("agenda", SCHEDULE) if len(calls) == 1 else publication("minutes", HELD)
        return {"items": [item], "notices": []}, None, event(name)

    response = ask(QUESTION, client=LLMClient(provider, provider="gigachat"), execute=search)
    return response, provider


@pytest.mark.parametrize("outcome", ["repaired", "malformed"])
def test_followup_search_and_bounded_repair_after_invalid_chronology(monkeypatch, outcome):
    response, provider = run_replay(monkeypatch, outcome)
    assert [tool["status"] for tool in response["trace"]] == ["ok", "ok"]
    assert len(provider.chronology_inputs) == 3
    first, followup = response["diagnostics"]["steps"]
    assert first["arguments"]["query"] == QUESTION
    assert "сентябр" not in followup["arguments"]["query"].casefold()
    checks = response["diagnostics"]["answer_checks"]
    assert checks[0]["validation"]["event_chronology"]["latest_held_date"] is None
    if outcome == "repaired":
        assert response["status"] == "ok" and response["answer"].count(HELD) == 1
        assert {s["id"] for s in response["sources"]} == {"W2"}
    else:
        assert response["status"] == "partial"
        assert response["stop_reason"] == "chronology_format_incomplete"
        assert response["claims"] == []
        assert (
            "EventChronology" not in response["answer"] and "model_type" not in response["answer"]
        )
        assert "CitedAnswer" not in provider.schemas
        assert (
            checks[-1]["validation"]["event_chronology"]["extraction"]["status"] == "invalid_format"
        )


def test_transport_failures_do_not_trigger_format_repair(monkeypatch):
    response, provider = run_replay(monkeypatch, "offline")
    assert response["stop_reason"] == "llm_unavailable" and len(provider.chronology_inputs) == 2
    assert "сбой соединения" in response["diagnostics"]["failure"].lower()


def test_failed_retrieval_is_visible_without_chronology_generation(monkeypatch):
    response, provider = run_replay(monkeypatch, "no_search")
    assert response["trace"] and all(tool["status"] == "error" for tool in response["trace"])
    assert not provider.chronology_inputs and not response["diagnostics"]["answer_checks"]
    assert response["stop_reason"] != "chronology_format_incomplete"
