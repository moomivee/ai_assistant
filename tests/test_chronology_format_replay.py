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
        self.answer_blocks = set()
        self.citable_ids = set()

    def generate(self, *, messages, schema, **kwargs):
        name = schema["title"]
        payload = json.loads(messages[-1]["content"])
        self.schemas.append(name)
        if name == "Plan":
            body = plan(fresh=True, latest_completed_event=True, rag_query=QUESTION).model_dump()
        elif "allowed_actions" in payload:
            body = self.decide(payload)
        elif name == "EventRelevance":
            body = {"off_topic": []}
        elif name == "EventChronology":
            body = self.classify(payload)
        elif name == "CitedAnswer":
            self.answer_blocks = set(payload["evidence"])
            fields = schema["$defs"]["CitedClaim"]["properties"]
            self.citable_ids = set(fields["evidence_ids"]["items"]["enum"])
            ids = payload["event_chronology"]["latest_evidence_ids"]
            body = dict(claims=[dict(text=HELD, evidence_ids=ids)], complete=True)
        elif name == "CitationReview":
            body = dict(supported=[0], coverage="sufficient", complete=True, feedback="")
        else:
            raise AssertionError(name)
        return GigaChatResponse(
            content=json.dumps(body, ensure_ascii=False), usage={}, model="scripted"
        )

    def decide(self, payload):
        query = "" if not payload["observations"] else "заседание совета сентябрь 2026"
        return action("web_search", query)

    def classify(self, payload):
        evidence = payload["evidence"]
        self.chronology_inputs.append(set(evidence))
        if self.outcome == "no_usable_rows":
            return {"blocks": []}
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
        assert response["stop_reason"] == "budget_exhausted"
        assert response["claims"] == []
        assert (
            "EventChronology" not in response["answer"] and "model_type" not in response["answer"]
        )
        assert "CitedAnswer" not in provider.schemas
        extraction = checks[-1]["validation"]["event_chronology"]["extraction"]
        assert extraction["status"] == "incomplete"
        assert extraction["reused_evidence_ids"] == ["W1:B1"]
        assert extraction["unparsed_evidence_ids"] == ["W2:B1"]


def test_transport_failures_do_not_trigger_format_repair(monkeypatch):
    response, provider = run_replay(monkeypatch, "offline")
    assert response["stop_reason"] == "llm_unavailable" and len(provider.chronology_inputs) == 2
    assert "сбой соединения" in response["diagnostics"]["failure"].lower()


def test_failed_retrieval_is_visible_without_chronology_generation(monkeypatch):
    response, provider = run_replay(monkeypatch, "no_search")
    assert response["trace"] and all(tool["status"] == "error" for tool in response["trace"])
    assert not provider.chronology_inputs and not response["diagnostics"]["answer_checks"]
    assert response["stop_reason"] != "chronology_format_incomplete"


def test_no_readable_chronology_after_repair_stops_before_a_second_search(monkeypatch):
    response, provider = run_replay(monkeypatch, "no_usable_rows")
    assert provider.chronology_inputs == [{"W1:B1"}, {"W1:B1"}]
    assert len(response["trace"]) == 1
    assert response["status"] == "partial" and not response["claims"]
    assert response["stop_reason"] == "chronology_format_incomplete"


class MixedChronologyReplay(CouncilReplay):
    """Replay four readable excerpts, one missing ID, then an unusable repair."""

    def decide(self, payload):
        if not payload["observations"]:
            return action("web_search", "последнее заседание газового совета")
        if self.outcome == "finish":
            return action("finish")
        return {
            **action("web_search", "gas council meeting official minutes"),
            "publisher_domain": "council.example",
        }

    def classify(self, payload):
        ids = list(payload["evidence"])
        self.chronology_inputs.append(set(ids))
        if payload.get("format_repair"):
            code = "refusal" if self.outcome == "refusal" else "empty_response"
            raise GigaChatError(code, "Провайдер не вернул разметку.")
        if "W6:B1" in ids:
            assert ids == ["W6:B1"]
            assessment = dict(
                status="assessed",
                events=[
                    dict(
                        kind="held", date="2026-11-17", quote=HELD, scope="совет", match="in_scope"
                    )
                ],
            )
            return {
                "blocks": {"W6:B1": assessment}
                if payload["keyed_blocks"]
                else [dict(evidence_id="W6:B1", **assessment)]
            }
        assert len(ids) == 5
        rows = [dict(evidence_id=key, status="uncertain", events=[]) for key in ids[:3]]
        rows.append(dict(evidence_id=ids[3], status="assessed", events=[]))
        rows.append(dict(source_id=ids[4], status="uncertain", events=[]))
        return {"blocks": rows}


@pytest.mark.parametrize("outcome", ["minutes", "empty", "finish", "refusal"])
def test_partly_readable_chronology_can_seek_independent_evidence(monkeypatch, outcome):
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    monkeypatch.setenv("WEB_DOMAINS", "example.org,council.example")
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setattr("src.clock._utc_now", lambda: datetime(2026, 11, 21, tzinfo=UTC))
    provider = MixedChronologyReplay(outcome)
    calls = []
    archive = [
        publication(str(i), text)
        for i, text in enumerate(
            (
                "Архив решений совета за прошлые годы.",
                "Члены совета обсуждают проведение встречи.",
                "О дате заседания объявят позднее.",
                "Совет утвердил регламент публикации протоколов.",
                "Декларация о сотрудничестве участников совета.",
            )
        )
    ]
    minutes = {**publication("minutes", HELD), "url": "https://council.example/minutes"}

    def search(registry, name, **arguments):
        assert name == "web_search"
        calls.append(arguments)
        items = archive if len(calls) == 1 else [minutes] if outcome == "minutes" else []
        return {"items": items, "notices": []}, None, event(name)

    result = ask(QUESTION, client=LLMClient(provider, provider="gigachat"), execute=search)
    assert provider.chronology_inputs[:2] == [{f"W{i}:B1" for i in range(1, 6)}, {"W5:B1"}]
    if outcome in {"finish", "refusal"}:
        assert len(calls) == 1 and len(provider.chronology_inputs) == 2
        assert not result["claims"] and HELD not in result["answer"]
        if outcome == "refusal":
            assert result["stop_reason"] == "llm_unavailable"
        else:
            assert result["decisions"][-1]["action"] == "finish"
        return
    assert len(calls) == 2
    assert calls[1]["publisher_domain"] == "council.example"
    assert calls[0]["query"] != calls[1]["query"]
    checks = result["diagnostics"]["answer_checks"]
    extraction = checks[0]["validation"]["event_chronology"]["extraction"]
    assert extraction["unparsed_evidence_ids"] == ["W5:B1"]
    assert extraction["repair_attempted"] and len(extraction["reviewed_evidence_ids"]) == 4
    assert result["status"] == "partial"
    if outcome == "minutes":
        assert provider.chronology_inputs[2:] == [{"W6:B1"}]
        assert provider.answer_blocks == provider.citable_ids
        assert "W5:B1" not in provider.citable_ids and "W6:B1" in provider.citable_ids
        assert HELD in result["answer"] and {s["url"] for s in result["sources"]} == {
            minutes["url"]
        }
        assert checks[-1]["validation"]["event_chronology"]["latest_held_date"] == "2026-11-17"
        assert "W5:B1" in checks[-1]["validation"]["event_chronology"]["unresolved_evidence_ids"]
    else:
        assert len(provider.chronology_inputs) == 2
        assert not result["claims"] and HELD not in result["answer"]
