"""Retrieved evidence must reach answer generation before another planning turn."""

import json

import httpx
from openai import OpenAI
from test_agent import REPORT, plan
from test_agent_loop import Scripted, action, event

from src.agent import answer_sources, ask
from src.integrations.llm import LLMClient
from src.models.schemas import CitedAnswer


def test_real_sdk_receives_report_text_and_ids_without_a_finish_decision():
    calls = []
    table = "Item | Amount\nOrion | 42\nVega | 28"

    def reply(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        name = body["response_format"]["json_schema"]["name"]
        calls.append((name, payload))
        assert name == "CitedAnswer"
        assert payload["question"] == "Сравни количества"
        assert payload["evidence"]["R1:B1"] == table
        assert payload["sources"][0]["id"] == "R1"
        result = {
            "claims": [{"evidence_ids": ["R1:B1"], "text": "Orion: 42; Vega: 28."}],
            "complete": True,
        }
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="local",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as sdk:
        response = ask(
            "Сравни количества",
            client=LLMClient(sdk),
            planner=lambda *_: plan(rag_query="amounts"),
            execute=lambda *a, **kw: ([{**REPORT, "text": table}], None, event("report_search")),
        )
    assert [name for name, _ in calls] == ["CitedAnswer"]
    assert response["status"] == "ok" and len(response["decisions"]) == 1
    assert response["decisions"][0]["origin"] == "plan_policy"
    assert response["diagnostics"]["steps"][0]["arguments"] == {"query": "amounts"}
    assert response["claims"][0]["evidence_ids"] == ["R1:B1"]
    check = response["diagnostics"]["answer_checks"][0]
    assert check["after_step"] == 1 and check["source_ids"] == ["R1"] and check["complete"]


def test_search_source_selection_cannot_discard_retrieved_evidence():
    assert answer_sources(
        {"sources": [REPORT], "decision": {"action": "report_search", "source_ids": ["R999"]}}
    ) == [REPORT]


def test_partial_answer_is_available_to_next_action_and_can_be_completed():
    class Client(Scripted):
        def generate(self, name, payload, schema):
            if name == "cited_answer":
                self.payloads.append((name, payload))
                return schema.model_validate(
                    {
                        "claims": [
                            {"text": value, "evidence_ids": [key]}
                            for key, value in payload["evidence"].items()
                        ],
                        "complete": len(payload["sources"]) == 2,
                    }
                )
            return super().generate(name, payload, schema)

    client = Client([action("report_search", "missing fact")])
    response = ask(
        "Два факта",
        client=client,
        planner=lambda *_: plan(rag_query="first fact"),
        execute=lambda r, n, **kw: ([{**REPORT, "text": kw["query"]}], None, event(n)),
    )
    second = [p for n, p in client.payloads if n == "decide"][0]
    assert second["current_answer"]["claims"][0]["text"] == "first fact"
    assert second["current_answer"]["complete"] is False
    assert response["status"] == "ok" and len(response["claims"]) == 2
    assert len(response["diagnostics"]["answer_checks"]) == 2


def test_irrelevant_sources_do_not_become_a_successful_answer():
    class NoEvidence(Scripted):
        def generate(self, name, payload, schema):
            if name == "cited_answer":
                self.payloads.append((name, payload))
                return CitedAnswer(claims=[], complete=False)
            return super().generate(name, payload, schema)

    client = NoEvidence(
        [
            action("report_search", "fact"),
            action("report_search", "fact"),
            action("report_search", "fact"),
        ]
    )
    response = ask(
        "Нет данных",
        client=client,
        planner=lambda *_: plan(),
        execute=lambda r, n, **kw: ([REPORT], None, event(n)),
    )
    assert response["status"] == "partial" and response["claims"] == []
    assert response["stop_reason"] == "no_progress"
    assert sum(name == "cited_answer" for name, _ in client.payloads) == 1


def test_source_handoff_does_not_skip_calculation_in_mixed_request():
    client = Scripted([action("report_search", "fact"), action("price_forecast")])

    from src.tools import run_tool

    def execute(registry, name, **args):
        return (
            ([REPORT], None, event(name))
            if name == "report_search"
            else run_tool(registry, name, **args)
        )

    response = ask(
        "Прогноз и контекст",
        client=client,
        planner=lambda *_: plan(intent="forecast", benchmark="WTI", months=2),
        execute=execute,
    )
    assert response["status"] == "ok" and response["forecast"]["benchmark"] == "WTI"
    assert [t["tool"] for t in response["trace"]] == ["report_search", "price_forecast"]
    assert len(response["diagnostics"]["answer_checks"]) == 1


def test_maximum_step_budget_allows_an_answer_check_after_each_tool(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "12")
    client = Scripted([action("report_search", f"fact {i}") for i in range(12)], complete=False)
    response = ask(
        "Нужно собрать несколько фактов",
        client=client,
        planner=lambda *_: plan(),
        execute=lambda r, n, **kw: ([{**REPORT, "text": kw["query"]}], None, event(n)),
    )
    assert response["status"] == "partial" and response["stop_reason"] == "budget_exhausted"
    assert len(response["trace"]) == len(response["diagnostics"]["answer_checks"]) == 12
