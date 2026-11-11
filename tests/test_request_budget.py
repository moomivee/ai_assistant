"""Request deadlines use a fake monotonic clock; these tests never sleep."""

import json

import httpx
import pytest
from openai import OpenAI
from test_agent import REPORT, plan
from test_agent_loop import Scripted, action, event

from src import budget
from src.agent import ask
from src.integrations.llm import LLMClient
from src.models.schemas import JudgeResult


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(budget, "monotonic", lambda: now[0])
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "10")
    return now


def test_request_deadline_resets_and_nested_scope_restores_parent(clock):
    assert budget.remaining_seconds() is None
    with budget.request_budget(10):
        clock[0] = 3
        with budget.request_budget(2):
            clock[0] = 6
            with pytest.raises(budget.RequestBudgetExceeded):
                budget.check_budget()
        assert budget.remaining_seconds() == 4
    assert budget.remaining_seconds() is None
    assert budget.bounded_timeout(25) == 25


def test_slow_plan_cannot_start_a_decision_or_tool_after_deadline(clock):
    client = Scripted([])

    def planner(*args):
        clock[0] = 11
        return plan()

    response = ask("Вопрос", client=client, planner=planner)
    assert client.payloads == []
    assert response["trace"] == [] and response["decisions"] == []
    assert response["stop_reason"] == "time_budget_exhausted"
    assert response["status"] == "unavailable"
    assert response["diagnostics"]["timeout_seconds"] == 10


def test_slow_tool_keeps_sources_but_does_not_start_another_generation(clock):
    client = Scripted([action("report_search", "fact")])

    def execute(registry, name, **args):
        clock[0] = 11
        return [REPORT], None, event(name)

    response = ask("Вопрос", client=client, execute=execute, planner=lambda *_: plan())
    assert len(client.payloads) == 1
    assert response["stop_reason"] == "time_budget_exhausted" and response["status"] == "partial"
    assert response["diagnostics"]["retrieved_sources"][0]["id"] == "R1"
    assert response["claims"] == []
    assert "Не удалось подготовить ответ" in response["answer"]


def test_grounded_partial_draft_survives_deadline_before_replanning(clock):
    class SlowAnswer(Scripted):
        def generate(self, name, payload, schema):
            result = super().generate(name, payload, schema)
            if name == "cited_answer":
                result.complete = False
                clock[0] = 11
            return result

    response = ask(
        "Вопрос",
        client=SlowAnswer([action("report_search", "fact"), action("finish")]),
        planner=lambda *_: plan(),
        execute=lambda *a, **kw: ([REPORT], None, event("report_search")),
    )
    assert response["stop_reason"] == "time_budget_exhausted" and response["status"] == "partial"
    assert response["claims"] and response["sources"][0]["id"] == "R1"
    assert REPORT["text"] in response["answer"]


def test_multiple_calls_inside_one_node_share_deadline(clock):
    class SlowClient:
        calls = 0

        def generate(self, *args):
            self.calls += 1
            clock[0] = 11
            return plan()

    client = SlowClient()

    def planner(bounded, *args):
        bounded.generate("plan", {}, object)
        return bounded.generate("repair", {}, object)

    response = ask("Вопрос", client=client, planner=planner)
    assert client.calls == 1
    assert response["stop_reason"] == "time_budget_exhausted"


def completion():
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
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "groundedness": 4,
                                "relevance": 3,
                                "tone": 4,
                                "completeness": 2,
                                "explanation": "ok",
                            }
                        ),
                    },
                }
            ],
        },
    )


def test_real_sdk_timeout_shrinks_with_remaining_budget(clock):
    seen = []

    def reply(req):
        seen.append(req.extensions["timeout"])
        return completion()

    with OpenAI(
        api_key="local",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as sdk:
        client = LLMClient(sdk)
        with budget.request_budget(10):
            client.generate("judge", {}, JudgeResult)
            clock[0] = 6
            client.generate("judge", {}, JudgeResult)
            clock[0] = 11
            with pytest.raises(budget.RequestBudgetExceeded):
                client.generate("judge", {}, JudgeResult)
    assert len(seen) == 2
    assert seen[0]["read"] == 10 and seen[1]["read"] == 4


def test_timeout_at_deadline_is_not_misreported_as_bad_model_output(clock):
    seen = []

    def reply(req):
        seen.append(req)
        clock[0] = 11
        raise httpx.ReadTimeout("slow generation", request=req)

    with OpenAI(
        api_key="local",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as sdk:
        with budget.request_budget(10), pytest.raises(budget.RequestBudgetExceeded):
            LLMClient(sdk).generate("judge", {}, JudgeResult)
    assert len(seen) == 1
