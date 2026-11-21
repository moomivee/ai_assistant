import json

import httpx
import pytest
from openai import OpenAI
from test_agent import executor, plan

from eval.run import RecordingPlanner
from src.integrations.llm import LLMClient, LLMUnavailable
from src.models.schemas import ForecastInputs, LatestEventScope, Plan
from src.planning import make_plan
from src.workflow import ask

QUESTION = "Спрогнозируй Brent на 3 месяца. Только расчёт"


def inputs(**changes):
    fields = dict(
        benchmark="Brent",
        benchmark_quote="Brent",
        months=3,
        discount=None,
        cut=None,
        sensitivity=None,
        parameters=[{"field": "months", "quote": "на 3 месяца"}],
    )
    return ForecastInputs(**{**fields, **changes})


class PlannerClient:
    def __init__(self, initial, extracted=None, fail=False):
        self.initial, self.extracted, self.fail = initial, extracted, fail
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append(name)
        if name == "plan":
            return self.initial
        assert name == "forecast" and schema is ForecastInputs
        if self.fail:
            raise LLMUnavailable("repair unavailable")
        return self.extracted


def test_exact_user_failure_repairs_schema_then_runs_real_forecast():
    seen = []
    bad = plan(intent="forecast", benchmark=None, months=3, wants_context=False)

    def respond(req):
        body = json.loads(req.content)
        seen.append(body)
        obj = bad if len(seen) == 1 else inputs()
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 1,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": obj.model_dump_json()},
                    }
                ],
            },
        )

    sdk = OpenAI(
        api_key="local",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    recorder = RecordingPlanner()
    response = ask(QUESTION, client=LLMClient(sdk), planner=recorder, execute=executor)
    assert response["status"] == "ok" and response["forecast"]["benchmark"] == "Brent"
    assert len(response["forecast"]["rows"]) == 3
    assert [t["tool"] for t in response["trace"]] == ["report_search", "price_forecast"]
    assert recorder.plan.benchmark == "Brent" and not recorder.plan.clarify
    assert [b["response_format"]["json_schema"]["name"] for b in seen] == ["Plan", "ForecastInputs"]
    assert all(b["response_format"]["json_schema"]["strict"] for b in seen)


def test_complete_plan_needs_only_one_call():
    c = PlannerClient(plan(intent="forecast", benchmark="Brent", months=3, wants_context=False))
    assert make_plan(c, QUESTION).clarify == ""
    assert c.calls == ["plan"]


@pytest.mark.parametrize(
    "question,fresh,latest",
    [
        ("Чем закончилось последнее голосование комитета?", True, True),
        ("Какие решения приняли участники, когда собирались в последний раз?", False, True),
        ("Каков действующий состав комитета?", True, False),
        ("Когда состоится следующее голосование?", True, False),
        ("Что решил комитет в феврале 2020 года?", False, False),
    ],
)
def test_legacy_research_plan_resolves_missing_event_scope(question, fresh, latest):
    class OlderPlanner(PlannerClient):
        def generate(self, name, payload, schema):
            if name == "latest_event_scope":
                self.calls.append(name)
                assert schema is LatestEventScope
                assert payload["question"] == question and payload["today"]
                return schema(latest_completed_event=latest)
            return super().generate(name, payload, schema)

    legacy = plan(fresh=fresh)
    client = OlderPlanner(
        Plan.model_validate(legacy.model_dump(exclude={"latest_completed_event"}))
    )
    result = make_plan(client, question)
    assert result.latest_completed_event is latest and result.fresh is (fresh or latest)
    assert client.calls == ["plan", "latest_event_scope"]


@pytest.mark.parametrize(
    "question,initial,expected",
    [
        ("Как прошло последнее голосование?", plan(latest_completed_event=True), True),
        ("Какова сегодняшняя квота?", plan(fresh=True, latest_completed_event=False), False),
        ("Что сказано в прошлогоднем отчёте?", plan(fresh=False), False),
        (
            QUESTION,
            plan(
                intent="forecast",
                benchmark="Brent",
                months=3,
                wants_context=False,
                latest_completed_event=True,
            ),
            False,
        ),
    ],
)
def test_resolved_event_scope_preserves_other_routes_without_extra_calls(
    question, initial, expected
):
    client = PlannerClient(initial)
    result = make_plan(client, question)
    assert result.latest_completed_event is expected and client.calls == ["plan"]
    if expected:
        assert result.fresh and result.wants_context


def test_standalone_help_command_works_without_calling_model():
    c = PlannerClient(None)
    result = make_plan(c, "  ЧТО   ты можешь?! ")
    assert result.intent == "capabilities"
    assert c.calls == [] and not result.clarify and not result.wants_context


def test_capability_paraphrase_uses_structured_intent_and_discards_unneeded_actions():
    c = PlannerClient(
        plan(
            intent="capabilities",
            rag_query="search",
            news=True,
            quotes=["Brent"],
            benchmark="Brent",
            clarify="Какой сорт?",
            skill="report_analysis",
        )
    )
    result = make_plan(c, "Первый раз здесь. С какими задачами к вам обращаются?")
    assert c.calls == ["plan"] and result.intent == "capabilities"
    assert not result.quotes and not result.rag_query and not result.clarify
    assert result.skill is None and result.benchmark is None
    assert not result.wants_context and not result.news and not result.fresh


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Что ты можешь сказать о добыче в России?", "research"),
        ("Что ты можешь? Спрогнозируй Brent на 3 месяца.", "forecast"),
        ("Что ты можешь приготовить на ужин?", "outside"),
    ],
)
def test_help_prefix_does_not_swallow_a_concrete_task(question, intent):
    c = PlannerClient(plan(intent=intent, benchmark="Brent", months=3, wants_context=False))
    assert make_plan(c, question).intent == intent
    assert c.calls == ["plan"]


@pytest.mark.parametrize("intent", ["research", "outside"])
def test_non_forecast_clarification_cannot_block_source_search(intent):
    question = "Каково содержание серы в Sokol? Только по загруженным отчётам."

    class Client(PlannerClient):
        def generate(self, name, payload, schema):
            if name == "source_access":
                self.calls.append(name)
                return schema(mode="reports_only", quote="Только по загруженным отчётам")
            return super().generate(name, payload, schema)

    c = Client(plan(intent=intent, clarify=question))
    result = make_plan(c, question)
    assert result.clarify == ""
    assert c.calls == (["plan", "source_access"] if intent == "research" else ["plan"])


def test_spurious_clarification_is_rechecked_and_removed():
    c = PlannerClient(
        plan(
            intent="forecast",
            benchmark="Brent",
            months=3,
            wants_context=False,
            clarify="Какой сорт рассчитать: Brent, WTI или Urals?",
        ),
        inputs(),
    )
    assert make_plan(c, QUESTION).clarify == "" and c.calls == ["plan", "forecast"]


def test_genuine_ambiguity_is_not_cleared():
    c = PlannerClient(
        plan(
            intent="forecast",
            benchmark="Brent",
            months=3,
            wants_context=False,
            clarify="Brent или WTI?",
        ),
        inputs(benchmark=None, benchmark_quote=""),
    )
    result = make_plan(c, "Спрогнозируй Brent или WTI на 3 месяца")
    assert result.benchmark is None and result.clarify


@pytest.mark.parametrize(
    "extracted",
    [
        inputs(benchmark=None, benchmark_quote=""),
        inputs(benchmark_quote="Brent"),
        inputs(benchmark="WTI", benchmark_quote="нефть"),
    ],
)
def test_unspecified_benchmark_has_no_default(extracted):
    c = PlannerClient(plan(intent="forecast", months=3, wants_context=False), extracted)
    r = make_plan(c, "Спрогнозируй нефть на 3 месяца")
    assert r.benchmark is None and r.clarify and c.calls == ["plan", "forecast"]


def test_missing_discount_stays_missing():
    c = PlannerClient(
        plan(intent="forecast", benchmark="Urals", months=3, wants_context=False),
        inputs(
            benchmark="Urals",
            benchmark_quote="Urals",
            discount=15,
            parameters=[{"field": "discount", "quote": "дисконт 15 долларов"}],
        ),
    )
    r = make_plan(c, "Спрогнозируй Urals на 3 месяца")
    assert r.discount is None and "дисконт" in r.clarify


def test_invented_horizon_is_rejected():
    c = PlannerClient(plan(intent="forecast", benchmark="Brent", wants_context=False), inputs())
    r = make_plan(c, "Спрогнозируй Brent")
    assert r.months is None and r.clarify


def test_repair_failure_does_not_run_tools():
    c = PlannerClient(plan(intent="forecast", months=3, wants_context=False), fail=True)
    r = ask(QUESTION, client=c, execute=executor)
    assert r["status"] == "unavailable" and not r["trace"] and c.calls == ["plan", "forecast"]


def test_validated_parameters_are_not_overwritten_by_repair():
    c = PlannerClient(plan(intent="forecast", months=3, wants_context=False), inputs(months=12))
    r = make_plan(c, QUESTION)
    assert r.benchmark == "Brent" and r.months == 3 and not r.clarify


def test_cyrillic_name_is_accepted_only_with_input_evidence():
    c = PlannerClient(
        plan(intent="forecast", months=3, wants_context=False), inputs(benchmark_quote="Брент")
    )
    assert make_plan(c, "Брент на 3 месяца").benchmark == "Brent"
