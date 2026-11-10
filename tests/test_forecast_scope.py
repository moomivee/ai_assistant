"""Workflow regressions with controlled LLM replies, not live Qwen evaluations."""

import pytest
from test_agent import WEB, FakeLLM, executor, plan

from src.integrations.llm import LLMUnavailable
from src.models.schemas import ForecastScope
from src.planning import make_plan
from src.workflow import ask

QUESTION = "Спрогнозируй цену Brent на 3 месяца."


def scope(**changes):
    return ForecastScope(
        **{
            **dict(context_request="", fresh=False, news=False, quote_request="", quotes=[]),
            **changes,
        }
    )


class ScopeClient(FakeLLM):
    def __init__(self, extracted=None, *, p=None, complete=False):
        super().__init__(
            p or plan(intent="forecast", benchmark="Brent", months=3, wants_context=True),
            complete=complete,
        )
        self.extracted = extracted or scope()
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append((name, payload))
        if name == "forecast_scope":
            assert schema is ForecastScope
            return self.extracted
        result = super().generate(name, payload, schema)
        if name == "answer":
            # Deliberately reproduce the contradictory output: assembly must not
            # treat an assessment of reports as a verdict on the calculation.
            result.limitations = [
                "Отсутствуют данные о прогнозе цен на Brent на 3 месяца.",
                "Отсутствуют данные о спросе и предложениях на нефть.",
            ]
        return result


def test_plain_forecast_removes_unrequested_context_and_web():
    client = ScopeClient(
        p=plan(
            intent="forecast",
            benchmark="Brent",
            months=3,
            wants_context=True,
            fresh=True,
            news=True,
            quotes=["Brent"],
        )
    )
    response = ask(QUESTION, client=client, execute=executor)
    assert [name for name, _ in client.calls] == ["plan", "forecast_scope"]
    assert [event["tool"] for event in response["trace"]] == ["report_search", "price_forecast"]
    assert response["status"] == "ok" and response["forecast"]["benchmark"] == "Brent"
    assert len(response["forecast"]["rows"]) == 3
    assert response["claims"] == [] and [s["id"] for s in response["sources"]] == ["M1"]
    assert "TAVILY" not in response["answer"] and "Отсутствуют данные" not in response["answer"]


@pytest.mark.parametrize("quote", [QUESTION, "объясни влияние санкций"])
def test_whole_forecast_or_invented_context_is_not_an_additional_request(quote):
    client = ScopeClient(scope(context_request=quote, fresh=True, news=True))
    resolved = make_plan(client, QUESTION)
    assert not resolved.wants_context and not resolved.fresh and not resolved.news


def test_explicit_news_is_assessed_separately_from_successful_calculation():
    context = "покажи свежие новости ОПЕК+"
    client = ScopeClient(scope(context_request=context, fresh=True, news=True))
    tool_args = []

    def capture(registry, name, **kwargs):
        tool_args.append((name, kwargs))
        return executor(registry, name, **kwargs)

    response = ask(QUESTION + " И " + context, client=client, execute=capture)
    assert [name for name, _ in tool_args] == ["report_search", "web_search", "price_forecast"]
    for name, payload in client.calls:
        if name == "answer":
            assert payload["question"] == context and payload["previous"] is None
            assert payload["task"] == "forecast_context"
    assert tool_args[0][1]["query"] == tool_args[1][1]["query"] == context
    assert response["forecast"] and response["status"] == "partial"
    assert response["answer"].startswith("Прогноз Brent на 3 мес.")
    assert f"Дополнительный вопрос «{context}»" in response["answer"]
    assert "Отсутствуют данные" not in response["answer"]
    assert "Источники подтверждают только часть запроса" not in response["answer"]


def test_complete_report_context_needs_no_web_and_ignores_stale_limitations():
    context = "расскажи о сернистости Sokol по отчёту"
    client = ScopeClient(scope(context_request=context), complete=True)
    response = ask(QUESTION + " И " + context, client=client, execute=executor)
    assert [event["tool"] for event in response["trace"]] == ["report_search", "price_forecast"]
    assert response["status"] == "ok" and response["claims"]
    assert [s["id"] for s in response["sources"]] == ["M1", "R1"]
    assert "Отсутствуют данные" not in response["answer"]


def test_report_only_mixed_forecast_preserves_source_restriction():
    context = "объясни влияние санкций"
    client = ScopeClient(
        scope(context_request=context),
        p=plan(intent="forecast", benchmark="Brent", months=3, report_only=True),
    )
    response = ask(
        QUESTION + " И " + context + ", только по отчётам", client=client, execute=executor
    )
    assert [event["tool"] for event in response["trace"]] == ["report_search", "price_forecast"]
    assert response["forecast"] and response["status"] == "partial"
    assert (
        context in response["answer"] and "Отсутствуют данные о прогнозе" not in response["answer"]
    )


def test_explicit_spot_quote_uses_quote_tool_without_general_web_search():
    request = "покажи последнюю опубликованную цену Brent"
    client = ScopeClient(scope(quote_request=request, quotes=["Brent"]))

    def quote_executor(registry, name, **kwargs):
        if name != "web_search":
            return executor(registry, name, **kwargs)
        assert kwargs["search"] is False and kwargs["quotes"] == ["Brent"]
        return (
            {"items": [{**WEB, "grade": "Brent", "price": 75.0}], "notices": []},
            None,
            {"tool": name, "engine": "fixture", "status": "ok", "ms": 0},
        )

    response = ask(QUESTION + " И " + request, client=client, execute=quote_executor)
    assert response["status"] == "ok" and response["forecast"]
    assert [name for name, _ in client.calls] == ["plan", "forecast_scope"]
    assert [event["tool"] for event in response["trace"]] == [
        "report_search",
        "web_search",
        "price_forecast",
    ]


@pytest.mark.parametrize(
    "quote_request,quotes",
    [("последняя цена Brent", ["Brent"]), (QUESTION, ["Brent"]), ("покажи текущую цену", ["WTI"])],
)
def test_quote_requires_actual_additional_clause_and_benchmark(quote_request, quotes):
    client = ScopeClient(scope(quote_request=quote_request, quotes=quotes))
    resolved = make_plan(
        client, QUESTION + " И покажи текущую цену" if quotes == ["WTI"] else QUESTION
    )
    assert resolved.quotes == [] and not resolved.fresh


def test_followup_can_consist_of_the_additional_question():
    context = "Объясни влияние санкций"
    client = ScopeClient(scope(context_request=context))
    resolved = make_plan(client, context, previous=QUESTION)
    assert resolved.wants_context and resolved.rag_query == context


def test_failed_calculation_is_not_presented_as_ready():
    client = ScopeClient()

    def broken_calculation(registry, name, **kwargs):
        if name == "price_forecast":
            return (
                None,
                "Не удалось загрузить историю цен.",
                {
                    "tool": name,
                    "engine": "fixture",
                    "status": "error",
                    "ms": 0,
                },
            )
        return executor(registry, name, **kwargs)

    response = ask(QUESTION, client=client, execute=broken_calculation)
    assert response["status"] == "partial" and response["forecast"] is None
    assert "Прогноз не рассчитан" in response["answer"]
    assert "Не удалось загрузить историю цен" in response["answer"]
    assert not response["sources"]


def test_scope_failure_does_not_silently_invent_a_route():
    class OfflineScope(ScopeClient):
        def generate(self, name, payload, schema):
            if name == "forecast_scope":
                raise LLMUnavailable("scope unavailable")
            return super().generate(name, payload, schema)

    response = ask(QUESTION, client=OfflineScope(), execute=executor)
    assert response["status"] == "unavailable" and response["trace"] == []


def test_unavailable_context_answer_does_not_remove_forecast():
    context = "объясни влияние санкций"

    class OfflineAnswer(ScopeClient):
        def generate(self, name, payload, schema):
            if name == "answer":
                raise LLMUnavailable("answer unavailable")
            return super().generate(name, payload, schema)

    response = ask(
        QUESTION + " И " + context,
        client=OfflineAnswer(scope(context_request=context)),
        execute=executor,
    )
    assert response["status"] == "partial" and response["forecast"]
    assert "Не удалось проверить ответ модели" in response["answer"]
    assert f"Дополнительный вопрос «{context}»" in response["answer"]
