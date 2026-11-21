"""Forecast clarifications over HTTP, with a scripted LLM and the real forecast tool."""

from shutil import copytree

import pytest
from fastapi.testclient import TestClient
from test_agent import plan

from src import agent
from src.api import app
from src.calculations.forecast import forecast as calculate
from src.config import ROOT, settings
from src.integrations.llm import LLMUnavailable

FIRST = "Спрогнозируй Urals"
HORIZON = "На 3 месяца"
CORRECTION = "Поправка, на 6 месяцев"
DISCOUNT = "12 долларов за баррель"


class LatestReplyModel:
    """Emit only information from this reply, so the fixture cannot supply missing memory."""

    def __init__(self):
        self.plan_calls = []

    def generate(self, name, payload, schema):
        question = payload.get("question")
        if name == "plan":
            self.plan_calls.append(question)
            fields = {
                FIRST: {"benchmark": "Urals"},
                HORIZON: {"months": 3, "parameters": [{"field": "months", "quote": HORIZON}]},
                CORRECTION: {
                    "months": 6,
                    "parameters": [{"field": "months", "quote": "на 6 месяцев"}],
                },
                DISCOUNT: {
                    "discount": 12,
                    "parameters": [{"field": "discount", "quote": DISCOUNT}],
                },
            }
            return plan(intent="forecast", rag_query="", wants_context=False, **fields[question])
        if name == "forecast":
            return schema(
                benchmark=None,
                benchmark_quote="",
                months=None,
                discount=None,
                cut=None,
                sensitivity=None,
                parameters=[],
            )
        assert name == "decide"
        action = "finish" if payload["calculation_available"] else "price_forecast"
        return schema(action=action, purpose="Рассчитать цену")


@pytest.fixture
def api(monkeypatch, tmp_path):
    data = tmp_path / "data"
    copytree(ROOT / "data/prices", data / "prices")
    monkeypatch.setenv("DATA_DIR", str(data))
    settings.cache_clear()
    model = LatestReplyModel()
    monkeypatch.setattr(agent, "llm", lambda: model)
    with TestClient(app) as client:
        yield client, model


def ask(client, question, **kwargs):
    response = client.post("/api/ask", json={"question": question, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def pending(response, word):
    assert response["status"] == "clarification" and word in response["answer"].lower()
    assert response["trace"] == [] and response["forecast"] is None
    assert response["dialogue_token"]
    return response["dialogue_token"]


@pytest.mark.parametrize("correct_months", [False, True], ids=["three-turns", "correct-horizon"])
def test_followups_remember_urals_and_only_compute_after_discount(api, correct_months):
    client, _ = api
    token = pending(ask(client, FIRST, remember_clarification=True), "месяц")
    token = pending(ask(client, HORIZON, dialogue_token=token), "дисконт")
    if correct_months:
        token = pending(ask(client, CORRECTION, dialogue_token=token), "дисконт")

    result = ask(client, DISCOUNT, dialogue_token=token)
    assert result["status"] == "ok" and not result.get("dialogue_token")
    assert [row["tool"] for row in result["trace"]] == ["price_forecast"]
    forecast = result["forecast"]
    assert (forecast["benchmark"], forecast["base"], forecast["discount"]) == ("Urals", "Brent", 12)
    months = 6 if correct_months else 3
    assert len(forecast["rows"]) == months
    brent = calculate("Brent", months)
    for urals_row, brent_row in zip(forecast["rows"], brent["rows"], strict=True):
        assert urals_row["month"] == brent_row["month"]
        assert urals_row["point"] == pytest.approx(brent_row["point"] - 12)
    assert (
        client.post("/api/ask", json={"question": DISCOUNT, "dialogue_token": token}).status_code
        == 409
    )


def test_separate_request_scope_and_stale_token_do_not_change_pending_dialogue(api):
    client, model = api
    start = pending(ask(client, FIRST, remember_clarification=True), "месяц")
    fresh = ask(client, HORIZON)
    assert fresh["status"] == "clarification" and "сорт" in fresh["answer"].lower()
    assert not fresh.get("dialogue_token")
    calls_before = list(model.plan_calls)
    response = client.post(
        "/api/ask", json={"question": HORIZON, "dialogue_token": start, "report_only": True}
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "dialogue_conflict"
    assert model.plan_calls == calls_before
    next_token = pending(ask(client, HORIZON, dialogue_token=start), "дисконт")
    assert next_token != start
    response = client.post("/api/ask", json={"question": CORRECTION, "dialogue_token": start})
    assert response.status_code == 409
    assert ask(client, DISCOUNT, dialogue_token=next_token)["status"] == "ok"


def test_pending_dialogue_can_retry_an_unavailable_model(api, monkeypatch):
    client, model = api
    token = pending(ask(client, FIRST, remember_clarification=True), "месяц")

    def offline(*args):
        raise LLMUnavailable("test offline")

    with monkeypatch.context() as patch:
        patch.setattr(model, "generate", offline)
        failed = ask(client, HORIZON, dialogue_token=token)
    assert failed["status"] == "unavailable" and failed["dialogue_token"] == token
    pending(ask(client, HORIZON, dialogue_token=token), "дисконт")


def test_previous_still_supports_stateless_api_clients(api, monkeypatch):
    client, model = api
    previous = "Спрогнозируй Urals на 3 месяца"
    original_generate = model.generate

    def legacy_model(name, payload, schema):
        if name != "plan":
            return original_generate(name, payload, schema)
        assert payload["previous"] == previous and payload["question"] == DISCOUNT
        return plan(
            intent="forecast",
            wants_context=False,
            benchmark="Urals",
            months=3,
            discount=12,
            parameters=[{"field": "discount", "quote": DISCOUNT}],
        )

    monkeypatch.setattr(model, "generate", legacy_model)
    result = ask(client, DISCOUNT, previous=previous)
    assert result["status"] == "ok" and not result.get("dialogue_token")
    assert result["forecast"]["discount"] == 12 and len(result["forecast"]["rows"]) == 3


def test_independent_forecast_replaces_history_before_another_clarification(api, monkeypatch):
    client, model = api
    first = pending(ask(client, FIRST, remember_clarification=True), "месяц")
    second = pending(ask(client, HORIZON, dialogue_token=first), "дисконт")
    alternative = "Новая задача. Посчитай прогноз WTI."
    original_generate = model.generate

    def new_task_model(name, payload, schema):
        if name == "plan" and payload["question"] == alternative:
            assert FIRST in payload["previous"] and HORIZON in payload["previous"]
            new_plan = plan(intent="forecast", benchmark="WTI", wants_context=False)
            return schema(**new_plan.model_dump(), new_task=True)
        if name == "plan" and payload["question"] == CORRECTION:
            assert payload["previous"] == alternative
            assert payload["pending_forecast"]["confirmed"] == {"benchmark": "WTI"}
        return original_generate(name, payload, schema)

    monkeypatch.setattr(model, "generate", new_task_model)
    third = pending(ask(client, alternative, dialogue_token=second), "месяц")
    result = ask(client, CORRECTION, dialogue_token=third)
    assert result["status"] == "ok" and result["forecast"]["benchmark"] == "WTI"
    assert result["forecast"]["discount"] is None and len(result["forecast"]["rows"]) == 6


def test_legacy_workflow_returns_token_for_pending_forecast(api, monkeypatch):
    from src import workflow

    client, model = api
    monkeypatch.setenv("AGENT_MODE", "workflow")
    settings.cache_clear()
    monkeypatch.setattr(workflow, "llm", lambda: model)
    token = pending(ask(client, FIRST, remember_clarification=True), "месяц")
    assert pending(ask(client, HORIZON, dialogue_token=token), "дисконт") != token
