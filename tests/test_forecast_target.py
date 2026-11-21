"""Forecast planning regressions through the agent loop and the real calculator."""

import pytest
from test_agent import plan
from test_planning import PlannerClient, inputs

from src.agent import ask


class ForecastClient(PlannerClient):
    def generate(self, name, payload, schema):
        if name == "decide":
            action = "finish" if payload["calculation_available"] else "price_forecast"
            return schema(action=action, purpose="Рассчитать прогноз")
        return super().generate(name, payload, schema)


def forecast_plan(grade, months=3, discount_quote=""):
    return plan(
        intent="forecast",
        benchmark=grade,
        months=months,
        wants_context=False,
        discount=12 if discount_quote else None,
        parameters=[{"field": "discount", "quote": discount_quote}] if discount_quote else [],
    )


@pytest.mark.parametrize("initial_grade", ["Brent", "Urals"])
def test_target_urals_and_reference_brent_reach_calculator_once(initial_grade):
    question = "Спрогнозируй Urals на 3 месяца при дисконте 12 долларов к Brent."
    initial = forecast_plan(initial_grade, discount_quote="при дисконте 12 долларов к Brent")
    # A role correction must not overwrite the already grounded horizon and discount.
    corrected = inputs(benchmark="Urals", benchmark_quote="Спрогнозируй Urals", months=8)
    client = ForecastClient(initial, corrected)
    result = ask(question, client=client)
    assert result["status"] == "ok"
    assert [row["tool"] for row in result["trace"]] == ["price_forecast"]
    calculated = result["forecast"]
    assert calculated["benchmark"] == "Urals" and calculated["base"] == "Brent"
    assert calculated["discount"] == 12 and len(calculated["rows"]) == 3
    assert client.calls == (["plan", "forecast"] if initial_grade == "Brent" else ["plan"])


def test_missing_discount_resolved_by_one_followup_matches_single_turn():
    question = "Спрогнозируй Urals на 3 месяца"
    initial = forecast_plan("Urals")
    incomplete = inputs(benchmark="Urals", benchmark_quote="Urals")
    first = ask(question, client=ForecastClient(initial, incomplete))
    assert first["status"] == "clarification" and "дисконт" in first["answer"].lower()
    assert first["trace"] == []

    reply = "12 долларов за баррель"
    resolved = ForecastClient(forecast_plan("Urals", discount_quote=reply))
    second = ask(reply, previous=first["question"], client=resolved)
    combined = ask(f"{question}, дисконт {reply}", client=ForecastClient(resolved.initial))
    assert second["status"] == combined["status"] == "ok"
    assert second["forecast"] == combined["forecast"] and second["forecast"]["discount"] == 12
    assert [row["tool"] for row in second["trace"]] == ["price_forecast"]
    assert resolved.calls == ["plan"]


@pytest.mark.parametrize("grade", ["WTI", "Brent"])
def test_plain_six_month_forecast_keeps_grade_without_role_recheck(grade):
    client = ForecastClient(forecast_plan(grade, months=6))
    result = ask(f"Какой прогноз для {grade} на 6 месяцев?", client=client)
    assert result["status"] == "ok" and client.calls == ["plan"]
    assert result["forecast"]["base"] == result["forecast"]["benchmark"] == grade
    assert result["forecast"]["discount"] is None and len(result["forecast"]["rows"]) == 6


@pytest.mark.parametrize(
    "grade,reference,initial_grade,extracted_grade,target_quote",
    [
        ("WTI", "Brent", "WTI", "WTI", "Спрогнозируй WTI"),
        ("Brent", "WTI", "Brent", "Brent", "Спрогнозируй Brent"),
        ("WTI", "Brent", "WTI", "Urals", "Спрогнозируй Urals"),
        ("WTI", "Urals", "WTI", "Urals", "Urals"),
        ("WTI", "Brent", "Urals", "WTI", "Спрогнозируй WTI"),
    ],
)
def test_unsupported_discount_never_triggers_a_different_calculation(
    grade, reference, initial_grade, extracted_grade, target_quote
):
    discount_clause = f"дисконте 12 долларов к {reference}"
    question = f"Спрогнозируй {grade} на 3 месяца при {discount_clause}."
    client = ForecastClient(
        forecast_plan(initial_grade, discount_quote=discount_clause),
        inputs(benchmark=extracted_grade, benchmark_quote=target_quote),
    )
    result = ask(question, client=client)
    assert result["status"] == "clarification" and result["trace"] == []
    assert result["forecast"] is None
    assert client.calls == ["plan", "forecast"]
    if grade == extracted_grade:
        assert grade in result["answer"] and "дисконт" in result["answer"].lower()


def test_urals_discount_to_wti_is_not_silently_treated_as_a_discount_to_brent():
    question = "Спрогнозируй Urals на 3 месяца при дисконте 12 долларов к WTI."
    clause = "12 долларов к WTI"
    client = ForecastClient(
        forecast_plan("Urals", discount_quote=clause),
        inputs(benchmark="Urals", benchmark_quote="Спрогнозируй Urals"),
    )
    result = ask(question, client=client)
    assert result["status"] == "clarification" and result["forecast"] is None
    assert result["trace"] == [] and "Brent" in result["answer"] and "WTI" in result["answer"]


def test_separate_wti_mention_does_not_invalidate_an_urals_discount_to_brent():
    question = "WTI уже посчитали. Теперь рассчитай Urals на 3 месяца с дисконтом 12 USD к Brent."
    client = ForecastClient(forecast_plan("Urals", discount_quote="12 USD к Brent"))
    result = ask(question, client=client)
    assert client.calls == ["plan"] and result["status"] == "ok"
    assert result["forecast"]["benchmark"] == "Urals" and result["forecast"]["discount"] == 12


def test_recheck_recovers_decimal_discount_with_a_separate_grade_mention():
    clause = "на 12,3 доллара дешевле Brent"
    question = f"WTI уже обсудили. Рассчитай Urals на 3 месяца, считая его {clause}."
    initial = forecast_plan("Brent", discount_quote=clause).model_copy(update={"discount": 12.3})
    client = ForecastClient(initial, inputs(benchmark="Urals", benchmark_quote="Рассчитай Urals"))
    result = ask(question, client=client)
    assert result["status"] == "ok" and client.calls == ["plan", "forecast"]
    assert result["forecast"]["benchmark"] == "Urals" and result["forecast"]["discount"] == 12.3


@pytest.mark.parametrize("reference", ["", " к Urals"])
def test_discount_does_not_supply_a_missing_target(reference):
    clause = "дисконтом 12 долларов" + reference
    client = ForecastClient(
        forecast_plan("Urals", discount_quote=clause),
        inputs(benchmark="Urals", benchmark_quote="Urals"),
    )
    result = ask(f"Какой прогноз цены нефти на 3 месяца с {clause}?", client=client)
    assert result["status"] == "clarification" and "сорт" in result["answer"].lower()
    assert result["forecast"] is None and result["trace"] == []
