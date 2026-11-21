import pytest
from test_agent import plan
from test_planning import PlannerClient, inputs

from src.forecast_memory import ForecastFollowup, forecast_memory, next_snapshot, previous_text
from src.planning import make_plan


def forecast(**values):
    return plan(**{"intent": "forecast", "wants_context": False, **values})


def start(question, **inputs):
    return next_snapshot(None, question, forecast(**inputs).model_dump())


def followup(memory, question, draft, repair=None):
    client = PlannerClient(draft, repair or inputs(months=None, benchmark=None))
    with forecast_memory(memory):
        resolved = make_plan(client, question, previous_text(memory))
        snapshot = next_snapshot(memory, question, resolved.model_dump())
    return resolved, snapshot


def test_repair_preserves_grounded_inputs_and_accounts_for_correction():
    first = start("Спрогнозируй Urals", benchmark="Urals", clarify="На сколько месяцев?")
    second, memory = followup(
        first,
        "На 3 месяца",
        forecast(months=3, parameters=[{"field": "months", "quote": "3 месяца"}]),
    )
    assert second.benchmark == "Urals" and second.months == 3 and "дисконт" in second.clarify
    corrected, memory = followup(
        memory,
        "Поправка — на 6 месяцев",
        forecast(months=6, parameters=[{"field": "months", "quote": "6 месяцев"}]),
    )
    assert corrected.months == 6 and corrected.benchmark == "Urals"
    # A stale, once-correct value quoted from the opening dialogue cannot undo the latest correction.
    resolved, _ = followup(
        memory,
        "12 долларов за баррель",
        forecast(
            months=3,
            discount=12,
            parameters=[
                {"field": "months", "quote": "3 месяца"},
                {"field": "discount", "quote": "12 долларов за баррель"},
            ],
        ),
    )
    assert (resolved.benchmark, resolved.months, resolved.discount) == ("Urals", 6, 12)
    assert not resolved.clarify


@pytest.mark.parametrize("horizon_evidence", [[], [{"field": "months", "quote": "12 долларов"}]])
def test_amount_in_answer_does_not_override_the_stored_horizon(horizon_evidence):
    memory = start("Спрогнозируй Urals на 3 месяца", benchmark="Urals", months=3)
    result, _ = followup(
        memory,
        "Дисконт 12 долларов к Brent",
        forecast(
            benchmark="Brent",
            months=12,
            discount=12,
            parameters=[*horizon_evidence, {"field": "discount", "quote": "12 долларов к Brent"}],
        ),
    )
    assert (result.benchmark, result.months, result.discount) == ("Urals", 3, 12)


def test_switching_grade_drops_its_old_discount():
    m = start(
        "Urals с дисконтом 12 USD",
        benchmark="Urals",
        discount=12,
        parameters=[{"field": "discount", "quote": "12 USD"}],
    )
    r, _ = followup(
        m,
        "Исправь сорт на WTI, на 3 месяца",
        forecast(
            benchmark="WTI",
            months=3,
            discount=12,
            parameters=[
                {"field": "months", "quote": "на 3 месяца"},
                {"field": "discount", "quote": "12 USD"},
            ],
        ),
    )
    assert r.benchmark == "WTI" and r.months == 3 and r.discount is None and not r.clarify


def test_cancelling_assumption_with_latest_evidence():
    m = start(
        "WTI на 3 месяца, сокращение 1 млн баррелей",
        benchmark="WTI",
        months=3,
        cut=1,
        parameters=[{"field": "cut", "quote": "сокращение 1 млн баррелей"}],
    )
    reply = "Без сокращения"
    draft = ForecastFollowup(
        **forecast().model_dump(), withdrawn=[{"field": "cut", "quote": reply}]
    )
    r, _ = followup(m, reply, draft)
    assert r.benchmark == "WTI" and r.months == 3 and r.cut is None and not r.clarify


def test_explicit_new_task_does_not_receive_superseded_evidence():
    m = start("Urals на 3 месяца", benchmark="Urals", months=3)
    client = PlannerClient(
        ForecastFollowup(**forecast(benchmark="Brent").model_dump(), new_task=True),
        inputs(months=None),
    )
    with forecast_memory(m):
        new = make_plan(client, "Новый прогноз для Brent", previous_text(m))
        snapshot = next_snapshot(m, "Новый прогноз для Brent", new.model_dump())
    assert new.months is None and new.clarify
    assert previous_text(snapshot) == "Новый прогноз для Brent"
    assert new.benchmark == "Brent"


def test_context_is_local_and_history_contains_only_bounded_user_text():
    initial = "Прогноз WTI? " + "а" * 1900
    m = start(initial, benchmark="WTI", clarify="На сколько месяцев? Допустимо от 1 до 12.")
    for i in range(12):
        _, m = followup(m, f"Пока думаю {i}.", forecast())
    assert m["turns"] == [initial, "Пока думаю 11."]
    assert "Допустимо" not in previous_text(m)
    assert m["confirmed"] == {"benchmark": "WTI"}
    fresh = make_plan(
        PlannerClient(forecast(months=2), inputs(benchmark=None, months=None)), "На 2 месяца"
    )
    assert fresh.benchmark is None


@pytest.mark.parametrize("quote", ["12", "12 долларов", "12 долларов к Brent"])
def test_short_amount_quote_cannot_switch_forecast_target(quote):
    m = start("Спрогнозируй Urals на 3 месяца", benchmark="Urals", months=3)
    result, _ = followup(
        m,
        "12 долларов к Brent",
        forecast(
            benchmark="Brent", discount=12, parameters=[{"field": "discount", "quote": quote}]
        ),
    )
    assert result.benchmark == "Urals" and result.discount == 12 and not result.clarify


def test_uncited_horizon_with_unit_supersedes_previous_horizon():
    m = start("Прогноз Urals на 3 месяца", benchmark="Urals", months=3)
    resolved, m = followup(m, "Исправь на шесть месяцев", forecast(months=6))
    assert resolved.months == 6 and "дисконт" in resolved.clarify
    assert m["evidence"]["months"] == "шесть месяцев"


def test_new_task_drops_previous_grade_hallucinated_by_model():
    m = start("Urals на 3 месяца", benchmark="Urals", months=3)
    candidate = ForecastFollowup(
        **forecast(benchmark="Urals", months=3, discount=3).model_dump(), new_task=True
    )
    resolved, _ = followup(
        m,
        "Теперь отдельный прогноз на четыре месяца",
        candidate,
        inputs(
            benchmark=None,
            benchmark_quote="",
            months=4,
            parameters=[{"field": "months", "quote": "четыре месяца"}],
        ),
    )
    assert resolved.benchmark is None and resolved.discount is None
    assert resolved.months == 4 and "сорт" in resolved.clarify


def test_grounded_report_subrequest_and_textual_scope_survive_until_calculation():
    clause = "объясни падение добычи только по загруженному отчёту"
    question = "Спрогнозируй Urals и " + clause
    memory = start(question, benchmark="Urals", wants_context=True, report_only=True)
    _, memory = followup(memory, "На 3 месяца", forecast(months=3))
    reply = "Дисконт 12 долларов"

    class Client(PlannerClient):
        def generate(self, name, payload, schema):
            if name == "forecast_scope":
                assert question in payload["previous"]
                return schema(
                    context_request=clause, fresh=False, news=False, quote_request="", quotes=[]
                )
            return super().generate(name, payload, schema)

    draft = forecast(discount=12, parameters=[{"field": "discount", "quote": reply}])
    with forecast_memory(memory):
        resolved = make_plan(Client(draft), reply, previous_text(memory))
    assert not resolved.clarify and resolved.months == 3 and resolved.benchmark == "Urals"
    assert resolved.wants_context and resolved.rag_query == clause
    assert resolved.report_only and not resolved.fresh


def test_omitted_discount_never_switches_to_reference_grade():
    memory = start("Прогноз Urals на 3 месяца", benchmark="Urals", months=3)
    resolved, _ = followup(
        memory,
        "12 долларов к Brent",
        forecast(benchmark="Brent"),
        inputs(
            benchmark="Brent",
            benchmark_quote="Brent",
            months=None,
            discount=12,
            parameters=[{"field": "discount", "quote": "12"}],
        ),
    )
    assert resolved.benchmark == "Urals" and resolved.discount == 12


def test_spurious_repair_cannot_lose_this_turns_complete_correction():
    m = start("Спрогнозируй Urals на 3 месяца", benchmark="Urals", months=3)
    resolved, _ = followup(
        m,
        "Лучше на 6 месяцев, дисконт 12 долларов",
        forecast(
            months=6,
            discount=12,
            parameters=[
                {"field": "months", "quote": "6 месяцев"},
                {"field": "discount", "quote": "12 долларов"},
            ],
            clarify="Какой сорт рассчитать?",
        ),
    )
    assert (resolved.months, resolved.discount, resolved.benchmark) == (6, 12, "Urals")
    assert not resolved.clarify
