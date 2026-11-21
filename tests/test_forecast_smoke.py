from eval.forecast_smoke import calculation, clarification
from src.calculations.forecast import forecast
from src.workflow import forecast_answer


def test_live_gate_checks_actual_arguments_and_catches_blocked_duplicates():
    args = dict(benchmark="Urals", months=3, discount=12, cut=None, sensitivity=None)
    result = forecast(**args)
    step = {"tool": "price_forecast", "status": "ok", "arguments": args.copy()}
    response = {
        "answer": forecast_answer(result),
        "status": "ok",
        "clarification": False,
        "forecast": result,
        "sources": [{"kind": "model", "id": "M1"}],
        "diagnostics": {"steps": [step]},
    }
    assert all(calculation(response, "Urals", 3, 12).values())
    step["arguments"]["benchmark"] = "Brent"
    assert not all(calculation(response, "Urals", 3, 12).values())
    step["arguments"]["benchmark"] = "Urals"
    response["diagnostics"]["steps"].append({**step, "status": "rejected"})
    assert not all(calculation(response, "Urals", 3, 12).values())


def test_live_gate_rejects_generic_clarification_for_unsupported_wti():
    response = {
        "status": "clarification",
        "clarification": True,
        "forecast": None,
        "answer": "Укажите дисконт.",
        "diagnostics": {"steps": [], "plan": {"benchmark": "WTI", "months": 3, "discount": 12}},
    }
    assert not all(clarification(response, "WTI", 12).values())
    response["answer"] = "Дисконт поддерживается только для Urals. Рассчитать обычный WTI?"
    assert all(clarification(response, "WTI", 12).values())
