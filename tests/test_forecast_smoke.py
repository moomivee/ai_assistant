import json
from io import BytesIO

from eval import core_smoke
from eval.forecast_smoke import calculation, clarification, memory_dialogue
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


def test_memory_smoke_uses_only_latest_token_and_keeps_it_out_of_console(monkeypatch, capsys):
    payloads = []
    questions = [
        "Спрогнозируй Urals",
        "На 3 месяца",
        "Измени горизонт на 6 месяцев",
        "12 долларов за баррель",
    ]

    def api(request, timeout):
        i = len(payloads)
        payloads.append(json.loads(request.data))
        if i < 3:
            months = (None, 3, 6)[i]
            response = {
                "status": "clarification",
                "clarification": True,
                "forecast": None,
                "dialogue_token": f"test-secret-{i}",
                "answer": "На сколько месяцев?" if months is None else "Какой дисконт?",
                "diagnostics": {"steps": [], "plan": {"benchmark": "Urals", "months": months}},
            }
        else:
            args = dict(benchmark="Urals", months=6, discount=12, cut=None, sensitivity=None)
            result = forecast(**args)
            response = {
                "answer": forecast_answer(result),
                "status": "ok",
                "clarification": False,
                "forecast": result,
                "sources": [{"kind": "model", "id": "M1"}],
                "diagnostics": {
                    "steps": [{"tool": "price_forecast", "status": "ok", "arguments": args}]
                },
            }
        return BytesIO(json.dumps({**response, "trace_id": f"trace-{i}"}).encode())

    monkeypatch.setattr(core_smoke, "urlopen", api)
    assert memory_dialogue("http://localhost", correct_horizon=True)
    assert payloads == [
        {"question": questions[0], "remember_clarification": True},
        *[
            {"question": q, "dialogue_token": f"test-secret-{i}"}
            for i, q in enumerate(questions[1:])
        ],
    ]
    output = capsys.readouterr().out
    assert "test-secret" not in output
    assert all(f"trace-{i}" in output for i in range(4))
