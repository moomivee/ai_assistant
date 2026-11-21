"""Live forecast routing checks against the running v3 API, using its configured LLM."""

import argparse
from urllib.error import URLError

from .core_smoke import ask_api, forecast_checks

CASES = [
    ("Спрогнозируй Urals на 3 месяца при дисконте 12 долларов к Brent.", "Urals", 3, 12),
    (
        "Если к Brent применяется дисконт 9 USD за баррель, какой прогноз для Urals на 2 месяца?",
        "Urals",
        2,
        9,
    ),
    ("Спрогнозируй Brent на 3 месяца.", "Brent", 3, None),
    ("Дай прогноз WTI на полгода.", "WTI", 6, None),
]
UNSUPPORTED = "Спрогнозируй WTI на 3 месяца с дисконтом 12 долларов к Brent."
DIALOG = "Спрогнозируй Urals на 3 месяца"


def calculation(response, target, months, discount):
    args = dict(benchmark=target, months=months, discount=discount, cut=None, sensitivity=None)
    steps = response["diagnostics"]["steps"]
    result = response.get("forecast") or {}
    return {
        "статус ok без уточнения": response["status"] == "ok" and not response["clarification"],
        "один успешный расчёт с точными аргументами": (
            len(steps) == 1
            and steps[0]["tool"] == "price_forecast"
            and steps[0]["status"] == "ok"
            and steps[0]["arguments"] == args
        ),
        "верны база расчёта и дисконт": (
            result.get("base") == ("Brent" if target == "Urals" else target)
            and result.get("discount") == discount
        ),
        **forecast_checks(response, target, months),
    }


def clarification(response, target, discount):
    plan = response["diagnostics"]["plan"]
    answer = response["answer"].casefold()
    return {
        "уточнение без инструментов и прогноза": (
            response["status"] == "clarification"
            and response["clarification"]
            and response["diagnostics"]["steps"] == []
            and response["forecast"] is None
        ),
        "сорт и горизонт сохранены": plan.get("benchmark") == target and plan.get("months") == 3,
        "ответ о дисконте": (
            plan.get("discount") == discount
            and "дисконт" in answer
            and (discount is None or (target.casefold() in answer and "urals" in answer))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    args = parser.parse_args()

    def check(question, scorer, *expected, previous=None):
        print(question, flush=True)
        try:
            response = ask_api(args.base_url, question, previous=previous)
            checks = scorer(response, *expected)
            failed = [name for name, ok in checks.items() if not ok]
            print("  ", response.get("mode"), response.get("trace_id"), ", ".join(failed) or "PASS")
            return not failed
        except (ValueError, TypeError, KeyError, URLError, TimeoutError) as error:
            print("  Ошибка:", error)
            return False

    passed = [
        check(q, calculation, grade, months, discount) for q, grade, months, discount in CASES
    ]
    passed.append(check(UNSUPPORTED, clarification, "WTI", 12))
    dialog = check(DIALOG, clarification, "Urals", None)
    if dialog:
        # The forecast UI sends the original user question in previous after clarification.
        dialog = check("12 долларов за баррель", calculation, "Urals", 3, 12, previous=DIALOG)
    passed.append(dialog)
    print(f"Сценариев пройдено: {sum(passed)}/{len(passed)}.")
    return int(not all(passed))


if __name__ == "__main__":
    raise SystemExit(main())
