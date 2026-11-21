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
        "диалог расчёта закрыт": response.get("dialogue_token") is None,
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


def clarification(response, target, discount, months=3):
    plan = response["diagnostics"]["plan"]
    answer = response["answer"].casefold()
    checks = {
        "уточнение без инструментов и прогноза": (
            response["status"] == "clarification"
            and response["clarification"]
            and response["diagnostics"]["steps"] == []
            and response["forecast"] is None
        ),
        "верны сорт, горизонт и дисконт": (
            plan.get("benchmark") == target
            and plan.get("months") == months
            and plan.get("discount") == discount
        ),
    }
    if months is None:
        checks["вопрос о горизонте"] = any(word in answer for word in ("месяц", "горизонт", "срок"))
    else:
        checks["вопрос о дисконте"] = "дисконт" in answer and (
            discount is None or (target.casefold() in answer and "urals" in answer)
        )
    return checks


def check(base_url, question, scorer, *expected, **request_fields):
    print(question, flush=True)
    try:
        response = ask_api(base_url, question, **request_fields)
        failed = [name for name, ok in scorer(response, *expected).items() if not ok]
        print("  ", response.get("mode"), response.get("trace_id"), ", ".join(failed) or "PASS")
        return None if failed else response
    except (ValueError, TypeError, KeyError, URLError, TimeoutError) as error:
        print("  Ошибка:", error)
        return None


def memory_dialogue(base_url, *, correct_horizon=False):
    turns = [("Спрогнозируй Urals", None), ("На 3 месяца", 3)]
    if correct_horizon:
        turns.append(("Измени горизонт на 6 месяцев", 6))
    token = None
    for question, months in turns:
        fields = {"remember_clarification": True} if token is None else {"dialogue_token": token}
        response = check(base_url, question, clarification, "Urals", None, months, **fields)
        if response is None:
            return False
        next_token = response.get("dialogue_token")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            print("   Для следующего уточнения ожидался новый идентификатор диалога.")
            return False
        token = next_token
    final = check(
        base_url,
        "12 долларов за баррель",
        calculation,
        "Urals",
        turns[-1][1],
        12,
        dialogue_token=token,
    )
    return final is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument(
        "--memory-only", action="store_true", help="Проверить два диалога с памятью уточнений"
    )
    args = parser.parse_args()

    passed = []
    if not args.memory_only:
        passed.extend(
            check(args.base_url, q, calculation, grade, months, discount) is not None
            for q, grade, months, discount in CASES
        )
        passed.append(check(args.base_url, UNSUPPORTED, clarification, "WTI", 12) is not None)
        legacy = check(args.base_url, DIALOG, clarification, "Urals", None)
        if legacy is not None:
            legacy = check(
                args.base_url,
                "12 долларов за баррель",
                calculation,
                "Urals",
                3,
                12,
                previous=DIALOG,
            )
        passed.append(legacy is not None)
    passed.append(memory_dialogue(args.base_url))
    passed.append(memory_dialogue(args.base_url, correct_horizon=True))
    print(f"Сценариев пройдено: {sum(passed)}/{len(passed)}.")
    return int(not all(passed))


if __name__ == "__main__":
    raise SystemExit(main())
