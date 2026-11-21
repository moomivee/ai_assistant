"""Check four public API answers, or replay their exported UI traces (stdlib only)."""

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

QUESTIONS = {
    "forecast": "Спрогнозируй среднемесячную цену Brent на 3 месяца.",
    "comparison": "Чем отличаются Urals и ESPO по плотности и содержанию серы?",
    "quote": "Какая последняя опубликованная спотовая цена Brent?",
    "mixed": "Чем по качеству отличаются Urals и ESPO и какова последняя цена Brent?",
}
# EIA Country Analysis Brief: Russia, July 2025, Table 2. Other assays need review.
EIA_TABLE = {"Urals": ((30, 32), (1.3, 1.5)), "ESPO": ((34, 37), (0.4, 0.6))}
RANGE_SEPARATOR = r"\s*(?:[-–—−]|to|до)\s*"
PRICE_WITH_UNIT = re.compile(
    r"(?<![\w.,−-])(\d+(?:[.,]\d+)?)\s*(?:USD|доллар[а-я]*|долл\.?|\$)"
    r"\s*(?:/|за|per)\s*(?:барр|barrel|bbl)",
    re.I,
)
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
GRADE = re.compile(r"\b(Urals|ESPO|Sokol|Varandey)\b", re.I)


def numbers(text):
    return {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", text)}


def usd_per_barrel(text):
    return bool(
        re.search(r"USD|доллар|долл\.|\$", text, re.I) and re.search(r"барр|barrel|bbl", text, re.I)
    )


def grade_sections(text, grade):
    parts = GRADE.split(text)
    return " ".join(
        part for name, part in zip(parts[1::2], parts[2::2]) if name.lower() == grade.lower()
    )


def quality_metrics(text, expected, *, table_row=False):
    (low_api, high_api), (low_sulfur, high_sulfur) = expected
    api = rf"(?<![\d.,]){low_api}{RANGE_SEPARATOR}{high_api}(?![\d.,])"
    low = str(low_sulfur).replace(".", "[.,]")
    high = str(high_sulfur).replace(".", "[.,]")
    sulfur = rf"(?<![\d.,]){low}\s*%?{RANGE_SEPARATOR}{high}\s*%"
    metrics = set()
    if re.search(rf"\bAPI\s*[:=]?\s*(?:от\s+)?{api}(?!\s*%)|{api}\s*°?\s*API\b", text, re.I):
        metrics.add("API")
    if table_row and re.search(rf"{api}\s+{sulfur}", text, re.I):
        # Flattened EIA table rows carry the API unit in the column header.
        metrics.add("API")
    if re.search(sulfur, text, re.I):
        metrics.add("sulfur")
    return metrics


def comparison_checks(response):
    answer = response["answer"]
    checks = {}
    for grade, gold in EIA_TABLE.items():
        verified = set()
        for claim in response.get("claims") or []:
            text = claim.get("text", "")
            if not text or text not in answer:
                continue
            citations = claim.get("citations") or [claim]
            for citation in citations:
                source_id = citation.get("source")
                source = next((s for s in response["sources"] if s["id"] == source_id), None)
                if not source or f"[{source_id}]" not in answer:
                    continue
                quote = citation.get("quote", "")
                excerpts = [source.get("text", ""), *(source.get("evidence") or {}).values()]
                if not quote or not any(
                    " ".join(quote.split()) in " ".join(e.split()) for e in excerpts
                ):
                    continue
                source_metrics = quality_metrics(grade_sections(quote, grade), gold, table_row=True)
                claim_metrics = quality_metrics(grade_sections(text, grade), gold)
                verified |= source_metrics & claim_metrics
        checks[f"{grade}: диапазоны °API и % серы привязаны к сорту и цитате"] = verified == {
            "API",
            "sulfur",
        }
    checks["указаны шкала API и содержание серы в процентах"] = bool(
        re.search(r"\bAPI\b", answer, re.I)
        and "%" in answer
        and re.search(r"сер[ауыни]|sulf", answer, re.I)
    )
    return checks


def valid_intervals(rows, as_of, months=3):
    try:
        last = date.fromisoformat(as_of + "-01")
        for row in rows:
            year, month = divmod(last.year * 12 + last.month, 12)
            last = date(year, month + 1, 1)
            if (
                row["month"] != last.strftime("%Y-%m")
                or not row["lower"] <= row["point"] <= row["upper"]
            ):
                return False
        return len(rows) == months
    except (KeyError, TypeError, ValueError):
        return False


def forecast_checks(response, benchmark="Brent", months=3):
    forecast = response.get("forecast") or {}
    rows = forecast.get("rows") or []
    last = rows[-1] if rows else {}
    answer = response["answer"]
    return {
        f"{benchmark}, последовательные месяцы с прогнозными интервалами ({months})": (
            forecast.get("benchmark") == benchmark
            and valid_intervals(rows, forecast.get("as_of"), months)
        ),
        "конечный месяц, цена, границы интервала и единицы видны в ответе": bool(
            last.get("month")
            and last["month"] in answer
            and usd_per_barrel(answer)
            and {last.get(k) for k in ("point", "lower", "upper")} <= numbers(answer)
        ),
        "ссылка на расчёт в ответе": any(
            s.get("kind") == "model" and f"[{s['id']}]" in answer for s in response["sources"]
        ),
    }


def quote_checks(response):
    answer = response["answer"]
    today = (response.get("diagnostics") or {}).get("time_context", {}).get(
        "today"
    ) or date.today().isoformat()
    found = False
    for source in response["sources"]:
        price = source.get("price")
        if (
            source.get("grade") != "Brent"
            or source.get("kind") != "web"
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or price <= 0
        ):
            continue
        try:
            day = date.fromisoformat(source["date"])
        except (ValueError, TypeError, KeyError):
            continue
        dates = (
            day.isoformat(),
            day.strftime("%d.%m.%Y"),
            f"{day.day} {MONTHS[day.month - 1]} {day.year}",
        )
        if day > date.fromisoformat(today) or not source.get("url"):
            continue
        if any(
            "brent" in sentence.lower()
            and any(d in sentence for d in dates)
            and any(float(p.replace(",", ".")) == price for p in PRICE_WITH_UNIT.findall(sentence))
            for paragraph in re.split(r"\n\s*\n", answer)
            if f"[{source['id']}]" in paragraph
            for sentence in re.split(r"(?<=[.!?;])\s+(?=[А-ЯA-Z])", paragraph)
        ):
            found = True
            break
    return {
        "цена Brent, дата наблюдения и ссылка совпали с данными котировки": found,
        "цена указана в USD за баррель": usd_per_barrel(answer),
    }


def score(case, response):
    diagnostics = response.get("diagnostics") or {}
    checks = {"статус ok": response.get("status") == "ok"}
    if case == "forecast":
        checks.update(forecast_checks(response))
    if case in {"comparison", "mixed"}:
        checks.update(comparison_checks(response))
    if case in {"quote", "mixed"}:
        checks.update(quote_checks(response))
    steps = diagnostics.get("steps") or []
    quote_provider_error = any(
        "не удалось получить котировку" in notice.lower() and "FRED" in notice
        for step in steps
        for notice in (step.get("notices") or [])
    )
    return {
        "quote_provider_error": quote_provider_error,
        "case": case,
        "question": QUESTIONS[case],
        "passed": all(checks.values()),
        "checks": checks,
        "trace_id": response.get("trace_id"),
        "status": response.get("status"),
        "stop_reason": response.get("stop_reason"),
        "failure": diagnostics.get("failure"),
        "sources": [
            {k: s.get(k) for k in ("id", "kind", "title", "date", "url", "page")}
            for s in response["sources"]
        ],
        "issues": [
            {
                "tool": s.get("tool"),
                "status": s.get("status"),
                "error": s.get("error"),
                "notices": s.get("notices"),
            }
            for s in steps
            if s.get("error") or s.get("notices")
        ],
        "answer": response["answer"],
    }


def saved_answers(directory):
    answers = {}
    for path in sorted(directory.glob("agent-trace-*.json")):
        response = json.loads(path.read_text(encoding="utf-8"))
        question = response.get("question")
        if question not in QUESTIONS.values():
            continue
        if question in answers:
            raise ValueError(
                f"В папке два трейса для вопроса «{question}». Разложите прогоны по разным папкам."
            )
        answers[question] = response
    return answers


def ask_api(base_url, question, previous=None):
    payload = {"question": question}
    if previous is not None:
        payload["previous"] = previous
    request = Request(
        base_url.rstrip("/") + "/api/ask",
        json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as result:
        return json.load(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base-url", default="http://127.0.0.1:8003")
    mode.add_argument(
        "--trace-dir", type=Path, help="Папка с экспортами agent-trace-*.json одного прогона"
    )
    parser.add_argument(
        "--output", type=Path, help="Новый JSON-файл с ответами, источниками и проверками"
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Файл уже существует: {args.output}")
    saved = saved_answers(args.trace_dir) if args.trace_dir is not None else None
    rows = []
    for case, question in QUESTIONS.items():
        try:
            if saved is None:
                response = ask_api(args.base_url, question)
            else:
                response = saved[question]
            row = score(case, response)
        except (ValueError, KeyError, TypeError, URLError, TimeoutError) as error:
            row = {"case": case, "passed": False, "error": str(error)}
        rows.append(row)
        print(f"{case}: {'PASS' if row['passed'] else 'CHECK'} · {question}", flush=True)
        for label, ok in row.get("checks", {}).items():
            print(f"  {'+' if ok else '-'} {label}")
        for issue in row.get("issues", []):
            print(
                f"  {issue['tool']} ({issue['status']}) {issue['error'] or ''} {' '.join(issue['notices'] or [])}"
            )
        if row.get("quote_provider_error"):
            print(
                "  Стенд сообщил об ошибке получения котировки EIA/FRED. Проверьте сообщение инструмента."
            )
        if row.get("failure") or row.get("error"):
            print(" ", row.get("failure") or row.get("error"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(rows, output, ensure_ascii=False, indent=2)
    passed = sum(row["passed"] for row in rows)
    print(
        f"{passed}/{len(rows)}. Сравнение сверяется с таблицей 2 EIA (2025). Другой источник или формат оцените вручную."
    )
    return int(passed != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
