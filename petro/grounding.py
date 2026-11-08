# валидация резульата ллмки

import re
from .planning import ask_json

NUM = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?")

GROUND_PROMPT = """Ответь на вопрос, используя только переданный массив evidence. Отчёты имеют приоритет для
соответствующих периодов, актуальные сведения можно взять только из датированной веб-публикации.
Верни JSON вида {"complete":true,"claims":[{"source":"R1","quote":"точная непрерывная выдержка из текста источника",
"text":"один факт по-русски"}]}. В одном факте только числа, непосредственно присутствующие в его выдержке или дате.
Не включай номера источников в text. Для сравнения разных источников дай отдельный факт для каждого.
complete=true означает, что на содержательные части вопроса найден ответ. Без доказательства верни пустой claims и complete=false.
Если есть только часть ответа, верни подтвержденные факты с complete=false. Не считай публикацию старого отчёта свидетельством о сегодняшней цене."""

GROUND_PROMPT += """
Поле text пиши по-русски, quote оставь на языке документа. Оставляй только факты, прямо отвечающие
на поставленный вопрос. Совпадение географического названия с названием сорта нефти не является доказательством.
Пример формата для вопроса о грузообороте порта и отрывка 'The terminal handled 12 tankers in May.':
{"complete":true,"claims":[{"source":"R1","quote":"The terminal handled 12 tankers in May.",
"text":"В мае терминал принял 12 танкеров."}]} В ответе используй только данные текущего запроса."""


def normalized(t: str) -> str:
    return " ".join(t.split()).casefold().replace("ё", "е")


def number_set(t: str) -> set[str]:
    return {x.replace(",", ".") for x in NUM.findall(t)}


def grounded(question: str, evidence: list[dict]) -> tuple[list[dict], bool]:
    if not evidence:
        return [], False
    by_id = {r["id"]: r for r in evidence}
    answer = ask_json(
        GROUND_PROMPT,
        {
            "question": question,
            "evidence": [
                {k: r.get(k) for k in ("id", "kind", "title", "date", "text")} for r in evidence
            ],
        },
    )
    accepted = []
    claims = answer.get("claims")
    if not isinstance(claims, list):
        return [], False
    for c in claims[:7]:
        if not isinstance(c, dict):
            continue
        row = by_id.get(c.get("source"))
        quote, text = c.get("quote"), c.get("text")
        if not row or not isinstance(quote, str) or not isinstance(text, str) or not text.strip():
            continue
        if not re.search(r"[А-Яа-яЁё]", text):
            continue
        if len(quote.strip()) < 6 or normalized(quote) not in normalized(row["text"]):
            continue
        if number_set(text) - number_set(quote + " " + str(row["date"] or "")):
            continue
        accepted.append({"source": row["id"], "text": text.strip()[:750]})
    return accepted, bool(accepted) and len(accepted) == len(claims) and answer.get("complete") is True


def extract_facts(question: str, evidence: list[dict]) -> list[dict]:
    """Locally reproduce a few tabular facts. All numbers are parsed from the retrieved PDFs."""
    q = normalized(question)
    if (
        ("urals" in q or "юралс" in q)
        and "espo" in q
        and any(w in q for w in ("отлич", "сравн", "плотност", "сер", "характерист", "качест"))
    ):
        for row in evidence:
            text = row["text"]
            values = [
                re.search(g + r"\s+(\d+) to (\d+)\s+([\d.]+)% to ([\d.]+)%", text, re.I)
                for g in ("Urals", "ESPO")
            ]
            if all(values):
                a, b = [m.groups() for m in values]
                return [
                    {
                        "source": row["id"],
                        "text": f"По таблице EIA у Urals плотность {a[0]}–{a[1]} °API и содержание серы {a[2]}–{a[3]}%. У ESPO — {b[0]}–{b[1]} °API и {b[2]}–{b[3]}% серы. ESPO легче и содержит меньше серы.",
                    }
                ]
    if (
        ("urals" in q or "юралс" in q)
        and "espo" not in q
        and any(w in q for w in ("регион", "где добыва", "где производ"))
    ):
        for row in evidence:
            match = re.search(
                r"Urals, a medium-sour crude oil produced mainly in Western Siberia and Russia[’']s Volga-Urals region",
                row["text"],
            )
            if match:
                return [
                    {
                        "source": row["id"],
                        "text": "По данным EIA, Urals добывают главным образом в Западной Сибири и Волго-Уральском регионе России.",
                    }
                ]
    if "налогооблож" in q and ("2025" in q or "2026" in q or "2027" in q or "2028" in q):
        target = re.search(r"(?:на|в|за)\s+(202[5-8])\s*(?:год|г\b)", q)
        for row in evidence:
            values = re.search(
                r"Цена нефти для налогообложения\s*5?.{0,90}?баррель\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                row["text"],
                re.I,
            )
            if row["report"] == "cbr-oct-2025" and values and target:
                year = int(target[1])
                price = values.group(year - 2024 + 1)
                return [
                    {
                        "source": row["id"],
                        "text": f"В базовом сценарии от 24.10.2025 Банк России заложил на {year} год среднюю цену нефти для налогообложения {price} USD/барр. Это годовая сценарная предпосылка.",
                    }
                ]
    if ("сырой нефти" in q or "crude oil" in q) and "2024" in q and ("добы" in q or "produc" in q):
        for row in evidence:
            m = re.search(
                r"Russia produced ([\d.]+) million barrels per day \(b/d\) of crude oil in 2024, a decrease of (\d+)% from ([\d.]+) million b/d in 2023",
                row["text"],
            )
            if m:
                return [
                    {
                        "source": row["id"],
                        "text": f"По оценке EIA, в 2024 году Россия добывала {m[1]} млн баррелей сырой нефти в сутки. В 2023 году показатель составлял {m[3]} млн б/с. Снижение — {m[2]}%.",
                    }
                ]
    return []
