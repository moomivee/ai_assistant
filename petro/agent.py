import re
from datetime import date
from functools import lru_cache
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError
from .config import settings
from .grounding import extract_facts, grounded, number_set
from .planning import PLAN_PROMPT, Plan, ask_json, contextual_question, local_plan
from .tools import AnalystRegistry, make_registry, run_tool

LLM_ERRORS = (ValueError, TypeError, IndexError, KeyError, OpenAIError)


class State(TypedDict, total=False):
    question: str
    registry: AnalystRegistry
    plan: Plan
    llm_ok: bool
    reports: list
    web: list
    facts: list
    complete: bool
    calc: dict
    notes: list
    trace: list
    answer: str
    sources: list


def route(state: State):
    q = state["question"]
    plan = local_plan(q)
    online = bool(settings().llm_key)
    notes = []
    if online:
        try:
            result = ask_json(PLAN_PROMPT, {"question": q, "today": date.today().isoformat()})
            if "intent" not in result:
                raise ValueError("missing intent")
            learned = Plan.model_validate(result)
            # Explicit scope, freshness and user-entered scenario values override inferred ones.
            learned.report_only |= plan.report_only
            if learned.report_only:
                learned.wants_context = True
            if plan.intent != "outside":
                if plan.fresh and learned.intent == "research":
                    learned.fresh = True
                if plan.intent == learned.intent == "forecast":
                    for field in ("benchmark", "discount", "cut", "sensitivity"):
                        v = getattr(plan, field)
                        if v is not None:
                            setattr(learned, field, v)
                    learned.clarify = plan.clarify or learned.clarify
                if learned.intent == "research" and plan.quotes:
                    learned.quotes = plan.quotes
            # Guard against invented numerical assumptions: only accept explicit inputs.
            if learned.intent == "forecast":
                numbers = {float(v) for v in number_set(q)}
                markers = {
                    "discount": r"дисконт|скидк|discount|spread",
                    "cut": r"сокращ|снижен|урез|cut|reduction",
                    "sensitivity": r"чувствительност|sensitivity|коэффициент",
                }
                for field, marker in markers.items():
                    v = getattr(learned, field)
                    if v is not None and (v not in numbers or not re.search(marker, q, re.I)):
                        setattr(learned, field, None)
                if not learned.benchmark:
                    learned.clarify = "Укажите сорт нефти для расчёта."
                if not 1 <= learned.months <= 12:
                    learned.clarify = "Укажите горизонт от 1 до 12 месяцев."
            if not learned.rag_query:
                learned.rag_query = plan.rag_query
            # Keep a table-specific cross-language search phrase for known series.
            if plan.rag_query != q and plan.intent != "outside":
                learned.rag_query = plan.rag_query
            plan = learned
        except LLM_ERRORS:
            online = False
            notes.append("Ответ LLM для маршрутизации недоступен. Включён локальный разбор запроса.")
    return {"plan": plan, "llm_ok": online, "notes": notes}


def retrieve(state: State):
    p = state["plan"]
    items, error, event = run_tool(
        state["registry"], "report_search", query=p.rag_query or state["question"]
    )
    return {
        "reports": items or [],
        "trace": [event],
        "notes": state["notes"] + ([error] if error else []),
    }


def assess(state: State):
    p, q, rows = state["plan"], state["question"], state["reports"]
    if not p.wants_context:
        return {"facts": [], "complete": False}
    known = extract_facts(q, rows)
    if known and (not state["llm_ok"] or not _multipart(q)):
        return {"facts": known, "complete": True}
    if state["llm_ok"] and rows:
        try:
            facts, complete = grounded(q, rows)
            return {"facts": facts or known, "complete": complete if facts else bool(known)}
        except LLM_ERRORS:
            return {
                "facts": known,
                "complete": bool(known),
                "llm_ok": False,
                "notes": state["notes"] + ["Не удалось проверить ответ LLM по выдержкам."],
            }
    return {"facts": known, "complete": bool(known)}


def _multipart(question: str) -> bool:
    return bool(
        re.search(
            r"(?:\?|,|\sи)\s*(?:како|каков|сколько|почему|как |что |где |кто |когда )", question, re.I
        )
    )


def internet(state: State):
    p = state["plan"]
    result, error, event = run_tool(
        state["registry"],
        "web_search",
        query=state["question"],
        fresh=p.fresh,
        quotes=p.quotes,
        search=p.news or not p.quotes,
    )
    result = result or {}
    return {
        "web": result.get("items", []),
        "trace": state["trace"] + [event],
        "notes": state["notes"] + result.get("notices", []) + ([error] if error else []),
    }


def calculate(state: State):
    p = state["plan"]
    data, error, event = run_tool(
        state["registry"],
        "price_forecast",
        benchmark=p.benchmark,
        months=p.months,
        discount=p.discount,
        cut=p.cut,
        sensitivity=p.sensitivity,
    )
    return {
        "calc": data,
        "trace": state["trace"] + [event],
        "notes": state["notes"] + ([error] if error else []),
    }


def after_route(state: State):
    p = state["plan"]
    return "finish" if p.intent == "outside" or p.clarify else "reports"


def after_assess(state: State):
    p = state["plan"]
    if p.report_only:
        return "finish"
    return "web" if p.fresh or p.news or p.quotes or not state["complete"] else "finish"


def forecast_answer(calc: dict):
    last = calc["rows"][-1]
    answer = (
        f"Прогноз {calc['benchmark']} на {len(calc['rows'])} мес. В {last['month']} модель ожидает "
        f"{last['point']:.2f} USD/барр. Ориентировочный 90%-й прогнозный интервал — "
        f"{last['lower']:.2f}–{last['upper']:.2f} USD/барр. [M1]\n\n"
        f"Расчёт начинается после {calc['as_of']}. Последняя среднемесячная цена {calc['base']} "
        f"составила {calc['last']:.2f} USD/барр. [M1]\n\n"
        f"Средняя абсолютная ошибка ансамбля для этого горизонта на {calc['origins']} исторических точках старта "
        f"составила {last['mae']:.2f} USD/барр. Для прогноза «цена останется последней» — {last['naive_mae']:.2f}. [M1]"
    )
    if calc["discount"] is not None:
        answer += f"\n\nУсловный Urals равен расчёту Brent минус заданные {calc['discount']:g} USD/барр. Интервал условен при фиксированном дисконте."
    if calc["cut"] is not None:
        answer += (
            f"\n\nСценарная надбавка ко всем месяцам — {calc['cut']:g} × {calc['sensitivity']:g} = "
            f"{calc['cut'] * calc['sensitivity']:g} USD/барр. Чувствительность задана пользователем."
        )
    from .forecast import month_number

    if month_number(date.today().isoformat()[:7]) - month_number(calc["as_of"]) > 2:
        answer += (
            "\n\nПоследнее наблюдение старше двух месяцев. Обновите CSV для прогноза от текущей даты."
        )
    return answer


def finish(state: State):
    p, notes = state["plan"], list(state["notes"])
    rows, web, facts = state.get("reports", []), state.get("web", []), list(state.get("facts", []))
    sources = []
    if p.intent == "outside":
        answer = "Я отвечаю на вопросы о нефтегазовом рынке. Могу, например, сравнить российские сорта нефти или рассчитать прогноз Brent на три месяца."
    elif p.clarify:
        answer = p.clarify
    elif p.intent == "forecast":
        calc = state.get("calc")
        answer = forecast_answer(calc) if calc else "Для такого расчёта нужны дополнительные данные."
        if calc:
            sources = [
                {
                    "id": "M1",
                    "kind": "model",
                    "title": f"Расчёт · EIA/FRED · {calc['base']}, среднемесячный спот",
                    "date": calc["as_of"],
                    "url": calc["source_url"],
                    "text": "ARIMA(1,1,0) и простое экспоненциальное сглаживание. Интервал калиброван по историческим ошибкам отдельно для каждого горизонта.",
                }
            ]
    else:
        general = [w for w in web if "price" not in w]
        if general and state["llm_ok"]:
            try:
                claims, covered = grounded(
                    state["question"], (rows if p.wants_context else []) + general
                )
                existing = {f["text"] for f in facts}
                facts.extend(c for c in claims if c["text"] not in existing)
                if claims and not covered:
                    notes.append("Источники подтверждают только часть запрошенных сведений.")
            except LLM_ERRORS:
                notes.append("Не удалось проверить синтез веб-публикаций.")
        if (
            not facts
            and rows
            and p.wants_context
            and not (p.fresh and not general)
            and not state.get("complete")
        ):
            if not general or p.report_only:
                notes.append(
                    "Точного ответа по фрагментам подтвердить не удалось. Для проверки показаны ближайшие страницы отчётов."
                )
                sources.extend(rows[:2])
        for w in web:
            if "price" in w:
                facts.append({"source": w["id"], "text": w["text"]})
        if general and not state["llm_ok"]:
            for w in general[:3]:
                if "publisher" in w:
                    facts.append(
                        {"source": w["id"], "text": f"{w['date']} · {w['publisher']} · «{w['title']}»."}
                    )
        if general and not any(f["source"] in {r["id"] for r in general} for f in facts):
            sources.extend(general[:3])
            notes.append(
                "Найдены веб-публикации. Для самостоятельной проверки открыты выдержки и даты."
            )
        lookup = {row["id"]: row for row in rows + web}
        lines = []
        for fact in facts:
            row = lookup.get(fact["source"])
            if row:
                lines.append(f"{fact['text']} [{row['id']}]")
                sources.append(row)
        if p.report_only and p.fresh:
            notes.append(
                "По запросу использованы только отчёты. Свежесть на сегодняшний день не проверена."
            )
        if p.fresh and not web and not p.report_only:
            notes.append("Актуальные сведения для этого запроса не подтверждены.")
        answer = "\n\n".join(lines) or "Подтверждённых сведений для полного ответа пока недостаточно."
    if notes:
        answer += "\n\n" + " ".join(dict.fromkeys(notes))
    return {"answer": answer, "sources": list({r["id"]: r for r in sources}.values())}


@lru_cache
def graph():
    g = StateGraph(State)
    for name, fn in [
        ("plan_request", route),
        ("reports", retrieve),
        ("assess", assess),
        ("web_lookup", internet),
        ("forecast", calculate),
        ("finish", finish),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "plan_request")
    g.add_conditional_edges("plan_request", after_route, {"finish": "finish", "reports": "reports"})
    g.add_conditional_edges(
        "reports", lambda s: s["plan"].intent == "forecast", {True: "forecast", False: "assess"}
    )
    g.add_conditional_edges("assess", after_assess, {"web": "web_lookup", "finish": "finish"})
    g.add_edge("forecast", "finish")
    g.add_edge("web_lookup", "finish")
    g.add_edge("finish", END)
    return g.compile()


def ask(question: str, previous: str | None = None) -> dict:
    question = question.strip()
    if not 1 <= len(question) <= 2000:
        raise ValueError("Введите вопрос длиной от 1 до 2000 символов.")
    question = contextual_question(question, previous)
    state = graph().invoke(
        {"question": question, "registry": make_registry(), "notes": [], "trace": []}
    )
    clarification = bool(state["plan"].clarify) or (
        state["plan"].intent == "forecast" and not state.get("calc")
    )
    return {
        "clarification": clarification,
        "question": question,
        "answer": state["answer"],
        "sources": state.get("sources", []),
        "trace": state.get("trace", []),
        "forecast": state.get("calc"),
        "intent": state["plan"].intent,
        "mode": settings().llm_model if state["llm_ok"] else "Локальный разбор",
    }
