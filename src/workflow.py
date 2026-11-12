"""Bounded LangGraph workflow; LLM plans, code enforces source priority."""

from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .answering import verified_answer
from .config import settings
from .integrations.llm import LLMUnavailable, llm
from .integrations.telemetry import callbacks, request_trace, span
from .logging import log
from .models.schemas import AgentResponse, EvidenceAnswer
from .planning import make_plan
from .tools import make_registry, run_tool
from .validation import normalized


class State(TypedDict, total=False):
    question: str
    previous: str | None
    plan: dict
    reports: list
    web: list
    draft: dict
    calc: dict | None
    notes: list[str]
    trace: list
    failure: str
    response: dict
    answer_repairs: int


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
    from .calculations.forecast import month_number

    if month_number(date.today().isoformat()[:7]) - month_number(calc["as_of"]) > 2:
        answer += "\n\nПоследнее наблюдение старше двух месяцев. Обновите CSV для прогноза от текущей даты."
    if last["mae"] >= last["naive_mae"]:
        answer += (
            "\n\nНа этом горизонте ансамбль не превосходит наивный прогноз по исторической MAE."
        )
    return (
        answer
        + "\n\nПрогнозный интервал не гарантирует будущую цену; сценарные допущения считаются фиксированными."
    )


def build_graph(client=None, registry=None, execute=run_tool, planner=make_plan):
    client, registry = client or llm(), registry or make_registry()

    def route(s):
        try:
            plan = planner(client, s["question"], s.get("previous"))
            return {"plan": plan.model_dump()}
        except LLMUnavailable as exc:
            return {"failure": str(exc)}

    def invoke(s, name, **kwargs):
        data, error, event = execute(registry, name, **kwargs)
        return data, {
            "trace": s["trace"] + [event],
            "notes": s["notes"] + ([error] if error else []),
        }

    def retrieve(s):
        data, update = invoke(s, "report_search", query=s["plan"]["rag_query"] or s["question"])
        return {**update, "reports": data or []}

    def assess(s):
        if not s["plan"]["wants_context"]:
            return {}
        is_forecast = s["plan"]["intent"] == "forecast"
        sources = s["reports"] + s["web"]
        empty = EvidenceAnswer(complete=False, claims=[], limitations=[]).model_dump()
        if not sources:
            return {"draft": empty}
        try:
            draft, repaired = verified_answer(
                client,
                {
                    "task": "forecast_context" if is_forecast else "research",
                    "question": s["plan"]["rag_query"] if is_forecast else s["question"],
                    "previous": None if is_forecast else s.get("previous"),
                    "today": date.today().isoformat() if s["plan"]["fresh"] else None,
                    "requires_fresh": s["plan"]["fresh"],
                    "sources": sources,
                },
                allow_repair=s.get("answer_repairs", 0) == 0,
            )
            return {
                "draft": draft.model_dump(),
                "answer_repairs": s.get("answer_repairs", 0) + int(repaired),
            }
        except LLMUnavailable:
            return {
                "draft": empty,
                "notes": s["notes"] + ["Не удалось проверить ответ модели по источникам."],
            }

    def internet(s):
        p = s["plan"]
        data, update = invoke(
            s,
            "web_search",
            query=p["rag_query"] if p["intent"] == "forecast" else s["question"],
            fresh=p["fresh"],
            quotes=p["quotes"],
            search=p["news"] or p["wants_context"] or not p["quotes"],
        )
        data = data or {}
        return {
            **update,
            "web": data.get("items", []),
            "notes": update["notes"] + data.get("notices", []),
        }

    def calculate(s):
        data, update = invoke(
            s,
            "price_forecast",
            **{k: s["plan"][k] for k in ("benchmark", "months", "discount", "cut", "sensitivity")},
        )
        return {**update, "calc": data}

    def after_route(s):
        p = s.get("plan")
        return (
            "finish" if s.get("failure") or p["intent"] == "outside" or p["clarify"] else "reports"
        )

    def after_assess(s):
        p = s["plan"]
        if not p["report_only"] and (
            p["fresh"]
            or p["news"]
            or p["quotes"]
            or (p["wants_context"] and not s["draft"]["complete"])
        ):
            return "web"
        return "forecast" if p["intent"] == "forecast" else "finish"

    def after_web(s):
        p = s["plan"]
        known = {(normalized(r["text"]), r.get("date")) for r in s["reports"]}
        novel = any((normalized(r["text"]), r.get("date")) not in known for r in s["web"])
        if p["wants_context"] and novel:
            return "synthesize"
        log("answer.web_synthesis_skipped", wants_context=p["wants_context"], new_context=novel)
        return "forecast" if p["intent"] == "forecast" else "finish"

    def finish(s):
        p, calc, notes = s.get("plan"), s.get("calc"), list(s["notes"])
        sources, claims, blocks = [], [], []
        status, intent = "partial", p["intent"] if p else "unknown"
        if s.get("failure"):
            blocks = [s["failure"] + " Проверьте подключение LLM и поддержку structured outputs."]
            status = "unavailable"
        elif p["intent"] == "outside":
            blocks = [
                "Этот запрос вне нефтегазовой специализации. Могу помочь с отраслевыми отчётами, ценами нефти и расчётом прогноза."
            ]
            status = "outside"
        elif p["clarify"]:
            blocks, status = [p["clarify"]], "clarification"
        else:
            # Calculation is the primary answer for a forecast. Its availability
            # never depends on whether a report contains the same prediction.
            if calc:
                blocks.append(forecast_answer(calc))
                sources.append(
                    {
                        "id": "M1",
                        "kind": "model",
                        "title": f"Расчёт · EIA/FRED · {calc['base']}",
                        "date": calc["as_of"],
                        "url": calc["source_url"],
                        "text": "ARIMA(1,1,0), экспоненциальное сглаживание; интервал по историческим ошибкам.",
                    }
                )
            lookup = {r["id"]: r for r in s["reports"] + s["web"]}
            claims = s["draft"]["claims"]
            for c in claims:
                r = lookup[c["source"]]
                label = "Отчёт" if r["kind"] == "report" else "Веб"
                blocks.append(
                    f"{c['text']} [{c['source']}] ({label}, {r['date'] or 'дата не указана'})"
                )
                sources.append(r)
            for r in s["web"]:
                if "price" in r and r["id"] not in {c["source"] for c in claims}:
                    blocks.append(r["text"] + f" [{r['id']}] (Веб, {r['date']})")
                    sources.append(r)
            enough = not p["wants_context"] or s["draft"]["complete"]
            if p["intent"] == "forecast":
                enough = enough and bool(calc)
                if not calc:
                    notes.append("Расчётный модуль не вернул прогноз.")
                if p["wants_context"] and not s["draft"]["complete"]:
                    notes.append(
                        f"Дополнительный вопрос «{p['rag_query']}» подтверждён источниками не полностью."
                    )
            if p["quotes"]:
                found = {r.get("grade") for r in s["web"] if "price" in r}
                missing = set(p["quotes"]) - found
                enough = enough and not missing
                if missing:
                    notes.append(
                        f"Последние опубликованные котировки {', '.join(sorted(missing))} не получены."
                    )
            if p["fresh"] and not s["web"]:
                enough = False
                notes.append("Актуальные данные для этого запроса не подтверждены.")
            if p["report_only"] and p["fresh"]:
                notes.append(
                    "Использованы только загруженные отчёты; свежесть на сегодняшний день не проверялась."
                )
            if not blocks:
                blocks, enough = (
                    [
                        "Прогноз не рассчитан."
                        if p["intent"] == "forecast"
                        else "Подтверждённых данных для ответа недостаточно."
                    ],
                    False,
                )
            # Unverified free-form missing-data claims are never shown as facts.
            if not enough and p["intent"] != "forecast":
                notes.append(
                    "Источники подтверждают только часть запроса или не содержат необходимых сведений."
                )
            status = "ok" if enough and not notes else "partial"
        blocks += list(dict.fromkeys(notes))
        return {
            "response": {
                "question": s["question"],
                "answer": "\n\n".join(blocks),
                "status": status,
                "intent": intent,
                "clarification": status == "clarification",
                "mode": settings().llm_model,
                "sources": list({r["id"]: r for r in sources}.values()),
                "claims": claims,
                "trace": s["trace"],
                "forecast": calc,
            }
        }

    def observed(name, fn):
        def call(s):
            with span("workflow." + name) as obs:
                result = fn(s)
                obs.update(output=result)
                return result

        return call

    g = StateGraph(State)
    for name, fn in [
        ("plan", route),
        ("reports", retrieve),
        ("assess", assess),
        ("web", internet),
        ("synthesize", assess),
        ("forecast", calculate),
        ("finish", finish),
    ]:
        g.add_node(name, observed(name, fn))
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_route, {"reports": "reports", "finish": "finish"})
    g.add_edge("reports", "assess")
    g.add_conditional_edges("assess", after_assess, {k: k for k in ("web", "forecast", "finish")})
    g.add_conditional_edges("web", after_web, {k: k for k in ("synthesize", "forecast", "finish")})
    g.add_conditional_edges(
        "synthesize",
        lambda s: "forecast" if s["plan"]["intent"] == "forecast" else "finish",
        {"forecast": "forecast", "finish": "finish"},
    )
    g.add_edge("forecast", "finish")
    g.add_edge("finish", END)
    return g.compile()


def ask(
    question: str,
    previous: str | None = None,
    *,
    client=None,
    registry=None,
    execute=run_tool,
    planner=make_plan,
) -> dict:
    question = question.strip()
    if not 1 <= len(question) <= 2000:
        raise ValueError("Введите вопрос длиной от 1 до 2000 символов.")
    with request_trace(question, previous) as (trace_id, obs):
        initial = {
            "question": question,
            "previous": previous,
            "reports": [],
            "web": [],
            "notes": [],
            "trace": [],
            "calc": None,
            "answer_repairs": 0,
            "draft": EvidenceAnswer(complete=False, claims=[], limitations=[]).model_dump(),
        }
        state = build_graph(client, registry, execute, planner).invoke(
            initial,
            config={
                "callbacks": callbacks(),
                "metadata": {"trace_id": trace_id},
                "recursion_limit": 15,
            },
        )
        response = AgentResponse.model_validate(
            {**state["response"], "trace_id": trace_id}
        ).model_dump()
        obs.update(
            output=response,
            metadata={"intent": response["intent"], "status": response["status"]},
            status="error" if response["status"] == "unavailable" else "ok",
        )
        return response
