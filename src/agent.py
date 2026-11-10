"""Bounded observe/act loop: the LLM chooses the next tool after every result."""

import json
from datetime import date
from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import create_model

from .config import settings
from .evidence import generate_answer, render_claims
from .integrations.llm import LLMUnavailable, llm
from .integrations.telemetry import callbacks, request_trace, span
from .models.schemas import AgentResponse, CitedAnswer, NextAction
from .planning import make_plan
from .tools import make_registry, run_tool
from .workflow import forecast_answer


class State(TypedDict, total=False):
    question: str
    previous: str | None
    plan: dict
    sources: list
    observations: list
    decisions: list
    trace: list
    calc: dict | None
    decision: dict
    signatures: list
    attempts: int
    feedback: str
    answer_attempts: int
    draft: dict
    evidence: dict
    failure: str
    stop_reason: str
    response: dict


@lru_cache(maxsize=32)
def action_schema(allowed):
    return create_model("AvailableAction", __base__=NextAction, action=(Literal[allowed], ...))


def decision_sources(sources):
    """Bound context and replace fragmented pages with their opened full page."""
    full = {(s.get("report"), s.get("page")) for s in sources if s.get("full_page")}
    result, remaining = [], 14000
    for source in sorted(sources, key=lambda s: not s.get("full_page", False)):
        if not source.get("full_page") and (source.get("report"), source.get("page")) in full:
            continue
        if remaining <= 0:
            break
        row = {k: source.get(k) for k in ("id", "report", "page", "title", "kind", "full_page")}
        row["text"] = source["text"][:remaining]
        remaining -= len(row["text"])
        result.append(row)
    return result


def allowed_actions(state):
    p = state["plan"]
    called = [event["tool"] for event in state["trace"]]
    actions = []
    if p["wants_context"] or p["quotes"]:
        actions.append("report_search")
        if any(s["kind"] == "report" for s in state["sources"]):
            actions.append("read_report")
        if not p["report_only"] and "report_search" in called:
            actions.append("web_search")
    if p["intent"] == "forecast" and not state["calc"]:
        actions.append("price_forecast")
    # Finishing without a required attempt cannot silently skip the user's task.
    required = (
        (p["intent"] != "forecast" or bool(state["calc"]))
        and (not p["wants_context"] or "report_search" in called)
        and (not (p["fresh"] and not p["report_only"]) or "web_search" in called)
    )
    if required:
        actions.append("finish")
    if state["observations"]:
        last = state["observations"][-1]
        if str(last.get("error") or "").startswith("Duplicate action rejected"):
            alternatives = [name for name in actions if name != last["tool"]]
            # Do not force completion after an empty search: a different query
            # is still useful when no other evidence-gathering action exists.
            if any(name != "finish" for name in alternatives):
                actions = alternatives
    return actions


def merge_sources(existing, incoming):
    result = [dict(s) for s in existing]
    known = {s.get("chunk_id") or (s["url"], s["text"]): s for s in result}
    for source in incoming:
        key = source.get("chunk_id") or (source["url"], source["text"])
        if key not in known:
            prefix = "R" if source["kind"] == "report" else "W"
            number = 1 + sum(s["kind"] == source["kind"] for s in result)
            item = {**source, "id": f"{prefix}{number}"}
            result.append(item)
            known[key] = item
    return result


def build_graph(client=None, registry=None, execute=run_tool, planner=make_plan):
    client, registry = client or llm(), registry or make_registry()

    def plan(state):
        try:
            return {"plan": planner(client, state["question"], state.get("previous")).model_dump()}
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def decide(state):
        try:
            decision = client.generate(
                "decide",
                {
                    "question": state["question"],
                    "previous": state.get("previous"),
                    "task": state["plan"],
                    "today": date.today().isoformat(),
                    "allowed_actions": allowed_actions(state),
                    "steps_remaining": settings().agent_max_steps - state["attempts"],
                    "observations": state["observations"],
                    "sources": decision_sources(state["sources"]),
                    "calculation_available": bool(state["calc"]),
                    "answer_feedback": state["feedback"],
                },
                action_schema(tuple(allowed_actions(state))),
            ).model_dump()
            return {
                "decision": decision,
                "decisions": state["decisions"] + [decision],
                "attempts": state["attempts"] + 1,
            }
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def act(state):
        d, p = state["decision"], state["plan"]
        name = d["action"]
        error, data, event = None, None, None
        if name not in allowed_actions(state):
            error = "Action rejected by policy; choose one of allowed_actions."
            args = {}
        elif name == "report_search":
            args = {"query": d["query"]}
        elif name == "read_report":
            args = {"report": d["report"], "page": d["page"]}
            if d["report"] not in {s.get("report") for s in state["sources"]}:
                error = "Use a registered report ID from retrieved sources."
        elif name == "web_search":
            args = {
                "query": d["query"],
                "fresh": p["fresh"],
                "quotes": p["quotes"],
                "search": p["news"] or p["wants_context"] or not p["quotes"],
            }
        elif name == "price_forecast":
            args = {k: p[k] for k in ("benchmark", "months", "discount", "cut", "sensitivity")}
        else:
            args = {}
        signature = name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
        if signature in state["signatures"]:
            error = "Duplicate action rejected. Change the query or read another page; do not retry unchanged."
        if not error:
            data, error, event = execute(registry, name, **args)
        observation = {"tool": name, "arguments": args, "error": error}
        update = {
            "signatures": state["signatures"] + [signature],
            "trace": state["trace"] + ([event] if event else []),
        }
        if name in ("report_search", "read_report") and data is not None:
            update["sources"] = merge_sources(state["sources"], data)
            observation["result_count"] = len(data)
        elif name == "web_search" and data:
            update["sources"] = merge_sources(state["sources"], data.get("items", []))
            observation.update(
                result_count=len(data.get("items", [])), notices=data.get("notices", [])
            )
        elif name == "price_forecast" and data:
            update["calc"] = data
            observation["result"] = {k: data[k] for k in ("benchmark", "as_of", "last")}
        update["observations"] = state["observations"] + [observation]
        return update

    def answer(state):
        p = state.get("plan")
        if not p or p["clarify"] or p["intent"] == "outside" or not p["wants_context"]:
            return {}
        try:
            selected = set(state.get("decision", {}).get("source_ids", []))
            sources = (
                [s for s in state["sources"] if s["id"] in selected]
                if selected
                else state["sources"]
            )
            draft, evidence, feedback = generate_answer(
                client,
                p["rag_query"] if p["intent"] == "forecast" else state["question"],
                sources,
                feedback=state["feedback"],
            )
            previous = CitedAnswer.model_validate(state["draft"])
            if not draft.complete and len(draft.claims) < len(previous.claims):
                draft = previous
                evidence = {**evidence, **state["evidence"]}
            return {
                "draft": draft.model_dump(),
                "evidence": evidence,
                "feedback": feedback,
                "answer_attempts": state["answer_attempts"] + 1,
            }
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def finish(state):
        p, calc = state.get("plan"), state["calc"]
        sources, claims, blocks = [], [], []
        intent = p["intent"] if p else "unknown"
        stop = state.get("stop_reason", "completed")
        if state.get("failure") and not state["sources"] and not calc:
            status, blocks = "unavailable", [state["failure"]]
        elif p and p["intent"] == "outside":
            status, blocks = (
                "outside",
                [
                    "Этот запрос вне нефтегазовой специализации. Могу помочь с отраслевыми отчётами, ценами нефти и расчётом прогноза."
                ],
            )
        elif p and p["clarify"]:
            status, blocks = "clarification", [p["clarify"]]
        else:
            draft = CitedAnswer.model_validate(state["draft"])
            blocks, claims, sources = render_claims(draft, state["evidence"], state["sources"])
            if calc:
                blocks.insert(0, forecast_answer(calc))
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
            web = [s for s in state["sources"] if s["kind"] == "web"]
            for source in web:
                if "price" in source and source["id"] not in {s["id"] for s in sources}:
                    blocks.append(source["text"] + f" [{source['id']}]")
                    sources.append(source)
            complete = bool(p) and (not p["wants_context"] or draft.complete)
            complete = complete and (intent != "forecast" or bool(calc))
            if p and p["quotes"]:
                complete = complete and set(p["quotes"]) <= {
                    s.get("grade") for s in web if "price" in s
                }
            if p and p["fresh"]:
                complete = complete and bool(web)
            if state["attempts"] >= settings().agent_max_steps and not complete:
                stop = "budget_exhausted"
            if state.get("failure"):
                complete = False
            if not blocks:
                blocks = ["Подтверждённых данных для ответа недостаточно."]
            if not complete:
                blocks.append(
                    "Запрос подтверждён не полностью. Доступных источников или результата проверки недостаточно для полного ответа."
                )
            if p and p["report_only"] and p["fresh"]:
                blocks.append(
                    "Использованы только загруженные отчёты; актуальность на сегодняшний день не проверялась."
                )
            if not complete:
                for observation in state["observations"]:
                    blocks.extend(observation.get("notices", []))
            status = "ok" if complete else "partial"
        return {
            "response": {
                "question": state["question"],
                "answer": "\n\n".join(dict.fromkeys(blocks)),
                "status": status,
                "intent": intent,
                "clarification": status == "clarification",
                "mode": getattr(client, "model", settings().llm_model),
                "sources": sources,
                "claims": claims,
                "trace": state["trace"],
                "forecast": calc,
                "decisions": state["decisions"],
                "stop_reason": stop,
                "agent_mode": "loop",
                "rejected_actions": sum(bool(o.get("error")) for o in state["observations"]),
            }
        }

    def after_plan(state):
        return (
            "finish"
            if state.get("failure")
            or state["plan"]["intent"] == "outside"
            or state["plan"]["clarify"]
            else "decide"
        )

    def after_decide(state):
        if state.get("failure"):
            return "finish"
        return (
            "answer"
            if state["decision"]["action"] == "finish" and "finish" in allowed_actions(state)
            else "act"
        )

    def after_act(state):
        return "answer" if state["attempts"] >= settings().agent_max_steps else "decide"

    def after_answer(state):
        if (
            not state.get("failure")
            and state["plan"]["wants_context"]
            and not state["draft"]["complete"]
            and state["answer_attempts"] < 2
            and state["attempts"] < settings().agent_max_steps
        ):
            return "decide"
        return "finish"

    def observed(name, fn):
        def call(state):
            with span("agent." + name) as obs:
                result = fn(state)
                obs.update(output=result, status="error" if result.get("failure") else "ok")
                return result

        return call

    graph = StateGraph(State)
    for name, fn in [
        ("plan", plan),
        ("decide", decide),
        ("act", act),
        ("answer", answer),
        ("finish", finish),
    ]:
        graph.add_node(name, observed(name, fn))
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, {x: x for x in ("finish", "decide")})
    graph.add_conditional_edges("decide", after_decide, {x: x for x in ("finish", "answer", "act")})
    graph.add_conditional_edges("act", after_act, {x: x for x in ("answer", "decide")})
    graph.add_conditional_edges("answer", after_answer, {x: x for x in ("finish", "decide")})
    graph.add_edge("finish", END)
    return graph.compile()


def ask(
    question, previous=None, *, client=None, registry=None, execute=run_tool, planner=make_plan
):
    if settings().agent_mode == "workflow":
        from .workflow import ask as baseline

        return baseline(
            question, previous, client=client, registry=registry, execute=execute, planner=planner
        )
    question = question.strip()
    if not 1 <= len(question) <= 2000:
        raise ValueError("Введите вопрос длиной от 1 до 2000 символов.")
    with request_trace(question, previous) as (trace_id, obs):
        state = build_graph(client, registry, execute, planner).invoke(
            {
                "question": question,
                "previous": previous,
                "sources": [],
                "observations": [],
                "decisions": [],
                "trace": [],
                "calc": None,
                "signatures": [],
                "attempts": 0,
                "answer_attempts": 0,
                "feedback": "",
                "draft": {"claims": [], "complete": False},
                "evidence": {},
            },
            config={
                "callbacks": callbacks(),
                "metadata": {"trace_id": trace_id},
                "recursion_limit": 2 * settings().agent_max_steps + 12,
            },
        )
        response = AgentResponse.model_validate(
            {**state["response"], "trace_id": trace_id}
        ).model_dump()
        obs.update(
            output=response,
            metadata={
                "intent": response["intent"],
                "status": response["status"],
                "agent_mode": "loop",
            },
            status="error" if response["status"] == "unavailable" else "ok",
        )
        return response
