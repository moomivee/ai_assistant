"""Bounded observe/act loop: the LLM chooses the next tool after every result."""

from contextlib import nullcontext
from datetime import date
from functools import lru_cache
from time import perf_counter
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import create_model

from .budget import (
    BudgetedClient,
    RequestBudgetExceeded,
    ResearchBudgetExceeded,
    check_budget,
    remaining_seconds,
    request_budget,
    research_phase,
)
from .config import settings
from .evidence import generate_answer, render_claims
from .integrations.llm import LLMUnavailable, llm
from .integrations.telemetry import callbacks, request_trace, span
from .models.schemas import AgentResponse, CitedAnswer, NextAction
from .planning import make_plan
from .progress import action_signature, check_assessment, evidence_key, source_key
from .tools import make_registry, run_tool
from .workflow import forecast_answer


class State(TypedDict, total=False):
    question: str
    previous: str | None
    plan: dict
    sources: list
    observations: list
    diagnostic_steps: list
    decisions: list
    trace: list
    calc: dict | None
    decision: dict
    action_results: dict
    no_progress_steps: int
    attempts: int
    feedback: str
    answer_attempts: int
    answer_context_ids: list
    answer_checks: list
    draft: dict
    evidence: dict
    failure: str
    stop_reason: str
    research_stopped: bool
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
        row = {
            k: source.get(k)
            for k in ("id", "report", "page", "title", "kind", "date", "url", "full_page")
        }
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
    # Repeated arguments do not disable an entire tool. A changed query must
    # remain available even when other tools are also allowed.
    return actions


def merge_sources(existing, incoming):
    result = [dict(s) for s in existing]
    known = {source_key(s): s for s in result}
    for source in incoming:
        key = source_key(source)
        if key not in known:
            prefix = "R" if source["kind"] == "report" else "W"
            number = 1 + sum(s["kind"] == source["kind"] for s in result)
            item = {**source, "id": f"{prefix}{number}"}
            result.append(item)
            known[key] = item
    return result


def answer_sources(state):
    # A search decision may contain spurious source_ids. Only an explicit
    # finish selection may narrow the sources sent to answer generation.
    decision = state.get("decision", {})
    selected = set(decision.get("source_ids", [])) if decision.get("action") == "finish" else set()
    return [s for s in state["sources"] if s["id"] in selected] if selected else state["sources"]


def needs_answer(state):
    if not state.get("plan", {}).get("wants_context"):
        return False
    ids = [s["id"] for s in answer_sources(state)]
    return bool(ids) and ids != state.get("answer_context_ids", [])


def task_complete(state):
    p = state.get("plan")
    if not p or (p["wants_context"] and not state["draft"]["complete"]):
        return False
    if p["intent"] == "forecast" and not state["calc"]:
        return False
    web = [s for s in state["sources"] if s["kind"] == "web"]
    return (not p["fresh"] or bool(web)) and set(p["quotes"]) <= {
        s.get("grade") for s in web if "price" in s
    }


def build_graph(client=None, registry=None, execute=run_tool, planner=make_plan):
    client, registry = BudgetedClient(client or llm()), registry or make_registry()

    def plan(state):
        try:
            return {"plan": planner(client, state["question"], state.get("previous")).model_dump()}
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def decide(state):
        try:
            actions = allowed_actions(state)
            if state["attempts"] == 0 and actions == ["report_search"]:
                # The model already supplied a search query in Plan, and the
                # report-first policy leaves no action to choose here.
                decision = NextAction(
                    action="report_search",
                    purpose="Первый поиск по запросу из проверенного плана; приоритет отчётов.",
                    query=state["plan"]["rag_query"] or state["question"],
                    report="",
                    page=None,
                ).model_dump()
                decision["origin"] = "plan_policy"
            else:
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
                        "current_answer": state["draft"],
                        "no_progress_steps": state["no_progress_steps"],
                        "no_progress_limit": settings().agent_no_progress_limit,
                        "seconds_remaining": remaining_seconds(),
                        "previous_assessment": {
                            k: state.get("decision", {}).get(k)
                            for k in (
                                "assessment",
                                "missing_facts",
                                "evidence_checks",
                                "unknown_source_ids",
                            )
                        },
                    },
                    action_schema(tuple(actions)),
                ).model_dump()
                decision["origin"] = "llm"
            decision.update(check_assessment(decision, state["sources"]))
            if (
                decision["assessment"] == "enough"
                and not decision["missing_facts"]
                and decision["assessment_references_valid"]
                and "finish" in allowed_actions(state)
                and decision["action"] != "finish"
            ):
                # Reconcile the model's own explicit, referenced assessment.
                # Record both actions; answer generation still validates claims.
                decision["model_action"] = decision["action"]
                decision["action"] = "finish"
                decision["source_ids"] = list(
                    dict.fromkeys(
                        key for check in decision["evidence_checks"] for key in check["source_ids"]
                    )
                )
                decision["controller_note"] = (
                    "Referenced assessment says enough; proceeding to answer."
                )
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
        signature = action_signature(name, args)
        previous = state["action_results"].get(signature) if not error else None
        if previous:
            error = "Duplicate action rejected. Reuse the existing result, change the arguments, or finish."
        if not error:
            data, error, event = execute(registry, name, **args)
        observation = {"tool": name, "arguments": args, "error": error}
        update = {
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
        incoming = (
            data
            if name in ("report_search", "read_report")
            else (data.get("items", []) if name == "web_search" and data else [])
        )
        keys = {source_key(s) for s in (incoming or [])}
        found_ids = [s["id"] for s in update.get("sources", []) if source_key(s) in keys]
        known_evidence = {evidence_key(s) for s in state["sources"]}
        new_ids = [
            s["id"]
            for s in update.get("sources", [])
            if evidence_key(s) not in known_evidence and s["text"].strip()
        ]
        progress = not error and bool(new_ids or (update.get("calc") and not state["calc"]))
        # First executed attempt establishes a baseline, even if empty. Leave
        # room for reformulation after an initial empty result and a duplicate.
        initial_attempt = not state["trace"] and event is not None
        stagnant = 0 if progress or initial_attempt else state["no_progress_steps"] + 1
        observation.update(
            source_ids=found_ids,
            new_source_ids=new_ids,
            new_evidence=progress,
            no_progress_steps=stagnant,
        )
        if previous:
            reused = previous["source_ids"]
            observation.update(duplicate_of=previous["step"], reused_source_ids=reused)
            observation["notices"] = [
                f"Attempt {previous['step']} already returned source IDs {', '.join(reused) or 'none'}. "
                + (f"That attempt failed: {previous['error']}. " if previous["error"] else "")
                + "Its result remains available; this duplicate added no evidence."
            ]
        elif event is not None:
            update["action_results"] = {
                **state["action_results"],
                signature: {"step": state["attempts"], "source_ids": found_ids, "error": error},
            }
        update.update(
            observations=state["observations"] + [observation], no_progress_steps=stagnant
        )
        if stagnant >= settings().agent_no_progress_limit:
            update["stop_reason"] = "no_progress"
            update["feedback"] = (
                state["feedback"]
                + "\nResearch stopped after repeated attempts without new evidence. "
                "Use only available sources; do not invent missing facts."
            ).strip()
        # Expose the same progress information to the inspector and the model.
        update["diagnostic_steps"] = state.get("diagnostic_steps", []) + [
            {
                **observation,
                "step": state["attempts"],
                "purpose": d.get("purpose", ""),
                "status": (event["status"] if event else "rejected"),
                "ms": event["ms"] if event else None,
            }
        ]
        return update

    def answer(state):
        p = state.get("plan")
        if not p or p["clarify"] or p["intent"] == "outside" or not p["wants_context"]:
            return {}
        sources = answer_sources(state)
        ids = [s["id"] for s in sources]
        start = perf_counter()
        check = {"after_step": state["attempts"], "source_ids": ids}

        def result_check(**values):
            return state["answer_checks"] + [
                {**check, **values, "ms": round((perf_counter() - start) * 1000, 2)}
            ]

        try:
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
                "answer_context_ids": ids,
                "answer_checks": result_check(
                    status="ok",
                    complete=draft.complete,
                    claims=len(draft.claims),
                    feedback=feedback,
                ),
            }
        except (LLMUnavailable, RequestBudgetExceeded) as exc:
            return {
                "failure": str(exc),
                "stop_reason": "time_budget_exhausted"
                if isinstance(exc, RequestBudgetExceeded)
                else "llm_unavailable",
                "answer_checks": result_check(status="error", error=str(exc)),
            }

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
            complete = task_complete(state)
            if (
                state["attempts"] >= settings().agent_max_steps
                and not complete
                and stop == "completed"
            ):
                stop = "budget_exhausted"
            if state.get("failure"):
                complete = False
            if not blocks:
                blocks = [
                    "Не удалось подготовить ответ по найденным источникам."
                    if state.get("failure") and state["sources"]
                    else "Подтверждённых данных для ответа недостаточно."
                ]
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
                    if not observation.get("duplicate_of"):
                        blocks.extend(observation.get("notices", []))
            status = "ok" if complete else "partial"
        if stop == "no_progress":
            blocks.append("Поиск остановлен: повторные попытки не добавили новых данных.")
        elif stop == "time_budget_exhausted":
            blocks.append("Достигнут общий лимит времени запроса. Обработка остановлена.")
        elif stop == "research_budget_exhausted":
            blocks.append(
                "Поиск остановлен, чтобы сохранить время на ответ по найденным источникам."
            )
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
                "diagnostics": {
                    "plan": p,
                    "steps": state.get("diagnostic_steps", []),
                    "retrieved_sources": [
                        {
                            **s,
                            "text": s["text"][:4000],
                            "text_truncated": len(s["text"]) > 4000,
                            "used_in_answer": s["id"] in {x["id"] for x in sources},
                        }
                        for s in state["sources"]
                    ],
                    "answer_feedback": state["feedback"],
                    "failure": state.get("failure", ""),
                    "no_progress_steps": state["no_progress_steps"],
                    "no_progress_limit": settings().agent_no_progress_limit,
                    "timeout_seconds": settings().agent_timeout_seconds,
                    "answer_reserve_seconds": min(
                        settings().agent_answer_reserve_seconds,
                        settings().agent_timeout_seconds / 2,
                    ),
                    "answer_checks": state["answer_checks"],
                },
            }
        }

    def after_plan(state):
        return (
            "finish"
            if state.get("failure")
            or state.get("research_stopped")
            or state["plan"]["intent"] == "outside"
            or state["plan"]["clarify"]
            else "decide"
        )

    def after_decide(state):
        if state.get("failure"):
            return "finish"
        if state.get("research_stopped"):
            return "answer" if needs_answer(state) else "finish"
        return (
            ("answer" if needs_answer(state) else "finish")
            if state["decision"]["action"] == "finish" and "finish" in allowed_actions(state)
            else "act"
        )

    def after_act(state):
        if state.get("failure"):
            return "finish"
        # Every new evidence set reaches the answer node immediately. Another
        # planning decision is not required to unlock already retrieved facts.
        if needs_answer(state):
            return "answer"
        if (
            task_complete(state)
            or state.get("research_stopped")
            or state.get("stop_reason") == "no_progress"
            or state["attempts"] >= settings().agent_max_steps
        ):
            return "finish"
        return "decide"

    def after_answer(state):
        if (
            not state.get("failure")
            and state.get("stop_reason", "completed") == "completed"
            and not state.get("research_stopped")
            and not task_complete(state)
            and state["attempts"] < settings().agent_max_steps
        ):
            return "decide"
        return "finish"

    def observed(name, fn):
        def call(state):
            with span("agent." + name) as obs:
                try:
                    with research_phase() if name in {"plan", "decide", "act"} else nullcontext():
                        if name != "finish":
                            check_budget()
                        result = fn(state)
                except ResearchBudgetExceeded as exc:
                    result = {"research_stopped": True, "stop_reason": "research_budget_exhausted"}
                    if not state.get("plan"):
                        result["failure"] = str(exc)
                except RequestBudgetExceeded as exc:
                    result = {"failure": str(exc), "stop_reason": "time_budget_exhausted"}
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
    graph.add_conditional_edges("act", after_act, {x: x for x in ("answer", "decide", "finish")})
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
    with (
        request_budget(settings().agent_timeout_seconds, settings().agent_answer_reserve_seconds),
        request_trace(question, previous) as (trace_id, obs),
    ):
        state = build_graph(client, registry, execute, planner).invoke(
            {
                "question": question,
                "previous": previous,
                "sources": [],
                "observations": [],
                "diagnostic_steps": [],
                "decisions": [],
                "trace": [],
                "calc": None,
                "action_results": {},
                "no_progress_steps": 0,
                "attempts": 0,
                "answer_attempts": 0,
                "answer_context_ids": [],
                "answer_checks": [],
                "feedback": "",
                "draft": {"claims": [], "complete": False},
                "evidence": {},
            },
            config={
                "callbacks": callbacks(),
                "metadata": {"trace_id": trace_id},
                # Each action can now visit decide -> act -> answer.
                "recursion_limit": 3 * settings().agent_max_steps + 12,
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
