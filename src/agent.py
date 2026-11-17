"""Bounded observe/act loop: the LLM chooses the next tool after every result."""

from contextlib import nullcontext
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
from .clock import clock_payload, request_clock
from .config import settings
from .evidence import generate_answer, render_claims
from .integrations.agent_skills import SkillUnavailable, skill_context
from .integrations.llm import LLMUnavailable, llm
from .integrations.telemetry import callbacks, request_trace, span
from .logging import log
from .models.schemas import AgentResponse, CitedAnswer, NextAction, WebTimeFilterChoice
from .planning import make_plan
from .progress import action_signature, check_assessment, context_blocks, evidence_key, source_key
from .request_context import current_scope, scope_payload
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


def repair_web_time_filter(client, question, previous, decision):
    """Let the model choose one window; never silently prefer a date or force web."""
    if (
        decision["action"] != "web_search"
        or decision.get("days") is None
        or not (decision.get("start_date") or decision.get("end_date"))
    ):
        return decision
    before = {key: decision.get(key) for key in ("days", "start_date", "end_date")}
    choice = client.generate(
        "web_time_filter",
        {
            "question": question,
            "previous": previous,
            **clock_payload(),
            "search_query": decision["query"],
            "proposed": before,
        },
        WebTimeFilterChoice,
    )
    after = {
        "days": before["days"] if choice.mode == "relative" else None,
        "start_date": before["start_date"] if choice.mode == "dates" else None,
        "end_date": before["end_date"] if choice.mode == "dates" else None,
    }
    return {
        **decision,
        **after,
        "argument_repair": {"mode": choice.mode, "before": before, "after": after},
    }


def recent_source_ids(state):
    """Prioritize newly obtained evidence, without treating retrieval time as truth."""
    return list(
        dict.fromkeys(
            source_id
            for observation in reversed(state.get("observations", []))
            for source_id in observation.get("new_source_ids", [])
        )
    )


def decision_sources(sources, *, priority_source_ids=()):
    """Share bounded context across sources instead of filling it with one PDF."""
    result = {}
    for source, _, _, end, block in context_blocks(
        sources, 14000, priority_source_ids=priority_source_ids
    ):
        row = result.setdefault(
            source["id"],
            {
                k: source.get(k)
                for k in (
                    "id",
                    "report",
                    "page",
                    "title",
                    "kind",
                    "date",
                    "url",
                    "full_page",
                    "document_id",
                    "version_id",
                    "block_id",
                    "locator",
                    "method",
                    "content_type",
                    "content_scope",
                    "date_basis",
                    "retrieved_at",
                    "local_url",
                )
            },
        )
        row["text"] = row.get("text", "") + block
        row["context_truncated"] = end < len(source["text"])
    return list(result.values())


def research_status(state):
    """Expose completed work without making the model reconstruct it from source text."""
    return {
        "completed_actions": [
            {"tool": row["tool"], "arguments": row["arguments"]}
            for row in state["observations"]
            if not row.get("error")
        ],
        "report_dates": sorted(
            {s["date"] for s in state["sources"] if s["kind"] == "report" and s.get("date")}
        ),
        "web_source_count": sum(s["kind"] == "web" for s in state["sources"]),
        "accepted_claim_count": len(state["draft"]["claims"]),
    }


def allowed_actions(state):
    p = state["plan"]
    called = [event["tool"] for event in state["trace"]]
    actions = []
    if p["wants_context"]:
        actions.append("report_search")
        if any(s["kind"] == "report" for s in state["sources"]):
            actions.append("read_report")
    if (p["wants_context"] or p["quotes"]) and not p["report_only"]:
        actions.append("web_search")
        if any(s["kind"] == "web" and "price" not in s for s in state["sources"]):
            actions.append("read_web")
    if p["intent"] == "forecast" and not state["calc"]:
        actions.append("price_forecast")
    # A model may stop with a partial answer after an unsuccessful attempt.
    # Source order is its decision; completeness is checked against evidence.
    evidence_attempted = any(name in called for name in ("report_search", "web_search"))
    required = (p["intent"] != "forecast" or bool(state["calc"])) and (
        not (p["wants_context"] or p["quotes"]) or evidence_attempted
    )
    if required or not actions:
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
    if p["fresh"] and p["report_only"]:
        # The user permits only a document snapshot, not verification as of now.
        return False
    web = [s for s in state["sources"] if s["kind"] == "web"]
    # A successful search is not proof of a current, relevant answer. For prose
    # answers generate_answer has already checked the actual cited evidence.
    if p["wants_context"] and state.get("feedback", "").strip():
        return False
    return set(p["quotes"]) <= {s.get("grade") for s in web if "price" in s}


def build_graph(client=None, registry=None, execute=run_tool, planner=make_plan):
    client, registry = BudgetedClient(client or llm()), registry or make_registry()

    def plan(state):
        try:
            value = planner(client, state["question"], state.get("previous"))
            if current_scope().report_only:
                value = value.model_copy(update={"report_only": True})
            return {"plan": value.model_dump()}
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def decide(state):
        try:
            actions = allowed_actions(state)
            if state["attempts"] == 0 and actions == ["report_search"]:
                # Only an explicit reports-only scope leaves one evidence tool.
                # Reuse the model's planned query without a redundant decision.
                decision = NextAction(
                    action="report_search",
                    purpose="Поиск по запросу из плана в выбранной пользователем базе документов.",
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
                        "user_scope": scope_payload(),
                        **clock_payload(),
                        "allowed_actions": allowed_actions(state),
                        "research_status": research_status(state),
                        "steps_remaining": settings().agent_max_steps - state["attempts"],
                        "observations": state["observations"],
                        "sources": decision_sources(
                            state["sources"], priority_source_ids=recent_source_ids(state)
                        ),
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
            if "web_search" in actions:
                decision = repair_web_time_filter(
                    client, state["question"], state.get("previous"), decision
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
            if d.get("block_id"):
                args["block_id"] = d["block_id"]
            if d.get("version_id"):
                args["version_id"] = d["version_id"]
            if d["report"] not in {s.get("report") for s in state["sources"]}:
                error = "Use a registered report ID from retrieved sources."
            elif d.get("version_id") and not any(
                s.get("report") == d["report"] and s.get("version_id") == d["version_id"]
                for s in state["sources"]
            ):
                error = "Use a version returned by report_search."
        elif name == "web_search":
            args = {
                "query": d["query"],
                "fresh": p["fresh"],
                "quotes": p["quotes"],
                "search": p["news"] or p["wants_context"] or not p["quotes"],
            }
            for key in ("topic", "days", "start_date", "end_date", "search_depth"):
                if d.get(key) not in (None, "", "general", "basic"):
                    args[key] = d[key]
        elif name == "read_web":
            args = {"url": d["url"], "query": d["query"]}
            if d["url"] not in {s["url"] for s in state["sources"] if s["kind"] == "web"}:
                error = "Read only a URL returned by web_search in this request."
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
        if d.get("argument_repair"):
            observation["argument_repair"] = d["argument_repair"]
        update = {
            "trace": state["trace"] + ([event] if event else []),
        }
        if name in ("report_search", "read_report") and data is not None:
            update["sources"] = merge_sources(state["sources"], data)
            observation["result_count"] = len(data)
        elif name in ("web_search", "read_web") and data:
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
            else (data.get("items", []) if name in ("web_search", "read_web") and data else [])
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
                fresh=p["fresh"],
                report_only=p["report_only"],
                priority_source_ids=recent_source_ids(state),
            )
            # The latest validation replaces the candidate, including an empty
            # rejection. Restoring an old complete draft can resurrect stale facts.
            return {
                "draft": draft.model_dump(),
                "evidence": evidence,
                "feedback": feedback,
                "answer_attempts": state["answer_attempts"] + 1,
                "answer_context_ids": ids,
                "answer_checks": result_check(
                    status="ok",
                    supplied_source_ids=ids,
                    source_ids=list(dict.fromkeys(row["source"] for row in evidence.values())),
                    evidence_ids=list(evidence),
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
                status = "partial" if blocks else "unavailable"
                blocks.append("Не удалось завершить обработку запроса. " + state["failure"])
            else:
                status = "ok" if complete else "partial"
                if not blocks:
                    blocks = ["Подтверждённых данных для ответа недостаточно."]
            if not complete and not state.get("failure"):
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
                "mode": getattr(client, "model", settings().active_model),
                "sources": sources,
                "claims": claims,
                "trace": state["trace"],
                "forecast": calc,
                "decisions": state["decisions"],
                "stop_reason": stop,
                "agent_mode": "loop",
                "rejected_actions": sum(bool(o.get("error")) for o in state["observations"]),
                "diagnostics": {
                    "time_context": clock_payload(),
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
                    selected = state.get("plan", {}).get("skill")
                    try:
                        skill_cm = skill_context(selected)
                        skill_cm.__enter__()
                    except SkillUnavailable:
                        log("skill.unavailable", skill=selected)
                        skill_cm = nullcontext()
                        skill_cm.__enter__()
                    try:
                        with (
                            research_phase() if name in {"plan", "decide", "act"} else nullcontext()
                        ):
                            if name != "finish":
                                check_budget()
                            result = fn(state)
                    finally:
                        skill_cm.__exit__(None, None, None)
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
        request_clock(),
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
            status="error" if response["diagnostics"]["failure"] else "ok",
        )
        return response
