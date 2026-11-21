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
from .event_search import EventSearchAction, event_publication_search, event_search_context
from .evidence import generate_answer, render_claims
from .forecast_memory import pending_forecast
from .integrations.agent_skills import SkillUnavailable, skill_context
from .integrations.llm import LLMUnavailable, llm
from .integrations.telemetry import callbacks, request_trace, span
from .integrations.web import DAILY, PAGE_DATE_NOTICE, SEARCH_DATE_NOTICE, WEB_QUERY_LIMIT
from .logging import log
from .models.schemas import AgentResponse, CitedAnswer, NextAction, WebTimeFilterChoice
from .planning import apply_report_scope, make_plan
from .progress import action_signature, check_assessment, context_blocks, evidence_key, source_key
from .request_context import plan_scope, scope_payload
from .request_progress import announce_quote_fallback
from .tools import make_registry, run_tool
from .workflow import capabilities_answer, forecast_answer


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
    event_analysis_cache: dict
    direct_quotes_attempted: bool
    quote_fallback_source_ids: list[str]
    draft: dict
    evidence: dict
    failure: str
    stop_reason: str
    research_stopped: bool
    response: dict


@lru_cache(maxsize=32)
def action_schema(allowed, *, event_followup=False):
    return create_model(
        "AvailableAction",
        __base__=EventSearchAction if event_followup else NextAction,
        action=(Literal[allowed], ...),
    )


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


def decision_sources(sources, *, priority_source_ids=(), excluded_ids=()):
    """Share bounded context across sources instead of filling it with one PDF."""
    result = {}
    if excluded_ids:
        excluded_ids = set(excluded_ids)
        sources = [source for source in sources if source["id"] not in excluded_ids]
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
        "semantic_review_enabled": settings().answer_review,
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
    if (
        state.get("direct_quotes_attempted")
        and p["wants_context"]
        and not quote_fallback_eligible(state)
    ):
        evidence_attempted = any(
            o["tool"] == "report_search"
            or (o["tool"] == "web_search" and o["arguments"].get("search"))
            for o in state["observations"]
        )
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
    # A hand-picked subset can hide a later occurrence discovered in this turn.
    # The chronology must compare the same cumulative pool after a finish action.
    selected = (
        set(decision.get("source_ids", []))
        if decision.get("action") == "finish"
        and not state.get("plan", {}).get("latest_completed_event")
        else set()
    )
    sources = state["sources"]
    if state.get("direct_quotes_attempted"):
        # A quote alone cannot answer an additional research request. When the
        # planner could not separate the clauses, evaluate all evidence together
        # once some actual research material is available.
        context_sources = [s for s in sources if "price" not in s]
        if state["plan"].get("context_request") or not context_sources:
            sources = context_sources
    return [s for s in sources if s["id"] in selected] if selected else sources


def needs_answer(state):
    if not (state.get("plan", {}).get("wants_context") or "quote_fallback_source_ids" in state):
        return False
    ids = [s["id"] for s in answer_sources(state)]
    return bool(ids) and ids != state.get("answer_context_ids", [])


def required_results_ready(state):
    """Require a calculation and a direct quote or reviewed spot evidence when requested."""
    p = state.get("plan")
    if not p or (p["intent"] == "forecast" and not state["calc"]):
        return False
    web = [s for s in state["sources"] if s["kind"] == "web"]
    if set(p["quotes"]) <= {s.get("grade") for s in web if "price" in s}:
        return True
    # A publication can substantiate a dated spot observation through the same
    # citation review as other prose. Keep its real source in the answer.
    fallback_sources = state.get("quote_fallback_source_ids")
    if not fallback_sources or not state["draft"]["complete"]:
        return False
    cited_sources = {
        state["evidence"][eid]["source"]
        for claim in state["draft"]["claims"]
        for eid in claim["evidence_ids"]
    }
    return bool(cited_sources.intersection(fallback_sources))


def quote_fallback_eligible(state):
    plan = state["plan"]
    return (
        state.get("direct_quotes_attempted", False)
        and not plan.get("context_request")
        and not plan["news"]
        and not required_results_ready(state)
    )


def can_finish_unreviewed(state):
    """Do not research repeatedly solely because the review was switched off."""
    checks = state.get("answer_checks", [])
    validation = checks[-1].get("validation", {}) if checks else {}
    if state.get("plan", {}).get("latest_completed_event"):
        chronology = validation.get("event_chronology") or {}
        if (
            not chronology.get("latest_held_date")
            or validation.get("chronology_matches_draft") is not True
            or chronology.get("overdue_scheduled_dates")
            or chronology.get("unresolved_evidence_ids")
            or chronology.get("missing_requested_context")
        ):
            return False
    return (
        not settings().answer_review
        and bool(state["draft"]["claims"])
        and validation.get("candidate_complete") is True
        and validation.get("candidate_claims") == validation.get("accepted_claims")
        and (required_results_ready(state) or state.get("direct_quotes_attempted"))
    )


def task_complete(state):
    p = state.get("plan")
    if not p or (p["wants_context"] and not state["draft"]["complete"]):
        return False
    if not required_results_ready(state):
        return False
    if p["fresh"] and p["report_only"]:
        # The user permits only a document snapshot, not verification as of now.
        return False
    # A successful search is not proof of a current, relevant answer. For prose
    # answers generate_answer has already checked the actual cited evidence.
    if p["wants_context"] and state.get("feedback", "").strip():
        return False
    return True


def build_graph(client=None, registry=None, execute=run_tool, planner=make_plan):
    client, registry = BudgetedClient(client or llm()), registry or make_registry()

    def plan(state):
        try:
            value = planner(client, state["question"], state.get("previous"))
            value = apply_report_scope(value, state["question"])
            pending = pending_forecast()
            return {
                "plan": value.model_dump(),
                "previous": None if pending and pending.new_task else state.get("previous"),
            }
        except LLMUnavailable as exc:
            return {"failure": str(exc), "stop_reason": "llm_unavailable"}

    def decide(state):
        try:
            actions = allowed_actions(state)
            p = state["plan"]
            last_validation = (
                state["answer_checks"][-1].get("validation", {}) if state["answer_checks"] else {}
            )
            event_context = event_search_context(state) if "web_search" in actions else None
            if (
                state["attempts"] == 0
                and p["intent"] == "research"
                and p["quotes"]
                and set(p["quotes"]) <= DAILY.keys()
                and "web_search" in actions
            ):
                decision = NextAction(
                    action="web_search",
                    purpose="Получить последнее опубликованное наблюдение EIA/FRED для "
                    + ", ".join(p["quotes"])
                    + ".",
                ).model_dump()
                decision.update(origin="plan_policy", quote_lookup=True)
            elif state["attempts"] == 0 and actions == ["report_search"]:
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
                        **({"event_search": event_context} if event_context is not None else {}),
                        "steps_remaining": settings().agent_max_steps - state["attempts"],
                        "observations": state["observations"],
                        "sources": decision_sources(
                            state["sources"],
                            priority_source_ids=recent_source_ids(state),
                            excluded_ids=last_validation.get("event_relevance", {}).get(
                                "excluded_source_ids", []
                            ),
                        ),
                        "calculation_available": bool(state["calc"]),
                        "answer_feedback": state["feedback"],
                        "answer_validation": last_validation,
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
                    action_schema(tuple(actions), event_followup=event_context is not None),
                ).model_dump()
                decision["origin"] = "llm"
            decision.update(check_assessment(decision, state["sources"]))
            if (
                decision["action"] == "web_search"
                and event_context is not None
                and not decision.get("quote_lookup")
            ):
                decision = event_publication_search(decision, event_context, state)
            if "web_search" in actions and not (event_context and event_context["followup"]):
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
        argument_repair = d.get("argument_repair")
        if name not in allowed_actions(state):
            error = "Action rejected by policy; choose one of allowed_actions."
            args = {}
        elif name == "report_search":
            args = {"query": d["query"]}
            if not d["query"].strip():
                # The model already supplied a query for the initial report search.
                # After text retrieval it must specify which remaining fact to look for.
                planned = p["rag_query"].strip()
                if planned and all("price" in source for source in state["sources"]):
                    args = {"query": planned}
                    argument_repair = {
                        "mode": "plan_rag_query",
                        "before": {"query": d["query"]},
                        "after": args.copy(),
                    }
                else:
                    error = "report_search requires a non-empty query. Provide the search text."
        elif name == "read_report":
            args = {"report": d["report"], "page": d["page"]}
            if d.get("block_id"):
                args["block_id"] = d["block_id"]
            if d.get("version_id"):
                args["version_id"] = d["version_id"]
            if d["report"] not in {s.get("report") for s in state["sources"]}:
                error = "Use a registered report ID from retrieved sources."
            elif (d["page"] is not None and d["page"] < 1) or (
                d["page"] is None and not d.get("block_id", "").strip()
            ):
                error = "read_report requires a positive page or a block_id from retrieved sources."
            elif d.get("version_id") and not any(
                s.get("report") == d["report"] and s.get("version_id") == d["version_id"]
                for s in state["sources"]
            ):
                error = "Use a version returned by report_search."
        elif name == "web_search":
            fallback = quote_fallback_eligible(state)
            args = {
                "query": d["query"],
                "fresh": p["fresh"],
                "quotes": [] if state.get("direct_quotes_attempted") else p["quotes"],
                "search": not d.get("quote_lookup")
                and (p["news"] or p["wants_context"] or not p["quotes"] or fallback),
            }
            if state.get("direct_quotes_attempted") and p.get("context_fresh") is not None:
                args["fresh"] = p["context_fresh"]
            if args["search"] and fallback:
                args.update(quote_fallback=True, fresh=True)
            if args["search"] and not d["query"].strip():
                planned = (p.get("context_request") or p["rag_query"]).strip()
                if (
                    p.get("latest_completed_event")
                    and planned
                    and not state["sources"]
                    and not argument_repair
                ):
                    args["query"] = " ".join(planned.split())[:WEB_QUERY_LIMIT]
                    argument_repair = {
                        "mode": "plan_web_query",
                        "before": {"query": d["query"]},
                        "after": {"query": args["query"]},
                    }
                else:
                    error = "web_search requires a non-empty query when searching publications."
            for key in (
                "topic",
                "days",
                "start_date",
                "end_date",
                "search_depth",
                "publisher_domain",
            ):
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
            if args.get("quote_fallback"):
                announce_quote_fallback()
            data, error, event = execute(registry, name, **args)
        observation = {"tool": name, "arguments": args, "error": error}
        if argument_repair:
            observation["argument_repair"] = argument_repair
        update = {
            "trace": state["trace"] + ([event] if event else []),
        }
        if d.get("quote_lookup"):
            update["direct_quotes_attempted"] = True
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
        if d.get("quote_lookup") and error and not observation.get("notices"):
            observation["notices"] = ["Не удалось получить котировку EIA/FRED. " + error]
        incoming = (
            data
            if name in ("report_search", "read_report")
            else (data.get("items", []) if name in ("web_search", "read_web") and data else [])
        )
        keys = {source_key(s) for s in (incoming or [])}
        found_ids = [s["id"] for s in update.get("sources", []) if source_key(s) in keys]
        if args.get("quote_fallback") and event is not None:
            update["quote_fallback_source_ids"] = found_ids
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
        if not p or p["clarify"] or p["intent"] == "outside" or not needs_answer(state):
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
            validation = {}
            question = state["question"]
            fresh = p["fresh"]
            if p["intent"] == "forecast":
                question = p["rag_query"]
            elif p.get("context_request"):
                # A dated quote has its own source and is rendered separately.
                # Review the remaining analytical question against its own period.
                question = p["context_request"]
                if p.get("context_fresh") is not None:
                    fresh = p["context_fresh"]
            chronology_options = {}
            if p.get("latest_completed_event"):
                event_cache = state.get("event_analysis_cache", {})
                relevance_cache = dict(event_cache.get("relevance", {}))
                chronology_cache = dict(event_cache.get("chronology", {}))
                last_check = state["answer_checks"][-1] if state["answer_checks"] else {}
                prior_dates = last_check.get("validation", {}).get("event_chronology", {})
                chronology_options = {
                    "latest_completed_event": True,
                    "relevance_cache": relevance_cache,
                    "chronology_cache": chronology_cache,
                    "retained_block_id": next(
                        iter(prior_dates.get("latest_evidence_ids", [])), None
                    ),
                    "allow_chronology_repair": not any(
                        check.get("validation", {}).get("chronology_repair_attempted")
                        for check in state["answer_checks"]
                    ),
                    "allow_format_repair": not any(
                        check.get("validation", {})
                        .get("event_chronology", {})
                        .get("extraction", {})
                        .get("repair_attempted")
                        for check in state["answer_checks"]
                    ),
                }
            draft, evidence, feedback = generate_answer(
                client,
                question,
                sources,
                feedback=state["feedback"],
                fresh=fresh or "quote_fallback_source_ids" in state,
                spot_grades=p["quotes"] if "quote_fallback_source_ids" in state else (),
                previous=state.get("previous") if p["intent"] == "research" else None,
                report_only=p["report_only"],
                priority_source_ids=recent_source_ids(state),
                diagnostics=validation,
                allow_citation_repair=not any(
                    previous.get("validation", {}).get("citation_repair_attempted")
                    for previous in state["answer_checks"]
                ),
                # A reply to an earlier clarification already carries the user's
                # choice. Missing evidence for that choice calls for research.
                allow_research_clarification=(
                    p["intent"] == "research"
                    and not p["quotes"]
                    and not p["news"]
                    and not state.get("previous")
                    and not any(check.get("claims") for check in state["answer_checks"])
                ),
                **chronology_options,
            )
            relevance = validation.get("event_relevance", {})
            excluded = set(relevance.get("excluded_source_ids", []))
            new_ids = (
                state["observations"][-1].get("new_source_ids", []) if state["observations"] else []
            )
            if new_ids and set(new_ids) <= excluded and not draft.complete and evidence:
                feedback += (
                    " Последняя выдача посвящена другим событиям. Проверьте первоисточник "
                    "запрошенной организации или смените поисковую формулировку."
                )
            # The latest validation replaces the candidate, including an empty
            # rejection. Restoring an old complete draft can resurrect stale facts.
            result = {
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
                    validation=validation,
                ),
            }
            if chronology_options:
                result["event_analysis_cache"] = {
                    "relevance": relevance_cache,
                    "chronology": chronology_cache,
                }
            extraction = validation.get("event_chronology", {}).get("extraction", {})
            # An incomplete comparison still has usable rows. Its evidence gaps
            # go back to the planner under the usual time and action limits.
            if extraction.get("status") == "invalid_format":
                result["stop_reason"] = "chronology_format_incomplete"
            elif validation.get("research_clarification"):
                result["stop_reason"] = "research_clarification"
            return result
        except (LLMUnavailable, RequestBudgetExceeded) as exc:
            return {
                "failure": str(exc),
                "stop_reason": "time_budget_exhausted"
                if isinstance(exc, RequestBudgetExceeded)
                else "llm_unavailable",
                "answer_checks": result_check(
                    status="error", error=str(exc), validation=validation
                ),
            }

    def finish(state):
        p, calc = state.get("plan"), state["calc"]
        sources, claims, blocks = [], [], []
        intent = p["intent"] if p else "unknown"
        stop = state.get("stop_reason", "completed")
        last_check = state["answer_checks"][-1] if state["answer_checks"] else {}
        validation = last_check.get("validation", {})
        citation_failure = (
            not state.get("failure")
            and validation.get("reason") == "local_validation_rejected"
            and bool(validation.get("candidate_claims"))
            and validation["candidate_claims"] == validation.get("citation_rejected")
        )
        if state.get("failure") and not state["sources"] and not calc:
            status, blocks = "unavailable", [state["failure"]]
        elif intent == "capabilities":
            status, blocks = "ok", [capabilities_answer(p["report_only"])]
        elif p and p["intent"] == "outside":
            status, blocks = (
                "outside",
                [
                    "Этот запрос вне нефтегазовой специализации. Могу помочь с отраслевыми отчётами, ценами нефти и расчётом прогноза."
                ],
            )
        elif p and p["clarify"]:
            status, blocks = "clarification", [p["clarify"]]
        elif stop == "research_clarification":
            status, blocks = "clarification", [validation["research_clarification"]["clarify"]]
        else:
            draft = CitedAnswer.model_validate(state["draft"])
            blocks, claims, sources = render_claims(draft, state["evidence"], state["sources"])
            chronology = validation.get("event_chronology") or {}
            if claims and chronology.get("latest_held_date"):
                blocks.insert(0, "По найденным источникам")
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
            complete = task_complete(state) and stop != "chronology_format_incomplete"
            unreviewed = (
                bool(
                    p
                    and (p["wants_context"] or "quote_fallback_source_ids" in state)
                    and draft.claims
                )
                and not settings().answer_review
            )
            if unreviewed:
                complete = False
                if stop == "completed" and can_finish_unreviewed(state):
                    stop = "review_disabled"
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
                if stop == "chronology_format_incomplete":
                    if chronology.get("scheduled_leads") and not chronology.get("latest_held_date"):
                        blocks.append(
                            "В найденных материалах есть анонс. Подтвердить дату последнего "
                            "состоявшегося события пока не удалось."
                        )
                    elif claims:
                        blocks.append(
                            "Приведены подтверждённые сведения. Сопоставить даты во всех "
                            "найденных материалах не удалось."
                        )
                    else:
                        blocks.append(
                            "Материалы найдены, но установить по ним дату последнего "
                            "состоявшегося события пока не удалось."
                        )
                elif chronology.get("overdue_scheduled_dates") and not complete:
                    dates = ", ".join(chronology["overdue_scheduled_dates"])
                    blocks.append(
                        f"В публикациях упомянуты более поздние даты в расписании ({dates}). "
                        "Пока не удалось подтвердить, состоялось ли запланированное событие."
                    )
                elif validation.get("reason") == "no_confirmed_event":
                    blocks.append(
                        "В найденных материалах не удалось подтвердить, когда состоялось событие."
                    )
                elif validation.get("reason") == "no_relevant_event_sources":
                    blocks.append(
                        "Найденные публикации относятся к другим событиям. Дату подтвердить не удалось."
                    )
                elif validation.get("reason") == "event_chronology_mismatch":
                    blocks.append(
                        "Не удалось связать выводы с датой последнего подтверждённого события. "
                        "В трейсе можно открыть фрагмент, по которому проверялась дата."
                    )
                elif claims and not complete and chronology.get("unresolved_evidence_ids"):
                    blocks.append(
                        "Для части найденных материалов не удалось установить дату или состав "
                        "участников. Более позднее событие остаётся возможным."
                    )
                elif citation_failure:
                    blocks.append(
                        "Источники найдены, но ссылки модели не совпали с фрагментами, "
                        "переданными для подготовки ответа. Утверждения с такими ссылками исключены."
                    )
                elif not blocks:
                    gaps = validation.get("missing_aspects", [])
                    if validation.get("reason") == "semantic_validation_rejected" and gaps:
                        blocks.append(
                            "Что показала проверка найденных материалов\n"
                            + "\n".join("• " + gap for gap in dict.fromkeys(gaps))
                        )
                    else:
                        blocks.append("Подтверждённых данных для ответа недостаточно.")
                elif not complete:
                    blocks.append(
                        "Проверка смысла и полноты ответа отключена в настройках. "
                        "Утверждения проверены только по ссылкам и числам; полнота ответа не подтверждена."
                        if unreviewed
                        else "Запрос подтверждён не полностью. Доступных источников или результата проверки недостаточно для полного ответа."
                    )
            if p and p["report_only"] and p["fresh"]:
                blocks.append(
                    "Использованы только загруженные отчёты; актуальность на сегодняшний день не проверялась."
                )
            if not complete:
                omit_date_notices = stop == "chronology_format_incomplete" or (
                    citation_failure
                    and not any(s["kind"] == "web" and not s.get("date") for s in sources)
                )
                for observation in state["observations"]:
                    if not observation.get("duplicate_of"):
                        blocks.extend(
                            notice
                            for notice in observation.get("notices", [])
                            if not (
                                omit_date_notices
                                and notice in (SEARCH_DATE_NOTICE, PAGE_DATE_NOTICE)
                            )
                        )
        if stop == "no_progress" and not citation_failure:
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
                    "semantic_review_enabled": settings().answer_review,
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
            or state["plan"]["intent"] in {"capabilities", "outside"}
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
        if (
            state["decision"].get("quote_lookup")
            and not state["plan"]["wants_context"]
            and not quote_fallback_eligible(state)
        ):
            return "finish"
        # Every new evidence set reaches the answer node immediately. Another
        # planning decision is not required to unlock already retrieved facts.
        if needs_answer(state):
            return "answer"
        if (
            "quote_fallback_source_ids" in state
            or task_complete(state)
            or can_finish_unreviewed(state)
            or state.get("research_stopped")
            or state.get("stop_reason") == "no_progress"
            or state["attempts"] >= settings().agent_max_steps
        ):
            return "finish"
        return "decide"

    def after_answer(state):
        if (
            not state.get("failure")
            and "quote_fallback_source_ids" not in state
            and state.get("stop_reason", "completed") == "completed"
            and not state.get("research_stopped")
            and not task_complete(state)
            and not can_finish_unreviewed(state)
            # A verified context answer can be returned alongside an explicit
            # missing quote. More research cannot retry the already queried feed.
            and not (
                state.get("direct_quotes_attempted")
                and state["draft"]["complete"]
                and not state["feedback"]
            )
            and state["attempts"] < settings().agent_max_steps
        ):
            return "decide"
        return "finish"

    def observed(name, fn):
        def call(state):
            with plan_scope(state.get("plan")), span("agent." + name) as obs:
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
                observed_result = {k: v for k, v in result.items() if k != "event_analysis_cache"}
                obs.update(
                    output=observed_result, status="error" if result.get("failure") else "ok"
                )
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
