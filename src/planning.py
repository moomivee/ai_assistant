"""Bounded structured checks of forecast inputs and combined research requests."""

from decimal import Decimal

from .clock import clock_payload
from .forecast_memory import ForecastFollowup, PendingForecast, merge_followup, pending_forecast
from .integrations.agent_skills import get_skill_catalog
from .integrations.llm import LLMUnavailable
from .integrations.web import DAILY
from .logging import log
from .models.schemas import (
    ForecastInputs,
    ForecastScope,
    LatestEventScope,
    Plan,
    ResearchQuoteScope,
)
from .request_context import current_scope, scope_payload
from .validation import (
    BENCHMARK_NAMES,
    discount_quote,
    named_grades,
    normalized,
    numbers,
    unsupported_discount,
    unsupported_discount_base,
    validate_plan,
)

FORECAST_FIELDS = ("benchmark", "months", "discount", "cut", "sensitivity")
HELP_COMMANDS = {"/help", "что ты можешь", "что ты умеешь", "чем ты можешь помочь"}


def resolve_forecast_scope(client, plan: Plan, payload: dict) -> Plan:
    """Verify proposed research separately from the numerical forecast task."""
    scope = client.generate("forecast_scope", payload, ForecastScope)
    inputs = [
        normalized(x).strip(" .!?;:") for x in (payload["question"], payload["previous"] or "")
    ]

    def subrequest(value):
        quote = normalized(value).strip(" .!?;:")
        # Evidence must be an actual additional clause, not the entire forecast
        # request copied into the research field. A follow-up may consist solely
        # of an additional question. Semantics use the schema, not keyword rules.
        valid = bool(quote) and any(
            quote in source and (quote != source or (i == 0 and bool(payload["previous"])))
            for i, source in enumerate(inputs)
        )
        if value and not valid:
            log("plan.scope_evidence_rejected")
        return value.strip() if valid else ""

    context = subrequest(scope.context_request)
    quote_request = subrequest(scope.quote_request)
    source_tokens = [
        word.strip(".,:;!?()[]«»\"'") for word in " ".join(inputs).replace("/", " ").split()
    ]
    quotes = [
        grade
        for grade in scope.quotes
        if quote_request and any(name in source_tokens for name in BENCHMARK_NAMES[grade])
    ]
    resolved = plan.model_copy(
        update={
            "wants_context": bool(context),
            "news": bool(context and scope.news),
            "fresh": bool((context and (scope.fresh or scope.news)) or quotes),
            "quotes": list(dict.fromkeys(quotes)),
            "rag_query": context or quote_request or plan.rag_query,
        }
    )
    log(
        "plan.forecast_scope",
        wants_context=resolved.wants_context,
        fresh=resolved.fresh,
        news=resolved.news,
        quotes=resolved.quotes,
    )
    return resolved


def discount_target_needs_review(plan: Plan, source: str) -> bool:
    if unsupported_discount(plan):
        return True
    if plan.benchmark != "Urals" or plan.discount is None:
        return False
    terms = normalized(discount_quote(plan))
    target_text = source.replace(terms, " ") if terms else source
    return plan.benchmark not in named_grades(target_text)


def discount_target_supported(
    plan: Plan, extracted: ForecastInputs, source: str, target_quote: str
) -> bool:
    """Check that the proposed target is identifiable outside the quoted discount terms."""
    if named_grades(target_quote) != {extracted.benchmark}:
        return False
    original_terms = normalized(discount_quote(plan))
    if named_grades(original_terms) == {extracted.benchmark} and not any(
        target_quote in part for part in source.split(original_terms)
    ):
        return False
    for terms in (original_terms, normalized(discount_quote(extracted))):
        if not terms or terms not in source or Decimal(str(plan.discount)) not in numbers(terms):
            continue
        # Exclude the reference in e.g. "12 dollars below Brent". A name that appears
        # only there cannot identify the requested price. Different numeric inputs
        # from the second reading never override the validated assumptions.
        remaining = source.split(terms)
        if not any(target_quote in part for part in remaining):
            continue
        if named_grades(" ".join(remaining)) == {extracted.benchmark}:
            return True
        # Additional clauses may mention other grades. In that case accept a specific
        # target clause when the earlier candidate is confirmed or is the quoted base.
        if len(target_quote.split()) > 1 and (
            plan.benchmark == extracted.benchmark or plan.benchmark in named_grades(terms)
        ):
            return True
    return False


def merge_forecast_inputs(
    plan: Plan, extracted: ForecastInputs, question, previous=None, *, review_target=False
):
    """Fill missing inputs and reconsider a target incompatible with a grounded discount."""
    candidate = plan.model_copy(deep=True)
    source = normalized((previous or "") + " " + question)
    quote = normalized(extracted.benchmark_quote)
    if review_target:
        candidate.benchmark = None
    if candidate.benchmark is None and extracted.benchmark is not None:
        if (
            quote
            and quote in source
            and extracted.benchmark in named_grades(quote)
            and (not review_target or discount_target_supported(plan, extracted, source, quote))
        ):
            candidate.benchmark = extracted.benchmark
    evidence = {e.field: e for e in candidate.parameters}
    new_evidence = {e.field: e for e in extracted.parameters}
    if review_target and extracted.discount == plan.discount and "discount" in new_evidence:
        # Keep a fuller verbatim citation (including the reference grade) for the
        # same validated amount, without replacing it with a different assumption.
        terms = normalized(new_evidence["discount"].quote)
        if normalized(discount_quote(plan)) in terms and terms in source:
            evidence["discount"] = new_evidence["discount"]
    for field in FORECAST_FIELDS[1:]:
        value = getattr(extracted, field)
        if getattr(candidate, field) is None and value is not None and field in new_evidence:
            setattr(candidate, field, value)
            evidence[field] = new_evidence[field]
    candidate.parameters = list(evidence.values())
    result = validate_plan(candidate, question, previous)
    if review_target and result.benchmark is None:
        result.clarify = (
            "Какой сорт нужно прогнозировать? Сценарий с дисконтом к Brent доступен для Urals."
        )
    return result


def verified_quote_scope(
    plan: Plan, scope: Plan | ResearchQuoteScope, question: str
) -> Plan | None:
    """Check grounding and independence before discarding either part of a request."""

    def clause(text):
        return normalized(text).strip(" .,:;!?…")

    source = clause(question)
    quote = clause(scope.quote_request)
    context = clause(scope.context_request)
    if not quote or quote not in source:
        return None
    if not context:
        if quote != source or plan.news:
            return None
        return plan.model_copy(
            update={
                "wants_context": False,
                "quote_request": question,
                "context_request": "",
                "context_fresh": False,
                "latest_completed_event": False,
            }
        )

    # Both clauses must occur at distinct positions in this user's question.
    if not any(context in part for part in source.split(quote, 1)):
        return None
    if scope.context_fresh is None and not plan.news:
        return None
    return plan.model_copy(
        update={
            "wants_context": True,
            "quote_request": scope.quote_request.strip(),
            "context_request": scope.context_request.strip(),
            "rag_query": scope.context_request.strip(),
            "context_fresh": bool(scope.context_fresh or plan.news or plan.latest_completed_event),
        }
    )


def resolve_research_quotes(client, plan: Plan, payload: dict) -> Plan:
    question = payload["question"]
    if not set(plan.quotes) <= DAILY.keys():
        # A published assessment (e.g. Urals) requires source research. Keep all
        # requested grades in the same question so a Brent feed cannot satisfy it.
        log("plan.quote_research_required", grades=plan.quotes)
        return plan.model_copy(
            update={
                "wants_context": True,
                "quotes": [],
                "rag_query": question,
                "context_request": "",
                "quote_request": "",
                "context_fresh": None,
            }
        )
    resolved = verified_quote_scope(plan, plan, question)
    if resolved is None:
        # Older prompts and ambiguous extra text (e.g. units) need one independent reading.
        try:
            scope = client.generate(
                "research_quote_scope",
                {
                    "question": question,
                    "previous": payload["previous"],
                    "quotes": plan.quotes,
                    **clock_payload(),
                },
                ResearchQuoteScope,
            )
            resolved = verified_quote_scope(plan, scope, question)
        except LLMUnavailable:
            log("plan.quote_scope_unavailable")
    if resolved is None:
        # No grounded separation. Keep the original research question and quote available.
        log("plan.quote_scope_unresolved")
        return plan.model_copy(
            update={
                "wants_context": True,
                "context_request": "",
                "quote_request": "",
                "context_fresh": None,
                "rag_query": question,
            }
        )
    log(
        "plan.quote_scope",
        wants_context=resolved.wants_context,
        context_fresh=resolved.context_fresh,
    )
    return resolved


def make_plan(client, question, previous=None, *, resolve_missing_event_scope=True) -> Plan:
    pending = pending_forecast()
    # Only an entire, unambiguous help question can bypass the semantic planner.
    if normalized(question).rstrip(" .?!…") in HELP_COMMANDS:
        if pending:
            pending.new_task = True
        log(
            "plan.validated",
            intent="capabilities",
            origin="help_command",
            needs_clarification=False,
        )
        return Plan(
            intent="capabilities",
            rag_query="",
            report_only=current_scope().report_only,
            fresh=False,
            news=False,
            wants_context=False,
            quotes=[],
            benchmark=None,
            months=None,
            discount=None,
            cut=None,
            sensitivity=None,
            parameters=[],
            clarify="",
        )
    payload = {
        "question": question,
        "previous": previous,
        **clock_payload(),
        "user_scope": scope_payload(),
        "available_skills": [
            s for s in get_skill_catalog() if s.get("enabled") and s.get("available", True)
        ],
    }
    withdrawn = set()
    if pending:
        snapshot = pending.snapshot
        payload["pending_forecast"] = {
            "confirmed": snapshot["confirmed"],
            "last_clarification": snapshot["clarification"],
        }
        followup = client.generate("plan", payload, ForecastFollowup)
        pending.new_task = getattr(followup, "new_task", False) or followup.intent != "forecast"
        pending.benchmark_change_quote = getattr(followup, "benchmark_change_quote", "")
        withdrawn = {
            p.field
            for p in getattr(followup, "withdrawn", [])
            if normalized(p.quote) and normalized(p.quote) in normalized(question)
        }
        raw = Plan.model_validate(
            followup.model_dump(include=set(Plan.model_fields), exclude_unset=True)
        )
        if pending.new_task:
            previous = payload["previous"] = None
            payload.pop("pending_forecast")
            if raw.intent == "forecast":
                # Confirm every input against the new task, including the grade and
                # horizon, which the ordinary single-turn validator allows to be implicit.
                if raw.benchmark not in named_grades(question):
                    raw.benchmark = None
                raw = merge_followup(
                    raw, question, PendingForecast({"confirmed": {}, "evidence": {}}), set()
                )
        raw = merge_followup(raw, question, pending, withdrawn)
    else:
        raw = client.generate("plan", payload, Plan)
    plan = validate_plan(raw, question, previous)
    if current_scope().report_only:
        plan = plan.model_copy(update={"report_only": True})
    research_quotes = plan.intent == "research" and bool(plan.quotes) and not plan.clarify
    if research_quotes:
        plan = resolve_research_quotes(client, plan, payload)
    # A legacy plan may omit temporal scope and even misclassify freshness.
    # Only unclassified research needs a second reading; quotes have their own check.
    if (
        resolve_missing_event_scope
        and plan.intent == "research"
        and plan.wants_context
        and plan.context_fresh is not False
        and not plan.clarify
        and "latest_completed_event" not in raw.model_fields_set
    ):
        scope = client.generate(
            "latest_event_scope",
            {"question": plan.context_request or question, "previous": previous, **clock_payload()},
            LatestEventScope,
        )
        plan.latest_completed_event = scope.latest_completed_event
        if scope.latest_completed_event:
            plan.fresh = True
    log(
        "plan.validated",
        intent=plan.intent,
        latest_completed_event=plan.latest_completed_event,
        benchmark=plan.benchmark,
        months=plan.months,
        needs_clarification=bool(plan.clarify),
    )
    if research_quotes:
        return plan
    if plan.context_request or plan.quote_request or plan.context_fresh is not None:
        plan = plan.model_copy(
            update={"quote_request": "", "context_request": "", "context_fresh": None}
        )
    if plan.intent != "forecast":
        return plan
    source = normalized((previous or "") + " " + question)
    review_target = discount_target_needs_review(plan, source)
    if unsupported_discount_base(plan) and not review_target:
        return plan

    if review_target or plan.clarify or raw.clarify:
        # At most one input repair. A genuinely absent parameter stays absent.
        log(
            "plan.repair_requested",
            missing=[f for f in FORECAST_FIELDS if getattr(plan, f) is None],
            review_discount_target=review_target,
        )
        extracted = client.generate(
            "forecast", {**payload, "review_discount_target": review_target}, ForecastInputs
        )
        if (
            raw.clarify
            and not plan.clarify
            and not review_target
            and (not pending or pending.new_task)
        ):
            # Without a pending clarification, recheck a possible ambiguity before
            # dismissing the model's question. A continuation already reconciled
            # the latest grounded corrections with previously supplied inputs.
            plan = plan.model_copy(update={**{f: None for f in FORECAST_FIELDS}, "parameters": []})
        plan = merge_forecast_inputs(
            plan, extracted, question, previous, review_target=review_target
        )
        if pending and not pending.new_task:
            # A second reading cannot revive a superseded value from an old turn.
            plan = validate_plan(
                merge_followup(plan, question, pending, withdrawn), question, previous
            )
        log(
            "plan.repaired",
            benchmark=plan.benchmark,
            months=plan.months,
            needs_clarification=bool(plan.clarify),
        )
    if not plan.clarify and (plan.wants_context or plan.fresh or plan.news or plan.quotes):
        plan = resolve_forecast_scope(client, plan, payload)
    return plan
