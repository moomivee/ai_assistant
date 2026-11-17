"""Bounded structured checks of forecast inputs and combined research requests."""

from .clock import clock_payload
from .integrations.agent_skills import get_skill_catalog
from .integrations.llm import LLMUnavailable
from .integrations.web import DAILY
from .logging import log
from .models.schemas import ForecastInputs, ForecastScope, Plan, ResearchQuoteScope
from .request_context import current_scope, scope_payload
from .validation import normalized, validate_plan

FORECAST_FIELDS = ("benchmark", "months", "discount", "cut", "sensitivity")
BENCHMARK_NAMES = {
    "Brent": ("brent", "брент"),
    "WTI": ("wti",),
    "Urals": ("urals", "уралс", "юралс"),
}


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


def merge_forecast_inputs(plan: Plan, extracted: ForecastInputs, question, previous=None):
    """Fill missing fields only; new values require quotes from the actual input."""
    candidate = plan.model_copy(deep=True)
    source = normalized((previous or "") + " " + question)
    quote = normalized(extracted.benchmark_quote)
    if candidate.benchmark is None and extracted.benchmark is not None:
        # This checks the model's evidence, not intent/routing by keywords.
        tokens = [word.strip(".,:;!?()[]«»\"'") for word in quote.replace("/", " ").split()]
        if (
            quote
            and quote in source
            and any(name in tokens for name in BENCHMARK_NAMES[extracted.benchmark])
        ):
            candidate.benchmark = extracted.benchmark
    evidence = {e.field: e for e in candidate.parameters}
    new_evidence = {e.field: e for e in extracted.parameters}
    for field in FORECAST_FIELDS[1:]:
        value = getattr(extracted, field)
        if getattr(candidate, field) is None and value is not None and field in new_evidence:
            setattr(candidate, field, value)
            evidence[field] = new_evidence[field]
    candidate.parameters = list(evidence.values())
    return validate_plan(candidate, question, previous)


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
            "context_fresh": bool(scope.context_fresh or plan.news),
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


def make_plan(client, question, previous=None) -> Plan:
    payload = {
        "question": question,
        "previous": previous,
        **clock_payload(),
        "user_scope": scope_payload(),
        "available_skills": [
            s for s in get_skill_catalog() if s.get("enabled") and s.get("available", True)
        ],
    }
    raw = client.generate("plan", payload, Plan)
    plan = validate_plan(raw, question, previous)
    if current_scope().report_only:
        plan = plan.model_copy(update={"report_only": True})
    log(
        "plan.validated",
        intent=plan.intent,
        benchmark=plan.benchmark,
        months=plan.months,
        needs_clarification=bool(plan.clarify),
    )
    if plan.intent == "research" and plan.quotes and not plan.clarify:
        return resolve_research_quotes(client, plan, payload)
    if plan.context_request or plan.quote_request or plan.context_fresh is not None:
        plan = plan.model_copy(
            update={"quote_request": "", "context_request": "", "context_fresh": None}
        )
    if plan.intent != "forecast":
        return plan

    if plan.clarify or raw.clarify:
        # At most one input repair. A genuinely absent parameter stays absent.
        log(
            "plan.repair_requested",
            missing=[f for f in FORECAST_FIELDS if getattr(plan, f) is None],
        )
        extracted = client.generate("forecast", payload, ForecastInputs)
        if raw.clarify and not plan.clarify:
            # Recheck a possible ambiguity before dismissing the clarification.
            plan = plan.model_copy(update={**{f: None for f in FORECAST_FIELDS}, "parameters": []})
        plan = merge_forecast_inputs(plan, extracted, question, previous)
        log(
            "plan.repaired",
            benchmark=plan.benchmark,
            months=plan.months,
            needs_clarification=bool(plan.clarify),
        )
    if not plan.clarify and (plan.wants_context or plan.fresh or plan.news or plan.quotes):
        plan = resolve_forecast_scope(client, plan, payload)
    return plan
