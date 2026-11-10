"""Plan validation with one evidence-grounded repair for incomplete forecasts."""

from datetime import date

from .logging import log
from .models.schemas import ForecastInputs, Plan
from .validation import normalized, validate_plan

FORECAST_FIELDS = ("benchmark", "months", "discount", "cut", "sensitivity")
BENCHMARK_NAMES = {
    "Brent": ("brent", "брент"),
    "WTI": ("wti",),
    "Urals": ("urals", "уралс", "юралс"),
}


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


def make_plan(client, question, previous=None) -> Plan:
    payload = {"question": question, "previous": previous, "today": date.today().isoformat()}
    raw = client.generate("plan", payload, Plan)
    plan = validate_plan(raw, question, previous)
    log(
        "plan.validated",
        intent=plan.intent,
        benchmark=plan.benchmark,
        months=plan.months,
        needs_clarification=bool(plan.clarify),
    )
    if plan.intent != "forecast" or not (plan.clarify or raw.clarify):
        return plan

    # At most one extra LLM call. A genuinely absent parameter stays absent.
    log("plan.repair_requested", missing=[f for f in FORECAST_FIELDS if getattr(plan, f) is None])
    extracted = client.generate("forecast", payload, ForecastInputs)
    if raw.clarify and not plan.clarify:
        # A model may ask a spurious question or signal genuine ambiguity even
        # with all fields filled. Re-extract these inputs before dismissing it.
        plan = plan.model_copy(update={**{f: None for f in FORECAST_FIELDS}, "parameters": []})
    repaired = merge_forecast_inputs(plan, extracted, question, previous)
    log(
        "plan.repaired",
        benchmark=repaired.benchmark,
        months=repaired.months,
        needs_clarification=bool(repaired.clarify),
    )
    return repaired
