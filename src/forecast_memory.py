"""User-grounded calculation inputs for one pending forecast clarification."""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal

from pydantic import Field

from .models.schemas import ParameterEvidence, Plan
from .validation import named_grades, normalized, numbers

NUMERIC_FIELDS = ("months", "discount", "cut", "sensitivity")


class ForecastFollowup(Plan):
    new_task: bool = Field(
        default=False,
        description="True only when the user starts an independent task instead of continuing the pending forecast.",
    )
    benchmark_change_quote: str = Field(
        default="",
        max_length=500,
        description="If changing the previous target grade, quote the user clause naming the new forecast target, separately from any discount reference.",
    )
    withdrawn: list[ParameterEvidence] = Field(
        default_factory=list,
        max_length=4,
        description="Previously confirmed assumptions the user explicitly cancels. Quote this turn verbatim.",
    )


@dataclass
class PendingForecast:
    snapshot: dict
    new_task: bool = False
    benchmark_change_quote: str = ""


_pending: ContextVar[PendingForecast | None] = ContextVar("pending_forecast", default=None)


@contextmanager
def forecast_memory(snapshot: dict | None):
    token = _pending.set(PendingForecast(snapshot) if snapshot else None)
    try:
        yield
    finally:
        _pending.reset(token)


def pending_forecast() -> PendingForecast | None:
    return _pending.get()


def previous_text(snapshot: dict) -> str:
    return "\n".join(snapshot["turns"])


def horizon_quote(text: str, value: int) -> str:
    """Recognize a horizon with its unit only when the planner omitted its quote."""
    names = (
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
        "десять",
        "одиннадцать",
        "двенадцать",
    )
    if not 1 <= value <= len(names):
        return ""
    match = re.search(
        rf"\b(?:{value}|{names[value - 1]})\s+(?:месяц\w*|мес\b|months?\b)",
        text,
        flags=re.IGNORECASE,
    )
    return match[0] if match else ""


def next_snapshot(previous_snapshot: dict | None, question: str, plan: dict) -> dict:
    """Keep the opening, latest reply and user turns supporting the current inputs."""
    pending = pending_forecast()
    if pending and pending.new_task:
        previous_snapshot = None
    turns = [*(previous_snapshot or {}).get("turns", ()), question]
    keep = {0, len(turns) - 1}
    review_scope = plan["wants_context"] or plan["fresh"] or plan["news"] or bool(plan["quotes"])
    # Preserve the turn that first introduced an additional market question. The normal
    # forecast_scope classifier will require its verbatim clause before using any tool.
    scope_turn = (previous_snapshot or {}).get("scope_turn") if review_scope else None
    if review_scope and scope_turn is None:
        scope_turn = question
    if scope_turn is not None:
        keep.add(turns.index(scope_turn))
    confirmed = {}
    evidence = {}
    citations = {entry["field"]: entry["quote"] for entry in plan["parameters"]}
    for field in ("benchmark", *NUMERIC_FIELDS):
        value = plan[field]
        if value is None or (field == "months" and not 1 <= value <= 12):
            continue
        if field in ("discount", "cut", "sensitivity") and value < 0:
            continue
        quote = citations.get(field)
        for i in reversed(range(len(turns))):
            text = turns[i]
            if field == "benchmark":
                supported = value in named_grades(text)
            elif quote:
                supported = normalized(quote) in normalized(text) and (
                    Decimal(str(value)) in numbers(quote)
                    or (field == "months" and not numbers(quote))
                )
            elif field == "months":
                quote = horizon_quote(text, value)
                supported = bool(quote)
            else:
                supported = False
            if supported:
                confirmed[field] = value
                if field != "benchmark":
                    evidence[field] = quote
                keep.add(i)
                break
    return {
        "turns": [turns[i] for i in sorted(keep)],
        "confirmed": confirmed,
        "report_only": plan["report_only"],
        "scope_turn": scope_turn,
        "evidence": evidence,
        "clarification": plan["clarify"],
    }


def merge_followup(
    plan: Plan, question: str, pending: PendingForecast, withdrawn: set[str]
) -> Plan:
    """Carry prior inputs unless a replacement is grounded in the current user turn."""
    if plan.intent != "forecast" or pending.new_task:
        return plan
    confirmed = pending.snapshot["confirmed"]
    evidence = pending.snapshot["evidence"]
    current = normalized(question)
    new_quotes = {row.field: row.quote for row in plan.parameters}
    discount_terms = normalized(new_quotes.get("discount", ""))
    # A named discount reference can occur in a reply without changing the forecast target.
    target_text = current.replace(discount_terms, " ") if discount_terms else current
    benchmark = plan.benchmark
    old_grade = confirmed.get("benchmark")
    horizon = horizon_quote(question, plan.months) if plan.months is not None else ""
    other_amounts = numbers(current.replace(normalized(horizon), " ", 1) if horizon else current)
    if benchmark not in named_grades(target_text):
        benchmark = old_grade
    elif (
        old_grade
        and benchmark != old_grade
        and (other_amounts or (discount_terms and discount_terms in current))
    ):
        # The model can quote just "12" out of "12 dollars below Brent". A simultaneous
        # switch of target needs its own clause, independent of the discount wording.
        target_clause = normalized(pending.benchmark_change_quote)
        if not (
            len(target_clause.split()) > 1
            and target_clause in target_text
            and named_grades(target_clause) == {benchmark}
        ):
            benchmark = old_grade

    if plan.months is not None:
        quote = normalized(new_quotes.get("months", ""))
        if horizon:
            new_quotes["months"] = horizon
        elif quote and any(
            new_quotes.get(field)
            and (quote in normalized(new_quotes[field]) or normalized(new_quotes[field]) in quote)
            and plan.months == getattr(plan, field)
            for field in ("discount", "cut", "sensitivity")
        ):
            # A currency/production amount is not a new horizon with the same number.
            new_quotes.pop("months")
    switched = bool(confirmed.get("benchmark") and benchmark != confirmed["benchmark"])
    values = {
        "benchmark": benchmark,
        "report_only": plan.report_only or pending.snapshot.get("report_only", False),
        "wants_context": plan.wants_context or bool(pending.snapshot.get("scope_turn")),
    }
    quotes = []
    for field in NUMERIC_FIELDS:
        quote = new_quotes.get(field, "")
        value = getattr(plan, field)
        if field in withdrawn:
            value = None
        elif value is None or not quote.strip() or normalized(quote) not in current:
            value = None if field == "discount" and switched else confirmed.get(field)
            quote = evidence.get(field, "")
        values[field] = value
        if value is not None and quote:
            quotes.append(ParameterEvidence(field=field, quote=quote))
    return plan.model_copy(update={**values, "parameters": quotes})
