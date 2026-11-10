"""Data checks, not a semantic regex router."""

import re
from decimal import Decimal

from .logging import log
from .models.schemas import EvidenceAnswer, Plan


def normalized(text):
    return " ".join(text.split()).casefold().replace("ё", "е")


def numbers(text):
    return {Decimal(x.replace(",", ".")) for x in re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", text)}


def validate_plan(plan: Plan, question: str, previous: str | None = None) -> Plan:
    p = plan.model_copy(deep=True)
    p.fresh = p.fresh or p.news or bool(p.quotes)
    if p.intent == "research" and (not p.quotes or p.news):
        p.wants_context = True
    if p.intent != "forecast":
        p.parameters = []
        p.benchmark = p.months = p.discount = p.cut = p.sensitivity = None
        return p
    # Forecast clarification follows validated inputs, not free-form model text.
    p.clarify = ""
    combined = normalized((previous or "") + " " + question)
    evidence = {e.field: e.quote for e in p.parameters}
    for field in ("months", "discount", "cut", "sensitivity"):
        value = getattr(p, field)
        if value is None:
            continue
        quote = evidence.get(field, "")
        if field == "months" and not quote:
            # The horizon can be checked against the original input. Monetary
            # and scenario assumptions still require a specific evidence quote.
            quote = (previous or "") + " " + question
        valid = bool(quote.strip()) and normalized(quote) in combined
        if field == "discount" and "%" in quote:
            valid = False
        if field != "months" or numbers(quote):
            valid = valid and Decimal(str(value)) in numbers(quote)
        if not valid:
            setattr(p, field, None)
            p.clarify = "Уточните параметры расчёта и единицы: они не подтверждены в запросе."
            log("plan.parameter_rejected", field=field)
    if p.benchmark is None:
        p.clarify = "Какой сорт рассчитать: Brent, WTI или Urals?"
    elif p.months is None or not 1 <= p.months <= 12:
        p.clarify = "На сколько месяцев нужен прогноз? Допустим горизонт от 1 до 12 месяцев."
    elif p.benchmark == "Urals" and p.discount is None:
        p.clarify = "Укажите дисконт Urals к Brent в USD за баррель."
    elif (p.cut is None) != (p.sensitivity is None):
        p.clarify = "Укажите сокращение в млн баррелей в сутки и чувствительность в USD/барр. на 1 млн баррелей в сутки."
    elif any(v is not None and v < 0 for v in (p.discount, p.cut, p.sensitivity)):
        p.clarify = "Дисконт, сокращение добычи и чувствительность должны быть неотрицательными."
    return p


def validate_claims(draft: EvidenceAnswer, sources: list[dict]) -> EvidenceAnswer:
    lookup = {s["id"]: s for s in sources}
    accepted = []
    for claim in draft.claims:
        source = lookup.get(claim.source)
        valid = source is not None and normalized(claim.quote) in normalized(source["text"])
        if valid:
            valid = numbers(claim.text) <= numbers(claim.quote + " " + (source["date"] or ""))
        if valid:
            accepted.append(claim)
        else:
            log("answer.claim_rejected", source_id=claim.source)
    return EvidenceAnswer(
        complete=draft.complete and bool(accepted) and len(accepted) == len(draft.claims),
        claims=accepted,
        limitations=draft.limitations,
    )
