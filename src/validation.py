"""Data checks, not a semantic regex router."""

import re
from decimal import Decimal

from .logging import log
from .models.schemas import EvidenceAnswer, ForecastInputs, Plan

BENCHMARK_NAMES = {
    "Brent": {"brent", "брент"},
    "WTI": {"wti"},
    "Urals": {"urals", "уралс", "юралс"},
}


def normalized(text):
    return " ".join(text.split()).casefold().replace("ё", "е")


def numbers(text):
    # A separator in "1.3%-1.5%" is not a negative sign. Keep unary minus
    # (including the Unicode minus) for actual negative changes and prices.
    text = text.replace("−", "-")
    text = re.sub(r"(?<=\d)(\s*%?)\s*[-–—]\s*(?=\d)", r"\1 ", text)
    return {Decimal(x.replace(",", ".")) for x in re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", text)}


def unsupported_discount(plan: Plan) -> bool:
    return plan.discount is not None and plan.benchmark in ("Brent", "WTI")


def named_grades(text: str) -> set[str]:
    words = set(re.findall(r"\w+", normalized(text)))
    return {grade for grade, names in BENCHMARK_NAMES.items() if words & names}


def discount_quote(inputs: Plan | ForecastInputs) -> str:
    # Match validate_plan's choice if a model returned the field twice.
    return next((p.quote for p in reversed(inputs.parameters) if p.field == "discount"), "")


def unsupported_discount_base(plan: Plan) -> bool:
    if plan.benchmark != "Urals" or plan.discount is None:
        return False
    return "WTI" in named_grades(discount_quote(plan))


def validate_plan(plan: Plan, question: str, previous: str | None = None) -> Plan:
    p = plan.model_copy(deep=True)
    if p.intent == "outside":
        # Canonicalize a semantic decision already made by the model; never
        # classify the question here using occupation names or keyword lists.
        p.rag_query = ""
        p.fresh = p.news = p.wants_context = False
        p.quotes = []
        p.skill = None
    p.fresh = p.fresh or p.news or bool(p.quotes)
    if p.intent == "research" and (not p.quotes or p.news):
        p.wants_context = True
    if p.intent != "forecast":
        options = {normalized(option).strip(" .!?…") for option in p.research_options}
        clarification = normalized(p.clarify).strip(" .!?…")
        if (
            p.intent == "research"
            and len(options - {""}) >= 2
            and clarification
            and clarification != normalized(question).strip(" .!?…")
        ):
            p.clarify = p.clarify.strip()
        else:
            if p.clarify:
                log("plan.unexpected_clarification_ignored", intent=p.intent)
            p.clarify = ""
            p.research_options = []
        p.parameters = []
        p.benchmark = p.months = p.discount = p.cut = p.sensitivity = None
        return p
    # Forecast clarification follows validated inputs, not free-form model text.
    p.clarify = ""
    p.research_options = []
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
    elif unsupported_discount(p):
        p.clarify = (
            f"Для {p.benchmark} доступен обычный прогноз. Дисконт к Brent применяется "
            f"только в сценарии Urals. Рассчитать {p.benchmark} без дисконта?"
        )
    elif unsupported_discount_base(p):
        p.clarify = (
            "В условии дисконта указан WTI. Для прогноза Urals расчёт использует Brent "
            "как базу. Какой дисконт к Brent заложить в USD за баррель?"
        )
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
        valid = (
            source is not None
            and bool(normalized(claim.quote))
            and normalized(claim.quote) in normalized(source["text"])
        )
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
