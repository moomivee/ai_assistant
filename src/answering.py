"""Grounded drafting, semantic review, and at most one focused repair."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, create_model

from .integrations.llm import LLMUnavailable
from .logging import log
from .models.schemas import AnswerReview, ClaimReview, DraftAnswer, EvidenceAnswer
from .validation import normalized, validate_claims


@lru_cache(maxsize=11)
def review_schema(count):
    """Constrain guided generation to the actual claims, not source indexes."""
    if not 0 <= count <= 10:
        raise ValueError("Expected between zero and ten claims.")
    item = ClaimReview
    if count:
        item = create_model(
            "ScopedClaimReview",
            __base__=ClaimReview,
            index=(Literal[tuple(range(count))], ...),
        )
    return create_model(
        "ScopedAnswerReview",
        __base__=AnswerReview,
        checks=(list[item], Field(min_length=count, max_length=count)),
    )


def compact_sources(sources):
    """Keep facts and citation identity; omit URLs, cache paths and rank metadata."""
    return [
        {key: row[key] for key in ("id", "kind", "title", "date", "page", "text") if key in row}
        for row in sources
    ]


def checked_review(draft, review, sources):
    """A review cannot bypass quote checks or invent repair evidence."""
    checks = {check.index: check.verdict for check in review.checks}
    well_formed = len(checks) == len(review.checks) and set(checks) == set(range(len(draft.claims)))
    if not well_formed:
        log("answer.review_invalid", claims=len(draft.claims), checks=len(review.checks))
    accepted = [
        claim for i, claim in enumerate(draft.claims) if well_formed and checks[i] == "supported"
    ]
    lookup = {s["id"]: s for s in sources}
    evidence = []
    seen = set()
    for item in review.repair_evidence:
        source = lookup.get(item.source)
        key = (item.source, normalized(item.quote))
        if source and key[1] and key[1] in normalized(source["text"]) and key not in seen:
            evidence.append(item)
            seen.add(key)
    result = EvidenceAnswer(
        complete=well_formed and review.complete and bool(accepted),
        claims=accepted,
        # Missing-data statements are not evidence. User-facing uncertainty is
        # assembled by the workflow and scoped to the actual request.
        limitations=[],
    )
    log(
        "answer.reviewed",
        accepted=len(accepted),
        rejected=len(draft.claims) - len(accepted),
        complete=result.complete,
        repair_evidence=len(evidence),
    )
    return result, evidence


def focused_sources(sources, evidence):
    """Keep each original ID, but show only verified relevant excerpts to repair."""
    selected = {}
    for item in evidence:
        selected.setdefault(item.source, []).append(item.quote)
    return [
        {**source, "text": "\n\n".join(selected[source["id"]])}
        for source in sources
        if source["id"] in selected
    ]


def verified_answer(client, payload, *, allow_repair=True):
    sources = compact_sources(payload["sources"])
    base = {**payload, "sources": sources}

    def review(draft, context, phase):
        result = client.generate(
            "answer_review",
            {
                **base,
                "sources": context,
                "phase": phase,
                "claim_indexes": list(range(len(draft.claims))),
                "draft": {
                    "claims": [
                        {"index": i, **claim.model_dump()} for i, claim in enumerate(draft.claims)
                    ]
                },
            },
            review_schema(len(draft.claims)),
        )
        return checked_review(draft, result, context)

    raw = client.generate("answer", {**base, "phase": "draft"}, DraftAnswer)
    draft = validate_claims(raw, sources)
    checked, evidence = review(draft, sources, "draft")
    if checked.complete or not allow_repair or not evidence:
        return checked, False

    context = focused_sources(sources, evidence)
    log("answer.repair_requested", sources=len(context))
    try:
        raw_repair = client.generate(
            "answer",
            {
                **base,
                "phase": "repair",
                "sources": context,
                "supported_claims": [claim.model_dump() for claim in checked.claims],
                "instruction": "Пересобери ответ только по этим проверенным цитатам. Ответь на все части question; не добавляй другие темы.",
            },
            DraftAnswer,
        )
        # Joined excerpts must not become invented contiguous quotations.
        ids = {source["id"] for source in context}
        repaired = validate_claims(raw_repair, [s for s in sources if s["id"] in ids])
        if not repaired.claims or repaired == draft:
            log("answer.repair_no_improvement")
            return checked, True
        final, _ = review(repaired, context, "repair")
    except LLMUnavailable:
        # Keep only claims that already passed review, never the unchecked repair.
        log("answer.repair_unavailable")
        return checked, True
    if final.complete:
        return final, True
    seen = {(c.source, normalized(c.quote), normalized(c.text)) for c in final.claims}
    final.claims.extend(
        c for c in checked.claims if (c.source, normalized(c.quote), normalized(c.text)) not in seen
    )
    final.claims = final.claims[:10]
    return final, True
