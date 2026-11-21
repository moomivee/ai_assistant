"""Address original evidence by IDs: models select blocks; code supplies quotations."""

from typing import Annotated

from pydantic import Field, create_model

from .budget import check_budget
from .clock import clock_payload
from .config import settings
from .models.schemas import CitationReview, CitedAnswer, CitedClaim
from .progress import context_blocks
from .request_context import scope_payload
from .validation import normalized, numbers


def _answer_schema(evidence):
    # Advertise only this call's block IDs to both transports. Keep the local
    # wire parser string-typed so an invalid ID can be diagnosed per claim.
    # Actual membership is checked below before any claim reaches the reviewer.
    block_id = Annotated[str, Field(json_schema_extra={"enum": list(evidence)})]
    claim = create_model(
        "CitedClaim",
        __base__=CitedClaim,
        evidence_ids=(list[block_id], CitedClaim.model_fields["evidence_ids"]),
    )
    return create_model(
        "CitedAnswer",
        __base__=CitedAnswer,
        claims=(list[claim], CitedAnswer.model_fields["claims"]),
    )


def _references_exist(claim, evidence):
    return bool(claim.evidence_ids) and all(key in evidence for key in claim.evidence_ids)


def catalog(sources, limit=18000, *, priority_source_ids=()):
    evidence = {}
    for source, i, start, end, block in context_blocks(
        sources, limit, priority_source_ids=priority_source_ids
    ):
        key = f"{source['id']}:B{i}"
        evidence[key] = {
            "source": source["id"],
            "text": block,
            "date": source.get("date"),
            "source_text_start": start,
            "source_text_end": end,
            **{
                k: source.get(k)
                for k in (
                    "title",
                    "kind",
                    "locator",
                    "page",
                    "url",
                    "local_url",
                    "document_id",
                    "version_id",
                    "block_id",
                    "method",
                    "content_type",
                    "content_scope",
                    "content_truncated",
                    "date_basis",
                    "retrieved_at",
                )
            },
        }
    return evidence


def generate_answer(
    client,
    question,
    sources,
    *,
    feedback="",
    previous=None,
    fresh=False,
    report_only=False,
    spot_grades=(),
    priority_source_ids=(),
    diagnostics: dict | None = None,
    allow_citation_repair=True,
    allow_research_clarification=False,
):
    # These counters describe validation, not model quality. Keep raw text and
    # rejected values out of diagnostics; None means generation did not finish.
    checks = diagnostics if diagnostics is not None else {}
    review_enabled = settings().answer_review
    checks.update(
        candidate_claims=None,
        candidate_complete=None,
        citation_rejected=0,
        numeric_rejected=0,
        semantic_rejected=0,
        accepted_claims=0,
        initial_citation_rejected=0,
        citation_repair_attempted=False,
        citation_repair_outcome="not_attempted",
        review_enabled=review_enabled,
        review_attempted=False,
        review_executed=False,
        coverage="unverified",
        missing_aspects=[],
        stage="evidence",
        reason="pending",
    )
    evidence = catalog(sources, priority_source_ids=priority_source_ids)
    if not evidence:
        checks["reason"] = "no_evidence"
        return CitedAnswer(claims=[], complete=False), evidence, "No evidence available"
    temporal_requirements = {"fresh": fresh, "report_only": report_only}
    if spot_grades:
        temporal_requirements["required_spot_grades"] = list(spot_grades)
    payload = {
        "question": question,
        **({"previous": previous} if previous else {}),
        **clock_payload(),
        "temporal_requirements": temporal_requirements,
        "user_scope": scope_payload(),
        "sources": [
            {
                k: s.get(k)
                for k in (
                    "id",
                    "title",
                    "date",
                    "url",
                    "page",
                    "kind",
                    "locator",
                    "version_id",
                    "method",
                    "content_type",
                    "content_scope",
                    "content_truncated",
                    "date_basis",
                    "retrieved_at",
                )
            }
            for s in sources
        ],
        "evidence": {key: row["text"] for key, row in evidence.items()},
        "feedback": feedback,
    }
    checks["stage"] = "answer_generation"
    schema = _answer_schema(evidence)
    raw = client.generate("cited_answer", payload, schema)
    invalid = sum(not _references_exist(claim, evidence) for claim in raw.claims)
    checks["initial_citation_rejected"] = invalid
    if allow_citation_repair and invalid and invalid == len(raw.claims):
        check_budget()
        checks.update(
            stage="citation_repair",
            candidate_claims=len(raw.claims),
            candidate_complete=raw.complete,
            citation_rejected=invalid,
            citation_repair_attempted=True,
            citation_repair_outcome="pending",
        )
        corrected_payload = dict(
            payload,
            citation_repair=(
                "Все предыдущие утверждения содержали хотя бы один неизвестный адрес блока. "
                "Повторно прочитай те же фрагменты. Для каждого подтверждённого утверждения "
                "скопируй в evidence_ids точные ключи объекта evidence. Поле id в sources "
                "обозначает документ или страницу и не адресует текстовый блок. "
                "Если ни один блок не подтверждает утверждение, пропусти его."
            ),
        )
        raw = client.generate("cited_answer", corrected_payload, schema)
        invalid = sum(not _references_exist(claim, evidence) for claim in raw.claims)
        if not raw.claims:
            outcome = "empty"
        elif invalid == 0:
            outcome = "ids_valid"
        elif invalid == len(raw.claims):
            outcome = "invalid_ids"
        else:
            outcome = "partially_valid_ids"
        checks["citation_repair_outcome"] = outcome
    checks["citation_rejected"] = invalid
    checks["candidate_claims"] = len(raw.claims)
    checks["candidate_complete"] = raw.complete
    if not raw.claims:
        checks["reason"] = "no_candidate_claims"
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "The model proposed no supported claims from the supplied evidence. "
            "Reassess whether the sources answer the question. Change the query or source "
            "to address a specific gap, or finish with the limitation; do not repeat the same search.",
        )
    checks["stage"] = "local_validation"
    valid = []
    for claim in raw.claims:
        if not _references_exist(claim, evidence):
            continue
        text = " ".join(
            evidence[key]["text"] + " " + (evidence[key]["date"] or "")
            for key in claim.evidence_ids
        )
        if numbers(claim.text) <= numbers(text):
            valid.append(claim)
        else:
            checks["numeric_rejected"] += 1
    if not valid:
        checks["reason"] = "local_validation_rejected"
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "The proposed citations contained unknown block IDs. No claim had all its citations "
            "in the supplied evidence. Do not repeat the same search; use a specific different "
            "source or page if it addresses an evidence gap, otherwise finish with this limitation."
            if invalid == len(raw.claims)
            else "No claims passed citation and numeric checks.",
        )
    if not review_enabled:
        # Valid IDs and matching numbers cannot establish currency. A fetched
        # web result may itself be old or unrelated to the claim.
        complete = raw.complete and len(valid) == len(raw.claims) and not fresh
        checks.update(
            accepted_claims=len(valid),
            stage="finished",
            reason="semantic_review_disabled",
        )
        return (
            CitedAnswer(claims=valid, complete=complete),
            evidence,
            "Freshness was not verified because semantic citation review is disabled. "
            "Do not present historical evidence as the latest state."
            if fresh
            else ""
            if complete
            else "Some requested facts remain unsupported. Search for the missing evidence or open the relevant page.",
        )
    checks["stage"] = "semantic_review"
    checks["review_attempted"] = True
    review = client.generate(
        "citation_review",
        {
            "question": question,
            **({"previous": previous} if previous else {}),
            **clock_payload(),
            "temporal_requirements": temporal_requirements,
            "allow_research_clarification": allow_research_clarification,
            "user_scope": scope_payload(),
            "claims": [
                {"index": i, "text": c.text, "evidence": [evidence[key] for key in c.evidence_ids]}
                for i, c in enumerate(valid)
            ],
        },
        CitationReview,
    )
    checks.update(
        review_executed=True,
        coverage=review.coverage,
        missing_aspects=review.missing_aspects,
    )
    indexes = review.supported
    well_formed = len(set(indexes)) == len(indexes) and all(0 <= i < len(valid) for i in indexes)
    if not well_formed:
        checks.update(semantic_rejected=len(valid), reason="invalid_review_indexes")
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "Citation review returned invalid or duplicate claim indexes. "
            "No claims passed semantic validation.",
        )
    supported = [c for i, c in enumerate(valid) if i in indexes]
    if (
        allow_research_clarification
        and not supported
        and not review.complete
        and review.coverage != "sufficient"
    ):
        options = [" ".join(value.split()) for value in review.research_options]
        labels = {normalized(value).strip(" .!?…") for value in options}
        if (
            2 <= len(options) <= 3
            and len(labels) == len(options)
            and "" not in labels
            and all(len(value) <= 140 for value in options)
        ):
            choices = "\n".join(f"{i}. {option}" for i, option in enumerate(options, 1))
            checks["research_clarification"] = {
                "clarify": f"Что именно вы хотите узнать?\n\n{choices}",
                "research_options": options,
            }
    complete = (
        raw.complete
        and review.complete
        and bool(supported)
        and len(supported) == len(valid) == len(raw.claims)
        and not review.feedback.strip()
        and review.coverage == "sufficient"
        and not review.missing_aspects
    )
    checks.update(
        semantic_rejected=len(valid) - len(supported),
        accepted_claims=len(supported),
        stage="finished",
        reason="complete"
        if complete
        else "semantic_validation_rejected"
        if not supported
        else "coverage_unverified"
        if review.coverage == "unverified"
        else "coverage_incomplete"
        if review.coverage != "sufficient" or review.missing_aspects
        else "incomplete",
    )
    feedback_parts = [review.feedback.strip()]
    if review.missing_aspects:
        feedback_parts.append("Missing requested aspects: " + "; ".join(review.missing_aspects))
    if review.coverage == "unverified":
        feedback_parts.append(
            "Coverage of the question was not verified. Check the requested relation, scope "
            "and time period before treating the answer as complete."
        )
    elif review.coverage == "partial" and not review.missing_aspects:
        feedback_parts.append(
            "The review found only partial coverage of the question. "
            "Identify the missing requested aspect and obtain direct evidence."
        )
    return (
        CitedAnswer(claims=supported, complete=complete),
        evidence,
        " ".join(part for part in feedback_parts if part)
        or (
            ""
            if complete
            else "Some requested facts or their currency remain unverified. "
            "Find direct evidence for the missing facts before finishing."
        ),
    )


def render_claims(draft, evidence, sources):
    lookup = {s["id"]: s for s in sources}
    blocks, claims, used = [], [], {}
    for claim in draft.claims:
        ids = list(dict.fromkeys(evidence[key]["source"] for key in claim.evidence_ids))
        blocks.append(claim.text + " " + " ".join(f"[{key}]" for key in ids))
        # Keep the public compatibility field quote as one original block. All
        # additional citations are exposed by evidence_ids and source.evidence.
        first = evidence[claim.evidence_ids[0]]
        claims.append(
            {
                "source": first["source"],
                "quote": first["text"][:1800],
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
                "citations": [
                    {
                        "evidence_id": eid,
                        "quote": evidence[eid]["text"],
                        **{k: v for k, v in evidence[eid].items() if k not in {"text", "date"}},
                    }
                    for eid in claim.evidence_ids
                ],
            }
        )
        for key in ids:
            if key not in used:
                used[key] = {**lookup[key], "evidence": {}}
            used[key]["evidence"].update(
                {
                    eid: evidence[eid]["text"]
                    for eid in claim.evidence_ids
                    if evidence[eid]["source"] == key
                }
            )
    return blocks, claims, list(used.values())
