"""Address original evidence by IDs: models select blocks; code supplies quotations."""

from .clock import clock_payload
from .config import settings
from .models.schemas import CitationReview, CitedAnswer
from .progress import context_blocks
from .request_context import scope_payload
from .validation import numbers


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
    fresh=False,
    report_only=False,
    priority_source_ids=(),
    diagnostics: dict | None = None,
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
    payload = {
        "question": question,
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
    raw = client.generate("cited_answer", payload, CitedAnswer)
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
        if any(key not in evidence for key in claim.evidence_ids):
            checks["citation_rejected"] += 1
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
            "No claims passed citation and numeric checks.",
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
            **clock_payload(),
            "temporal_requirements": temporal_requirements,
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
