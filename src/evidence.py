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
):
    evidence = catalog(sources, priority_source_ids=priority_source_ids)
    if not evidence:
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
                    "date_basis",
                    "retrieved_at",
                )
            }
            for s in sources
        ],
        "evidence": {key: row["text"] for key, row in evidence.items()},
        "feedback": feedback,
    }
    raw = client.generate("cited_answer", payload, CitedAnswer)
    valid = []
    for claim in raw.claims:
        if any(key not in evidence for key in claim.evidence_ids):
            continue
        text = " ".join(
            evidence[key]["text"] + " " + (evidence[key]["date"] or "")
            for key in claim.evidence_ids
        )
        if numbers(claim.text) <= numbers(text):
            valid.append(claim)
    if not valid:
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "No claims passed citation and numeric checks.",
        )
    if not settings().answer_review:
        # Valid IDs and matching numbers cannot establish currency. A fetched
        # web result may itself be old or unrelated to the claim.
        complete = raw.complete and len(valid) == len(raw.claims) and not fresh
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
    indexes = review.supported
    well_formed = len(set(indexes)) == len(indexes) and all(0 <= i < len(valid) for i in indexes)
    if not well_formed:
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
    )
    return (
        CitedAnswer(claims=supported, complete=complete),
        evidence,
        review.feedback.strip()
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
