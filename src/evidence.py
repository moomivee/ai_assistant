"""Address original evidence by IDs: models select blocks; code supplies quotations."""

from datetime import date

from .config import settings
from .models.schemas import CitationReview, CitedAnswer
from .progress import context_sources
from .request_context import scope_payload
from .validation import numbers


def catalog(sources, limit=18000):
    selected = context_sources(sources)
    evidence, size = {}, 0
    for source in selected:
        # Cut exact substrings, preferably on a line boundary. No whitespace
        # rewriting: quotes and offsets stay auditable against extracted text.
        text, start, chunks = source["text"], 0, []
        while start < len(text):
            end = min(start + 1600, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + 800, end)
                if boundary > start:
                    end = boundary + 1
            block = text[start:end]
            if block.strip():
                chunks.append((start, end, block))
            start = end
        for i, (start, end, block) in enumerate(chunks, 1):
            if size + len(block) > limit:
                break
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
            size += len(block)
    return evidence


def generate_answer(client, question, sources, *, feedback=""):
    evidence = catalog(sources)
    if not evidence:
        return CitedAnswer(claims=[], complete=False), evidence, "No evidence available"
    payload = {
        "question": question,
        "today": date.today().isoformat(),
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
        complete = raw.complete and len(valid) == len(raw.claims)
        return (
            CitedAnswer(claims=valid, complete=complete),
            evidence,
            ""
            if complete
            else "Some requested facts remain unsupported. Search for the missing evidence or open the relevant page.",
        )
    review = client.generate(
        "citation_review",
        {
            "question": question,
            "claims": [
                {"index": i, "text": c.text, "evidence": [evidence[key] for key in c.evidence_ids]}
                for i, c in enumerate(valid)
            ],
        },
        CitationReview,
    )
    indexes = review.supported
    well_formed = len(set(indexes)) == len(indexes) and all(0 <= i < len(valid) for i in indexes)
    supported = [c for i, c in enumerate(valid) if well_formed and i in indexes]
    return (
        CitedAnswer(
            claims=supported,
            complete=review.complete
            and bool(supported)
            and len(supported) == len(valid) == len(raw.claims),
        ),
        evidence,
        review.feedback,
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
