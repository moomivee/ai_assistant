"""Address original evidence by IDs: models select blocks; code supplies quotations."""

from .config import settings
from .models.schemas import CitationReview, CitedAnswer
from .validation import numbers


def catalog(sources, limit=18000):
    full = {(s.get("report"), s.get("page")) for s in sources if s.get("full_page")}
    selected = sorted(sources, key=lambda s: not s.get("full_page", False))
    evidence, size = {}, 0
    for source in selected:
        if not source.get("full_page") and (source.get("report"), source.get("page")) in full:
            continue
        # Blocks, rather than isolated lines, retain headers and rows together
        # and avoid forcing the model to copy brittle line indexes.
        chunks, block = [], ""
        for line in source["text"].splitlines():
            line = line.strip()
            for start in range(0, len(line), 1600):
                part = line[start : start + 1600]
                if block and len(block) + len(part) + 1 > 1600:
                    chunks.append(block)
                    block = ""
                block += ("\n" if block else "") + part
        if block:
            chunks.append(block)
        for i, block in enumerate(chunks, 1):
            if size + len(block) > limit:
                break
            key = f"{source['id']}:B{i}"
            evidence[key] = {"source": source["id"], "text": block, "date": source.get("date")}
            size += len(block)
    return evidence


def generate_answer(client, question, sources, *, feedback=""):
    evidence = catalog(sources)
    if not evidence:
        return CitedAnswer(claims=[], complete=False), evidence, "No evidence available"
    payload = {
        "question": question,
        "sources": [
            {k: s.get(k) for k in ("id", "title", "date", "page", "kind")} for s in sources
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
            complete=review.complete and bool(supported) and len(valid) == len(raw.claims),
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
            }
        )
        for key in ids:
            used[key] = {
                **lookup[key],
                "evidence": {
                    eid: item["text"] for eid, item in evidence.items() if item["source"] == key
                },
            }
    return blocks, claims, list(used.values())
