"""Observable progress and action identity; no domain-specific answer rules."""

import json
import unicodedata


def action_signature(name, arguments):
    canonical = dict(arguments)
    if "query" in canonical:
        canonical["query"] = " ".join(
            unicodedata.normalize("NFKC", canonical["query"]).casefold().split()
        )
    return name + ":" + json.dumps(canonical, sort_keys=True, ensure_ascii=False)


def source_key(source):
    return (
        source.get("document_id", source.get("report")),
        source.get("version_id"),
        source.get("chunk_id") or (source["url"], source["text"]),
    )


def page_key(source):
    """Only real pages can replace smaller fragments of the same version."""
    if source.get("kind") != "report" or source.get("page") is None:
        return None
    return (
        source.get("document_id", source.get("report")),
        source.get("version_id"),
        source["page"],
    )


def context_sources(sources):
    """Opening a long page must never hide an already retrieved relevant excerpt."""
    short_pages = [
        s
        for s in sources
        if s.get("full_page") and page_key(s) and not s.get("truncated") and len(s["text"]) <= 6000
    ]
    selected = []
    for source in sources:
        covered = not source.get("full_page") and any(
            page_key(source) == page_key(page)
            and " ".join(source["text"].split()) in " ".join(page["text"].split())
            for page in short_pages
        )
        if not covered:
            selected.append(source)
    # Short opened pages keep table headers together. Long pages come after
    # retrieved excerpts so their prefix cannot exhaust the context budget.
    return sorted(
        selected,
        key=lambda s: (
            bool(s.get("full_page") and (s.get("truncated") or len(s["text"]) > 6000)),
            not s.get("full_page", False),
        ),
    )


def evidence_key(source):
    # Reassigned IDs, whitespace and another URL alone are not new facts.
    # This is content novelty, not semantic relevance or correctness.
    return source["kind"], source.get("date"), " ".join(source["text"].split())


def check_assessment(decision, sources):
    known = {s["id"] for s in sources}
    checks = decision.get("evidence_checks", [])
    referenced = set(decision.get("source_ids", []))
    for check in checks:
        referenced.update(check["source_ids"])
    return {
        "unknown_source_ids": sorted(referenced - known),
        "assessment_references_valid": bool(checks)
        and all(check["source_ids"] for check in checks)
        and referenced <= known,
    }
