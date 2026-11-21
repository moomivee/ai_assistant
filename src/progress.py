"""Observable progress and action identity; no domain-specific answer rules."""

import json
import unicodedata
from collections import deque


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


def context_sources(sources, *, priority_source_ids=()):
    """Opening a long page must never hide an already retrieved relevant excerpt."""
    short_pages = [
        s
        for s in sources
        if s.get("full_page") and page_key(s) and not s.get("truncated") and len(s["text"]) <= 6000
    ]
    priorities = {source_id: rank for rank, source_id in enumerate(priority_source_ids)}
    selected = []
    for source in sources:
        covering = [
            page
            for page in short_pages
            if not source.get("full_page")
            and page_key(source) == page_key(page)
            and " ".join(source["text"].split()) in " ".join(page["text"].split())
        ]
        if not covering:
            selected.append(source)
        elif source["id"] in priorities:
            for page in covering:
                priorities[page["id"]] = min(
                    priorities.get(page["id"], len(priorities)), priorities[source["id"]]
                )
    # Short opened pages keep table headers together. Long pages come after
    # retrieved excerpts so their prefix cannot exhaust the context budget.
    return sorted(
        selected,
        key=lambda s: (
            priorities.get(s["id"], len(priorities)),
            bool(s.get("full_page") and (s.get("truncated") or len(s["text"]) > 6000)),
            not s.get("full_page", False),
        ),
    )


def _source_blocks(source):
    """Stable exact slices, independent of context limit and source priority."""
    text, start, block_no = source["text"], 0, 0
    while start < len(text):
        end = min(start + 1600, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + 800, end)
            if boundary <= start:
                # Prefer a token boundary when a page has no line breaks. A
                # normal date or numeric value must not be split at the cap.
                boundary = max(text.rfind(" ", start, end), text.rfind("\t", start, end))
            if boundary > start:
                end = boundary + 1
        block = text[start:end]
        if block.strip():
            block_no += 1
            yield block_no, start, end, block
        start = end


def context_blocks(sources, limit, *, priority_source_ids=(), retained_block_id=None):
    """Pack one exact block per source per round, giving new evidence a turn."""
    ordered = context_sources(sources, priority_source_ids=priority_source_ids)
    pending = deque((source, iter(_source_blocks(source))) for source in ordered)
    selected, remaining = [], limit
    # Keep one dated citation comparable across rounds while leaving room for new sources.
    if retained_block_id:
        source_id, _, number = retained_block_id.partition(":B")
        source = next((row for row in ordered if row["id"] == source_id), None)
        retained = (
            next((block for block in _source_blocks(source) if str(block[0]) == number), None)
            if source
            else None
        )
        if retained and len(retained[3]) <= remaining:
            selected.append((source, *retained))
            remaining -= len(retained[3])
        else:
            retained_block_id = None
    while pending and remaining > 0:
        source, blocks = pending.popleft()
        block = next(blocks, None)
        if block is None:
            continue
        block_no, start, end, text = block
        if retained_block_id and f"{source['id']}:B{block_no}" == retained_block_id:
            pending.append((source, blocks))
            continue
        if len(text) > remaining:
            # Do not replace a full block by an orphan tail or change its ID.
            continue
        selected.append((source, block_no, start, end, text))
        remaining -= len(text)
        pending.append((source, blocks))
    return selected


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
