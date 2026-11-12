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
    return source.get("chunk_id") or (source["url"], source["text"])


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
