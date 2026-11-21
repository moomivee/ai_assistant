"""Compare dated occurrences in the supplied evidence for a latest-event question."""

import json
from collections import Counter
from hashlib import sha256
from typing import Literal

from pydantic import Field, ValidationError, create_model

from .budget import check_budget
from .clock import clock_payload
from .event_dates import mentioned_dates
from .integrations.llm import InvalidModelOutput, StructuredOutputError
from .models.schemas import StrictModel


class EventCandidate(StrictModel):
    kind: Literal["held", "scheduled"]
    date: str
    quote: str
    scope: str
    match: Literal["in_scope", "different_scope", "uncertain"]


class EventAssessment(StrictModel):
    status: Literal["assessed", "uncertain"]
    events: list[EventCandidate]


class EventBlock(EventAssessment):
    evidence_id: str


class EventChronology(StrictModel):
    blocks: list[EventBlock]
    requested_context: list[Literal["next_scheduled", "previous_held"]] = Field(
        default_factory=list
    )


# Six normal source excerpts fit in one generation. The work cap also applies if a
# catalog consists of many tiny excerpts instead of full text blocks.
BATCH_SIZE = 6
MAX_BATCHES = 4


def _keyed_schema(ids):
    records = create_model(
        "EventAssessments",
        __base__=StrictModel,
        **{key: (EventAssessment, ...) for key in ids},
    )
    return create_model("EventChronology", __base__=EventChronology, blocks=(records, ...))


def _read_blocks(client, payload, *, keyed=False):
    """Recover independent rows by explicit block ID, never by their array position."""
    allowed = payload["evidence"].keys()
    schema = _keyed_schema(allowed) if keyed else EventChronology
    issues = []
    try:
        data = client.generate("event_chronology", payload, schema).model_dump()
    except StructuredOutputError as exc:
        issues = exc.issues
        try:
            data = json.loads(exc.response_content)
        except (TypeError, ValueError):
            data = None
    except InvalidModelOutput:
        # The provider exhausted its empty/truncated response fallback. Earlier
        # batches still count; only this batch has no usable rows to recover.
        return {}, None, [], Counter(unusable_completion=1, missing_ids=len(allowed))

    raw = data.get("blocks") if isinstance(data, dict) else None
    # Accept the keyed representation only when the application asked for it.
    # Validate each sibling independently, so a malformed row cannot erase a good one.
    if keyed and isinstance(raw, dict):
        rows = [
            {"evidence_id": key, **value}
            if isinstance(value, dict) and "evidence_id" not in value
            else None
            for key, value in raw.items()
        ]
    elif isinstance(raw, list):
        rows = raw
    else:
        return {}, None, issues, Counter(invalid_envelope=1, missing_ids=len(allowed))
    objects = [
        row for row in rows if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
    ]
    counts = Counter(row["evidence_id"] for row in objects)
    accepted = {}
    for obj in objects:
        key = obj["evidence_id"]
        if key in allowed and counts[key] == 1:
            try:
                accepted[key] = EventBlock.model_validate(obj)
            except ValidationError:
                continue
    try:
        context = EventChronology.model_validate({**data, "blocks": []}).requested_context
    except ValidationError:
        context = None
    accounting = Counter(
        empty_blocks=int(not rows),
        unaddressed_rows=len(rows) - len(objects),
        unknown_ids=sum(n for key, n in counts.items() if key not in allowed),
        duplicate_ids=sum(1 for key in allowed if counts[key] > 1),
        missing_ids=sum(1 for key in allowed if not counts[key]),
    )
    return accepted, context, issues, +accounting


def build_event_chronology(
    client, question, evidence, *, previous=None, allow_format_repair=True, cache=None
):
    clock = clock_payload()
    as_of = clock["today"]
    payload = {"question": question, **({"previous": previous} if previous else {}), **clock}
    inputs = {
        key: {
            "text": block["text"],
            "title": block.get("title"),
            "url": block.get("url"),
            "publication_date": block.get("date"),
        }
        for key, block in evidence.items()
    }
    # Never reuse a classification across questions or days. Copy the nested map so
    # an interrupted graph step cannot mutate the prior step's snapshot in place.
    scope = [question, previous, as_of]
    prior = cache if cache is not None and cache.get("scope") == scope else {}
    saved = dict(prior.get("rows", {}))
    context = prior.get("requested_context")
    keyed = prior.get("keyed_blocks", False)
    reviews = {}
    fingerprints = {}
    for key, entry in inputs.items():
        fingerprint = sha256(json.dumps(entry, ensure_ascii=False).encode()).hexdigest()
        fingerprints[key] = fingerprint
        if key in saved and saved[key]["fingerprint"] == fingerprint:
            reviews[key] = EventBlock.model_validate(saved[key]["row"])
    reused = list(reviews)
    pending = [key for key in inputs if key not in reviews]
    capacity = BATCH_SIZE * MAX_BATCHES
    deferred = pending[capacity:]
    pending = pending[:capacity]
    repair_attempted = False
    issues = []
    address_errors = Counter()
    batch_sizes = []
    for start in range(0, len(pending), BATCH_SIZE):
        keys = pending[start : start + BATCH_SIZE]
        batch_sizes.append(len(keys))
        batch_payload = {
            **payload,
            "evidence": {key: inputs[key] for key in keys},
            "keyed_blocks": keyed,
        }
        check_budget()
        rows, roles, faults, addressing = _read_blocks(client, batch_payload, keyed=keyed)
        issues.extend(faults)
        address_errors.update(addressing)
        if context is None and rows:
            context = roles
        missing = [key for key in keys if key not in rows]
        if allow_format_repair and not repair_attempted and (missing or context is None):
            repair_attempted = True
            check_budget()
            # A model that loses the association between a list item and its source
            # gets preassigned keys for this request, including later search rounds.
            if missing and not rows and not faults:
                keyed = True
            repair_payload = {
                **payload,
                "evidence": {key: inputs[key] for key in missing},
                "format_repair": {
                    "expected_evidence_ids": missing,
                    "validation_errors": faults,
                    "address_errors": dict(addressing),
                    "keyed_blocks": keyed,
                },
            }
            corrected, roles, faults, addressing = _read_blocks(client, repair_payload, keyed=keyed)
            rows.update(corrected)
            issues.extend(faults)
            address_errors.update(addressing)
            if context is None and (corrected or not missing):
                context = roles
        reviews.update(rows)
        for key, record in rows.items():
            saved[key] = {"fingerprint": fingerprints[key], "row": record.model_dump()}
    if cache is not None:
        cache.update(
            scope=scope,
            rows=saved,
            requested_context=context,
            keyed_blocks=keyed or repair_attempted,
        )
    missing = [key for key in pending if key not in reviews]
    if not evidence:
        context = []
    unresolved = set()
    held, scheduled = [], []
    for key, source in evidence.items():
        block = reviews.get(key)
        if block is None or block.status == "uncertain":
            unresolved.add(key)
            continue
        original = " ".join(source["text"].split())
        for item in block.events:
            quote = " ".join(item.quote.split())
            if (
                not quote
                or quote not in original
                or {day.isoformat() for day in mentioned_dates(quote)} != {item.date}
                or not item.scope.strip()
                or item.match == "uncertain"
            ):
                unresolved.add(key)
                continue
            if item.match == "different_scope":
                continue
            if item.kind == "held" and item.date > as_of:
                unresolved.add(key)
                continue
            row = {
                "date": item.date,
                "evidence_id": key,
                "quote": quote,
                "scope": item.scope.strip(),
            }
            events = held if item.kind == "held" else scheduled
            if row not in events:
                events.append(row)
    held.sort(key=lambda item: item["date"], reverse=True)
    scheduled.sort(key=lambda item: item["date"])
    latest = held[0]["date"] if held else None
    return {
        "as_of": as_of,
        "latest_held_date": latest,
        "latest_evidence_ids": list(
            dict.fromkeys(item["evidence_id"] for item in held if item["date"] == latest)
        ),
        "requested_context": list(dict.fromkeys(context or ())),
        "held_events": held,
        "scheduled_events": scheduled,
        "unresolved_evidence_ids": sorted(unresolved),
        "extraction": {
            "status": (
                "ok"
                if not missing and context is not None
                else "incomplete"
                if reviews
                else "invalid_format"
            ),
            "repair_attempted": repair_attempted,
            "issues": issues,
            "address_errors": dict(address_errors),
            "unparsed_evidence_ids": missing,
            "deferred_evidence_ids": deferred,
            "reused_evidence_ids": reused,
            "reviewed_evidence_ids": [key for key in pending if key in reviews],
            "batch_sizes": batch_sizes,
        },
    }
