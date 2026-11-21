"""Compare dated occurrences in the supplied evidence for a latest-event question."""

import re
from collections import defaultdict
from datetime import date
from typing import Literal

from pydantic import Field

from .clock import clock_payload
from .models.schemas import StrictModel


class EventCandidate(StrictModel):
    kind: Literal["held", "scheduled"]
    date: str
    quote: str
    scope: str
    match: Literal["in_scope", "different_scope", "uncertain"]


class EventBlock(StrictModel):
    evidence_id: str
    status: Literal["assessed", "uncertain"]
    events: list[EventCandidate]


class EventChronology(StrictModel):
    blocks: list[EventBlock]
    requested_context: list[Literal["next_scheduled", "previous_held"]] = Field(
        default_factory=list
    )


# The model classifies events and their scope. Code checks that its ISO date
# is an actual full date in the cited passage, including localized month names.
_MONTHS = (
    "january jan января янв",
    "february feb февраля фев февр",
    "march mar марта мар",
    "april apr апреля апр",
    "may мая",
    "june jun июня июн",
    "july jul июля июл",
    "august aug августа авг",
    "september sept sep сентября сен сент",
    "october oct октября окт",
    "november nov ноября ноя нояб",
    "december dec декабря дек",
)
_MONTH_NUMBER = {word: i for i, names in enumerate(_MONTHS, 1) for word in names.split()}
_MONTH = "(?:" + "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(\d{4}))?(?!\w)"
_DAY_FIRST = re.compile(rf"(?<!\w){_DAY}\s+(?:of\s+)?({_MONTH}){_YEAR}", re.I)
_MONTH_FIRST = re.compile(rf"(?<!\w)({_MONTH})\s+{_DAY}{_YEAR}", re.I)


def mentioned_dates(text: str, *, default_year: int | None = None) -> set[date]:
    """Find explicit dates; a caller can supply a year for day + named month in user-facing prose."""
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.I)
    parts = [
        (year, month, day)
        for year, month, day in re.findall(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", text)
    ]
    parts.extend(
        (year, month, day)
        for day, month, year in re.findall(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)", text)
    )
    parts.extend(
        (year or default_year, _MONTH_NUMBER[month.lower().rstrip(".")], day)
        for day, month, year in _DAY_FIRST.findall(text)
    )
    parts.extend(
        (year or default_year, _MONTH_NUMBER[month.lower().rstrip(".")], day)
        for month, day, year in _MONTH_FIRST.findall(text)
    )
    found = set()
    for year, month, day in parts:
        if not year:
            continue
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return found


def build_event_chronology(client, question, evidence, *, previous=None):
    clock = clock_payload()
    as_of = clock["today"]
    payload = {
        "question": question,
        **({"previous": previous} if previous else {}),
        **clock,
        "evidence": {
            key: {
                "text": block["text"],
                "title": block.get("title"),
                "url": block.get("url"),
                "publication_date": block.get("date"),
            }
            for key, block in evidence.items()
        },
    }
    chronology = (
        client.generate("event_chronology", payload, EventChronology)
        if evidence
        else EventChronology(blocks=[])
    )
    reviews_by_id = defaultdict(list)
    for block in chronology.blocks:
        reviews_by_id[block.evidence_id].append(block)
    unresolved = set()
    held, scheduled = [], []
    for key, source in evidence.items():
        reviews = reviews_by_id[key]
        if len(reviews) != 1 or reviews[0].status == "uncertain":
            unresolved.add(key)
            continue
        original = " ".join(source["text"].split())
        for item in reviews[0].events:
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
        "requested_context": list(dict.fromkeys(chronology.requested_context)),
        "held_events": held,
        "scheduled_events": scheduled,
        "unresolved_evidence_ids": sorted(unresolved),
    }
