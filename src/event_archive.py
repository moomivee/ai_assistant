"""Distinguish a documented historical occurrence from evidence that it is still the latest."""

from .clock import clock_today
from .config import settings
from .event_recovery import publication_query
from .event_search import requested_publication_window
from .event_subject import canonical_event_question, organizer_domain, requested_event_publisher
from .integrations.web import WEB_QUERY_LIMIT, allowed, publication_domains, web_date
from .models.schemas import NextAction

ARCHIVE_EVENT_NOTICE = "Более поздняя встреча не подтверждена. По архивным публикациям нельзя установить, что указанное событие было последним."


def archive_event_gap(state, validation):
    """An old press account establishes a date, but cannot cover the years since publication."""
    plan = state["plan"]
    if (
        not plan.get("latest_completed_event")
        or plan.get("report_only")
        or plan.get("fresh") is False
    ):
        return None
    question = state["question"]
    trusted = settings().domains
    source_limit, _ = requested_event_publisher(question, plan, trusted)
    if source_limit:
        return None
    if requested_publication_window(plan.get("event_publication_window"), question):
        return None
    subject = plan.get("context_request") or question
    organizer = organizer_domain(subject, plan, trusted, state.get("previous") or "")
    if len(set(trusted)) < 2:
        return None
    chronology = validation.get("event_chronology") or {}
    event_date = web_date(chronology.get("latest_held_date"))
    ids = chronology.get("latest_evidence_ids", [])
    if not event_date or not ids:
        return None
    today = clock_today()
    archive_ids = []
    publishers = set()
    for key in ids:
        block = state["evidence"].get(key)
        if not block or block.get("kind") != "web" or allowed(block.get("url", ""), (organizer,)):
            return None
        publisher = next(
            (d for d in sorted(trusted, key=len) if allowed(block.get("url", ""), (d,))), None
        )
        if not publisher:
            return None
        publishers.add(publisher)
        published = web_date(block.get("date"))
        if published is not None:
            if (today - published).days <= 365:
                return None
            archive_ids.append(key)
    if not archive_ids:
        return None
    omit = next(iter(publishers)) if len(publishers) == 1 else ""
    if omit:
        try:
            publication_domains("", trusted, omit=omit)
        except ValueError:
            return None
    return {
        "organizer_domain": organizer,
        "archive_publisher": omit,
        "event_date": event_date.isoformat(),
        "evidence_ids": archive_ids,
        "notice": ARCHIVE_EVENT_NOTICE,
        "dated_claim": {
            "text": f"В архивной публикации зафиксирована дата состоявшегося события — {event_date:%d.%m.%Y}.",
            "evidence_ids": archive_ids[:12],
        },
    }


def archive_verification_search(state, gap):
    """Check the organizer, then other publishers once, within the agent's ordinary step budget."""
    if not gap:
        return None
    searches = [
        step["arguments"]
        for step in state.get("observations", [])
        if step["tool"] == "web_search" and step["arguments"].get("search")
    ]
    targeted = {args.get("publisher_domain") for args in searches}
    organizer = gap["organizer_domain"]
    omit = gap["archive_publisher"]
    question = canonical_event_question(state["plan"].get("context_request") or state["question"])
    if organizer and organizer not in targeted:
        scope = {"publisher_domain": organizer}
        purpose = "Проверить по публикациям организатора, состоялось ли более позднее событие."
        query = publication_query(
            question[:WEB_QUERY_LIMIT], question=question, organizer_domain=organizer
        )
    elif omit and not any(args.get("omit_publisher_domain") for args in searches):
        alternatives = publication_domains("", settings().domains, omit=omit)
        if set(alternatives) <= targeted:
            return None
        scope = {"omit_publisher_domain": omit}
        purpose = "Проверить дату по независимым публикациям других разрешённых издателей."
        query = question[:WEB_QUERY_LIMIT]
    else:
        return None
    decision = NextAction(action="web_search", purpose=purpose, query=query).model_dump()
    return {
        **decision,
        **scope,
        "origin": "event_recovery",
        "controller_note": "Архивная публикация подтверждает дату события, но не его актуальность.",
    }
