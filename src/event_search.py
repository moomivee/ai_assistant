"""Keep follow-up publication searches grounded in the original event question."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field

from .config import settings
from .integrations.web import WEB_QUERY_LIMIT, allowed
from .models.schemas import NextAction
from .validation import normalized


class EventSearchAction(NextAction):
    event_search_goal: Literal["outcome", "archive", "coverage"] = Field(
        default="outcome", description="Для повторного веб-поиска выбери тип нужной публикации."
    )
    event_language: Literal["ru", "en"] = Field(
        default="ru", description="Язык поисковых терминов для выбранного типа публикации."
    )
    event_publisher_id: str = Field(
        default="",
        max_length=32,
        description="ID найденного веб-источника для поиска по его сайту. Пусто для поиска среди всех доверенных издателей.",
    )
    event_lead_date: str = Field(
        default="",
        max_length=10,
        description="Только для проверки конкретного события: одна из lead_dates. Для открытого поиска оставь пустым.",
    )
    event_subject: str = Field(
        default="",
        max_length=200,
        description="При длинном вопросе скопируй краткое название события или участников. Если это продолжение, можно взять название из предыдущего вопроса.",
    )


_PUBLICATION_TERMS = {
    "outcome": {"ru": "официальное сообщение об итогах", "en": "official statement results"},
    "archive": {"ru": "архив официальных сообщений", "en": "official press releases archive"},
    "coverage": {"ru": "итоги независимые сообщения", "en": "outcome news coverage"},
}


def event_search_context(state):
    plan = state["plan"]
    if (
        not plan.get("latest_completed_event")
        or plan.get("report_only")
        or state["draft"]["complete"]
    ):
        return None
    if not any(
        observation["tool"] in ("report_search", "read_report", "web_search", "read_web")
        and not observation.get("error")
        and (observation["tool"] != "web_search" or observation["arguments"].get("search"))
        for observation in state["observations"]
    ):
        return None
    chronology = next(
        (
            check["validation"]["event_chronology"]
            for check in reversed(state["answer_checks"])
            if check.get("validation", {}).get("event_chronology")
        ),
        {},
    )
    lead_dates = set(chronology.get("overdue_scheduled_dates", []))
    if date := chronology.get("latest_held_date"):
        lead_dates.add(date)
    publisher_pages = {}
    for source in state["sources"]:
        if source["kind"] == "web" and allowed(source.get("url", ""), settings().domains):
            publisher_pages.setdefault(source["url"], []).append(source["id"])
    return {
        "subject": plan.get("context_request") or state["question"],
        "lead_dates": sorted(lead_dates),
        "publisher_pages": publisher_pages,
    }


def event_publication_search(decision, context, state):
    """Encode a discovery objective or a verifiable dated lead as a publication query."""
    goal = decision.get("event_search_goal", "outcome")
    language = decision.get("event_language", "ru")
    subject = " ".join(context["subject"].split())
    term = " ".join(decision.get("event_subject", "").split())
    selected = normalized(term)
    if selected and any(
        selected in normalized(text) for text in (subject, state.get("previous") or "")
    ):
        if selected not in normalized(subject):
            subject = f"{term} {subject}"
    else:
        term = ""
    qualifiers = [_PUBLICATION_TERMS[goal][language]]
    date = decision.get("event_lead_date", "")
    if date in context["lead_dates"]:
        qualifiers.append(date)
    publisher = decision.get("event_publisher_id", "")
    for url, source_ids in context["publisher_pages"].items():
        if publisher in source_ids:
            qualifiers.append("site:" + urlsplit(url).hostname)
            break
    suffix = " " + " ".join(qualifiers)
    available = WEB_QUERY_LIMIT - len(suffix)
    if len(subject) > available:
        subject = (term or subject)[:available].rstrip()
    # A lead is an event date, never a publication window. Retrospective results
    # and later cancellations/corrections must remain eligible for the same query.
    before = {k: decision.get(k) for k in ("query", "days", "start_date", "end_date")}
    after = {"query": subject + suffix, "days": None, "start_date": None, "end_date": None}
    return {
        **decision,
        **after,
        "argument_repair": {"mode": "event_search", "before": before, "after": after},
    }
