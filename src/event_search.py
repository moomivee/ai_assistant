"""Keep follow-up publication searches grounded in the original event question."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field

from .clock import clock_today
from .config import settings
from .event_dates import open_event_query
from .integrations.web import WEB_QUERY_LIMIT, allowed
from .models.schemas import NextAction
from .validation import normalized


class EventSearchAction(NextAction):
    publisher_domain: str = Field(
        default="",
        max_length=253,
        description="Домен первоисточника из trusted_publishers. Можно выбрать до первого результата. Пусто для поиска по всем разрешённым издателям.",
    )
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
    followup = any(
        observation["tool"] in ("report_search", "read_report", "web_search", "read_web")
        and not observation.get("error")
        and (observation["tool"] != "web_search" or observation["arguments"].get("search"))
        for observation in state["observations"]
    )
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
    trusted = settings().domains
    validation = state["answer_checks"][-1].get("validation", {}) if state["answer_checks"] else {}
    excluded = set(validation.get("event_relevance", {}).get("excluded_source_ids", []))
    publisher_pages = {}
    for source in state["sources"]:
        if (
            source["kind"] == "web"
            and source["id"] not in excluded
            and allowed(source.get("url", ""), trusted)
        ):
            publisher_pages.setdefault(source["url"], []).append(source["id"])
    return {
        "trusted_publishers": list(trusted),
        "followup": followup,
        "lead_dates": sorted(lead_dates),
        "publisher_pages": publisher_pages,
    }


def event_publication_search(decision, context, state):
    """Preserve the model's search terms and pass the publisher as a provider filter."""
    before = {
        k: decision.get(k) for k in ("query", "publisher_domain", "days", "start_date", "end_date")
    }
    question = state["plan"].get("context_request") or state["question"]
    term = " ".join(decision.get("event_subject", "").split())
    if not any(normalized(term) in normalized(t) for t in (question, state.get("previous") or "")):
        term = ""
    lead = decision.get("event_lead_date", "")
    if lead not in context["lead_dates"]:
        lead = ""
    proposed = " ".join((before["query"] or "").split())
    initial_query = proposed or term or state["plan"]["rag_query"].strip()
    if proposed and term and normalized(term) not in normalized(proposed):
        initial_query = term + " " + proposed
    if not proposed and context["followup"] and initial_query:
        goal = decision.get("event_search_goal", "outcome")
        language = decision.get("event_language", "ru")
        initial_query += " " + _PUBLICATION_TERMS[goal][language]
    query = open_event_query(initial_query, question=question, lead_date=lead)
    # A confirmed lead is optional. No historical publication cutoff follows from it.
    suffix = " " + lead if lead and lead not in query else ""
    query = query[: WEB_QUERY_LIMIT - len(suffix)].rstrip() + suffix
    publisher = (before["publisher_domain"] or "").strip().lower()
    if publisher not in context["trusted_publishers"]:
        publisher = next(
            (
                urlsplit(url).hostname
                for url, ids in context["publisher_pages"].items()
                if decision.get("event_publisher_id") in ids
            ),
            "",
        )
    # Only the user's requested publication period carries over across searches.
    # With no explicit bounds, search_web accepts undated leads for page reading
    # and still rejects known publication dates later than the request date.
    start = end = None
    window = state["plan"].get("event_publication_window")
    if window and (quote := normalized(window["quote"])) and quote in normalized(state["question"]):
        start, end = window.get("start_date"), window.get("end_date")
        if end:
            end = min(end, clock_today().isoformat())
    after = {
        **before,
        "query": query,
        "publisher_domain": publisher,
        "days": None,
        "start_date": start,
        "end_date": end,
    }
    return {
        **decision,
        **after,
        "argument_repair": {"mode": "event_search", "before": before, "after": after},
    }
