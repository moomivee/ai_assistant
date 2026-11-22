"""Recover one useful action when event discovery repeats an inconclusive search."""

from .config import settings
from .integrations.web import WEB_QUERY_LIMIT, allowed
from .validation import normalized


def _search_key(arguments):
    # A different retrieval depth alone does not address the missing event date.
    return (
        normalized(arguments.get("query", "")),
        arguments.get("publisher_domain") or "",
        arguments.get("topic") or "general",
        arguments.get("days"),
        arguments.get("start_date"),
        arguments.get("end_date"),
    )


def recover_event_search(decision, context, state, actions):
    """Leave new proposals alone; use validation feedback only for a repeated search."""
    validation = state["answer_checks"][-1].get("validation", {}) if state["answer_checks"] else {}
    if not context["followup"] or validation.get("reason") != "no_confirmed_event":
        return decision
    searches = [
        (step, row)
        for step, row in enumerate(state["observations"], 1)
        if row["tool"] == "web_search" and row["arguments"].get("search")
    ]
    key = _search_key(decision)
    repeated = next(
        (
            (step, row)
            for step, row in reversed(searches)
            if not row.get("error") and _search_key(row["arguments"]) == key
        ),
        None,
    )
    if not repeated:
        return decision
    step, previous = repeated
    before = {
        field: decision.get(field)
        for field in (
            "action",
            "query",
            "publisher_domain",
            "days",
            "start_date",
            "end_date",
            "topic",
            "search_depth",
        )
    }
    repair = {
        "mode": "event_recovery",
        "repeated_step": step,
        "before": before,
        "normalization": decision.get("argument_repair"),
    }
    opened = {
        row["arguments"].get("url") for row in state["observations"] if row["tool"] == "read_web"
    }
    # Extract may have followed a redirect from a different discovered URL.
    opened.update(s["url"] for s in state["sources"] if s.get("content_type") == "extracted")
    relevance = validation.get("event_relevance", {})
    candidates = (
        set(previous.get("source_ids", []))
        .intersection(relevance.get("kept_source_ids", []))
        .difference(relevance.get("unverified_source_ids", []))
    )
    publisher = decision.get("publisher_domain")
    for source in state["sources"]:
        url = source.get("url")
        if (
            "read_web" in actions
            and source["id"] in candidates
            and source["id"] in context["publisher_pages"].get(url, [])
            and source.get("content_type") == "snippet"
            and (source.get("provider") == "tavily" or settings().web_key)
            and url not in opened
            and (not publisher or allowed(url, (publisher,)))
        ):
            after = {"action": "read_web", "url": url, "query": ""}
            repair["source_id"] = source["id"]
            purpose = "Открыть найденную публикацию и проверить, указаны ли в ней даты события."
            break
    else:
        terms = (
            "organizer official communiqué"
            if decision.get("event_language") == "en"
            else "итоговое коммюнике организатора"
        )
        query = decision["query"]
        if normalized(terms) in normalized(query):
            return decision
        query = query[: WEB_QUERY_LIMIT - len(terms) - 1].rstrip() + " " + terms
        after = {**before, "query": query}
        if any(_search_key(row["arguments"]) == _search_key(after) for _, row in searches):
            return decision
        purpose = "Найти итоговое коммюнике организатора с датой состоявшегося события."
    repair["after"] = after
    return {
        **decision,
        **after,
        "purpose": purpose,
        "origin": "event_recovery",
        "model_action": decision["action"],
        "model_purpose": decision["purpose"],
        "controller_note": f"Поиск повторяет шаг {step}. Проверка не подтвердила дату события.",
        "argument_repair": repair,
    }
