"""Choose a new publication lead after inconclusive latest-event discovery."""

import re
from datetime import date
from urllib.parse import urlsplit

from .clock import clock_today
from .config import settings
from .event_subject import organizer_search_terms
from .integrations.web import WEB_QUERY_LIMIT, allowed, publication_domains, web_date
from .validation import normalized


def _search_key(arguments):
    # A different retrieval depth alone does not address the missing event date.
    return (
        normalized(arguments.get("query", "")),
        arguments.get("publisher_domain") or "",
        arguments.get("omit_publisher_domain") or "",
        arguments.get("topic") or "general",
        arguments.get("days"),
        arguments.get("start_date"),
        arguments.get("end_date"),
    )


def _minor_rewording(first, second):
    if first[1:] != second[1:]:
        return False
    a, b = set(first[0].split()), set(second[0].split())
    changed = a ^ b
    # Reordered words or a speculative year do not introduce a new lead.
    # One substantive word (e.g. the name of a committee) certainly can.
    return bool(a and b) and (
        not changed
        or (len(changed) == 1 and all(len(token) == 4 and token.isdecimal() for token in changed))
    )


def _publisher(url, domains):
    # Group e.g. www and mobile subdomains under the configured publisher.
    return next(
        (domain for domain in sorted(domains, key=len) if allowed(url, (domain,))),
        urlsplit(url).hostname,
    )


def _archive_path_date(url):
    """An editorial date in a URL is a discovery hint only; never cite or store it."""
    path = urlsplit(url).path
    match = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$)", path)
    if match:
        year, month, day = match.groups()
    elif match := re.search(r"/(\d{1,2})/(\d{1,2})/(\d{4})(?:/|$)", path):
        day, month, year = match.groups()
    else:
        return None
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _topic_index(url):
    # Tag landing pages aggregate otherwise unrelated headlines and story dates.
    return bool(re.search(r"/(?:tags?|topics?)/[^/]+/?$", urlsplit(url).path, re.I))


def _chronology_leads(state, validation):
    """Source IDs of event-specific blocks; general topic relevance alone is insufficient."""
    chronology = validation.get("event_chronology") or {}
    blocks = set(chronology.get("unresolved_evidence_ids", ()))
    blocks.update(chronology.get("reported_excerpts", {}))
    blocks.update(row.get("evidence_id") for row in chronology.get("scheduled_leads", ()))
    blocks.difference_update(chronology.get("other_scope_blocks", ()))
    return {state.get("evidence", {}).get(key, {}).get("source") for key in blocks} - {None}


def publication_query(query, *, organizer_domain="", question="", language="ru"):
    """Preserve the query and add publication vocabulary for the selected site."""
    alias = organizer_search_terms(question, organizer_domain) if organizer_domain else ""
    english = organizer_domain in {"opec.org", "iea.org", "eia.gov"} or (
        language == "en" and organizer_domain != "cbr.ru"
    )
    terms = "official communiqué" if english else "итоговое коммюнике организатора"
    additions = [
        term for term in (alias, terms) if term and normalized(term) not in normalized(query)
    ]
    if not additions:
        return query
    suffix = " " + " ".join(additions)
    return query[: WEB_QUERY_LIMIT - len(suffix)].rstrip() + suffix


def recover_event_search(decision, context, state, actions):
    """Keep novel actions; spend recovery steps on distinct, relevant evidence."""
    requested_domain = context.get("requested_domain")
    if context.get("source_limit") and not requested_domain:
        # A free-form constraint can exclude publishers or allow multiple sites.
        # Do not invent a different scope for the model's proposed action.
        return decision
    validation = state["answer_checks"][-1].get("validation", {}) if state["answer_checks"] else {}
    if not context["followup"] or validation.get("reason") not in {
        "no_confirmed_event",
        "no_relevant_event_sources",
        "no_evidence",
    }:
        return decision
    searches = [
        (step, row)
        for step, row in enumerate(state["observations"], 1)
        if row["tool"] == "web_search" and row["arguments"].get("search") and not row.get("error")
    ]
    last = state["observations"][-1] if state["observations"] else {}
    organizer = context.get("organizer_domain")
    empty_primary = bool(
        organizer
        and not context.get("source_limit")
        and last.get("tool") == "web_search"
        and last["arguments"].get("publisher_domain") == organizer
        and (
            last.get("new_evidence") is False or last.get("result_count") == 0 or last.get("error")
        )
    )
    key = _search_key(decision)
    read = decision["action"] == "read_web"
    finish = decision["action"] == "finish"
    if finish:
        matches = searches
    elif read:
        page_ids = {s["id"] for s in state["sources"] if s.get("url") == decision["url"]}
        matches = [row for row in searches if page_ids.intersection(row[1].get("source_ids", []))]
    else:
        matches = [
            row for row in searches if _minor_rewording(_search_key(row[1]["arguments"]), key)
        ]
        if not matches and empty_primary and _minor_rewording(_search_key(last["arguments"]), key):
            matches = searches
    if not matches:
        return decision
    step, previous = matches[-1]
    # A repeated lookup at an empty official site or a premature finish may
    # refer back to no hits at all. Compare it with the earlier broad discovery.
    if empty_primary and previous["arguments"].get("publisher_domain") == organizer:
        step, previous = next(
            (
                row
                for row in reversed(searches)
                if not row[1]["arguments"].get("publisher_domain")
                and not row[1]["arguments"].get("omit_publisher_domain")
            ),
            (step, previous),
        )
    original = previous["arguments"]
    base = original if read or finish or empty_primary else decision
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
    if read:
        before["url"] = decision["url"]
    filters = {
        field: base.get(field)
        for field in ("publisher_domain", "days", "start_date", "end_date", "topic", "search_depth")
    }
    domains = context.get("trusted_publishers", ())
    publisher = requested_domain or base.get("publisher_domain")
    if requested_domain:
        filters["publisher_domain"] = requested_domain
    if organizer not in domains or (publisher and publisher != organizer):
        organizer = None

    relevance = validation.get("event_relevance", {})
    source_ids = set(previous.get("source_ids", ()))
    # A later primary or independent search can contribute a better lead even
    # when the model proposes its original broad query again.
    for lookup_step, row in searches:
        args = row["arguments"]
        new_scope = (organizer and args.get("publisher_domain") == organizer) or (
            not publisher and args.get("omit_publisher_domain")
        )
        if (
            lookup_step > step
            and new_scope
            and all(args.get(k) == filters[k] for k in ("days", "start_date", "end_date"))
        ):
            source_ids.update(row.get("source_ids", ()))
    ids = source_ids.intersection(relevance.get("kept_source_ids", ()))
    ids.difference_update(relevance.get("unverified_source_ids", ()))
    batch = [
        s
        for s in state["sources"]
        if s["id"] in ids
        and s["id"] in context["publisher_pages"].get(s.get("url"), ())
        and s.get("content_type") == "snippet"
        and (not publisher or allowed(s["url"], (publisher,)))
    ]
    opened = {
        row["arguments"]["url"]
        for row in state["observations"]
        if row["tool"] == "read_web" and row["arguments"].get("url")
    }
    opened.update(s["url"] for s in state["sources"] if s.get("content_type") == "extracted")
    opened_publishers = {_publisher(url, domains) for url in opened}
    today = clock_today()

    # A year-old publication can document the latest event. Its age is a reason
    # to seek independent evidence, never an event date or an exclusion window.
    def archived(source):
        day = web_date(source.get("date")) or _archive_path_date(source["url"])
        return (
            day is not None
            and (today - day).days > 365
            and not any(filters[k] for k in ("start_date", "end_date", "days"))
        )

    def primary(source):
        return bool(organizer and allowed(source["url"], (organizer,)))

    candidates = [
        s
        for s in batch
        if "read_web" in actions
        and s["url"] not in opened
        and (s.get("provider") == "tavily" or settings().web_key)
    ]
    leads = _chronology_leads(state, validation)
    batch_leads = ids.intersection(leads)

    def weak(source):
        return (
            archived(source)
            or _topic_index(source["url"])
            or bool(batch_leads and source["id"] not in leads and not web_date(source.get("date")))
        )

    candidates.sort(
        key=lambda s: (
            not weak(s),
            primary(s),
            _publisher(s["url"], domains) not in opened_publishers,
            web_date(s.get("date")) or date.min,
        ),
        reverse=True,
    )
    exhausted = not candidates or all(
        weak(s) or _publisher(s["url"], domains) in opened_publishers for s in candidates
    )
    empty_read = last["tool"] == "read_web" and (
        last.get("new_evidence") is False
        or last.get("result_count") == 0
        or bool(last.get("error"))
    )
    targeted = (
        empty_primary
        or bool(organizer)
        and any(row["arguments"].get("publisher_domain") == organizer for _, row in searches)
    )
    primary_candidate = next(
        (s for s in candidates if primary(s) and not _topic_index(s["url"])), None
    )
    # A failed speculative extraction consumes the same budget as a good read.
    # Reserve the next step for a different route before the no-progress limit.
    pivot = (
        organizer and not targeted and (exhausted or empty_read or finish) and not primary_candidate
    )
    # A one-off reformulation also works for an event with no known official
    # domain. Preserve any user-scoped publisher, query window and time budget.
    refocus = (
        empty_read
        and not primary_candidate
        and not any(web_date(s.get("date")) and not weak(s) for s in candidates)
    )
    press_lookup = (
        empty_primary
        and exhausted
        and not publisher
        and not any(row["arguments"].get("omit_publisher_domain") for _, row in searches)
        and any(site != organizer for site in domains)
    )
    if finish and not (pivot or primary_candidate or refocus or press_lookup):
        return decision
    exact = key == _search_key(original) and not read and not finish
    proposed = next((s for s in batch if s["url"] == decision.get("url")), None)
    weak_read = read and (
        proposed is None
        or weak(proposed)
        or _publisher(proposed["url"], domains) in opened_publishers
    )
    if not exact and not exhausted and not weak_read and not (pivot or refocus or finish):
        return decision

    repair = {
        "mode": "event_recovery",
        "repeated_step": step,
        "before": before,
        "normalization": decision.get("argument_repair"),
    }
    omit = ""
    if (
        exhausted
        and not pivot
        and not publisher
        and len(set(domains)) >= 2
        and not any(filters[k] for k in ("days", "start_date", "end_date"))
        and not any(row["arguments"].get("omit_publisher_domain") for _, row in searches)
    ):
        batch_ids = set(previous.get("source_ids", ()))
        publishers = {
            _publisher(s["url"], domains)
            for s in state["sources"]
            if s["id"] in batch_ids and s.get("url") and allowed(s["url"], domains)
        }
        if len(publishers) == 1:
            omit = publishers.pop()
    if press_lookup and not omit:
        omit = organizer
    if omit:
        try:
            publication_domains("", tuple(domains), omit=omit)
        except ValueError:
            omit = ""
    if read and not (pivot or refocus or omit) and proposed and proposed["url"] not in opened:
        # A deliberately selected, unopened page may answer a different part of
        # the question. Keep it unless we have genuinely better evidence.
        alternative = next((s for s in candidates if s["url"] != proposed["url"]), None)
        if not weak_read or not alternative or (archived(alternative) and not primary(alternative)):
            return decision
        if not archived(proposed) and not primary(alternative):
            return decision
    candidate = (
        primary_candidate
        if finish
        else next(
            (s for s in candidates if _publisher(s["url"], domains) not in opened_publishers), None
        )
    )
    if candidate and not (pivot or refocus or omit):
        if read and candidate["url"] == decision["url"]:
            return decision
        after = {"action": "read_web", "url": candidate["url"], "query": ""}
        repair["source_id"] = candidate["id"]
        purpose = "Открыть найденную публикацию и проверить, указаны ли в ней даты события."
    else:
        if press_lookup and omit:
            alias = organizer_search_terms(state.get("question", ""), organizer)
            terms = (
                f" {alias} event date outcome news coverage"
                if alias
                else " дата и итоги события публикации СМИ"
            )
            query = base["query"][: WEB_QUERY_LIMIT - len(terms)].rstrip() + terms
        else:
            query = publication_query(
                base["query"],
                organizer_domain=organizer if pivot else "",
                question=state.get("question", ""),
                language=decision.get("event_language", "ru"),
            )
        after = {"action": "web_search", **filters, "query": query}
        if pivot:
            after["publisher_domain"] = organizer
        elif omit:
            after["omit_publisher_domain"] = omit
        out_of_scope = (
            read and requested_domain and not allowed(decision["url"], (requested_domain,))
        )
        if "web_search" not in actions or (
            not out_of_scope
            and any(_search_key(row["arguments"]) == _search_key(after) for _, row in searches)
        ):
            return decision
        purpose = (
            "Проверить дату события в других разрешённых изданиях."
            if omit
            else "Найти итоговое коммюнике организатора с датой состоявшегося события."
        )
    repair["after"] = after
    return {
        **decision,
        **after,
        "purpose": purpose,
        "origin": "event_recovery",
        "model_action": decision["action"],
        "model_purpose": decision["purpose"],
        "controller_note": f"Материалы поиска на шаге {step} не подтвердили дату события.",
        "argument_repair": repair,
    }
