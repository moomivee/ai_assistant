"""Ground event discovery in the named institution; distinguish joint and members-only events."""

import re


def canonical_event_question(text: str) -> str:
    """Normalize the Cyrillic spelling while retaining the user's participants and question."""
    return re.sub(r"\bопэк\b", "ОПЕК", text, flags=re.I)


# Publisher identity is stable across meetings. Results, dates, meeting numbers
# and even the members of a voluntary group must come from the publications.
_OPEC = r"(?:OPEC|ОПЕК|OPEP)"
_OPEC_FULL = r"(?:Organi[sz]ation of (?:the )?Petroleum Exporting Countries|Организаци\w* стран\s*[-–—]?\s*экспорт[её]ров нефти)"
_ALLIANCE = re.compile(
    rf"\b{_OPEC}(?:\s*\+|[ -]+(?:plus|плюс)\b|\s*(?:and|&|/)\s*non[-‑– ]OPEC\b"
    r"|\s+и\s+(?:стран\w*\s*,?\s*)?не\s+входящ\w*\s+в\s+ОПЕК\b)",
    re.I,
)
_MEMBERS = re.compile(rf"\b(?:{_OPEC}|{_OPEC_FULL})\b", re.I)
_CONFERENCE = re.compile(rf"\b(?:{_OPEC}\s+Conference|конференци\w*\s+{_OPEC})\b", re.I)
_MEETING = re.compile(
    r"\b(?:meeting|conference|ministerial|summit|съезд\w*|встреч\w*|заседани\w*|саммит\w*|конференци\w*)\b",
    re.I,
)
_REGISTRY = {
    "opec.org": _MEMBERS,
    "iea.org": re.compile(
        r"\b(?:IEA|МЭА|International Energy Agency|Международн\w+ энергетическ\w+ агентств\w*)\b",
        re.I,
    ),
    "eia.gov": re.compile(
        r"\b(?:EIA|(?:U\.?S\.? )?Energy Information Administration|Управлени\w+ энергетическ\w+ информаци\w+ США)\b",
        re.I,
    ),
    "cbr.ru": re.compile(
        r"\b(?:ЦБ РФ|ЦБ России|Банк\w* России|Банк\w* Российской Федерации|Bank of Russia)\b", re.I
    ),
}


def organizer_domain(question, plan, trusted, previous=""):
    """An official-site search is a recovery option, never an expansion of the allowlist."""
    if (limit := plan.get("event_source_limit")) and limit.casefold() in question.casefold():
        return ""
    question = canonical_event_question(question)
    domains = {domain for domain, pattern in _REGISTRY.items() if pattern.search(question)}
    suggestion = plan.get("event_organizer") or {}
    quote = canonical_event_question(suggestion.get("quote", "").strip())
    domain = suggestion.get("domain", "").strip().lower()
    query = canonical_event_question(plan.get("rag_query", ""))
    if quote and domain in trusted and quote.casefold() in question.casefold():
        # The model can identify an organizer explicitly named by hostname. A media
        # domain inferred solely by the model is not evidence that it ran the event.
        if re.search(rf"(?<![\w.-])(?:www\.)?{re.escape(domain)}(?![\w.-])", quote, re.I):
            domains.add(domain)
        elif not domains and domain in _REGISTRY and _REGISTRY[domain].search(query):
            # An explicitly grounded alias and its normalized planning query
            # agree on a known organizer (e.g. "МЭА" in a nonstandard spelling).
            domains.add(domain)
    if not domains and previous:
        # A reference like "when did they last meet?" can inherit an entity from
        # the plan only if present in the prior dialogue and in its new query.
        prior = canonical_event_question(previous)
        domains = {
            site
            for site, pattern in _REGISTRY.items()
            if pattern.search(query) and pattern.search(prior)
        }
    return next(iter(domains)) if len(domains) == 1 and domains <= set(trusted) else ""


def _bodies(text):
    canonical = canonical_event_question(text)
    members_text, count = _ALLIANCE.subn(" ", canonical)
    return bool(_MEMBERS.search(members_text)), bool(count)


def other_event_body(question, title, excerpt):
    """Reject explicit cross-body substitutions; mixed/ambiguous articles still get reviewed."""
    question = canonical_event_question(question)
    title = canonical_event_question(title)
    excerpt = canonical_event_question(excerpt)
    requested = [site for site, name in _REGISTRY.items() if name.search(question)]
    if len(requested) == 1 and _MEETING.search(title):
        target = _REGISTRY[requested[0]]
        if (
            not target.search(title)
            and not target.search(excerpt)
            and any(name.search(title) for site, name in _REGISTRY.items() if site != requested[0])
        ):
            return True
    members, alliance = _bodies(question)
    if members == alliance:
        return False
    headline_members, headline_alliance = _bodies(title)
    body_members, body_alliance = _bodies(excerpt)
    if members:
        return headline_alliance and not (headline_members or body_members)
    # News sometimes shortens the alliance's name to OPEC. Only the explicitly
    # named members-only Conference establishes a contradictory scope here.
    return bool(_CONFERENCE.search(title)) and not (headline_alliance or body_alliance)


def organizer_search_terms(question, domain):
    """English terms for the organizer's own archive; keep any specific group in scope."""
    question = canonical_event_question(question)
    if domain not in _REGISTRY or not _REGISTRY[domain].search(question):
        return ""
    if domain != "opec.org":
        return {"iea.org": "IEA", "eia.gov": "EIA", "cbr.ru": "Bank of Russia"}[domain]
    members, alliance = _bodies(question)
    if members and not alliance:
        return "OPEC Conference" if _CONFERENCE.search(question) else "OPEC"
    if alliance and not members:
        if re.search(
            r"\b(?:JMMC|Joint Ministerial Monitoring Committee|мониторингов\w* комитет\w*)\b",
            question,
            re.I,
        ):
            return "OPEC+ Joint Ministerial Monitoring Committee"
        if re.search(r"\b(?:добровольн\w*|voluntar\w*)\b", question, re.I):
            return "OPEC+ voluntary production adjustments"
        if re.search(r"\b(?:министерск\w*|ministerial)\b", question, re.I):
            return "OPEC and non-OPEC Ministerial Meeting"
        return "OPEC+"
    return ""


# Used only for exact single-publisher instructions (not topic classification).
# A quoted multi-publisher or negative restriction remains free-form.
_PUBLISHER_ALIASES = {
    "rbc.ru": ("РБК", "RBC"),
    "reuters.com": ("Reuters", "Рейтер", "Рейтерс"),
    "interfax.ru": ("Интерфакс", "Interfax"),
    "ft.com": ("Financial Times", "FT"),
    "theguardian.com": ("The Guardian", "Guardian"),
    "euronews.com": ("Euronews", "Евроньюс"),
    "opec.org": ("ОПЕК", "ОПЕК+", "OPEC", "OPEC+"),
    "iea.org": ("МЭА", "IEA"),
    "eia.gov": ("EIA",),
}
_ONLY_PUBLISHER = re.compile(
    r"(?:используй(?:те)?\s+|ищи(?:те)?\s+|смотри(?:те)?\s+|use\s+)?"
    r"(?:только|исключительно|only)\s+"
    r"(?:(?:по данным|на сайте|материалы|публикации|источник|в|из|по|from|on)\s+)?"
    r"(?P<name>[^,;:!?.\n]+(?:\.[a-z]{2,})?)",
    re.I,
)


def requested_event_publisher(question, plan, trusted):
    """A name is an exclusive publisher only if the user's words explicitly make it one."""
    clause = (plan.get("event_source_limit") or "").strip()
    if clause.casefold() not in question.casefold():
        clause = ""
    aliases = {
        spelling.casefold(): domain
        for domain in trusted
        for spelling in (domain, *_PUBLISHER_ALIASES.get(domain, ()))
    }
    wording = clause or question
    match = _ONLY_PUBLISHER.search(wording)
    if match:
        name = match["name"].strip(' «»"').casefold()
        # A comma can continue a list. A negated "only" is not an exclusive
        # attribution even when just one recognizable outlet follows it.
        position = question.casefold().find(match[0].casefold())
        negated = re.search(r"\b(?:не|not)\s+$", question[:position], re.I)
        tail = wording[match.end() :].lstrip()
        if name in aliases and not negated and (not tail or tail[0] in ".!?;"):
            return clause or match[0], aliases[name]
    return clause, ""
