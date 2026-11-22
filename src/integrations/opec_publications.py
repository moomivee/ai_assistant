"""Read meeting communiqués from the organizer's press index when explicitly requested."""

import asyncio
import re
from contextlib import closing
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import urljoin

import httpx
from lxml import etree, html

from ..budget import check_budget
from ..event_subject import canonical_event_question, organizer_search_terms
from ..logging import log

_INDEX = "https://www.opec.org/press-releases.html"
_RELEASE = re.compile(r"https://www\.opec\.org/pr-detail/[A-Za-z0-9-]+\.html")
_ANNUAL = re.compile(r"https://www\.opec\.org/press-releases-(\d{4})\.html")
_CONTENT_LIMIT = 6000
_MAX_DETAILS = 4
_SCOPE = {
    "OPEC": {"conference"},
    "OPEC Conference": {"conference"},
    "OPEC+": {"voluntary", "ministerial", "monitoring", "joint"},
    "OPEC+ voluntary production adjustments": {"voluntary"},
    "OPEC and non-OPEC Ministerial Meeting": {"ministerial"},
    "OPEC+ Joint Ministerial Monitoring Committee": {"monitoring"},
}


# The archive is a catalog of recurring internal meetings. A bilateral meeting,
# a named workshop, or a question about an unrelated occurrence needs an open search.
_CALENDAR_WORD = re.compile(
    r"(?:когда|где|как(?:ой|ого|ая|ие)?|дней|день|дат[а-я]*|числ[а-я]*|назов[а-я]*|укаж[а-я]*|"
    r"был[а-я]*|прош[а-я]*|прох[а-я]*|состоя[а-я]*|соб[а-я]*|пров[а-я]*|планир[а-я]*|"
    r"назнач[а-я]*|последн[а-я]*|предыдущ[а-я]*|следующ[а-я]*|очередн[а-я]*|актуальн[а-я]*|"
    r"встреч[а-я]*|заседа[а-я]*|совещани[а-я]*|конференц[а-я]*|министерск[а-я]*|министр[а-я]*|"
    r"мониторингов[а-я]*|комитет[а-я]*|добровольн[а-я]*|групп[а-я]*|участник[а-я]*|стран[а-я]*|"
    r"реш[а-я]*|итог[а-я]*|последстви[а-я]*|результа[а-я]*|квот[а-я]*|добыч[а-я]*|"
    r"опек|плюс|что|каков[а-я]*|на|в|у|и|по|её|ее|этом|ней|нем|ли|их|они|это|году|раз|там|"
    r"when|where|was|were|will|be|did|do|does|the|a|of|at|by|on|in|and|their|it|what|which|"
    r"take|place|last|latest|most|recent|previous|next|date|outcomes?|decisions?|held|"
    r"meet(?:s|ings?)?|met|conferences?|ministerial|ministers?|voluntary|participating|"
    r"producers?|countries|group|joint|monitoring|committee|jmmc|opec|non|plus)"
)
_MEETING = re.compile(
    r"встреч|заседан|совещани|конференц|собира|собра|комитет|meet|met\b|conference|jmmc", re.I
)


def _families(query):
    # Unrecognized subjects (e.g. another country, a named congress) must retain
    # their qualifier in Tavily rather than be silently broadened into this catalog.
    words = re.findall(r"[^\W\d_]+", canonical_event_question(query).casefold())
    if not _MEETING.search(query) or any(not _CALENDAR_WORD.fullmatch(word) for word in words):
        return None
    return _SCOPE.get(organizer_search_terms(query, "opec.org"))


def _family(title):
    title = title.casefold()
    if "non-opec ministerial meeting" in title:
        return "ministerial"
    if "ministerial monitoring committee" in title or "jmmc" in title:
        return "monitoring"
    if re.search(r"\bopec conference\b|\bconference of (?:the )?opec member", title):
        return "conference"
    if "declaration of cooperation" in title and "meeting" in title:
        return "joint"
    # Voluntary producers' headlines often name the participants without saying
    # OPEC+. Confirm that attribution from the article before making it evidence.
    if "voluntary" in title or (
        title.count(",") >= 2 and ("market stability" in title or "production" in title)
    ):
        return "voluntary"
    if "opec+" in title and "meet" in title:
        return "joint"
    return None


async def _fetch_html(client, url, deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        return b""
    check_budget()
    try:
        async with asyncio.timeout(remaining):
            async with client.stream(
                "GET", url, timeout=remaining, follow_redirects=False
            ) as response:
                response.raise_for_status()
                if "text/html" not in response.headers.get("content-type", "").lower():
                    return b""
                buf = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(buf) + len(chunk) > 1024 * 1024:
                        return b""
                    buf.extend(chunk)
                return bytes(buf)
    except (TimeoutError, httpx.HTTPError) as exc:
        log("web.official_read_failed", provider="opec_press", error_type=type(exc).__name__)
        return b""


async def _page(client, url, deadline):
    raw = await _fetch_html(client, url, deadline)
    if raw:
        try:
            return html.fromstring(raw, parser=html.HTMLParser(encoding="utf-8", no_network=True))
        except (etree.ParserError, ValueError):
            pass
    return None


def _entries(page, families, start, end):
    rows = []
    for item in page.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " pritem ")]'):
        title = item.xpath("normalize-space(h1)")
        family = _family(title)
        url = urljoin(_INDEX, item.xpath("string(a/@href)"))
        if family not in families or not _RELEASE.fullmatch(url):
            continue
        try:
            day = datetime.strptime(
                item.xpath("normalize-space(p[1])").split("|")[0].strip(), "%d %B %Y"
            ).date()
        except ValueError:
            continue
        if day <= end and (not start or start <= day):
            rows.append((day, family, url))
    return sorted(rows, key=lambda row: row[0], reverse=True)


async def _article(client, url, family, published, deadline):
    page = await _page(client, url, deadline)
    if page is None:
        return None
    title = page.xpath('normalize-space(//article//*[@id="articleHeadline"])')
    body = page.xpath('//article//*[@id="articleContent"]')
    if not body or _family(title) != family:
        return None
    for node in body[0].xpath(".//script|.//style"):
        node.drop_tree()
    paragraphs = body[0].xpath(".//p|.//li[not(.//p)]") or body
    lines = [" ".join(p.text_content().split()) for p in paragraphs]
    intro = page.xpath('normalize-space(//article//*[@id="articleIntro"])')
    text = "\n".join(s for s in [intro, *lines] if s)
    if not text:
        return None
    if family == "voluntary" and not re.search(r"\bOPEC\s*\+|\bnon[- ]OPEC\b", text, re.I):
        return None
    return dict(
        kind="web",
        provider="opec_press",
        title=title[:190],
        url=url,
        date=published.isoformat(),
        date_basis="publisher_archive",
        text=text[:_CONTENT_LIMIT],
        content_type="extracted",
        content_truncated=len(text) > _CONTENT_LIMIT,
        content_limit=_CONTENT_LIMIT,
        retrieved_at=datetime.now(UTC).isoformat(),
        cache_hit=False,
    )


def search_opec_releases(query, *, limit, start, end, today, deadline):
    """Return dated publisher text, never promoting an index headline to an event."""
    families = _families(query)
    if not families or limit <= 0 or deadline <= monotonic():
        return []
    headers = {"User-Agent": "oil-gas-analyst (+https://github.com/moomivee/ai_assistant)"}

    async def search():
        async with httpx.AsyncClient(headers=headers) as client:
            return await _lookup(client, families, limit, start, min(end or today, today), deadline)

    # Like the other deadline-bound network tools, this runs in the sync agent worker.
    with closing(asyncio.new_event_loop()) as loop:
        return loop.run_until_complete(search())


async def _lookup(client, families, limit, start, end, deadline):
    page = await _page(client, _INDEX, deadline)
    if page is None:
        return []
    rows = _entries(page, families, start, end)
    if not rows:
        archives = []
        for href in page.xpath("//a/@href"):
            url = urljoin(_INDEX, href)
            match = _ANNUAL.fullmatch(url)
            if match and (start.year if start else 1) <= int(match[1]) <= end.year:
                archives.append((int(match[1]), url))
        if archives:
            older = await _page(client, max(archives)[1], deadline)
            if older is not None:
                rows = _entries(older, families, start, end)
    # Compare different meeting formats before trying older releases of the same
    # format. Otherwise many monthly releases can hide the full ministerial meeting.
    seen_families = set()
    first, rest = [], []
    for row in rows:
        (rest if row[1] in seen_families else first).append(row)
        seen_families.add(row[1])
    results = await asyncio.gather(
        *(
            _article(client, url, family, published, deadline)
            for published, family, url in (first + rest)[:_MAX_DETAILS]
        )
    )
    return [item for item in results if item is not None][:limit]
