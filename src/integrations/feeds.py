# ограниченный поиск свежих публикаций в RSS EIA и Банка России

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

from ..config import settings
from ..logging import log

FEEDS = {
    "EIA": "https://www.eia.gov/rss/todayinenergy.xml",
    "Банк России": "https://www.cbr.ru/rss/eventrss",
}
TOPICS = [
    ("росси", "russia"),
    ("уралс", "юралс", "urals"),
    ("кита", "china", "chinese"),
    ("инди", "india"),
    ("опек", "opec"),
    ("санкци", "sanction"),
    ("экспорт", "export"),
    ("переработ", "refin", "нпз"),
    ("добыч", "production"),
    ("спг", "lng"),
    ("нефт", "oil", "crude", "petroleum"),
    ("газ", "natural gas"),
]


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain(x: str) -> str:
    parser = PlainText()
    parser.feed(x)
    return " ".join(" ".join(parser.parts).split())


def search_feeds(query: str, trusted, parse_date) -> list[dict]:
    log("web.rss_fallback", feeds=len(FEEDS))
    cfg = settings()
    q = query.casefold()
    groups = [g for g in TOPICS if any(term in q for term in g)]
    if not groups:
        return []
    if "eia" in q or "эиа" in q:
        feeds = {"EIA": FEEDS["EIA"]}
    elif "банк россии" in q or "цб" in q:
        feeds = {"Банк России": FEEDS["Банк России"]}
    else:
        feeds = FEEDS

    def fetch(entry):
        publisher, feed = entry
        if not trusted(feed, cfg.domains):
            return []
        try:
            response = httpx.get(feed, timeout=cfg.timeout, follow_redirects=True)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (httpx.HTTPError, ElementTree.ParseError) as exc:
            log(
                "web.rss_failed",
                logging.WARNING,
                publisher=publisher,
                error_type=type(exc).__name__,
            )
            return []
        found = []
        for row in root.findall("./channel/item"):
            day = parse_date(row.findtext("pubDate", ""))
            if not day or not 0 <= (date.today() - day).days <= cfg.news_days:
                continue
            parts = urlsplit(row.findtext("link", ""))
            if parts.scheme not in ("http", "https"):
                continue
            url = urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))
            if not trusted(url, cfg.domains):
                continue
            title = _plain(row.findtext("title", ""))
            text = _plain(row.findtext("description", ""))[:1100]
            sample = (title + " " + text).casefold()
            if not all(any(term in sample for term in g) for g in groups):
                continue
            found.append(
                {
                    "kind": "web",
                    "title": title,
                    "date": day.isoformat(),
                    "url": url,
                    "text": (title + ". " + text)[:1350],
                    "publisher": publisher,
                }
            )
        return found

    with ThreadPoolExecutor(max_workers=2) as pool:
        hits = [r for batch in pool.map(fetch, feeds.items()) for r in batch]
    return sorted(hits, key=lambda r: r["date"], reverse=True)[:4]
