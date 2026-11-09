import csv
import io
import math
from datetime import date
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..logging import log
from .feeds import search_feeds

DAILY = {"Brent": "DCOILBRENTEU", "WTI": "DCOILWTICO"}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def allowed(url: str, domains: tuple[str, ...]) -> bool:
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    return (
        parts.scheme == "https"
        and not parts.username
        and not parts.password
        and any(host == d or host.endswith("." + d) for d in domains)
    )


def web_date(raw: str) -> date | None:
    try:
        return date.fromisoformat(raw[:10])
    except (TypeError, ValueError):
        try:
            return parsedate_to_datetime(raw).date()
        except (TypeError, ValueError, OverflowError):
            return None


def latest_spot(grade: str) -> dict:
    log("web.quote_requested", provider="EIA/FRED", grade=grade)
    ticker = DAILY[grade]
    response = httpx.get(
        FRED_CSV,
        params={"id": ticker},
        timeout=settings().timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    observations = []
    for row in reader:
        day = web_date(row.get("observation_date") or row.get("DATE") or "")
        try:
            price = float(row[ticker])
        except (ValueError, TypeError, KeyError):
            continue
        if day and day <= date.today() and math.isfinite(price):
            observations.append((day, price))
    if not observations:
        raise ValueError("FRED не вернул наблюдений с датой.")
    day, price = max(observations)
    return {
        "kind": "web",
        "title": f"EIA через FRED · {grade} spot",
        "date": day.isoformat(),
        "url": f"https://fred.stlouisfed.org/series/{ticker}",
        "grade": grade,
        "price": price,
        "text": f"Последнее опубликованное наблюдение {grade} spot за {day.isoformat()} — {price:.2f} USD/барр. Ряд дневной, публикуется с задержкой.",
    }


def search_web(query: str, fresh: bool) -> list[dict]:
    log(
        "web.search_requested",
        provider="tavily" if settings().web_key else "rss",
        fresh=fresh,
    )
    cfg = settings()
    if not cfg.web_key and fresh:
        return search_feeds(query, allowed, web_date)
    if not cfg.web_key:
        raise ValueError(
            "Для свободного веб-поиска укажите TAVILY_API_KEY в .env. Последние наблюдения Brent и WTI доступны через FRED без ключа."
        )
    payload = {
        "query": query[:400],
        "max_results": 6,
        "include_domains": list(cfg.domains),
        "include_answer": False,
        "topic": "news" if fresh else "general",
        "include_published_date": True,
    }
    if fresh:
        payload["days"] = cfg.news_days
    response = httpx.post(
        "https://api.tavily.com/search",
        json=payload,
        headers={"Authorization": "Bearer " + cfg.web_key},
        timeout=cfg.timeout,
    )
    response.raise_for_status()
    hits, seen = [], set()
    for item in response.json().get("results", []):
        url, snippet = (
            str(item.get("url") or ""),
            str(item.get("content") or "").strip(),
        )
        published = web_date(item.get("published_date") or "")
        if not allowed(url, cfg.domains) or url in seen or not snippet:
            continue
        if published and published > date.today():
            continue
        if fresh and (not published or (date.today() - published).days > cfg.news_days):
            continue
        seen.add(url)
        hits.append(
            {
                "kind": "web",
                "title": str(item.get("title") or urlparse(url).hostname)[:190],
                "date": published.isoformat() if published else None,
                "url": url,
                "text": snippet[:1500],
            }
        )
    log("web.search_completed", provider="tavily", accepted=len(hits))
    return hits


def web_lookup(query: str, fresh: bool, quotes: list[str], search: bool) -> dict:
    rows, notices = [], []
    for grade in dict.fromkeys(quotes):
        if grade not in DAILY:
            search = True
            continue
        try:
            rows.append(latest_spot(grade))
        except (httpx.HTTPError, ValueError) as exc:
            notices.append(f"Источник котировки {grade} недоступен ({type(exc).__name__}).")
            search = True
    if search:
        try:
            rows += search_web(query, fresh)
        except ValueError as exc:
            notices.append(str(exc))
        except (httpx.HTTPError, KeyError, TypeError) as exc:
            notices.append(f"Веб-поиск не выполнен ({type(exc).__name__}).")
    if not rows and not notices:
        notices.append("В доверенных источниках не найдено подходящей публикации с нужной датой.")
    return {
        "items": [dict(r, id=f"W{i}") for i, r in enumerate(rows, 1)],
        "notices": notices,
    }
