import csv
import io
import ipaddress
import math
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from ..budget import bounded_timeout, check_budget
from ..clock import clock_today
from ..config import settings
from ..logging import log
from . import web_cache
from .feeds import search_feeds

DAILY = {"Brent": "DCOILBRENTEU", "WTI": "DCOILWTICO"}
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SEARCH_TEXT_LIMIT = 2000
EXTRACT_TEXT_LIMIT = 12000


def allowed(url: str, domains: tuple[str, ...]) -> bool:
    """Accept public HTTPS publishers only; this is also used by the RSS fallback."""
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 32 for c in url):
        return False
    if "\\" in url:
        return False
    try:
        parts = urlparse(url)
        host = (parts.hostname or "").lower().rstrip(".")
        if (
            parts.scheme != "https"
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
            or not host
            or "%" in parts.netloc
        ):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
                return False
        else:
            if not address.is_global:
                return False
        return any(
            host == d.lower().rstrip(".") or host.endswith("." + d.lower().rstrip("."))
            for d in domains
            if d
        )
    except (ValueError, UnicodeError):
        return False


def web_date(raw: str | None) -> date | None:
    if not isinstance(raw, str):
        return None
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
        timeout=bounded_timeout(settings().timeout),
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
        if day and day <= clock_today() and math.isfinite(price):
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


def _date_bounds(days, start_date, end_date):
    today = clock_today()
    if days is not None:
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 3650:
            raise ValueError("days должен быть целым числом от 1 до 3650.")
        if start_date is not None or end_date is not None:
            raise ValueError("Задайте days либо start_date/end_date, но не оба варианта.")
        return today - timedelta(days=days), today
    parsed = []
    for label, raw in (("start_date", start_date), ("end_date", end_date)):
        if raw is None:
            parsed.append(None)
            continue
        try:
            day = date.fromisoformat(raw)
            if day.isoformat() != raw or day > today:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError(f"{label}: нужна дата YYYY-MM-DD не позже сегодняшней.") from None
        parsed.append(day)
    start, end = parsed
    if start and end and start > end:
        raise ValueError("start_date не может быть позже end_date.")
    return start, end


def validate_search_options(
    *,
    topic="general",
    days=None,
    start_date=None,
    end_date=None,
    max_results=6,
    search_depth="basic",
):
    """Invalid arguments are tool errors, not a successfully executed empty search."""
    if topic not in {"general", "news"} or search_depth not in {"basic", "advanced"}:
        raise ValueError("Неизвестный topic или search_depth для веб-поиска.")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 6
    ):
        raise ValueError("max_results должен быть целым числом от 1 до 6.")
    return _date_bounds(days, start_date, end_date)


def _within_window(published, start, end):
    if published is None:
        return not (start or end)
    return (
        published <= clock_today()
        and (not start or published >= start)
        and (not end or published <= end)
    )


def _retrieved_at():
    return datetime.now(UTC).isoformat()


def _evidence(item, *, text, content_type, retrieved_at, limit):
    published = web_date(item.get("published_date") or item.get("date"))
    url = item["url"]
    return {
        "kind": "web",
        "title": str(item.get("title") or urlparse(url).hostname)[:190],
        "date": published.isoformat() if published else None,
        "date_basis": "provider_estimate" if published else "unknown",
        "url": url,
        "text": text[:limit],
        "content_type": content_type,
        "retrieved_at": retrieved_at,
        "content_truncated": len(text) > limit,
        "content_limit": limit,
    }


def search_web(
    query: str,
    fresh: bool,
    *,
    topic: str = "general",
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int = 6,
    search_depth: str = "basic",
) -> list[dict]:
    """Search trusted publishers. Recency windows must be explicitly requested.

    `fresh` selects a shorter cache TTL and permits the limited RSS fallback. It
    does not imply that an older, still latest announcement must be discarded.
    Returned text is a search snippet, not evidence that the whole page was read.
    """
    check_budget()
    cfg = settings()
    query = " ".join(query.split())[:400]
    if not query:
        raise ValueError("Для веб-поиска нужен непустой запрос.")
    start, end = validate_search_options(
        topic=topic,
        days=days,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        search_depth=search_depth,
    )
    log("web.search_requested", provider="tavily" if cfg.web_key else "rss", fresh=fresh)
    if not cfg.web_key:
        if fresh:
            retrieved = _retrieved_at()
            return [
                dict(
                    row,
                    content_type="snippet",
                    retrieved_at=retrieved,
                    provider="rss",
                    cache_hit=False,
                )
                for row in search_feeds(query, allowed, web_date)
                if _within_window(web_date(row.get("date")), start, end)
            ][:max_results]
        raise ValueError(
            "Для свободного веб-поиска укажите TAVILY_API_KEY в .env. "
            "Последние наблюдения Brent и WTI доступны через FRED без ключа."
        )
    if not cfg.domains:
        raise ValueError("Для веб-поиска не настроены доверенные домены.")
    payload = {
        "query": query,
        "max_results": max_results,
        "include_domains": sorted(set(cfg.domains)),
        "include_domains_mode": "restrict",
        "include_answer": False,
        "include_raw_content": False,
        "topic": topic,
        "search_depth": search_depth,
        "auto_parameters": False,
        "include_published_date": True,
    }
    if start:
        payload["start_date"] = start.isoformat()
    if end:
        payload["end_date"] = end.isoformat()
    if start or end:
        payload["filter_by_published_date"] = True
    if fresh:
        # Refresh the provider's cached page, not an arbitrary publication window.
        payload["max_age_hours"] = 1
    cache_params = dict(payload, fresh=fresh)
    cached = web_cache.get("tavily.search", cache_params)
    if cached is not None:
        hits = cached.get("items", [])
        if isinstance(hits, list):
            log("web.cache_hit", operation="search", accepted=len(hits))
            return [dict(row, provider="tavily", cache_hit=True) for row in hits]
    response = httpx.post(
        "https://api.tavily.com/search",
        json=payload,
        headers={"Authorization": "Bearer " + cfg.web_key},
        timeout=bounded_timeout(cfg.timeout),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Tavily вернул некорректный список результатов.")
    retrieved = _retrieved_at()
    hits, seen = [], set()
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        url, snippet = str(item.get("url") or ""), str(item.get("content") or "").strip()
        published = web_date(item.get("published_date"))
        if not allowed(url, cfg.domains) or url in seen or not snippet:
            continue
        if not _within_window(published, start, end):
            continue
        seen.add(url)
        hits.append(
            dict(
                _evidence(
                    dict(item, url=url),
                    text=snippet,
                    content_type="snippet",
                    retrieved_at=retrieved,
                    limit=SEARCH_TEXT_LIMIT,
                ),
                provider="tavily",
                cache_hit=False,
            )
        )
        if len(hits) >= max_results:
            break
    web_cache.put("tavily.search", cache_params, {"items": hits}, ttl=300 if fresh else 3600)
    log("web.search_completed", provider="tavily", accepted=len(hits))
    return hits


def _http_notice(exc, operation):
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return f"{operation}: Tavily отклонил авторизацию. Проверьте TAVILY_API_KEY."
        if status in (429, 432, 433):
            return f"{operation}: достигнут лимит Tavily; проверьте квоту в личном кабинете."
        return f"{operation}: Tavily вернул HTTP {status}."
    return f"{operation} не выполнено ({type(exc).__name__})."


def read_web(url: str, query: str = "") -> dict:
    """Read one discovered trusted URL through Tavily Extract, never a local fetch.

    With query, Tavily returns up to five relevant source chunks; without query
    it extracts the page. The local content limit applies in both cases. Source
    text is untrusted data: it is returned as evidence, never executed as code.
    """
    check_budget()
    cfg = settings()
    if not allowed(url, cfg.domains):
        return {
            "items": [],
            "notices": ["Чтение разрешено только для HTTPS-страниц доверенных источников."],
        }
    if not cfg.web_key:
        return {"items": [], "notices": ["Для чтения веб-страницы нужен TAVILY_API_KEY в .env."]}
    query = " ".join(query.split())[:400]
    payload = {
        "urls": [url],
        "extract_depth": "basic",
        "format": "text",
        "include_published_date": True,
    }
    if query:
        payload.update(query=query, chunks_per_source=5)
    cache_params = dict(payload, trusted_domains=sorted(set(cfg.domains)))
    cached = web_cache.get("tavily.extract", cache_params)
    if cached is not None:
        log("web.cache_hit", operation="extract", accepted=len(cached.get("items", [])))
        return {
            **cached,
            "items": [
                dict(row, provider="tavily", cache_hit=True) for row in cached.get("items", [])
            ],
        }
    log("web.extract_requested", provider="tavily", host=urlparse(url).hostname)
    try:
        response = httpx.post(
            "https://api.tavily.com/extract",
            json=payload,
            headers={"Authorization": "Bearer " + cfg.web_key},
            timeout=bounded_timeout(cfg.timeout),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("invalid extract response")
    except httpx.HTTPError as exc:
        return {"items": [], "notices": [_http_notice(exc, "Чтение страницы")]}
    except (ValueError, TypeError):
        return {"items": [], "notices": ["Tavily вернул некорректный ответ при чтении страницы."]}
    rows, notices = [], []
    retrieved = _retrieved_at()
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        final_url = item.get("url") or ""
        if not allowed(final_url, cfg.domains):
            notices.append("Tavily вернул адрес вне доверенных источников; содержимое отклонено.")
            continue
        text = str(item.get("raw_content") or "").strip()
        published = web_date(item.get("published_date"))
        if published and published > clock_today():
            notices.append("Источник с будущей датой публикации отклонён.")
            continue
        if not text:
            continue
        row = _evidence(
            dict(item, url=final_url),
            text=text,
            content_type="extracted",
            retrieved_at=retrieved,
            limit=EXTRACT_TEXT_LIMIT,
        )
        row.update(
            id="W1",
            content_scope="query_chunks" if query else "page",
            requested_url=url,
            provider="tavily",
            cache_hit=False,
        )
        rows.append(row)
        if row["content_truncated"]:
            notices.append(f"Текст страницы ограничен {EXTRACT_TEXT_LIMIT} символами.")
        if query:
            notices.append(
                "Показаны фрагменты страницы, отобранные Tavily по запросу, до 5 фрагментов."
            )
        if row["date"] is None:
            notices.append(
                "Дата публикации страницы не установлена; время получения не является датой публикации."
            )
        break
    if data.get("failed_results"):
        notices.append("Tavily не смог полностью извлечь запрошенную страницу.")
    if not rows and not notices:
        notices.append("Tavily не вернул доступного текста страницы.")
    result = {"items": rows, "notices": notices}
    if rows:
        web_cache.put("tavily.extract", cache_params, result, ttl=900)
    log("web.extract_completed", provider="tavily", accepted=len(rows))
    return result


def web_lookup(query: str, fresh: bool, quotes: list[str], search: bool, **search_options) -> dict:
    # Validate before the provider-error handlers below (and before FRED calls).
    # The tool registry must record invalid arguments as an error, not status=ok.
    validate_search_options(**search_options)
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
            rows += search_web(query, fresh, **search_options)
            if fresh and not settings().web_key:
                notices.append(
                    "Доступен только ограниченный поиск по RSS EIA и Банка России; это не полный веб-поиск."
                )
        except ValueError as exc:
            notices.append(str(exc))
        except httpx.HTTPError as exc:
            notices.append(_http_notice(exc, "Веб-поиск"))
        except (KeyError, TypeError):
            notices.append("Веб-поиск вернул некорректный ответ.")
    if any(row.get("content_type") == "snippet" and not row.get("date") for row in rows):
        notices.append(
            "У части веб-источников дата публикации неизвестна; актуальность этих сведений не подтверждена."
        )
    if not rows and not notices:
        notices.append(
            "В доверенных источниках не найдено подходящей публикации с заданными параметрами."
        )
    return {
        "items": [dict(r, id=f"W{i}") for i, r in enumerate(rows, 1)],
        "notices": notices,
    }
