import asyncio
from contextlib import contextmanager
from datetime import date
from time import monotonic

import httpx
import pytest

from src import tools
from src.budget import ResearchBudgetExceeded, request_budget, research_phase
from src.config import settings
from src.integrations import feeds, web
from src.integrations.http_deadline import request_before


def quote(*grades, publications=False):
    return tools.run_tool(
        tools.make_registry(),
        "web_search",
        query="Market structure" if publications else "",
        fresh=True,
        quotes=list(grades),
        search=publications,
    )


def test_fred_failure_is_bounded_and_does_not_trigger_publication_search(monkeypatch):
    attempts = []

    def timeout(method, url, **kwargs):
        attempts.append(kwargs)
        raise httpx.ReadTimeout("sensitive proxy details")

    monkeypatch.setattr(web.httpx, "stream", timeout)
    monkeypatch.setattr(web, "search_web", lambda *a, **k: pytest.fail("No publication request"))
    with request_budget(5, answer_reserve=2), research_phase():
        data, error, event = quote("Brent")
    assert len(attempts) == 1 and 0 < attempts[0]["timeout"] <= 1.5
    assert event["status"] == "error" and error == data["notices"][0]
    assert data["items"] == [] and "Brent" in error and "время ожидания" in error
    assert "sensitive" not in error


def test_failed_quote_keeps_other_evidence_with_independent_status(monkeypatch):
    @contextmanager
    def stream(method, url, params, **kwargs):
        code = params["id"]
        response = httpx.Response(
            503 if code == web.DAILY["Brent"] else 200,
            request=httpx.Request(method, url),
            text=f"observation_date,{code}\n2025-03-03,67.0\n",
        )
        yield response

    monkeypatch.setattr(web.httpx, "stream", stream)
    monkeypatch.setattr(
        web, "search_web", lambda *a, **k: [{"url": "https://www.eia.gov/", "text": "Market."}]
    )
    data, error, event = quote("Brent", "WTI", publications=True)
    assert error is None and event["status"] == "ok"
    assert [r.get("grade") for r in data["items"]] == ["WTI", None]
    assert "Brent" in data["notices"][0] and "HTTP 503" in data["notices"][0]


def test_fred_inactive_series_reports_actual_last_observation(monkeypatch):
    attempts = []
    monkeypatch.setattr(web, "clock_today", lambda: date(2026, 11, 17))

    @contextmanager
    def stream(method, url, params, timeout, **kwargs):
        attempts.append((params, timeout))
        observations = "2026-11-17,.\n2026-11-18,72\n2026-11-16,nan\n"
        if "cosd" not in params:
            observations += "2026-01-15,61.5\n2026-01-14,62\n"
        yield httpx.Response(
            200,
            request=httpx.Request(method, url),
            text="DATE,DCOILBRENTEU\n" + observations,
        )

    monkeypatch.setattr(web.httpx, "stream", stream)
    data, error, event = quote("Brent")
    assert event["status"] == "ok" and error is None
    assert [(args.get("cosd"), args["coed"]) for args, _ in attempts] == [
        ("2026-10-03", "2026-11-17"),
        (None, "2026-11-17"),
    ]
    assert 0 < attempts[1][1] <= attempts[0][1] <= web.QUOTE_TIMEOUT
    assert data["items"][0]["price"] == 61.5 and data["items"][0]["date"] == "2026-01-15"


def test_unexpected_handler_error_survives_registry_without_raw_message(monkeypatch):
    def broken(query):
        raise RuntimeError("sensitive internal URL")

    monkeypatch.setitem(tools.FUNCTIONS, "report_search", (tools.ReportArgs, broken, "fixture"))
    data, error, event = tools.run_tool(tools.make_registry(), "report_search", query="Brent")
    assert data is None and event["status"] == "error"
    assert "RuntimeError" in error and "report_search" in error
    assert "sensitive" not in error


def test_quote_survives_exhausted_publication_budget(monkeypatch):
    monkeypatch.setattr(web, "latest_spot", lambda grade: {"grade": grade, "price": 69.5})

    def exhausted(*args, **kwargs):
        raise ResearchBudgetExceeded()

    monkeypatch.setattr(web, "search_web", exhausted)
    data, error, event = quote("Brent", publications=True)
    assert error is None and event["status"] == "ok"
    assert data["items"][0]["price"] == 69.5
    assert data["notices"] == [str(ResearchBudgetExceeded())]


@pytest.mark.parametrize("research_seconds,expected", [(120, 15), (3, 3)])
def test_fallback_tavily_timeout_respects_research_and_does_not_change_normal_search(
    monkeypatch, research_seconds, expected
):
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    monkeypatch.setenv("HTTP_TIMEOUT", "90")
    settings.cache_clear()
    timeouts = []

    def fallback(deadline, method, url, **kwargs):
        timeouts.append(deadline - monotonic())
        raise httpx.ReadTimeout("provider details")

    def general(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": []})

    monkeypatch.setattr(web, "request_before", fallback)
    monkeypatch.setattr(web.httpx, "post", general)
    with request_budget(research_seconds), research_phase():
        data, error, event = tools.run_tool(
            tools.make_registry(),
            "web_search",
            query="Brent price",
            fresh=True,
            quotes=[],
            search=True,
            quote_fallback=True,
        )
    assert event["status"] == "error" and data["items"] == []
    assert "запасного поиска котировки" in error and "provider details" not in error
    assert 0 < timeouts[0] <= expected
    assert web.search_web("other research", True) == [] and timeouts[1] == 90


def test_fallback_feeds_share_deadline_and_keep_successful_feed(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    settings.cache_clear()
    deadlines = {}

    def feed(deadline, method, url, **kwargs):
        deadlines[url] = deadline
        if url == feeds.FEEDS["Банк России"]:
            raise httpx.ReadTimeout("slow feed")
        xml = (
            "<rss><channel><item><title>Oil prices</title>"
            f"<pubDate>{web.clock_today().isoformat()}</pubDate>"
            "<link>https://www.eia.gov/news</link></item></channel></rss>"
        )
        return httpx.Response(200, text=xml, request=httpx.Request(method, url))

    monkeypatch.setattr(feeds, "request_before", feed)
    result = web.web_lookup("oil", True, [], True, quote_fallback=True)
    assert set(deadlines) == set(feeds.FEEDS.values()) and len(set(deadlines.values())) == 1
    assert len(result["items"]) == 1 and result["items"][0]["provider"] == "rss"


def test_fallback_deadline_cancels_in_flight_body_read(monkeypatch):
    cancelled = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            try:
                while True:
                    yield b"data"
                    await asyncio.sleep(0.005)
            finally:
                cancelled.append(True)

    transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=SlowBody()))
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
    with pytest.raises(httpx.ReadTimeout, match="deadline"):
        request_before(monotonic() + 0.1, "GET", "https://example.test/test")
    assert cancelled
