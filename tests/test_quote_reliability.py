from contextlib import contextmanager
from datetime import date

import httpx
import pytest

from src import tools
from src.budget import ResearchBudgetExceeded, request_budget, research_phase
from src.integrations import web


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
