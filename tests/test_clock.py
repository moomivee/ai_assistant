import json
from contextvars import Context
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from src import clock
from src.config import settings
from src.integrations.llm import LLMClient


@pytest.mark.parametrize(
    "zone,instant,today,year,offset",
    [
        ("Europe/Moscow", "2026-12-31T21:01:00+00:00", "2027-01-01", 2027, "+03:00"),
        ("UTC", "2026-12-31T21:01:00+00:00", "2026-12-31", 2026, "+00:00"),
        ("America/New_York", "2027-01-01T02:00:00+00:00", "2026-12-31", 2026, "-05:00"),
    ],
)
def test_request_date_uses_configured_timezone_at_year_boundary(
    monkeypatch, zone, instant, today, year, offset
):
    monkeypatch.setenv("APP_TIMEZONE", zone)
    settings.cache_clear()
    monkeypatch.setattr(clock, "_utc_now", lambda: datetime.fromisoformat(instant))
    with clock.request_clock():
        reading = clock.clock_payload()
    assert reading["today"] == today
    assert reading["current_year"] == year
    assert reading["timezone"] == zone
    assert reading["current_time"].endswith(offset)


def test_nested_request_retains_clock_and_new_request_gets_new_date(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    instant = [datetime(2026, 12, 31, 20, 59, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    with clock.request_clock():
        first = clock.clock_payload()
        instant[0] = datetime(2026, 12, 31, 21, 1, tzinfo=UTC)
        with clock.request_clock():
            assert clock.clock_payload() == first
        assert clock.clock_payload() == first
    with clock.request_clock():
        assert clock.clock_payload()["today"] == "2027-01-01"
    assert first["today"] == "2026-12-31"


def test_request_clock_is_restored_after_error_and_isolated_between_contexts(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "UTC")
    settings.cache_clear()
    instant = [datetime(2026, 11, 17, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    with pytest.raises(RuntimeError), clock.request_clock():
        assert clock.clock_payload()["today"] == "2026-11-17"
        instant[0] = datetime(2026, 11, 18, tzinfo=UTC)
        assert Context().run(clock.clock_payload)["today"] == "2026-11-18"
        assert clock.clock_payload()["today"] == "2026-11-17"
        raise RuntimeError("request failed")
    with clock.request_clock():
        assert clock.clock_payload()["today"] == "2026-11-18"


@pytest.mark.parametrize("value", ["Not/A_Timezone", "", "../Europe/Moscow"])
def test_invalid_timezone_fails_with_configuration_guidance(monkeypatch, value):
    monkeypatch.setenv("APP_TIMEZONE", value)
    settings.cache_clear()
    with pytest.raises(ValidationError, match="APP_TIMEZONE"):
        settings()


def test_time_and_review_defaults(monkeypatch):
    monkeypatch.delenv("APP_TIMEZONE", raising=False)
    monkeypatch.delenv("ANSWER_SELF_REVIEW", raising=False)
    settings.cache_clear()
    assert settings().app_timezone == "Europe/Moscow"
    assert settings().answer_review is True


class Output(BaseModel):
    ok: bool


class CaptureProvider:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        return self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(parsed=Output(ok=True), refusal=None))
            ],
            usage=None,
        )

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content='{"ok":true}', model="mock", usage={}, attempts=())


@pytest.mark.parametrize("provider", ["gigachat", "openai"])
def test_every_generation_and_trace_get_trusted_request_time(monkeypatch, provider):
    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    instant = [datetime(2026, 12, 31, 20, 59, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    backend = CaptureProvider()
    client = LLMClient(backend, provider=provider)
    untrusted = {
        "question": "Расскажи о встрече в 2024 году",
        "today": "2020-01-01",
        "current_year": 2020,
        "current_time": "untrusted",
        "timezone": "untrusted",
    }
    with clock.request_clock():
        expected = clock.clock_payload()
        for stage in ("plan", "decide", "cited_answer", "citation_review"):
            assert client.generate(stage, untrusted, Output).ok
            instant[0] = datetime(2026, 12, 31, 21, 1, tzinfo=UTC)
    for call in backend.calls:
        system = call["messages"][0]["content"]
        payload = json.loads(call["messages"][1]["content"])
        assert all(payload[key] == value for key, value in expected.items())
        assert payload["question"] == untrusted["question"]
        assert "сегодня 2026-12-31, текущий год 2026" in system
        assert "часовой пояс Europe/Moscow" in system
        assert "исторический период" in system
    assert untrusted["today"] == "2020-01-01"  # Caller data was not mutated.
    traces = [json.loads(line) for line in settings().trace_file.read_text().splitlines()]
    assert len(traces) == 4
    for trace in traces:
        assert trace["metadata"]["clock"] == expected
        assert all(trace["input"][key] == value for key, value in expected.items())


def test_planning_payload_uses_request_clock_for_injected_clients(monkeypatch):
    from test_agent import plan

    from src.planning import make_plan

    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    monkeypatch.setattr(clock, "_utc_now", lambda: datetime(2026, 12, 31, 21, 1, tzinfo=UTC))
    calls = []

    class Planner:
        def generate(self, stage, payload, schema):
            calls.append(payload)
            return plan()

    with clock.request_clock():
        make_plan(Planner(), "Как устроена добыча нефти?")
    assert calls[0]["today"] == "2027-01-01"
    assert calls[0]["current_year"] == 2027
    assert calls[0]["timezone"] == "Europe/Moscow"


def test_legacy_workflow_freezes_time_across_plan_and_answer(monkeypatch):
    from test_agent import FakeLLM, executor

    from src.workflow import ask

    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    instant = [datetime(2026, 12, 31, 20, 59, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    seen = []

    class Client(FakeLLM):
        def generate(self, name, payload, schema):
            seen.append(payload)
            result = super().generate(name, payload, schema)
            instant[0] = datetime(2026, 12, 31, 21, 1, tzinfo=UTC)
            return result

    result = ask("Каково содержание серы Sokol?", client=Client(), execute=executor)
    assert result["status"] == "ok"
    assert all(payload["today"] == "2026-12-31" for payload in seen)
    assert all(payload["current_year"] == 2026 for payload in seen)
    assert clock.clock_today().isoformat() == "2027-01-01"


def test_search_filters_use_same_reference_day_as_prompts(monkeypatch):
    from datetime import date

    from src.integrations.web import _within_window, validate_search_options

    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    instant = [datetime(2026, 12, 31, 21, 1, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    with clock.request_clock():
        start, end = validate_search_options(days=1)
        assert start == date(2026, 12, 31)
        assert end.isoformat() == clock.clock_payload()["today"] == "2027-01-01"
        instant[0] = datetime(2027, 1, 2, tzinfo=UTC)
        assert _within_window(date(2027, 1, 1), None, None)
        assert not _within_window(date(2027, 1, 2), None, None)
        with pytest.raises(ValueError, match="не позже сегодняшней"):
            validate_search_options(end_date="2027-01-02")


def test_rss_worker_threads_use_parent_request_day(monkeypatch):
    from datetime import date

    import httpx

    from src.integrations import feeds

    monkeypatch.setenv("APP_TIMEZONE", "Europe/Moscow")
    settings.cache_clear()
    instant = [datetime(2026, 12, 31, 21, 1, tzinfo=UTC)]
    monkeypatch.setattr(clock, "_utc_now", lambda: instant[0])
    monkeypatch.setattr(feeds, "FEEDS", {"EIA": "https://www.eia.gov/rss/test"})

    def get(url, **kwargs):
        # A worker with a fresh clock would now accept this tomorrow-dated item.
        instant[0] = datetime(2027, 1, 2, tzinfo=UTC)
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            text=(
                "<rss><channel><item><title>Oil report</title><pubDate>2027-01-02</pubDate>"
                "<link>https://www.eia.gov/oil</link><description>Oil news.</description>"
                "</item></channel></rss>"
            ),
        )

    monkeypatch.setattr(feeds.httpx, "get", get)
    with clock.request_clock():
        assert feeds.search_feeds("oil", lambda *_: True, date.fromisoformat) == []
