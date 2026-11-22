from datetime import UTC, date, datetime

import pytest

from src.config import settings
from src.integrations.llm import llm
from src.integrations.telemetry import langfuse_client


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    # Unit tests simulate external services; archive-specific cases supply publisher HTML.
    async def empty_archive(*args, **kwargs):
        return b""

    monkeypatch.setattr("src.integrations.opec_publications._fetch_html", empty_archive)
    for name, value in {
        "LLM_API_KEY": "",
        "LLM_PROVIDER": "openai",
        "GIGACHAT_CREDENTIALS": "",
        "GIGACHAT_CA_BUNDLE": "",
        "GIGACHAT_MODEL": "GigaChat-2",
        "GIGACHAT_SCOPE": "GIGACHAT_API_PERS",
        "GIGACHAT_BASE_URL": "https://api.giga.chat/v1",
        "JUDGE_API_KEY": "",
        "JUDGE_BASE_URL": "",
        "JUDGE_MODEL": "",
        "LANGFUSE_ENABLED": "false",
        "TRACE_CONTENT": "true",
        "TRACE_FILE": str(tmp_path / "trace.jsonl"),
        "CACHE_DIR": str(tmp_path / "cache"),
        "PROMPT_SOURCE": "local",
        "AGENT_MODE": "loop",
        "APP_TIMEZONE": "Europe/Moscow",
        "AGENT_MAX_STEPS": "6",
        "AGENT_NO_PROGRESS_LIMIT": "2",
        "AGENT_TIMEOUT_SECONDS": "180",
        "AGENT_ANSWER_RESERVE_SECONDS": "60",
        "ANSWER_SELF_REVIEW": "false",
    }.items():
        monkeypatch.setenv(name, value)
    settings.cache_clear()
    llm.cache_clear()
    langfuse_client.cache_clear()
    yield
    settings.cache_clear()
    llm.cache_clear()
    langfuse_client.cache_clear()


@pytest.fixture
def web_today(monkeypatch):
    # Same instant, different calendar days in UTC and the configured Moscow zone.
    monkeypatch.setattr("src.clock._utc_now", lambda: datetime(2026, 11, 21, 22, tzinfo=UTC))
    return date(2026, 11, 22)
