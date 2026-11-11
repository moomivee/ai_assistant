import pytest

from src.config import settings
from src.integrations.llm import llm
from src.integrations.telemetry import langfuse_client


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for name, value in {
        "LLM_API_KEY": "",
        "LANGFUSE_ENABLED": "false",
        "TRACE_CONTENT": "true",
        "TRACE_FILE": str(tmp_path / "trace.jsonl"),
        "CACHE_DIR": str(tmp_path / "cache"),
        "PROMPT_SOURCE": "local",
        "AGENT_MODE": "loop",
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
