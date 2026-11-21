import pytest
from test_agent import plan

from src.agent import ask
from src.config import settings


@pytest.mark.parametrize("mode", ["loop", "workflow"])
@pytest.mark.parametrize("report_only", [False, True])
def test_capabilities_finish_with_help_without_research(mode, report_only, monkeypatch):
    monkeypatch.setenv("AGENT_MODE", mode)
    monkeypatch.setenv("TAVILY_API_KEY", "")
    settings.cache_clear()

    def unexpected_tool(*args, **kwargs):
        pytest.fail("A help request must not call a research or calculation tool.")

    response = ask(
        "Расскажи, какие вопросы я могу здесь задать.",
        client=object(),  # Only the injected planner is needed.
        planner=lambda *_: plan(intent="capabilities", report_only=report_only),
        execute=unexpected_tool,
    )
    assert response["intent"] == "capabilities" and response["status"] == "ok"
    assert (
        response["trace"]
        == response["decisions"]
        == response["sources"]
        == response["claims"]
        == []
    )
    assert response["forecast"] is None and not response["clarification"]
    answer = response["answer"]
    assert "отчёте EIA" in answer and "документ" in answer
    if report_only:
        assert "Только база знаний" in answer
    else:
        assert "Спрогнозируй" in answer and "EIA и Банка России" in answer
    assert "вне нефтегазовой специализации" not in answer
