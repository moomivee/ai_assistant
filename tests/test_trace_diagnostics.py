"""The inspector exposes the same execution progress as the model receives."""

from test_agent import REPORT, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask
from src.integrations.llm import LLMUnavailable
from src.models.schemas import AgentResponse


def test_diagnostics_link_decisions_to_actual_results_and_shared_progress():
    other = {**REPORT, "text": "Other evidence. " * 400, "url": "https://eia.gov/other"}
    client = Scripted(
        [
            action("report_search", "Sokol sulfur"),
            {**action("finish"), "source_ids": ["R1"]},
        ],
        complete=[False, True],
    )
    response = ask(
        "Сера Sokol?",
        client=client,
        planner=lambda *_: plan(report_only=True),
        execute=lambda *a, **kw: ([REPORT, other], None, event("report_search")),
    )
    diag = response["diagnostics"]
    assert diag["plan"]["report_only"] is True
    step = diag["steps"][0]
    assert step["step"] == 1 and step["arguments"] == {"query": "Sokol sulfur"}
    assert step["status"] == "ok" and step["ms"] == 1
    assert step["source_ids"] == ["R1", "R2"] and step["result_count"] == 2
    assert diag["retrieved_sources"][0]["used_in_answer"] is True
    assert diag["retrieved_sources"][1]["used_in_answer"] is False
    assert len(diag["retrieved_sources"][1]["text"]) == 4000
    assert diag["retrieved_sources"][1]["text_truncated"] is True
    payloads = [p for name, p in client.payloads if name == "decide"]
    observation = payloads[1]["observations"][0]
    for key in ("source_ids", "new_source_ids", "new_evidence", "no_progress_steps"):
        assert observation[key] == step[key]
    assert observation["new_evidence"] is True
    assert "diagnostics" not in payloads[1]


def test_rejected_attempts_have_no_invented_execution_time(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    response = ask(
        "Найди факт",
        client=Scripted([action("report_search", "same")] * 3),
        planner=lambda *_: plan(report_only=True),
        execute=lambda *a, **kw: ([], None, event("report_search")),
    )
    steps = response["diagnostics"]["steps"]
    assert len(response["trace"]) == 1 and len(steps) == 3
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert [s["status"] for s in steps] == ["ok", "rejected", "rejected"]
    assert all(s["ms"] is None and s["error"] for s in steps[1:])
    assert response["stop_reason"] == "no_progress"


def test_tool_error_is_distinct_from_policy_rejection(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")
    response = ask(
        "Найди факт",
        client=Scripted([action("report_search", "fact"), action("finish")]),
        planner=lambda *_: plan(report_only=True),
        execute=lambda *a, **kw: (
            None,
            "Search failed",
            {**event("report_search"), "status": "error"},
        ),
    )
    step = response["diagnostics"]["steps"][0]
    assert step["status"] == "error" and step["error"] == "Search failed" and step["ms"] == 1


def test_unavailable_plan_has_an_honest_empty_trace_and_legacy_contract_still_valid():
    def unavailable(*args):
        raise LLMUnavailable("Модель недоступна")

    response = ask("Вопрос", client=Scripted([]), planner=unavailable)
    assert response["status"] == "unavailable"
    assert response["diagnostics"]["plan"] is None
    assert response["diagnostics"]["steps"] == []
    assert response["diagnostics"]["failure"] == "Модель недоступна"
    del response["diagnostics"]
    assert AgentResponse.model_validate(response).diagnostics is None
