"""Scope execution contracts with HTTP fixtures, not live intent-quality scores."""

import json

import pytest
from test_agent import WEB, plan
from test_agent_loop import event
from test_gigachat_client import auth_reply, completion
from test_gigachat_client import client as gigachat_client

from eval.metrics import route_metrics
from eval.run import load_cases, validate_data
from src.agent import ask
from src.config import ROOT

DATA = ROOT / "eval/data/scope_regressions.jsonl"


def test_authored_scope_dataset_loads_in_the_actual_evaluator():
    summary = validate_data(DATA)
    assert summary["cases"] == 14
    assert summary["groups"] == 7
    assert summary["splits"] == {"dev": 14, "test": 0}
    assert summary["retrieval_cases"] == 0


@pytest.mark.parametrize(
    "question",
    ["какая зарплата у нефтегазового аналитика?", "Найди вакансии аналитика в нефтяной компании."],
)
def test_native_outside_plan_stops_before_tools_even_with_leftover_search_flags(question):
    generated = []

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        generated.append(json.loads(request.content)["response_format"]["schema"]["title"])
        return completion(
            plan(
                intent="outside",
                skill="web_research",
                rag_query=question,
                fresh=True,
                news=True,
                wants_context=True,
                quotes=["Brent"],
            ).model_dump()
        )

    def execute(*args, **kwargs):
        pytest.fail("A structured outside decision must not execute any tool")

    client, native = gigachat_client(reply)
    try:
        result = ask(question, client=client, execute=execute)
    finally:
        native.close()
    assert generated == ["Plan"]
    assert result["status"] == result["intent"] == "outside"
    assert result["trace"] == result["decisions"] == result["claims"] == []
    parsed = result["diagnostics"]["plan"]
    assert not parsed["fresh"] and not parsed["wants_context"] and not parsed["news"]
    assert parsed["quotes"] == [] and parsed["rag_query"] == "" and parsed["skill"] is None


def test_wage_cost_question_can_reach_web_when_model_selects_research():
    generated, executed = [], []
    fact = "Growth in payroll costs increases oil production costs when productivity is unchanged."

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        title = json.loads(request.content)["response_format"]["schema"]["title"]
        generated.append(title)
        if title == "Plan":
            return completion(plan(rag_query="payroll and oil production costs").model_dump())
        if title == "AvailableAction":
            return completion(
                {
                    "action": "web_search",
                    "purpose": "Check cost mechanism",
                    "query": "payroll and oil production costs",
                }
            )
        assert title == "CitedAnswer"
        return completion({"claims": [{"text": fact, "evidence_ids": ["W1:B1"]}], "complete": True})

    def execute(registry, name, **arguments):
        executed.append((name, arguments))
        return {"items": [{**WEB, "text": fact}], "notices": []}, None, event(name)

    client, native = gigachat_client(reply)
    try:
        result = ask(
            "Как рост зарплаты работников влияет на себестоимость добычи нефти?",
            client=client,
            execute=execute,
        )
    finally:
        native.close()
    assert generated == ["Plan", "AvailableAction", "CitedAnswer"]
    assert result["intent"] == "research" and result["status"] == "ok"
    assert [name for name, _ in executed] == ["web_search"]
    assert result["claims"][0]["text"] == fact


def test_scope_gold_rejects_the_observed_salary_search_route():
    row = next(row for row in load_cases(DATA) if row["id"] == "scope-salary-role")
    response = {
        "trace": [{"tool": "web_search", "status": "ok"}],
        "intent": "research",
        "status": "partial",
        "agent_mode": "loop",
    }
    metrics = route_metrics(row["expected"], response, {"fresh": True, "wants_context": True})
    assert metrics["intent_correct"] == 0
    assert metrics["tool_policy_pass"] == 0
    assert metrics["forbidden_tool_violation"] == 1
