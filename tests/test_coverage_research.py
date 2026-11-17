"""Research continuation contracts with scripted decisions, not live LLM accuracy."""

import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import event

from src.agent import ask
from src.config import settings

SNIPPET = "В справочнике названы маркерные сорта Orion и Vega."
EXPORTS = "Экспорт вымышленной страны включает сорта Orion и Vega."
MISSING = "Подтверждение экспорта, а не только известности названий сортов."


def action(tool, query="экспорт вымышленной страны", **extra):
    return {"action": tool, "query": query, "purpose": "Проверить недостающий факт", **extra}


def review(coverage, missing=()):
    return {
        "supported": [0],
        # Intentionally contradictory in the partial case: the coverage
        # assessment must prevent this optimistic boolean from ending research.
        "complete": True,
        "feedback": "",
        "coverage": coverage,
        "missing_aspects": list(missing),
    }


class ResearchClient:
    def __init__(self, actions, reviews=()):
        self.actions, self.reviews = iter(actions), iter(reviews)
        self.payloads = []

    def generate(self, name, payload, schema):
        self.payloads.append((name, payload))
        if name == "decide":
            return schema.model_validate(next(self.actions))
        if name == "cited_answer":
            evidence = payload["evidence"]
            key = next(
                (key for key, text in evidence.items() if text == EXPORTS), next(iter(evidence))
            )
            return schema.model_validate(
                {
                    "claims": [{"text": evidence[key], "evidence_ids": [key]}],
                    "complete": True,
                }
            )
        if name == "citation_review":
            return schema.model_validate(next(self.reviews))
        raise AssertionError(name)


def enable_review(monkeypatch, enabled=True):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true" if enabled else "false")
    settings.cache_clear()


@pytest.mark.parametrize(
    "first_tool,next_tool",
    [
        ("web_search", "read_web"),
        ("web_search", "report_search"),
        ("report_search", "web_search"),
    ],
)
def test_partial_coverage_reaches_next_decision_and_allows_either_source_order(
    monkeypatch, first_tool, next_tool
):
    enable_review(monkeypatch)
    second_action = action(next_tool, "подтверждение экспорта")
    if next_tool == "read_web":
        second_action["url"] = WEB["url"]
    client = ResearchClient(
        [action(first_tool), second_action],
        [review("partial", [MISSING]), review("sufficient")],
    )
    calls = []

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        source = {
            **(REPORT if name == "report_search" else WEB),
            "text": SNIPPET if len(calls) == 1 else EXPORTS,
            "content_type": "extracted" if name == "read_web" else "snippet",
        }
        data = [source] if name == "report_search" else {"items": [source], "notices": []}
        return data, None, event(name)

    result = ask(
        "Какими сортами нефти торгует вымышленная страна?",
        planner=lambda *_: plan(),
        client=client,
        execute=execute,
    )

    assert [name for name, _ in calls] == [first_tool, next_tool]
    assert [entry["tool"] for entry in result["trace"]] == [first_tool, next_tool]
    assert result["status"] == "ok" and EXPORTS in result["answer"]
    assert SNIPPET not in result["answer"]
    checks = result["diagnostics"]["answer_checks"]
    assert checks[0]["claims"] == 1 and not checks[0]["complete"]
    assert checks[0]["validation"]["coverage"] == "partial"
    assert checks[1]["complete"] and checks[1]["validation"]["coverage"] == "sufficient"
    decisions = [payload for name, payload in client.payloads if name == "decide"]
    assert decisions[1]["answer_validation"]["missing_aspects"] == [MISSING]
    assert MISSING in decisions[1]["answer_feedback"]
    assert not decisions[1]["current_answer"]["complete"]
    assert len([name for name, _ in client.payloads if name == "citation_review"]) == 2


@pytest.mark.parametrize("tool", ["report_search", "web_search"])
def test_sufficient_narrow_answer_does_not_force_another_source(monkeypatch, tool):
    enable_review(monkeypatch)
    client = ResearchClient([action(tool)], [review("sufficient")])
    source = {**(REPORT if tool == "report_search" else WEB), "text": SNIPPET}
    calls = []

    def execute(registry, name, **arguments):
        calls.append(name)
        data = [source] if name == "report_search" else {"items": [source], "notices": []}
        return data, None, event(name)

    result = ask(
        "Какие два маркерных сорта названы в справочнике?",
        planner=lambda *_: plan(),
        client=client,
        execute=execute,
    )
    assert result["status"] == "ok" and calls == [tool]
    assert [name for name, _ in client.payloads] == ["decide", "cited_answer", "citation_review"]
    assert result["diagnostics"]["answer_checks"][0]["validation"]["coverage"] == "sufficient"


def test_disabled_review_ends_unverified_current_answer_without_repeated_search(monkeypatch):
    enable_review(monkeypatch, False)
    client = ResearchClient([action("web_search")])
    calls = []

    def execute(registry, name, **arguments):
        calls.append(name)
        return {"items": [WEB], "notices": []}, None, event(name)

    result = ask(
        "Что происходит с поставками сейчас?",
        planner=lambda *_: plan(fresh=True),
        client=client,
        execute=execute,
    )
    assert result["status"] == "partial" and result["stop_reason"] == "review_disabled"
    assert calls == ["web_search"]
    assert [name for name, _ in client.payloads] == ["decide", "cited_answer"]
    assert "отключена" in result["answer"] and result["claims"]
    validation = result["diagnostics"]["answer_checks"][0]["validation"]
    assert validation["candidate_complete"] is True
    assert validation["review_enabled"] is False and validation["review_executed"] is False
    assert validation["coverage"] == "unverified"


def test_disabled_review_preserves_pending_calculation_after_context_first(monkeypatch):
    enable_review(monkeypatch, False)
    client = ResearchClient([action("web_search"), action("price_forecast", "")])
    calls = []
    calculation = {
        "benchmark": "Brent",
        "base": "Brent",
        "as_of": "2026-10",
        "last": 70.0,
        "origins": 12,
        "discount": None,
        "cut": None,
        "sensitivity": None,
        "source_url": "https://example.org/fixture.csv",
        "rows": [
            {
                "month": "2026-11",
                "point": 71.0,
                "lower": 60.0,
                "upper": 80.0,
                "mae": 5.0,
                "naive_mae": 6.0,
            }
        ],
    }

    def execute(registry, name, **arguments):
        calls.append(name)
        data = calculation if name == "price_forecast" else {"items": [WEB], "notices": []}
        return data, None, event(name)

    result = ask(
        "Спрогнозируй Brent на месяц и объясни текущие факторы поставок.",
        planner=lambda *_: plan(intent="forecast", benchmark="Brent", months=1, fresh=True),
        client=client,
        execute=execute,
    )
    assert calls == ["web_search", "price_forecast"]
    assert result["status"] == "partial" and result["stop_reason"] == "review_disabled"
    assert result["forecast"] == calculation and "[M1]" in result["answer"]
    assert [name for name, _ in client.payloads] == ["decide", "cited_answer", "decide"]
    assert "price_forecast" in client.payloads[-1][1]["allowed_actions"]
