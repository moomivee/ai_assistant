"""Observed conflicting date arguments must be repaired before the provider call."""

from datetime import date, timedelta

import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import action, event

from src.agent import ask, repair_web_time_filter
from src.budget import RequestBudgetExceeded
from src.integrations.llm import LLMUnavailable
from src.integrations.web import validate_search_options
from src.models.schemas import NextAction


def proposal(**overrides):
    return NextAction.model_validate(
        {
            **action("web_search", "latest OPEC+ meeting"),
            "days": 30,
            "start_date": (date.today() - timedelta(days=31)).isoformat(),
            "end_date": date.today().isoformat(),
            "search_depth": "advanced",
            **overrides,
        }
    ).model_dump()


class ChoosingModel:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append((name, payload))
        if name == "decide":
            return schema.model_validate(proposal())
        if name == "web_time_filter":
            if isinstance(self.mode, Exception):
                raise self.mode
            return schema.model_validate({"mode": self.mode})
        if name == "cited_answer":
            keys = [key for key in payload["evidence"] if key.startswith("W")]
            return schema.model_validate(
                {
                    "claims": [
                        {"text": payload["evidence"][key], "evidence_ids": [key]} for key in keys
                    ],
                    "complete": bool(keys),
                }
            )
        if name == "citation_review":
            return schema.model_validate(
                {"supported": list(range(len(payload["claims"]))), "complete": True, "feedback": ""}
            )
        raise AssertionError(name)


@pytest.mark.parametrize("mode", ["unbounded", "relative", "dates"])
def test_model_chooses_exclusive_window_without_changing_query_or_tool(mode):
    client, before = ChoosingModel(mode), proposal()
    repaired = repair_web_time_filter(client, "Latest meeting?", "previous turn", before)
    validate_search_options(**{k: repaired[k] for k in ("days", "start_date", "end_date")})
    assert before == proposal()  # Caller data is not modified.
    assert repaired["action"] == before["action"]
    assert repaired["query"] == before["query"]
    assert repaired["search_depth"] == before["search_depth"]
    assert len(client.calls) == 1
    assert client.calls[0][1]["question"] == "Latest meeting?"
    assert client.calls[0][1]["previous"] == "previous turn"
    assert repaired["days"] == (30 if mode == "relative" else None)
    for key in ("start_date", "end_date"):
        assert repaired[key] == (before[key] if mode == "dates" else None)
    assert repaired["argument_repair"]["mode"] == mode
    assert repaired["argument_repair"]["before"] == client.calls[0][1]["proposed"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"days": None},
        {"start_date": None, "end_date": None},
        {"start_date": "", "end_date": ""},
        {"action": "report_search"},
        {"action": "finish"},
    ],
)
def test_valid_windows_and_other_tools_do_not_add_an_llm_call(overrides):
    client, original = ChoosingModel("unbounded"), proposal(**overrides)
    assert repair_web_time_filter(client, "Question", None, original) == original
    assert client.calls == []


@pytest.mark.parametrize("mode", ["unbounded", "relative", "dates"])
def test_loop_executes_repaired_action_and_records_both_versions(mode, monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    client, called = ChoosingModel(mode), []

    def execute(registry, name, **args):
        called.append((name, args))
        if name == "report_search":
            return [REPORT], None, event(name)
        assert name == "web_search"
        validate_search_options(**{k: args.get(k) for k in ("days", "start_date", "end_date")})
        return {"items": [WEB], "notices": []}, None, event(name)

    response = ask(
        "Когда была последняя встреча ОПЕК+?",
        client=client,
        planner=lambda *_: plan(fresh=True),
        execute=execute,
    )
    assert [name for name, _ in called] == ["web_search"]
    assert response["status"] == "ok"
    assert response["rejected_actions"] == 0
    assert [name for name, _ in client.calls].count("web_time_filter") == 1
    decision = response["decisions"][0]
    step = response["diagnostics"]["steps"][0]
    assert step["argument_repair"] == decision["argument_repair"]
    assert step["argument_repair"]["before"]["days"] == 30
    assert step["arguments"] == called[0][1]


@pytest.mark.parametrize(
    "failure,stop",
    [
        (LLMUnavailable("repair unavailable"), "llm_unavailable"),
        (RequestBudgetExceeded(), "time_budget_exhausted"),
    ],
)
def test_failed_repair_never_sends_invalid_arguments(failure, stop):
    called = []

    def execute(registry, name, **args):
        called.append(name)
        assert name == "report_search"
        return [REPORT], None, event(name)

    response = ask(
        "Последние новости?",
        client=ChoosingModel(failure),
        planner=lambda *_: plan(fresh=True),
        execute=execute,
    )
    assert called == []
    assert response["status"] == "unavailable"
    assert response["stop_reason"] == stop
