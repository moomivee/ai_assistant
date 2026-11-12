"""Regression tests for context delivery and model choice, not LLM quality scores."""

from datetime import date

import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask, decision_sources
from src.evidence import generate_answer


def test_opened_page_keeps_dates_and_urls_in_decision_context():
    fragment = {**REPORT, "report": "eia", "page": 24}
    page = {
        **fragment,
        "id": "R2",
        "full_page": True,
        "text": "References: a meeting held in 2024.",
    }
    web = {**WEB, "date": None}
    context = decision_sources([fragment, page, web])
    assert [s["id"] for s in context] == ["R2", "W1"]
    assert context[0]["date"] == page["date"]
    assert context[0]["url"] == page["url"]
    assert context[0]["page"] == 24 and context[0]["full_page"]
    assert context[1]["date"] is None  # Unknown dates must not become today's date.
    assert context[1]["url"] == web["url"]


def test_answer_generation_gets_today_and_original_source_dates():
    class Capture:
        def generate(self, name, payload, schema):
            assert name == "cited_answer"
            assert payload["today"] == date.today().isoformat()
            assert payload["sources"][0]["date"] == REPORT["date"]
            assert payload["sources"][0]["url"] == REPORT["url"]
            assert payload["evidence"]["R1:B1"] == REPORT["text"]
            return schema(claims=[], complete=False)

    draft, _, _ = generate_answer(Capture(), "Когда было последнее событие?", [REPORT])
    assert not draft.complete and not draft.claims


@pytest.mark.parametrize(
    "choice",
    [
        action("web_search", "organisation latest meeting official announcement"),
        action("report_search", "different historical evidence"),
        action("read_report", report="eia", page=24),
    ],
)
def test_stale_context_leaves_next_tool_to_model_even_if_plan_missed_freshness(choice):
    called = []
    old = {**REPORT, "report": "eia", "page": 24, "text": "References to past meetings."}

    def execute(registry, name, **args):
        called.append(name)
        if len(called) == 1:
            data = [old]
        elif name == "web_search":
            data = {"items": [WEB], "notices": []}
        else:
            data = [{**REPORT, "url": "https://eia.gov/other.pdf", "text": "Additional fact."}]
        return data, None, event(name)

    client = Scripted([choice], complete=[False, True])
    ask(
        "Когда была последняя встреча?",
        client=client,
        execute=execute,
        planner=lambda *_: plan(fresh=False, rag_query="meeting"),
    )
    payload = next(p for name, p in client.payloads if name == "decide")
    assert payload["today"] == date.today().isoformat()
    assert payload["sources"][0]["date"] == old["date"]
    assert payload["sources"][0]["url"] == old["url"]
    assert payload["task"]["fresh"] is False
    assert {"web_search", "report_search", "read_report", "finish"} <= set(
        payload["allowed_actions"]
    )
    assert called == ["report_search", choice["action"]]
