"""Regression tests for context delivery and model choice, not LLM quality scores."""

import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import answer_sources, ask, can_finish_unreviewed, decision_sources
from src.clock import clock_payload
from src.evidence import generate_answer


@pytest.mark.parametrize("latest,expected_ids", [(True, ["R1", "W1"]), (False, ["R1"])])
def test_finish_selection_cannot_hide_newer_event_candidate(latest, expected_ids):
    state = {
        "plan": {"latest_completed_event": latest},
        "decision": {"action": "finish", "source_ids": ["R1"]},
        "sources": [REPORT, WEB],
    }
    assert [s["id"] for s in answer_sources(state)] == expected_ids


@pytest.mark.parametrize(
    "chronology,matched,may_finish",
    [
        ({"latest_held_date": None}, False, False),
        ({"latest_held_date": "2026-01-01", "unresolved_evidence_ids": ["W2:B1"]}, True, False),
        (
            {"latest_held_date": "2026-01-01", "missing_requested_context": ["next date"]},
            True,
            False,
        ),
        (
            {"latest_held_date": "2026-01-01", "overdue_scheduled_dates": ["2026-03-20"]},
            True,
            False,
        ),
        ({"latest_held_date": "2026-01-01"}, False, False),
        ({"latest_held_date": "2026-01-01"}, True, True),
    ],
)
def test_unreviewed_answer_only_ends_research_after_verifying_event_chronology(
    chronology, matched, may_finish
):
    state = {
        "plan": plan(latest_completed_event=True).model_dump(),
        "draft": {"claims": ["unreviewed candidate"]},
        "sources": [WEB],
        "answer_checks": [
            {
                "validation": {
                    "candidate_complete": True,
                    "candidate_claims": 1,
                    "accepted_claims": 1,
                    "event_chronology": chronology,
                    "chronology_matches_draft": matched,
                }
            }
        ],
    }
    assert can_finish_unreviewed(state) is may_finish


def test_opened_page_keeps_dates_and_urls_in_decision_context():
    fragment = {**REPORT, "report": "eia", "page": 24}
    page = {
        **fragment,
        "id": "R2",
        "full_page": True,
        "text": REPORT["text"] + "\nReferences: a meeting held in 2024.",
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
            assert payload["today"] == clock_payload()["today"]
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

    client = Scripted([action("report_search", "meeting"), choice], complete=[False, True])
    ask(
        "Когда была последняя встреча?",
        client=client,
        execute=execute,
        planner=lambda *_: plan(fresh=False, rag_query="meeting"),
    )
    payload = [p for name, p in client.payloads if name == "decide"][1]
    assert payload["today"] == clock_payload()["today"]
    assert payload["sources"][0]["date"] == old["date"]
    assert payload["sources"][0]["url"] == old["url"]
    assert payload["task"]["fresh"] is False
    assert payload["research_status"]["completed_actions"] == [
        {"tool": "report_search", "arguments": {"query": "meeting"}}
    ]
    assert payload["research_status"]["report_dates"] == [old["date"]]
    assert payload["research_status"]["web_source_count"] == 0
    assert {"web_search", "report_search", "read_report", "finish"} <= set(
        payload["allowed_actions"]
    )
    assert called == ["report_search", choice["action"]]
