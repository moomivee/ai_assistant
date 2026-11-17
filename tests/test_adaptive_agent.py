"""Deterministic routing/validation regressions, not a live model quality score."""

from datetime import UTC, datetime

import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src import clock
from src.agent import allowed_actions, ask, decision_sources, task_complete
from src.config import settings
from src.models.schemas import CitedAnswer, CitedClaim


def initial(**changes):
    return {
        "plan": plan(**changes).model_dump(),
        "trace": [],
        "sources": [],
        "calc": None,
        "draft": {"claims": [], "complete": False},
        "feedback": "",
    }


def test_capabilities_depend_on_scope_instead_of_mandatory_source_order():
    assert allowed_actions(initial()) == ["report_search", "web_search"]
    assert allowed_actions(initial(fresh=True)) == ["report_search", "web_search"]
    assert allowed_actions(initial(report_only=True)) == ["report_search"]
    assert allowed_actions(initial(wants_context=False, quotes=["Brent"])) == ["web_search"]
    assert allowed_actions(initial(intent="forecast", wants_context=False)) == ["price_forecast"]


def test_web_presence_cannot_override_failed_answer_validation():
    state = initial(fresh=True)
    state.update(
        sources=[REPORT, WEB],
        draft={"claims": [{"text": "Old answer", "evidence_ids": ["R1:B1"]}], "complete": True},
        feedback="No claims passed citation and numeric checks.",
    )
    assert not task_complete(state)
    state["draft"]["complete"] = False
    state["feedback"] = ""
    assert not task_complete(state)


def test_latest_with_reports_only_cannot_be_announced_verified_today():
    state = initial(fresh=True, report_only=True)
    state["draft"] = {"claims": [{"text": "Historical fact"}], "complete": True}
    assert not task_complete(state)


def test_new_web_evidence_reaches_decision_despite_large_earlier_reports():
    old = [{**REPORT, "id": f"R{i}", "text": ("Old report section. " * 1200)} for i in range(10)]
    result = decision_sources(old + [WEB], priority_source_ids=["W1"])
    assert result[0]["id"] == "W1" and result[0]["text"] == WEB["text"]
    assert any(row["context_truncated"] for row in result)
    assert sum(len(row["text"]) for row in result) <= 14000


@pytest.mark.parametrize("first_tool", ["report_search", "web_search"])
def test_failed_semantic_review_can_lead_to_new_search_and_correct_evidence(
    monkeypatch, first_tool
):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    monkeypatch.setattr(clock, "_utc_now", lambda: datetime(2026, 11, 17, 12, tzinfo=UTC))
    # Fictional dated sources: dates are fixtures, not statements about OPEC.
    old = {**REPORT, "text": "References: Council briefing, July 2025.", "date": "2025-07-24"}
    new = {
        **WEB,
        "text": "On 16 November 2026 the fictional council met and kept output unchanged.",
        "date": "2026-11-16",
        "url": "https://example.org/council/2026-11-16",
    }

    class Reviewer(Scripted):
        def generate(self, name, payload, schema):
            if name == "cited_answer":
                self.payloads.append((name, payload))
                key = next(
                    (k for k, text in payload["evidence"].items() if text == new["text"]), None
                )
                return CitedAnswer(
                    claims=[
                        CitedClaim(
                            evidence_ids=[key or next(iter(payload["evidence"]))],
                            text=(
                                new["text"]
                                if key
                                else "Последняя встреча состоялась в июле 2025 года."
                            ),
                        )
                    ],
                    complete=True,
                )
            if name == "citation_review":
                self.payloads.append((name, payload))
                valid = payload["claims"][0]["evidence"][0]["text"] == new["text"]
                return schema(
                    supported=[0] if valid else [],
                    complete=valid,
                    feedback=""
                    if valid
                    else "Bibliography does not establish the latest meeting; verify current events.",
                )
            return super().generate(name, payload, schema)

    client = Reviewer(
        [
            action(first_tool, "council meeting"),
            action("web_search", "council latest meeting official statement"),
        ]
    )
    calls = []

    def execute(registry, name, **args):
        calls.append((name, args))
        source = old if len(calls) == 1 else new
        data = (
            [source]
            if name == "report_search"
            else {"items": [{**source, "kind": "web"}], "notices": []}
        )
        return data, None, event(name)

    result = ask(
        "Когда было последнее заседание вымышленного совета и что решили?",
        client=client,
        planner=lambda *_: plan(fresh=True),
        execute=execute,
    )
    assert [name for name, _ in calls] == [first_tool, "web_search"]
    assert result["status"] == "ok" and "2025" not in result["answer"]
    assert result["sources"][0]["url"] == new["url"]
    checks = result["diagnostics"]["answer_checks"]
    assert checks[0]["claims"] == 0 and not checks[0]["complete"]
    assert checks[1]["complete"] and not checks[1]["feedback"]
    assert all(p["today"] == "2026-11-17" for _, p in client.payloads)
    assert result["diagnostics"]["time_context"]["timezone"] == "Europe/Moscow"
    second_decision = [p for n, p in client.payloads if n == "decide"][1]
    assert "Bibliography" in second_decision["answer_feedback"]
    assert not second_decision["current_answer"]["complete"]


def test_request_date_remains_fixed_across_midnight_and_all_graph_stages(monkeypatch):
    moments = iter(
        [
            datetime(2026, 12, 31, 20, 59, tzinfo=UTC),
            datetime(2026, 12, 31, 21, 1, tzinfo=UTC),
        ]
    )
    monkeypatch.setattr(clock, "_utc_now", lambda: next(moments))
    client = Scripted([action("web_search", "oil facts")])
    result = ask(
        "Факт об отрасли",
        client=client,
        planner=lambda *_: plan(),
        execute=lambda *a, **kw: ({"items": [WEB], "notices": []}, None, event("web_search")),
    )
    assert all(p["today"] == "2026-12-31" for _, p in client.payloads)
    assert result["diagnostics"]["time_context"]["current_year"] == 2026
    assert clock.clock_payload()["today"] == "2027-01-01"


def test_unavailable_web_can_finish_partial_without_mandatory_pdf_search():
    client = Scripted([action("web_search", "latest event"), action("finish")])
    response = ask(
        "Последнее событие?",
        client=client,
        planner=lambda *_: plan(fresh=True),
        execute=lambda *a, **kw: (
            {"items": [], "notices": ["Search unavailable"]},
            None,
            event("web_search"),
        ),
    )
    assert [x["tool"] for x in response["trace"]] == ["web_search"]
    assert response["status"] == "partial" and not response["claims"]
    assert "Search unavailable" in response["answer"]
