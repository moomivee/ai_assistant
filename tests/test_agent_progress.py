"""General stopping/recovery behavior with scripted decisions, not a Qwen eval."""

import pytest
from test_agent import REPORT, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask


def report(text, **kwargs):
    return {**REPORT, "report": "fixture-report", "text": text, **kwargs}


def test_search_read_ping_pong_stops_and_reuses_results():
    calls = []
    client = Scripted(
        [
            action("report_search", "composition"),
            action("read_report", report="fixture-report", page=6),
            action("read_report", report="fixture-report", page=6),
            action("report_search", "composition"),
        ],
        complete=False,
    )

    def execute(registry, name, **args):
        calls.append(name)
        text = "Composition table" if name == "report_search" else "Composition table\nNew fact"
        return [report(text, full_page=name == "read_report")], None, event(name)

    response = ask(
        "Сравни состав",
        client=client,
        execute=execute,
        planner=lambda *_: plan(rag_query="composition"),
    )
    assert calls == ["report_search", "read_report"]
    assert len(response["decisions"]) == 5
    assert response["stop_reason"] == "no_progress"
    steps = response["diagnostics"]["steps"]
    assert [s["no_progress_steps"] for s in steps] == [0, 1, 0, 1, 2]
    assert steps[1]["duplicate_of"] == 1 and steps[1]["reused_source_ids"] == ["R1"]
    assert steps[3]["duplicate_of"] == 3 and steps[3]["reused_source_ids"] == ["R2"]
    assert steps[4]["ms"] is None
    assert response["claims"] and "New fact" in response["answer"]
    assert "Attempt" not in response["answer"]


def test_duplicate_does_not_disable_changed_search_when_other_tools_are_available():
    client = Scripted(
        [
            action("report_search", "broad"),
            action("report_search", "specific missing property"),
            action("finish"),
        ],
        complete=[False, True],
    )
    calls = []

    def execute(registry, name, **args):
        calls.append(args["query"])
        return [report(args["query"])], None, event(name)

    response = ask(
        "Сравни свойства",
        client=client,
        execute=execute,
        planner=lambda *_: plan(rag_query="broad"),
    )
    third = [p for n, p in client.payloads if n == "decide"][1]
    assert {"report_search", "read_report", "web_search", "finish"} <= set(third["allowed_actions"])
    assert calls == ["broad", "specific missing property"]
    assert response["diagnostics"]["steps"][2]["new_evidence"] is True
    assert response["stop_reason"] == "completed"


def test_new_queries_and_urls_with_same_content_do_not_extend_loop():
    calls = []
    client = Scripted([action("report_search", q) for q in ("second", "third")], complete=False)

    def execute(registry, name, **args):
        calls.append(args["query"])
        return (
            [report("Same   fact", url="https://example.org/" + args["query"])],
            None,
            event(name),
        )

    response = ask(
        "Найди факт", client=client, execute=execute, planner=lambda *_: plan(rag_query="first")
    )
    assert len(calls) == 3
    assert response["stop_reason"] == "no_progress"
    assert [s["new_evidence"] for s in response["diagnostics"]["steps"]] == [True, False, False]
    assert response["diagnostics"]["no_progress_steps"] == 2


def test_query_case_and_whitespace_do_not_bypass_duplicate_guard():
    response = ask(
        "Найди факт",
        client=Scripted([action("report_search", q) for q in (" OIL  sulfur ", "oil\nsulfur")]),
        planner=lambda *_: plan(report_only=True, rag_query="Oil sulfur"),
        execute=lambda *a, **kw: ([], None, event("report_search")),
    )
    assert len(response["trace"]) == 1
    assert response["stop_reason"] == "no_progress" and response["status"] == "partial"
    assert response["claims"] == [] and response["sources"] == []
    assert "Attempt" not in response["answer"]


@pytest.mark.parametrize(
    "assessment,missing,ids,finish_early",
    [
        ("enough", [], ["R1"], True),
        ("enough", ["Another property"], ["R1"], False),
        ("enough", [], ["R999"], False),
        ("enough", [], [], False),
        ("need_evidence", ["Another property"], ["R1"], False),
    ],
)
def test_referenced_sufficiency_can_finish_but_missing_or_invalid_evidence_cannot(
    assessment, missing, ids, finish_early
):
    decision = {
        **action("report_search", "another property"),
        "assessment": assessment,
        "missing_facts": missing,
        "evidence_checks": [{"fact": "Property found", "source_ids": ids}],
    }
    client = Scripted([decision, action("finish")], complete=False)
    calls = []

    def execute(registry, name, **args):
        calls.append(args["query"])
        return [report(args["query"])], None, event(name)

    response = ask(
        "Свойства?", client=client, execute=execute, planner=lambda *_: plan(rag_query="property")
    )
    second = response["decisions"][1]
    assert len(calls) == (1 if finish_early else 2)
    if finish_early:
        assert second["action"] == "finish" and second["model_action"] == "report_search"
        assert second["controller_note"] and response["claims"]
    else:
        assert second["action"] == "report_search"
        assert "model_action" not in second
    assert second["unknown_source_ids"] == (["R999"] if ids == ["R999"] else [])


def test_sufficiency_does_not_skip_required_fresh_web_check():
    client = Scripted(
        [
            {
                **action("web_search", "supply today"),
                "assessment": "enough",
                "evidence_checks": [{"fact": "Historical supply", "source_ids": ["R1"]}],
            },
            action("finish"),
        ]
    )
    calls = []

    def execute(registry, name, **args):
        calls.append(name)
        data = (
            [report("Historical supply")]
            if name == "report_search"
            else {"items": [], "notices": []}
        )
        return data, None, event(name)

    response = ask(
        "Спрос сегодня?",
        client=client,
        execute=execute,
        planner=lambda *_: plan(fresh=True, rag_query="supply"),
    )
    assert calls == ["report_search", "web_search"]
    assert response["status"] == "partial"
    assert response["decisions"][1]["action"] == "web_search"
