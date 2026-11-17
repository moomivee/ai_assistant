"""Routing labels describe necessary evidence, not an imposed sequence of searches."""

import json

import pytest

from eval.build_adaptive_seed import DATA, build
from eval.metrics import route_metrics
from eval.run import load_cases
from eval.smoke import check_result


def expected(**values):
    return {
        "routing_policy": "adaptive-v3",
        "tools": ["report_search", "web_search"],
        "required_tools": ["web_search"],
        "required_tool_groups": [],
        "intent": "research",
        "statuses": ["ok", "partial"],
        "plan_checks": {"fresh": True},
        **values,
    }


def response(*tools, status="ok", checks=None):
    return {
        "trace": [{"tool": tool, "status": "ok"} for tool in tools],
        "intent": "research",
        "status": status,
        "agent_mode": "loop",
        "diagnostics": {"answer_checks": checks or []},
    }


def test_adaptive_latest_can_start_and_finish_with_web_without_rag():
    metrics = route_metrics(expected(), response("web_search", "read_web"), {"fresh": True})
    assert metrics["tool_policy_pass"] == metrics["required_tool_success"] == 1
    assert metrics["tool_sequence_exact"] == metrics["rag_before_web"] == 0
    assert metrics["adaptive_routing_policy"] == 1


@pytest.mark.parametrize("tool", ["web_search", "report_search"])
def test_general_research_allows_either_evidence_source(tool):
    gold = expected(required_tools=[], required_tool_groups=[["report_search", "web_search"]])
    metrics = route_metrics(gold, response(tool), {"fresh": True})
    assert metrics["tool_policy_pass"] == metrics["required_tool_recall"] == 1
    assert route_metrics(gold, response(), {})["tool_policy_pass"] == 0


def test_predicted_plan_never_relaxes_current_or_combined_gold_requirements():
    wrong_plan = {"fresh": False, "report_only": True}
    assert route_metrics(expected(), response("report_search"), wrong_plan)["tool_policy_pass"] == 0
    gold = expected(required_tools=["report_search", "web_search"])
    assert route_metrics(gold, response("web_search"), wrong_plan)["tool_policy_pass"] == 0
    assert (
        route_metrics(gold, response("web_search", "report_search"), wrong_plan)["tool_policy_pass"]
        == 1
    )


def test_report_only_and_discovery_restrictions_remain_enforced():
    gold = expected(required_tools=["report_search"], plan_checks={"report_only": True})
    assert route_metrics(gold, response("report_search", "web_search"), {})["tool_policy_pass"] == 0
    assert (
        route_metrics(expected(), response("read_web", "web_search"), {})["tool_policy_pass"] == 0
    )


def test_failed_required_tool_is_not_a_successful_route():
    result = response("web_search")
    result["trace"][0]["status"] = "error"
    metrics = route_metrics(expected(), result, {"fresh": True})
    assert metrics["tool_policy_pass"] == 1
    assert metrics["required_tool_success"] == 0


@pytest.mark.parametrize("tool_status, passed", [("error", False), ("ok", True)])
def test_live_smoke_requires_successful_tool_even_for_allowed_partial_answer(tool_status, passed):
    gold = expected()
    answer = response(
        "web_search", status="partial", checks=[{"complete": False, "feedback": "Missing facts"}]
    )
    answer["trace"][0]["status"] = tool_status
    metrics = route_metrics(gold, answer, {"fresh": True})
    assert metrics["tool_policy_pass"] == metrics["status_correct"] == 1
    result = {"rows": [{"case_id": "latest", "response": answer, "metrics": metrics}]}
    check = check_result(result, [{"id": "latest", "expected": gold}])[0]
    assert check["route_ok"] is passed and check["passed"] is passed
    assert not check["semantic_quality_independently_measured"]


def test_stale_draft_resurrection_cannot_pass_final_review_consistency():
    # Observed trace: any web hit enabled success despite rejected final claims.
    result = response(
        "report_search",
        "web_search",
        checks=[{"complete": True, "feedback": "No claims passed citation and numeric checks."}],
    )
    metrics = route_metrics(expected(), result, {"fresh": True})
    assert metrics["tool_policy_pass"] == 1
    assert metrics["final_review_consistency"] == 0
    result["status"] = "partial"
    assert route_metrics(expected(), result, {})["final_review_consistency"] == 1


def test_adaptive_seed_is_reproducible_and_preserves_retrieval_gold_and_splits():
    old = load_cases(DATA / "cases.jsonl")
    new = load_cases(DATA / "cases_v3.jsonl")
    assert new == build() and len(new) == 100
    assert sum(bool(row["relevant"]) for row in new) == 40
    for before, after in zip(old, new, strict=True):
        for key in ("id", "question", "relevant", "split", "group", "retrieval_query"):
            assert before[key] == after[key]
        assert after["expected"]["routing_policy"] == "adaptive-v3"
        if after["category"] == "web":
            assert after["expected"]["required_tools"] == ["web_search"]
        elif after["category"] == "mixed":
            assert after["expected"]["required_tools"] == ["report_search", "web_search"]


def test_adaptive_regressions_are_authored_gold_not_a_claim_of_measured_quality():
    rows = load_cases(DATA / "adaptive_regressions.jsonl")
    assert len(rows) == 10 and len({r["id"] for r in rows}) == 10
    assert all(r["split"] == "dev" for r in rows)
    assert all(r["provenance"]["human_review"] == "pending" for r in rows)
    latest = next(r for r in rows if r["id"] == "adaptive-latest-meeting")
    assert latest["expected"]["required_tools"] == ["web_search"]
    assert "июлем 2025" in latest["expected"]["answer_rubric"]
    # Dataset includes no manufactured observed model answers.
    assert "observed_response" not in json.dumps(rows)
