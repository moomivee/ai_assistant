"""Coverage rubrics are semantic gold; routes alone cannot establish completeness."""

import pytest

from eval.metrics import route_metrics
from eval.run import load_cases, validate_data
from src.config import ROOT

DATA = ROOT / "eval/data/coverage_regressions.jsonl"


def case(case_id):
    return next(row for row in load_cases(DATA) if row["id"] == case_id)


def response(*tools, status="ok"):
    return {
        "trace": [{"tool": tool, "status": "ok"} for tool in tools],
        "intent": "research",
        "status": status,
        "agent_mode": "loop",
    }


def test_coverage_cases_validate_as_authored_dev_rubrics_without_fabricated_contexts():
    summary = validate_data(DATA)
    assert summary["splits"]["test"] == 0
    assert summary["retrieval_cases"] == 0
    for row in load_cases(DATA):
        assert row["provenance"]["human_review"] == "pending"
        assert "observed_response" not in row


@pytest.mark.parametrize("tool", ["report_search", "web_search"])
def test_broad_and_explicit_example_requests_do_not_require_both_searches(tool):
    for case_id in ("coverage-russia-overview", "coverage-two-examples"):
        gold = case(case_id)["expected"]
        metrics = route_metrics(gold, response(tool), {"wants_context": True, "fresh": False})
        assert metrics["tool_policy_pass"] == metrics["required_tool_success"] == 1
    # The trace's web-first route is allowed. Its incomplete answer must be
    # assessed against the rubric/evidence, not relabelled as a routing failure.


def test_current_supply_requires_web_but_historical_supply_does_not():
    wrong_plan = {"fresh": False, "report_only": True}
    current = case("coverage-current-exports")["expected"]
    historical = case("coverage-historical-exports")["expected"]
    assert route_metrics(current, response("report_search"), wrong_plan)["tool_policy_pass"] == 0
    assert route_metrics(current, response("web_search"), wrong_plan)["tool_policy_pass"] == 1
    assert route_metrics(historical, response("report_search"), wrong_plan)["tool_policy_pass"] == 1


def test_current_report_only_question_keeps_web_forbidden_and_status_partial():
    gold = case("coverage-current-report-only")["expected"]
    metrics = route_metrics(gold, response("report_search"), {"report_only": False})
    assert metrics["tool_policy_pass"] == 1 and metrics["status_correct"] == 0
    assert (
        route_metrics(gold, response("report_search", status="partial"), {})["status_correct"] == 1
    )
    assert route_metrics(gold, response("report_search", "web_search"), {})["tool_policy_pass"] == 0


def test_explicit_report_and_current_supply_comparison_needs_both_in_either_order():
    gold = case("coverage-combined-supply")["expected"]
    assert route_metrics(gold, response("web_search"), {})["tool_policy_pass"] == 0
    for tools in (("report_search", "web_search"), ("web_search", "report_search")):
        assert route_metrics(gold, response(*tools), {})["tool_policy_pass"] == 1


@pytest.mark.parametrize(
    "validation",
    [
        {"coverage": "partial"},
        {"coverage": "unverified"},
        {"coverage": "sufficient", "missing_aspects": ["Не проверен состав поставок."]},
        {"review_enabled": False},
        {"review_executed": False},
    ],
)
def test_ok_cannot_hide_missing_coverage_or_disabled_review(validation):
    gold = case("coverage-russia-overview")["expected"]
    actual = response("web_search")
    actual["diagnostics"] = {
        "answer_checks": [{"complete": True, "feedback": "", "validation": validation}]
    }
    assert route_metrics(gold, actual, {})["final_review_consistency"] == 0
    actual["status"] = "partial"
    assert route_metrics(gold, actual, {})["final_review_consistency"] == 1
