"""Deterministic metrics; no model-generated gold labels."""

from statistics import mean

from src.validation import normalized


def relevant(chunk, gold):
    return (
        chunk.get("report") == gold["report"]
        and chunk.get("page") == gold["page"]
        and normalized(gold["quote"]) in normalized(chunk["text"])
    )


def retrieval_metrics(ranked, golds):
    if not golds:
        raise ValueError("Retrieval evaluation requires gold evidence")
    positions = [
        i for i, chunk in enumerate(ranked[:10], 1) if any(relevant(chunk, g) for g in golds)
    ]
    return {
        "hit@1": float(1 in positions),
        "hit@5": float(any(i <= 5 for i in positions)),
        "mrr@10": 1 / min(positions) if positions else 0.0,
        "evidence_recall@5": sum(any(relevant(c, g) for c in ranked[:5]) for g in golds)
        / len(golds),
    }


def route_metrics(expected, response, plan):
    actual = [t["tool"] for t in response["trace"]]
    wanted = expected["tools"]
    forbidden = expected.get("forbidden", [])
    web_i = actual.index("web_search") if "web_search" in actual else None
    required = expected.get("required_tools", wanted)
    # A pure numerical forecast needs no report lookup. Retain the historical
    # exact sequence only as a diagnostic, not a quality gate for an agent loop.
    if (
        response.get("agent_mode") == "loop"
        and expected.get("intent") == "forecast"
        and expected.get("plan_checks", {}).get("wants_context") is False
        and not expected.get("plan_checks", {}).get("quotes")
    ):
        required = [tool for tool in required if tool != "report_search"]
    required_ok = set(required) <= set(actual) if required else not actual
    policy_ok = not set(actual) & set(forbidden) and (
        web_i is None or ("report_search" in actual and actual.index("report_search") < web_i)
    )
    successful = {t["tool"] for t in response["trace"] if t.get("status", "ok") == "ok"}
    return {
        "tool_policy_pass": float(required_ok and policy_ok),
        "required_tool_success": float(set(required) <= successful),
        "tool_calls": len(actual),
        "rejected_actions": response.get("rejected_actions", 0),
        "decision_count": len(response.get("decisions", [])),
        "duplicate_actions": sum(
            bool(step.get("duplicate_of"))
            for step in (response.get("diagnostics") or {}).get("steps", [])
        ),
        "stopped_no_progress": float(response.get("stop_reason") == "no_progress"),
        "stopped_time_budget": float(response.get("stop_reason") == "time_budget_exhausted"),
        "stopped_research_budget": float(
            response.get("stop_reason") == "research_budget_exhausted"
        ),
        "answer_attempts": len((response.get("diagnostics") or {}).get("answer_checks", [])),
        "tool_sequence_exact": float(actual == wanted),
        "required_tool_recall": len(set(actual) & set(required)) / len(set(required))
        if required
        else float(not actual),
        "forbidden_tool_violation": float(bool(set(actual) & set(forbidden))),
        "rag_before_web": float(
            web_i is None or ("report_search" in actual and actual.index("report_search") < web_i)
        ),
        "intent_correct": float(response["intent"] == expected["intent"]),
        "status_correct": float(response["status"] in expected["statuses"]),
        "plan_fields_correct": float(
            bool(plan) and all(plan.get(k) == v for k, v in expected.get("plan_checks", {}).items())
        ),
    }


def mean_metrics(rows):
    keys = sorted({k for row in rows for k in row})
    return {k: mean(row[k] for row in rows if k in row) for k in keys}
