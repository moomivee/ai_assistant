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
    forbidden = set(expected.get("forbidden", []))
    if "web_search" in forbidden or expected.get("plan_checks", {}).get("report_only"):
        forbidden.update({"web_search", "read_web"})
    web_i = next((i for i, tool in enumerate(actual) if tool in {"web_search", "read_web"}), None)
    required = expected.get("required_tools", wanted)
    groups = expected.get("required_tool_groups", [])
    adaptive = expected.get("routing_policy") == "adaptive-v3"
    # A pure numerical forecast needs no report lookup. Retain the historical
    # exact sequence only as a diagnostic, not a quality gate for an agent loop.
    if (
        not adaptive
        and response.get("agent_mode") == "loop"
        and expected.get("intent") == "forecast"
        and expected.get("plan_checks", {}).get("wants_context") is False
        and not expected.get("plan_checks", {}).get("quotes")
    ):
        required = [tool for tool in required if tool != "report_search"]
    requirements = [set([tool]) for tool in required] + [set(group) for group in groups]
    required_ok = all(group & set(actual) for group in requirements) if requirements else not actual
    rag_first = web_i is None or (
        "report_search" in actual and actual.index("report_search") < web_i
    )
    policy_ok = not set(actual) & forbidden and (adaptive or rag_first)
    if "read_web" in actual:
        policy_ok = (
            policy_ok
            and "web_search" in actual
            and actual.index("web_search") < actual.index("read_web")
        )
    successful = {t["tool"] for t in response["trace"] if t.get("status", "ok") == "ok"}
    metrics = {
        "tool_policy_pass": float(required_ok and policy_ok),
        "required_tool_success": float(
            all(group & successful for group in requirements) if requirements else not actual
        ),
        "tool_calls": len(actual),
        "rejected_actions": response.get("rejected_actions", 0),
        "decision_count": len(response.get("decisions", [])),
        "model_decision_count": sum(
            d.get("origin") != "plan_policy" for d in response.get("decisions", [])
        ),
        "policy_decision_count": sum(
            d.get("origin") == "plan_policy" for d in response.get("decisions", [])
        ),
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
        "required_tool_recall": sum(bool(group & set(actual)) for group in requirements)
        / len(requirements)
        if requirements
        else float(not actual),
        "forbidden_tool_violation": float(bool(set(actual) & set(forbidden))),
        "rag_before_web": float(rag_first),
        "adaptive_routing_policy": float(adaptive),
        "intent_correct": float(response["intent"] == expected["intent"]),
        "status_correct": float(response["status"] in expected["statuses"]),
        "plan_fields_correct": float(
            bool(plan) and all(plan.get(k) == v for k, v in expected.get("plan_checks", {}).items())
        ),
    }
    checks = (response.get("diagnostics") or {}).get("answer_checks", [])
    if checks:
        last = checks[-1]
        # This is a consistency check, not independent semantic evaluation.
        # A web call or a matching source date cannot establish answer freshness.
        metrics["final_review_consistency"] = float(
            response.get("status") != "ok"
            or (
                bool(last.get("complete"))
                and not (last.get("feedback") or "").strip()
                and not last.get("error")
            )
        )
    return metrics


def mean_metrics(rows):
    keys = sorted({k for row in rows for k in row})
    return {k: mean(row[k] for row in rows if k in row) for k in keys}
