"""Harness tests with scripted LLMs. Live model quality is measured separately."""

import pytest
from test_agent import REPORT, WEB, plan

from eval.metrics import route_metrics
from src.agent import ask, merge_sources
from src.evidence import catalog, generate_answer
from src.models.schemas import CitedAnswer, CitedClaim, NextAction
from src.tools import make_registry, read_report, run_tool


def action(name, query="", report="", page=None):
    return dict(action=name, query=query, report=report, page=page, purpose="Check evidence")


class Scripted:
    def __init__(self, actions):
        self.actions, self.payloads = iter(actions), []

    def generate(self, name, payload, schema):
        self.payloads.append((name, payload))
        if name == "decide":
            return NextAction.model_validate(next(self.actions))
        if name == "cited_answer":
            key = next(iter(payload["evidence"]))
            return schema.model_validate(
                {
                    "claims": [{"text": payload["evidence"][key], "evidence_ids": [key]}],
                    "complete": True,
                }
            )
        if name == "citation_review":
            return schema.model_validate(
                {"supported": list(range(len(payload["claims"]))), "complete": True, "feedback": ""}
            )
        raise AssertionError(name)


def event(name):
    return {"tool": name, "engine": "test.fixture", "status": "ok", "ms": 1}


def test_replans_after_empty_search_and_reads_full_page():
    calls = []

    def execute(registry, name, **args):
        calls.append((name, args))
        data = (
            []
            if len(calls) == 1
            else [{**REPORT, "report": "eia", "full_page": name == "read_report"}]
        )
        return data, None, event(name)

    client = Scripted(
        [
            action("report_search", "broad"),
            action("report_search", "focused"),
            action("read_report", report="eia", page=6),
            action("finish"),
        ]
    )
    response = ask(
        "Сера Sokol?", client=client, execute=execute, planner=lambda *_: plan(report_only=True)
    )
    assert response["status"] == "ok"
    assert [x[0] for x in calls] == ["report_search", "report_search", "read_report"]
    assert calls[0][1] != calls[1][1]
    second = [p for n, p in client.payloads if n == "decide"][1]
    assert second["observations"][0]["result_count"] == 0
    assert response["agent_mode"] == "loop" and response["claims"][0]["evidence_ids"]


def test_model_cannot_skip_report_priority():
    called = []

    def execute(registry, name, **args):
        called.append(name)
        return (
            ([REPORT] if name == "report_search" else {"items": [WEB], "notices": []}),
            None,
            event(name),
        )

    client = Scripted(
        [
            action("web_search", "news"),
            action("report_search", "oil"),
            action("web_search", "news"),
            action("finish"),
        ]
    )
    response = ask(
        "Новости нефти", client=client, execute=execute, planner=lambda *_: plan(fresh=True)
    )
    assert called == ["report_search", "web_search"]
    assert [
        x for n, p in client.payloads if n == "decide" for x in p["observations"] if x.get("error")
    ]
    assert response["status"] == "ok"


def test_repeated_actions_stop_at_budget(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_NO_PROGRESS_LIMIT", "12")
    calls = []

    def execute(registry, name, **args):
        calls.append(name)
        return [], None, event(name)

    client = Scripted([action("report_search", "same")] * 3)
    response = ask(
        "Найди факт", client=client, execute=execute, planner=lambda *_: plan(report_only=True)
    )
    assert calls == ["report_search"]
    assert len(response["decisions"]) == 3
    assert response["status"] == "partial" and response["stop_reason"] == "budget_exhausted"


def test_pure_forecast_uses_calculator_without_forced_search():
    p = plan(intent="forecast", benchmark="WTI", months=2, wants_context=False)
    client = Scripted([action("price_forecast"), action("finish")])
    response = ask("Рассчитай WTI на 2 месяца", client=client, planner=lambda *_: p)
    assert [x["tool"] for x in response["trace"]] == ["price_forecast"]
    assert response["forecast"]["benchmark"] == "WTI" and len(response["forecast"]["rows"]) == 2
    assert response["status"] == "ok"


def test_report_only_web_attempt_is_not_executed(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "2")

    def execute(*args, **kwargs):
        pytest.fail("Forbidden tool must not reach executor")

    response = ask(
        "Только PDF",
        client=Scripted([action("web_search", "oil")] * 2),
        execute=execute,
        planner=lambda *_: plan(report_only=True),
    )
    assert response["trace"] == [] and response["status"] == "partial"


def test_source_ids_do_not_collide_between_searches():
    rows = merge_sources(
        [REPORT], [{**REPORT, "url": "https://eia.gov/another.pdf", "text": "Another fact"}]
    )
    assert [x["id"] for x in rows] == ["R1", "R2"]
    assert len(merge_sources(rows, [REPORT])) == 2


@pytest.mark.parametrize("text,keys", [("Сера 99%.", ["R1:B1"]), ("Сера 0.2%.", ["invented"])])
def test_unsupported_numbers_and_fake_citations_rejected(text, keys):
    class Client:
        def generate(self, name, payload, schema):
            assert name == "cited_answer"
            return CitedAnswer(claims=[CitedClaim(text=text, evidence_ids=keys)], complete=True)

    draft, _, _ = generate_answer(Client(), "Сера?", [REPORT])
    assert not draft.claims and not draft.complete


def test_full_page_supersedes_cut_fragments_in_model_context():
    sources = [
        {**REPORT, "report": "r", "page": 1},
        {
            **REPORT,
            "id": "R2",
            "report": "r",
            "page": 1,
            "full_page": True,
            "text": "Header\nSokol 0.2%",
        },
    ]
    evidence = catalog(sources)
    assert set(evidence) == {"R2:B1"}
    assert evidence["R2:B1"]["text"] == "Header\nSokol 0.2%"


def test_read_report_uses_registered_file_and_preserves_rows():
    page = read_report("eia-russia-2025", 6)[0]
    assert page["page"] == 6 and page["full_page"]
    assert "Urals" in page["text"] and "ESPO" in page["text"]
    with pytest.raises(ValueError):
        read_report("../../outside", 1)
    data, error, event_ = run_tool(
        make_registry(), "read_report", report="eia-russia-2025", page=9999
    )
    assert data is None and error and event_["status"] == "error"


def test_eval_accepts_valid_alternative_routes_without_loosening_policy():
    expected = {
        "tools": ["report_search"],
        "forbidden": ["web_search"],
        "intent": "research",
        "statuses": ["ok"],
    }
    response = {
        "trace": [event("report_search"), event("read_report"), event("report_search")],
        "intent": "research",
        "status": "ok",
        "agent_mode": "loop",
    }
    metrics = route_metrics(expected, response, {"intent": "research"})
    assert metrics["tool_sequence_exact"] == 0 and metrics["tool_policy_pass"] == 1
    response["trace"].append(event("web_search"))
    assert route_metrics(expected, response, {})["tool_policy_pass"] == 0


def test_answer_context_uses_model_selected_source_ids():
    other = {
        **REPORT,
        "id": "R2",
        "url": "https://eia.gov/other",
        "text": "Unrelated refinery fact.",
    }
    client = Scripted(
        [action("report_search", "Sokol"), {**action("finish"), "source_ids": ["R1"]}]
    )
    result = ask(
        "Сера Sokol",
        client=client,
        planner=lambda *_: plan(report_only=True),
        execute=lambda *a, **kw: ([REPORT, other], None, event("report_search")),
    )
    payload = next(p for n, p in client.payloads if n == "cited_answer")
    assert all(key.startswith("R1:") for key in payload["evidence"])
    assert result["status"] == "ok"


def test_api_returns_new_agent_contract(monkeypatch):
    from fastapi.testclient import TestClient

    from src import api

    client = Scripted([action("report_search", "Sokol"), action("finish")])
    monkeypatch.setattr(
        api,
        "ask",
        lambda q, p: ask(
            q,
            p,
            client=client,
            planner=lambda *_: plan(report_only=True),
            execute=lambda *a, **kw: ([REPORT], None, event("report_search")),
        ),
    )
    with TestClient(api.app) as http:
        result = http.post("/api/ask", json={"question": "Сера Sokol"}).json()
    assert result["agent_mode"] == "loop" and result["status"] == "ok"
    assert result["claims"][0]["evidence_ids"] and result["sources"][0]["evidence"]


def test_action_schema_contains_only_current_capabilities():
    from src.agent import action_schema

    schema = action_schema(("read_report", "finish")).model_json_schema()
    assert schema["properties"]["action"]["enum"] == ["read_report", "finish"]


def test_eval_counts_execution_failure_in_denominator(monkeypatch, tmp_path):
    import json

    from eval import run as runner
    from src import agent

    row = {
        "id": "broken",
        "group": "broken",
        "category": "research",
        "split": "dev",
        "question": "test",
        "previous": None,
        "relevant": [],
        "expected": {
            "intent": "research",
            "tools": ["report_search"],
            "statuses": ["ok"],
            "answer_rubric": "Answer supported by report",
        },
    }
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(row) + "\n")

    def fail(*a, **kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(agent, "ask", fail)
    result = runner.run("agent", data=data, output=tmp_path / "result.json")
    assert len(result["rows"]) == 1 and result["rows"][0]["execution_error"] == "RuntimeError"
    assert result["summary"]["response_contract_valid_rate"] == 0
    assert result["summary"]["task_success_rate"] is None


def test_failed_repair_does_not_erase_previously_grounded_facts():
    class Client(Scripted):
        answers = 0

        def generate(self, name, payload, schema):
            if name == "cited_answer":
                self.answers += 1
                key = next(iter(payload["evidence"]))
                return CitedAnswer(
                    claims=[
                        CitedClaim(
                            evidence_ids=[key],
                            text="Сера 0.2%." if self.answers == 1 else "Сера 99%.",
                        )
                    ],
                    complete=False,
                )
            return super().generate(name, payload, schema)

    client = Client(
        [
            action("report_search", "Sokol"),
            action("finish"),
            action("read_report", report="eia", page=6),
            action("finish"),
        ]
    )
    response = ask(
        "Sokol?",
        client=client,
        planner=lambda *_: plan(report_only=True),
        execute=lambda r, n, **kw: ([{**REPORT, "report": "eia"}], None, event(n)),
    )
    assert (
        response["status"] == "partial"
        and "0.2%" in response["answer"]
        and "99%" not in response["answer"]
    )


def test_blocks_keep_table_headers_and_rows_together():
    table = "API gravity | Sulfur\nUrals 30 to 32 1.3% to 1.5%\nESPO 34 to 37 0.4% to 0.6%"
    evidence = catalog([{**REPORT, "text": table}])
    assert list(evidence) == ["R1:B1"] and evidence["R1:B1"]["text"] == table


def test_self_review_is_explicit_opt_in(monkeypatch):
    from src.config import settings

    class Client(Scripted):
        pass

    c = Client([])
    generate_answer(c, "Sokol?", [REPORT])
    assert [n for n, p in c.payloads] == ["cited_answer"]
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    c = Client([])
    generate_answer(c, "Sokol?", [REPORT])
    assert [n for n, p in c.payloads] == ["cited_answer", "citation_review"]


def test_no_tool_expected_rejects_unnecessary_execution():
    expected = {"tools": [], "intent": "outside", "statuses": ["outside"]}
    response = {
        "trace": [event("report_search")],
        "intent": "outside",
        "status": "outside",
        "agent_mode": "loop",
    }
    assert route_metrics(expected, response, {"intent": "outside"})["tool_policy_pass"] == 0


def test_predicted_plan_cannot_relax_gold_tool_requirements():
    expected = {
        "tools": ["report_search", "price_forecast"],
        "intent": "forecast",
        "statuses": ["ok"],
        "plan_checks": {"wants_context": True},
    }
    response = {
        "trace": [event("price_forecast")],
        "intent": "forecast",
        "status": "ok",
        "agent_mode": "loop",
    }
    wrong_plan = {"intent": "forecast", "wants_context": False, "quotes": []}
    metrics = route_metrics(expected, response, wrong_plan)
    assert metrics["tool_policy_pass"] == 0 and metrics["plan_fields_correct"] == 0


def test_duplicate_empty_search_still_allows_a_different_query():
    queries = []

    def execute(registry, name, **args):
        queries.append(args["query"])
        return ([] if args["query"] == "broad" else [REPORT]), None, event(name)

    client = Scripted(
        [
            action("report_search", "broad"),
            action("report_search", "broad"),
            action("report_search", "focused"),
            action("finish"),
        ]
    )
    response = ask(
        "Сера Sokol?", client=client, execute=execute, planner=lambda *_: plan(report_only=True)
    )
    assert queries == ["broad", "focused"]
    assert response["status"] == "ok" and response["rejected_actions"] == 1
