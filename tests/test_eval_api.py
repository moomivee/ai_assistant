from fastapi.testclient import TestClient
from test_agent import FakeLLM, executor

from eval.metrics import retrieval_metrics, route_metrics
from eval.run import validate_data
from eval.traces import candidates, link_cases
from src.agent import ask
from src.api import app
from src.config import settings


def test_gold_dataset():
    result = validate_data()
    assert result["cases"] == 100 and result["retrieval_cases"] == 40
    assert sum(result["categories"].values()) == 100


def test_overlap_does_not_inflate_recall():
    gold = [
        {"report": "r", "page": 1, "quote": "first fact"},
        {"report": "r", "page": 2, "quote": "second fact"},
    ]
    ranked = [{"report": "r", "page": 1, "text": "first fact"}] * 5
    metrics = retrieval_metrics(ranked, gold)
    assert metrics["hit@1"] == 1 and metrics["evidence_recall@5"] == 0.5


def test_routing_order():
    e = {
        "tools": ["report_search", "web_search"],
        "forbidden": ["price_forecast"],
        "intent": "research",
        "statuses": ["ok"],
    }
    r = {
        "trace": [
            {"tool": "web_search"},
            {"tool": "report_search"},
            {"tool": "price_forecast"},
        ],
        "intent": "research",
        "status": "ok",
    }
    m = route_metrics(e, r, {"intent": "research"})
    assert (
        m["tool_sequence_exact"] == 0
        and m["rag_before_web"] == 0
        and m["forbidden_tool_violation"] == 1
    )


def test_api_contract_and_request_ids():
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.headers["x-request-id"]
        r = client.post("/api/ask", json={"question": "Brent"})
        assert r.status_code == 200 and r.json()["status"] == "unavailable"
        assert client.post("/api/ask", json={"question": ""}).status_code == 422
        assert client.post("/api/ask", json={"question": "   "}).status_code == 422
        assert client.get("/reports/not-a-file.pdf").status_code == 404


def test_trace_candidates_are_not_gold(tmp_path):
    ask("test question", client=FakeLLM(), execute=executor)
    rows = list(candidates(settings().trace_file))
    assert len(rows) == 1 and rows[0]["expected"] is None and rows[0]["review"] == "pending"
    cases = [
        {
            "question": "test question",
            "previous": None,
            "expected": {"intent": "research"},
        }
    ]
    event = {
        "input": {"question": "test question", "previous": None},
        "trace_id": "abc",
    }
    linked = link_cases(cases, [event])
    assert linked[0]["expected"] == cases[0]["expected"] and linked[0]["observed_trace_ids"] == [
        "abc"
    ]


def test_eval_button(monkeypatch):
    from eval import ui

    ui._jobs.clear()
    monkeypatch.setattr(ui, "run", lambda **kwargs: {"summary": {"n": 1}})
    with TestClient(ui.app) as client:
        assert client.get("/").status_code == 200
        r = client.post("/runs", json={"mode": "agent", "limit": 1})
        assert r.status_code == 202
        p = client.get("/runs/" + r.json()["id"]).json()
        assert p["status"] == "done" and p["summary"]["n"] == 1
        assert client.post("/runs", json={"mode": "routing", "judge": True}).status_code == 422
        assert client.get("/runs/unknown/result").status_code == 404
