import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.agent import ask
from src.config import settings
from src.integrations.llm import LLMUnavailable
from src.models.schemas import EvidenceAnswer, Plan
from src.tools import make_registry, run_tool


def plan(**kwargs):
    return Plan(
        **{
            **dict(
                intent="research",
                rag_query="Sokol",
                fresh=False,
                report_only=False,
                wants_context=True,
                news=False,
                quotes=[],
                benchmark=None,
                months=None,
                discount=None,
                cut=None,
                sensitivity=None,
                parameters=[],
                clarify="",
            ),
            **kwargs,
        }
    )


REPORT = {
    "id": "R1",
    "kind": "report",
    "title": "EIA",
    "date": "2025-07-24",
    "url": "https://eia.gov/report.pdf",
    "page": 6,
    "text": "Sokol: sulfur 0.2% to 0.3%.",
}
WEB = {
    "id": "W1",
    "kind": "web",
    "title": "EIA news",
    "date": "2026-11-09",
    "url": "https://eia.gov/news",
    "text": "Fresh news about oil supplies.",
}


class FakeLLM:
    def __init__(self, p=None, complete=True, fail=False):
        self.p, self.complete, self.fail = p or plan(), complete, fail

    def generate(self, name, payload, schema):
        if self.fail:
            raise LLMUnavailable("test offline")
        if name == "plan":
            return self.p
        source = payload["sources"][0]
        return EvidenceAnswer(
            complete=self.complete,
            claims=[
                {
                    "source": source["id"],
                    "quote": source["text"],
                    "text": source["text"],
                }
            ],
            limitations=[],
        )


def executor(registry, name, **args):
    if name == "price_forecast":
        return run_tool(registry, name, **args)
    data = [REPORT] if name == "report_search" else {"items": [WEB], "notices": []}
    return data, None, {"tool": name, "engine": "fixture", "status": "ok", "ms": 1}


@pytest.mark.parametrize(
    "p,complete,wanted,status",
    [
        (plan(), True, ["report_search"], "ok"),
        (plan(), False, ["report_search", "web_search"], "partial"),
        (plan(fresh=True), True, ["report_search", "web_search"], "ok"),
        (plan(fresh=True, report_only=True), True, ["report_search"], "partial"),
        (plan(intent="outside"), True, [], "outside"),
        (
            plan(intent="forecast", benchmark="Urals", months=3, wants_context=False),
            True,
            [],
            "clarification",
        ),
        (
            plan(intent="forecast", benchmark="Brent", months=3, wants_context=False),
            True,
            ["report_search", "price_forecast"],
            "ok",
        ),
    ],
)
def test_graph_branches(p, complete, wanted, status):
    response = ask("Спрогнозируй Brent на 3 месяца", client=FakeLLM(p, complete), execute=executor)
    assert [t["tool"] for t in response["trace"]] == wanted
    assert response["status"] == status
    if "price_forecast" in wanted:
        assert len(response["forecast"]["rows"]) == 3
        assert (
            response["forecast"]["rows"][-1]["lower"]
            <= response["forecast"]["rows"][-1]["point"]
            <= response["forecast"]["rows"][-1]["upper"]
        )
        assert response["trace"][-1]["engine"] == "ouroboros.ToolRegistry"
        assert "[M1]" in response["answer"]


def test_no_regex_fallback():
    response = ask("Прогноз Brent на 3 месяца", client=FakeLLM(fail=True), execute=executor)
    assert response["status"] == "unavailable" and response["trace"] == []


def test_concurrent_request_traces():
    with ThreadPoolExecutor(max_workers=3) as pool:
        answers = list(
            pool.map(
                lambda i: ask(f"Question {i}", client=FakeLLM(), execute=executor),
                range(3),
            )
        )
    events = [json.loads(line) for line in settings().trace_file.read_text().splitlines()]
    ids = {a["trace_id"] for a in answers}
    assert len(ids) == 3 and {e["trace_id"] for e in events} == ids
    for trace in ids:
        rows = [e for e in events if e["trace_id"] == trace]
        roots = [e for e in rows if e["name"] == "agent.request"]
        assert len(roots) == 1
        assert all(
            e["parent_id"] == roots[0]["span_id"] for e in rows if e["name"].startswith("workflow.")
        )


def test_local_content_opt_out(monkeypatch):
    monkeypatch.setenv("TRACE_CONTENT", "false")
    settings.cache_clear()
    ask("PRIVATE-QUESTION", client=FakeLLM(), execute=executor)
    text = settings().trace_file.read_text()
    assert "PRIVATE-QUESTION" not in text and REPORT["text"] not in text
    assert "workflow.plan" in text


def test_registry_rejects_invalid_forecast():
    result, error, event = run_tool(make_registry(), "price_forecast", benchmark="Brent", months=99)
    assert result is None and error and event["status"] == "error"


def test_quote_request_skips_unnecessary_synthesis():
    calls = []

    class QuoteClient(FakeLLM):
        def generate(self, name, payload, schema):
            calls.append(name)
            return super().generate(name, payload, schema)

    def quotes(registry, name, **args):
        if name == "report_search":
            return executor(registry, name, **args)
        assert args["search"] is False
        return (
            {
                "items": [
                    {
                        **WEB,
                        "grade": "Brent",
                        "price": 75.0,
                        "text": "Последнее наблюдение Brent 75 USD/барр.",
                    }
                ],
                "notices": [],
            },
            None,
            {"tool": name, "engine": "fixture", "status": "ok", "ms": 0},
        )

    p = plan(quotes=["Brent"], fresh=True, wants_context=False)
    response = ask("Последняя дневная цена Brent", client=QuoteClient(p), execute=quotes)
    assert calls == ["plan"] and response["status"] == "ok"
    assert [t["tool"] for t in response["trace"]] == ["report_search", "web_search"]
