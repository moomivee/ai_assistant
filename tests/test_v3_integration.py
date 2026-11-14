"""Cross-component contracts; scripted LLM assertions are not model-quality evals."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask, decision_sources
from src.api import app
from src.evidence import catalog, render_claims
from src.models.schemas import CitedAnswer
from src.request_context import current_scope, request_scope
from src.tools import read_report
from src.upload_limits import UploadLimitMiddleware


def test_explicit_scope_survives_graph_and_cannot_be_overridden():
    seen = []

    def execute(registry, name, **args):
        seen.append((name, current_scope()))
        return [REPORT], None, event(name)

    with request_scope(["selected-document"], True):
        result = ask(
            "Сера Urals?",
            client=Scripted([]),
            execute=execute,
            planner=lambda *_: plan(report_only=False),
        )
    assert result["status"] == "ok"
    assert result["diagnostics"]["plan"]["report_only"] is True
    assert seen[0][1].document_ids == ("selected-document",)
    assert current_scope().document_ids == ()


def test_read_web_requires_discovered_url_and_keeps_extraction():
    calls = []
    extracted = {**WEB, "text": "Detailed source text", "content_type": "extracted"}

    def execute(registry, name, **args):
        calls.append(name)
        data = (
            []
            if name == "report_search"
            else {"items": [WEB if name == "web_search" else extracted]}
        )
        return data, None, event(name)

    client = Scripted(
        [
            action("web_search", "specific fact"),
            {**action("read_web"), "url": WEB["url"]},
        ],
        complete=[False, True],
    )
    result = ask("Какой факт?", client=client, execute=execute, planner=lambda *_: plan())
    assert result["status"] == "ok"
    assert calls == ["report_search", "web_search", "read_web"]
    assert len(result["diagnostics"]["retrieved_sources"]) == 2
    assert "read_web" in [p for n, p in client.payloads if n == "decide"][-1]["allowed_actions"]


def test_unseen_web_url_is_rejected_before_executor():
    calls = []

    def execute(registry, name, **args):
        calls.append(name)
        return ([] if name == "report_search" else {"items": [WEB]}), None, event(name)

    client = Scripted(
        [
            action("web_search", "specific fact"),
            {**action("read_web"), "url": "https://eia.gov/not-discovered"},
            action("finish"),
        ],
        complete=False,
    )
    result = ask("Какой факт?", client=client, execute=execute, planner=lambda *_: plan())
    assert calls == ["report_search", "web_search"]
    assert any("Read only" in (s["error"] or "") for s in result["diagnostics"]["steps"])


def test_document_page_keys_keep_versions_and_nonpdf_sources():
    old = {**REPORT, "document_id": "d", "version_id": "old", "page": 1}
    new = {**old, "id": "R2", "version_id": "new", "full_page": True}
    sheet = {**REPORT, "id": "R3", "page": None, "locator": {"sheet": "Facts", "range": "A1:B3"}}
    assert {s["id"] for s in decision_sources([old, new, sheet, WEB])} == {"R1", "R2", "R3", "W1"}
    assert len(catalog([old, new, sheet, WEB])) == 4


def test_citations_are_exact_used_substrings_with_version_and_locator():
    text = "  Header  \n\nSulfur   1.5%\n" + "Z" * 2000
    source = {
        **REPORT,
        "text": text,
        "version_id": "immutable",
        "document_id": "doc",
        "locator": {"sheet": "Grades", "range": "A1:B3"},
        "page": None,
    }
    evidence = catalog([source])
    key = next(iter(evidence))
    row = evidence[key]
    assert row["text"] == text[row["source_text_start"] : row["source_text_end"]]
    draft = CitedAnswer(claims=[{"text": "Sulfur 1.5%", "evidence_ids": [key]}], complete=True)
    _, claims, used = render_claims(draft, evidence, [source])
    citation = claims[0]["citations"][0]
    assert citation["quote"] in text
    assert citation["version_id"] == "immutable"
    assert citation["locator"]["range"] == "A1:B3"
    assert list(used[0]["evidence"]) == [key]


def test_read_uploaded_csv_keeps_cells_and_scope(tmp_path, monkeypatch):
    from src.knowledge.store import KnowledgeStore

    store = KnowledgeStore(tmp_path / "knowledge")
    document = store.upload("grades.csv", b"grade,sulfur\nUrals,1.5\n", "Grades")
    store.process(document["current_version_id"])
    monkeypatch.setattr("src.knowledge.store.get_store", lambda: store)
    block = store.blocks(document["id"])[0]
    source = read_report(document["id"], block_id=block["block_id"])[0]
    assert source["page"] is None and source["locator"]
    assert source["version_id"] == document["current_version_id"]
    with request_scope(["other-document"]), pytest.raises(ValueError, match="scope"):
        read_report(document["id"], block_id=block["block_id"])


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"101")]])
def test_upload_body_rejected_before_multipart_parser(headers):
    called, sent = [], []
    messages = iter([{"type": "http.request", "body": b"x" * 101, "more_body": False}])

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def downstream(*args):
        called.append(True)

    asyncio.run(
        UploadLimitMiddleware(downstream, 100)(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/knowledge/documents",
                "headers": headers,
            },
            receive,
            send,
        )
    )
    assert not called
    assert sent[0]["status"] == 413


def test_api_rejects_unknown_scope_before_llm(monkeypatch):
    monkeypatch.setattr("src.api.ask", lambda *_: pytest.fail("Must not call LLM"))
    response = TestClient(app).post("/api/ask", json={"question": "Q", "document_ids": ["missing"]})
    assert response.status_code == 422


def test_api_unknown_note_has_clear_error():
    response = TestClient(app).post("/api/ask", json={"question": "Q", "note_id": "f" * 32})
    assert response.status_code in {404, 422}


def test_opening_truncated_page_keeps_relevant_late_excerpt():
    fragment = {**REPORT, "text": "Sulfur 1.5%", "report": "d", "page": 1, "version_id": "v"}
    page = {**fragment, "id": "R2", "full_page": True, "truncated": True, "text": "Intro " * 4000}
    assert any("Sulfur" in row["text"] for row in decision_sources([fragment, page]))
    assert any("Sulfur" in row["text"] for row in catalog([fragment, page]).values())


def test_selected_native_skill_enters_llm_context_and_is_reset():
    from src.integrations.agent_skills import current_skill_context

    class SkillClient(Scripted):
        def generate(self, name, payload, schema):
            context = current_skill_context()
            assert context["metadata"]["name"] == "report_analysis"
            assert context["text"] and context["metadata"]["hash"]
            return super().generate(name, payload, schema)

    response = ask(
        "Сера Urals?",
        client=SkillClient([]),
        execute=lambda _, name, **args: ([REPORT], None, event(name)),
        planner=lambda *_: plan(skill="report_analysis"),
    )
    assert response["status"] == "ok"
    assert current_skill_context()["text"] == ""


def test_review_cannot_mark_complete_after_discarding_claim(monkeypatch):
    from src.config import settings
    from src.evidence import generate_answer

    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()

    class Reviewer:
        def generate(self, name, payload, schema):
            if name == "cited_answer":
                key = next(iter(payload["evidence"]))
                return schema(
                    claims=[
                        {"text": "First fact", "evidence_ids": [key]},
                        {"text": "Second fact", "evidence_ids": [key]},
                    ],
                    complete=True,
                )
            return schema(supported=[0], complete=True, feedback="second unsupported")

    draft, _, _ = generate_answer(Reviewer(), "Two facts?", [REPORT])
    assert len(draft.claims) == 1 and not draft.complete


def test_eval_counts_read_web_as_web_and_requires_discovery():
    from eval.metrics import route_metrics

    expected = {
        "tools": ["report_search"],
        "intent": "research",
        "statuses": ["ok"],
        "forbidden": ["web_search"],
    }
    response = {
        "trace": [event("report_search"), event("read_web")],
        "status": "ok",
        "intent": "research",
        "agent_mode": "loop",
    }
    metrics = route_metrics(expected, response, {})
    assert metrics["tool_policy_pass"] == 0 and metrics["forbidden_tool_violation"] == 1
    expected["forbidden"] = []
    assert route_metrics(expected, response, {})["tool_policy_pass"] == 0
    response["trace"].insert(1, event("web_search"))
    assert route_metrics(expected, response, {})["tool_policy_pass"] == 1
