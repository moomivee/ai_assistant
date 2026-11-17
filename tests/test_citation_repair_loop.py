"""Citation repair shares the evidence catalogue and the entire request's retry budget."""

from test_agent import WEB, plan
from test_agent_loop import action, event

from src.agent import ask
from src.config import settings

EXPORT = "В отчёте о поставках указан сорт Orion."


class CitationLoopClient:
    def __init__(self, decisions, citations):
        self.decisions = iter(decisions)
        self.citations = iter(citations)
        self.catalogues = []
        self.reviews = []

    def generate(self, name, payload, schema):
        if name == "decide":
            return schema.model_validate(next(self.decisions))
        if name == "cited_answer":
            catalogue = payload["evidence"]
            shape = schema.model_json_schema()
            ref = shape["properties"]["claims"]["items"]["$ref"]
            claim = shape["$defs"][ref.rsplit("/", 1)[-1]]
            allowed = claim["properties"]["evidence_ids"]["items"]["enum"]
            assert set(allowed) == set(catalogue)
            self.catalogues.append(dict(catalogue))
            return schema.model_validate(
                {
                    "claims": [{"text": EXPORT, "evidence_ids": [next(self.citations)]}],
                    "complete": True,
                }
            )
        assert name == "citation_review"
        self.reviews.append(payload["claims"])
        return schema.model_validate(
            {"feedback": "", "supported": [0], "coverage": "sufficient", "complete": True}
        )


def execute_web(registry, name, **args):
    assert name == "web_search"
    source = {
        **WEB,
        "url": WEB["url"] + "/" + args["query"],
        "text": EXPORT + " " + args["query"],
    }
    return {"items": [source], "notices": []}, None, event(name)


def test_corrected_block_reaches_review_without_another_search(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    client = CitationLoopClient([action("web_search", "exports")], ["W1", "W1:B1"])
    result = ask(
        "Какой сорт указан в поставках?",
        planner=lambda *_: plan(),
        client=client,
        execute=execute_web,
    )

    assert [item["tool"] for item in result["trace"]] == ["web_search"]
    assert len(client.catalogues) == 2 and client.catalogues[0] == client.catalogues[1]
    assert len(client.reviews) == 1 and client.reviews[0][0]["evidence"][0]["source"] == "W1"
    assert result["status"] == "ok" and result["claims"][0]["evidence_ids"] == ["W1:B1"]
    assert result["diagnostics"]["answer_checks"][0]["validation"]["citation_repair_attempted"]


def test_second_search_uses_new_catalogue_without_resetting_repair_budget(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    client = CitationLoopClient(
        [action("web_search", "exports"), action("web_search", "shipments"), action("finish")],
        ["W1", "W1", "W2"],
    )
    result = ask(
        "Какие поставки описаны?", planner=lambda *_: plan(), client=client, execute=execute_web
    )

    assert [item["tool"] for item in result["trace"]] == ["web_search", "web_search"]
    assert [set(c) for c in client.catalogues] == [{"W1:B1"}, {"W1:B1"}, {"W1:B1", "W2:B1"}]
    assert result["status"] == "partial" and not result["claims"] and not result["sources"]
    assert not client.reviews
    assert EXPORT not in result["answer"]
    assert [
        check["validation"]["citation_repair_attempted"]
        for check in result["diagnostics"]["answer_checks"]
    ] == [True, False]
