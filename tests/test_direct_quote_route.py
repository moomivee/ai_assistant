from contextlib import contextmanager

import httpx
import pytest
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, action, event

from src.agent import ask
from src.integrations import web


def quote_plan(**updates):
    return plan(**{"quotes": ["Brent"], "fresh": True, "wants_context": False, **updates})


def test_latest_observation_uses_registry_without_requesting_publication_options(monkeypatch):
    urls = []

    @contextmanager
    def fred_response(method, url, **options):
        assert url == web.FRED_CSV and options["params"]["id"] == "DCOILBRENTEU"
        urls.append(url)
        yield httpx.Response(
            200,
            text="observation_date,DCOILBRENTEU\n2025-03-04,74.32\n2025-03-05,.\n",
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(web.httpx, "stream", fred_response)
    client = Scripted([])
    result = ask(
        "Последняя опубликованная цена Brent?", client=client, planner=lambda *_: quote_plan()
    )

    assert client.payloads == [] and urls == [web.FRED_CSV]
    assert result["status"] == "ok" and result["trace"][0]["engine"] == "ouroboros.ToolRegistry"
    assert "2025-03-04" in result["answer"] and "74.32 USD/барр." in result["answer"]
    assert result["sources"][0]["url"].endswith("/DCOILBRENTEU")
    (step,) = result["diagnostics"]["steps"]
    assert step["arguments"] == {"quotes": ["Brent"], "query": "", "fresh": True, "search": False}
    assert result["decisions"][0]["origin"] == "plan_policy"


def test_model_can_finish_after_failed_direct_quote_and_keep_provider_reason():
    calls = []

    def execute(registry, name, **arguments):
        calls.append(name)
        return {"items": [], "notices": []}, "HTTP 503", {**event(name), "status": "error"}

    result = ask(
        "Спот Brent?",
        planner=lambda *_: quote_plan(),
        client=Scripted([action("finish")]),
        execute=execute,
    )
    assert calls == ["web_search"] and result["status"] == "partial"
    assert "EIA/FRED" in result["answer"] and "HTTP 503" in result["answer"]
    assert result["sources"] == []


@pytest.mark.parametrize("wants_context", [False, True])
@pytest.mark.parametrize("has_observation", [False, True])
def test_failed_quote_gets_one_optional_publication_search(
    monkeypatch, wants_context, has_observation
):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    calls = []
    source = {
        **WEB,
        "text": "Последняя опубликованная спотовая цена Brent на 2025-03-04 — 74.32 USD/барр.",
        "date": "2025-03-05",
    }

    def execute(registry, name, **args):
        calls.append(args)
        assert name == "web_search"
        if args["quotes"]:
            return {"items": [], "notices": ["EIA/FRED: таймаут."]}, "timeout", event(name)
        assert args["quote_fallback"] and args["fresh"] and args["search"]
        return {"items": [source] if has_observation else [], "notices": []}, None, event(name)

    client = Scripted([action("web_search", "Brent spot latest observation")])
    result = ask(
        "Какая последняя опубликованная спотовая цена Brent? Укажи дату наблюдения и источник.",
        client=client,
        planner=lambda *_: quote_plan(wants_context=wants_context),
        execute=execute,
    )
    assert len(calls) == 2
    if has_observation:
        assert result["status"] == "ok"
        assert "74.32" in result["answer"] and "2025-03-04" in result["answer"]
        assert [s["url"] for s in result["sources"]] == [source["url"]]
        for stage, payload in client.payloads:
            if stage in ("cited_answer", "citation_review"):
                assert payload["temporal_requirements"] == {
                    "fresh": True,
                    "report_only": False,
                    "required_spot_grades": ["Brent"],
                }
    else:
        assert result["status"] == "partial" and not result["claims"] and not result["sources"]
        assert "EIA/FRED" in result["answer"]
        assert [stage for stage, _ in client.payloads] == ["decide"]


def test_pure_quote_cannot_override_report_scope():
    question = "Последняя цена Brent только из загруженных документов."

    def reports_only(registry, name, **args):
        assert name == "report_search" and args["query"] == question
        return [], None, event(name)

    result = ask(
        question,
        client=Scripted([action("finish")]),
        planner=lambda *_: quote_plan(report_only=True),
        execute=reports_only,
    )
    assert result["status"] == "partial" and [row["tool"] for row in result["trace"]] == [
        "report_search"
    ]
    assert "актуальность на сегодняшний день не проверялась" in result["answer"]


def test_quote_fallback_excludes_a_futures_price_rejected_by_review(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")

    class ReviewSpot(Scripted):
        def generate(self, name, payload, schema):
            if name == "citation_review":
                assert payload["temporal_requirements"]["required_spot_grades"] == ["WTI"]
                return schema.model_validate(
                    {
                        "supported": [],
                        "complete": False,
                        "coverage": "partial",
                        "feedback": "Приведена фьючерсная цена.",
                        "missing_aspects": ["Спотовая цена WTI с датой наблюдения."],
                    }
                )
            return super().generate(name, payload, schema)

    futures = {**WEB, "text": "Фьючерс WTI на закрытии стоил 74.32 USD/барр."}

    def execute(registry, name, **args):
        if args["quotes"]:
            return {"items": [], "notices": ["EIA/FRED: таймаут."]}, "timeout", event(name)
        return {"items": [futures], "notices": []}, None, event(name)

    result = ask(
        "Назови последний спот WTI.",
        planner=lambda *_: quote_plan(quotes=["WTI"]),
        client=ReviewSpot([action("web_search", "WTI spot last observation")]),
        execute=execute,
    )
    assert result["status"] == "partial" and "74.32" not in result["answer"]
    assert not result["claims"] and not result["sources"]


@pytest.mark.parametrize("context_tool", ["report_search", "web_search"])
def test_mixed_quote_failure_keeps_context_route_open(monkeypatch, context_tool):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    calls = []
    reason = "EIA/FRED: HTTP 503."

    def execute(registry, name, **args):
        calls.append((name, args))
        if args.get("quotes"):
            assert not args["search"]
            return {"items": [], "notices": [reason]}, reason, {**event(name), "status": "error"}
        data = [REPORT] if name == "report_search" else {"items": [WEB], "notices": []}
        return data, None, event(name)

    client = Scripted([action(context_tool, "facts about crude oil")])
    result = ask(
        "Какие свойства у нефти Sokol и какова последняя цена Brent?",
        planner=lambda *_: quote_plan(
            wants_context=True, context_request="Какие свойства у нефти Sokol?", context_fresh=False
        ),
        client=client,
        execute=execute,
    )
    assert [name for name, _ in calls] == ["web_search", context_tool]
    if context_tool == "web_search":
        assert calls[1][1]["search"] and not calls[1][1]["quotes"] and not calls[1][1]["fresh"]
    assert result["status"] == "partial" and result["claims"]
    assert (REPORT if context_tool == "report_search" else WEB)["text"] in result["answer"]
    assert reason in result["answer"]


def test_available_quote_survives_empty_context(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    published = {**WEB, "grade": "Brent", "price": 74.32, "text": "Brent 74.32 USD/барр."}

    def execute(registry, name, **args):
        if args.get("quotes"):
            return {"items": [published], "notices": []}, None, event(name)
        assert name == "report_search"
        return [], None, event(name)

    client = Scripted([action("report_search", "Sokol sulfur"), action("finish")])
    result = ask(
        "Какие свойства у нефти Sokol и какова последняя цена Brent?",
        planner=lambda *_: quote_plan(wants_context=True),
        client=client,
        execute=execute,
    )
    assert result["status"] == "partial" and published["text"] in result["answer"]
    assert [n for n, _ in client.payloads] == ["decide", "decide"]
    assert not result["claims"] and len(result["sources"]) == 1


def test_mixed_success_uses_separate_evidence_for_price_and_context(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    context = "Какие свойства у нефти Sokol?"
    quote = {**WEB, "grade": "Brent", "price": 74.32, "text": "Brent 74.32 USD/барр."}

    def execute(registry, name, **args):
        if name == "web_search":
            assert args["quotes"] == ["Brent"] and not args["search"]
            return {"items": [quote], "notices": []}, None, event(name)
        assert name == "report_search"
        return [REPORT], None, event(name)

    client = Scripted([action("report_search", "Sokol properties")])
    result = ask(
        context + " Приведи и последнюю цену Brent.",
        planner=lambda *_: quote_plan(
            wants_context=True, context_request=context, context_fresh=False
        ),
        client=client,
        execute=execute,
    )
    draft = next(payload for name, payload in client.payloads if name == "cited_answer")
    assert draft["question"] == context and list(draft["evidence"].values()) == [REPORT["text"]]
    assert result["status"] == "ok"
    assert quote["text"] in result["answer"] and REPORT["text"] in result["answer"]
    assert {row["id"] for row in result["sources"]} == {"R1", "W1"}
