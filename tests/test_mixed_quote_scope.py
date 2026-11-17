import json

import pytest
from test_agent import REPORT, plan
from test_agent_loop import Scripted, action, event
from test_gigachat_client import auth_reply, completion
from test_gigachat_client import client as gigachat_client

from src.agent import ask
from src.integrations.llm import LLMUnavailable
from src.planning import make_plan

QUALITY = "Чем по качеству отличаются Urals и ESPO"
PRICE = "какова последняя цена Brent"
QUESTION = f"{QUALITY} и {PRICE}?"


@pytest.mark.parametrize("grades", [["Urals"], ["Brent", "Urals"]])
def test_grade_without_daily_feed_stays_a_research_question(grades):
    question = f"Укажи последнюю цену {' и '.join(grades)}."

    class Client(Scripted):
        def generate(self, name, payload, schema):
            if name == "plan":
                return plan(
                    quotes=grades, wants_context=False, quote_request=question, context_fresh=False
                )
            if name == "decide":
                task = payload["task"]
                assert task["wants_context"] and task["fresh"] and not task["quotes"]
                assert task["rag_query"] == question
            return super().generate(name, payload, schema)

    def execute(registry, name, **args):
        assert name == "web_search" and args["search"] and args["fresh"] and not args["quotes"]
        return {"items": [], "notices": []}, None, event(name)

    client = Client([action("web_search", question), action("finish")])
    result = ask(question, client=client, execute=execute)
    assert len(result["trace"]) == 1 and result["decisions"][0]["origin"] == "llm"


def test_mixed_question_keeps_report_answer_when_quote_provider_fails(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    used = []
    fact = "В этом примере ESPO легче Urals и содержит меньше серы."
    reason = "EIA/FRED: HTTP 503."

    class Client(Scripted):
        def generate(self, name, payload, schema):
            if name == "plan":
                return plan(quotes=["Brent"], fresh=True, wants_context=False)
            if name == "research_quote_scope":
                return schema(quote_request=PRICE, context_request=QUALITY, context_fresh=False)
            if name == "cited_answer":
                assert payload["question"] == QUALITY
            return super().generate(name, payload, schema)

    def execute(registry, name, **args):
        used.append((name, args))
        if args.get("quotes"):
            return {"items": [], "notices": [reason]}, reason, {**event(name), "status": "error"}
        assert name == "report_search"
        return [{**REPORT, "text": fact}], None, event(name)

    result = ask(QUESTION, client=Client([action("report_search", QUALITY)]), execute=execute)
    assert [name for name, _ in used] == ["web_search", "report_search"]
    assert used[0][1]["quotes"] == ["Brent"] and used[0][1]["search"] is False
    assert result["status"] == "partial" and fact in result["answer"] and reason in result["answer"]
    assert result["claims"][0]["evidence_ids"] and result["sources"][0]["page"] == REPORT["page"]


@pytest.mark.parametrize("malformed_scope", [False, True])
def test_legacy_price_only_plan_for_mixed_question_recovers_comparison(malformed_scope):
    generated = []

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        body = json.loads(request.content)
        name = body["response_format"]["schema"]["title"]
        generated.append(name)
        if name == "Plan":
            # Actual regression: quote present, explicit comparison lost, new fields absent.
            old_fields = plan(
                quotes=["Brent"],
                wants_context=False,
                fresh=True,
                report_only=malformed_scope,
            ).model_dump(exclude={"quote_request", "context_request", "context_fresh"})
            return completion(old_fields)
        assert name == "ResearchQuoteScope"
        assert json.loads(body["messages"][-1]["content"])["question"] == QUESTION
        if malformed_scope:
            return completion({"unexpected_field": True})
        return completion(
            {"quote_request": PRICE, "context_request": QUALITY, "context_fresh": False}
        )

    client, native = gigachat_client(reply)
    try:
        result = make_plan(client, QUESTION)
    finally:
        native.close()
    assert generated == ["Plan", "ResearchQuoteScope"]
    assert result.quotes == ["Brent"] and result.wants_context and result.fresh
    if malformed_scope:
        assert result.report_only and result.rag_query == QUESTION
        assert result.context_request == "" and result.context_fresh is None
    else:
        assert result.context_request == result.rag_query == QUALITY
        assert result.quote_request == PRICE and result.context_fresh is False


@pytest.mark.parametrize(
    "context,fresh",
    [
        (QUALITY, False),
        ("что решила ОПЕК+ на последней встрече", True),
    ],
)
def test_explicit_context_and_price_clauses_override_wrong_routing_flag(context, fresh):
    class Client:
        def generate(self, name, payload, schema):
            assert name == "plan"
            return plan(
                quotes=["Brent"],
                wants_context=False,
                quote_request=PRICE,
                context_request=context,
                context_fresh=fresh,
            )

    result = make_plan(Client(), f"{PRICE} и {context}?")
    assert result.wants_context and result.fresh
    assert result.context_fresh is fresh and result.rag_query == context


@pytest.mark.parametrize("first_clause", ["whole", "цены WTI и Brent"])
def test_units_politeness_and_multiple_prices_are_a_single_quote_task(first_clause):
    question = "Пожалуйста, укажи последние цены WTI и Brent в долларах за баррель, со ссылкой."
    calls = []

    class Client:
        def generate(self, name, payload, schema):
            calls.append(name)
            if name == "plan":
                return plan(
                    quotes=["WTI", "Brent"],
                    wants_context=True,
                    report_only=True,
                    quote_request=question if first_clause == "whole" else first_clause,
                )
            return schema(quote_request=question, context_request="", context_fresh=False)

    result = make_plan(Client(), question)
    assert calls == (["plan"] if first_clause == "whole" else ["plan", "research_quote_scope"])
    assert result.quotes == ["WTI", "Brent"] and result.report_only
    assert result.wants_context is False and result.context_request == ""


@pytest.mark.parametrize("failed", [False, True])
def test_unconfirmed_scope_cannot_remove_two_tasks_or_reuse_previous_question(failed):
    class Client:
        def generate(self, name, payload, schema):
            if name == "plan":
                return plan(
                    quotes=["Brent"],
                    wants_context=False,
                    quote_request=PRICE,
                    context_request="за прошлый год",
                    context_fresh=False,
                )
            assert name == "research_quote_scope"
            if failed:
                raise LLMUnavailable("offline")
            # These excerpts overlap and cannot establish separate goals.
            return schema(quote_request=PRICE, context_request=PRICE, context_fresh=False)

    result = make_plan(Client(), QUESTION, previous="за прошлый год")
    assert result.wants_context and result.quotes == ["Brent"]
    assert result.context_request == "" and result.context_fresh is None
    assert result.rag_query == QUESTION
