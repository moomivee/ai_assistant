"""Action contracts accept omitted irrelevant fields, never invent tool arguments."""

import json

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError
from test_agent import REPORT, WEB, plan
from test_agent_loop import Scripted, event
from test_gigachat_client import auth_reply, completion
from test_gigachat_client import client as gigachat_client

from src import tools
from src.agent import action_schema, ask
from src.integrations.llm import LLMClient, LLMUnavailable
from src.models.schemas import NextAction


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "web_search",
            "purpose": "Проверить последнее событие",
            "query": "latest meeting",
        },
        {"action": "report_search", "purpose": "Найти характеристику", "query": "sulfur"},
        {"action": "finish", "purpose": "Доступных доказательств недостаточно"},
        {"action": "price_forecast", "purpose": "Рассчитать проверенные параметры плана"},
    ],
)
def test_native_gigachat_accepts_action_without_unrelated_report_fields(payload):
    sent = []

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        sent.append(json.loads(request.content))
        return completion(payload)

    client, native = gigachat_client(reply)
    try:
        result = client.generate("decide", {}, action_schema((payload["action"],)))
    finally:
        native.close()
    assert result.action == payload["action"]
    assert result.report == "" and result.page is None
    assert result.query == payload.get("query", "")
    assert len(sent) == 1  # No retry or fabricated report/page is needed.
    required = sent[0]["response_format"]["schema"]["required"]
    assert "action" in required and "purpose" in required
    assert not {"report", "page", "query"} & set(required)


def test_openai_sdk_retains_strict_wire_schema_and_accepts_local_defaults():
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "action-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "finish",
                                    "purpose": "No evidence available",
                                }
                            ),
                        },
                    }
                ],
            },
        )

    native = OpenAI(
        api_key="test-only",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    )
    try:
        result = LLMClient(native, provider="openai").generate(
            "decide", {}, action_schema(("finish",))
        )
    finally:
        native.close()
    assert result.action == "finish" and result.page is None and result.query == ""
    wire = requests[0]["response_format"]["json_schema"]
    assert wire["strict"] is True
    assert set(wire["schema"]["required"]) == set(wire["schema"]["properties"])


@pytest.mark.parametrize("extra", [{"report_name": "made-up"}, {"action": "delete_file"}])
def test_missing_irrelevant_defaults_do_not_accept_unknown_actions_or_fields(extra):
    with pytest.raises(ValidationError):
        action_schema(("web_search", "finish")).model_validate(
            {
                "action": "web_search",
                "purpose": "Check evidence",
                "query": "oil",
                **extra,
            }
        )


def test_gigachat_wrong_field_is_still_a_structured_error():
    def reply(request):
        return (
            auth_reply()
            if request.url.path.endswith("oauth")
            else completion(
                {
                    "action": "web_search",
                    "purpose": "Check evidence",
                    "query": "oil",
                    "invented": True,
                }
            )
        )

    client, native = gigachat_client(reply)
    try:
        with pytest.raises(LLMUnavailable):
            client.generate("decide", {}, action_schema(("web_search",)))
    finally:
        native.close()


@pytest.mark.parametrize(
    "missing",
    [{"report": "eia"}, {"page": 6}, {"report": "eia", "page": 0}, {"report": "eia", "page": -1}],
)
def test_read_report_missing_required_arguments_is_rejected_then_replanned(missing):
    calls = []

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        if name == "read_report":
            assert arguments["report"] == "eia" and arguments["page"] == 6
        source = {**REPORT, "report": "eia"}
        if name == "read_report":
            source.update(text="Full page.\n" + REPORT["text"], full_page=True)
        return [source], None, event(name)

    client = Scripted(
        [
            {"action": "read_report", "purpose": "Read details", **missing},
            {"action": "read_report", "purpose": "Use discovered page", "report": "eia", "page": 6},
        ],
        complete=[False, True],
    )
    result = ask(
        "Сера Sokol по отчёту",
        client=client,
        execute=execute,
        planner=lambda *_: plan(report_only=True),
    )
    assert [name for name, _ in calls] == ["report_search", "read_report"]
    # This transport fixture disables semantic review; no verified completeness is claimed.
    assert result["diagnostics"]["semantic_review_enabled"] is False
    assert result["status"] == "partial" and result["rejected_actions"] == 1
    assert result["diagnostics"]["steps"][1]["status"] == "rejected"
    assert result["decisions"][1]["page"] == missing.get("page")
    assert result["decisions"][1]["report"] == missing.get("report", "")


@pytest.mark.parametrize("omitted", ["", " \t"])
def test_initial_report_search_uses_model_planned_query_when_action_omits_it(omitted):
    question = "Чем отличаются Urals и ESPO по плотности и содержанию серы?"
    query = "плотность и содержание серы в Urals и ESPO"
    calls = []

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        return [REPORT], None, event(name)

    client = Scripted([{"action": "report_search", "purpose": "Find comparison", "query": omitted}])
    result = ask(
        question,
        client=client,
        execute=execute,
        planner=lambda *_: plan(rag_query=" " + query),
    )
    assert calls == [("report_search", {"query": query})]
    assert result["rejected_actions"] == 0 and result["decisions"][0]["query"] == omitted
    step = result["diagnostics"]["steps"][0]
    assert step["arguments"] == {"query": query}
    assert step["argument_repair"] == {
        "mode": "plan_rag_query",
        "before": {"query": omitted},
        "after": {"query": query},
    }


def test_initial_repaired_report_query_is_deduplicated_like_an_explicit_query():
    calls = []

    def execute(registry, name, **arguments):
        calls.append(arguments)
        return [], None, event(name)

    client = Scripted(
        [
            {"action": "report_search", "purpose": "Find comparison"},
            {"action": "report_search", "purpose": "Retry", "query": "Sokol"},
            {"action": "finish", "purpose": "No usable evidence"},
        ]
    )
    result = ask("Сера Sokol?", client=client, execute=execute, planner=lambda *_: plan())
    assert calls == [{"query": "Sokol"}]
    assert result["diagnostics"]["steps"][1]["duplicate_of"] == 1


def test_price_observation_does_not_replace_first_context_search(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    query = "содержание серы в Sokol"
    quote = {**WEB, "grade": "Brent", "price": 74.32, "text": "Brent 74.32 USD/барр."}
    calls = []

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        if name == "web_search":
            return {"items": [quote], "notices": []}, None, event(name)
        return [REPORT], None, event(name)

    result = ask(
        "Сколько серы в нефти Sokol и какова последняя цена Brent?",
        planner=lambda *_: plan(
            rag_query=query,
            quotes=["Brent"],
            fresh=True,
            context_request="Сколько серы в нефти Sokol?",
            context_fresh=False,
        ),
        client=Scripted([{"action": "report_search", "purpose": "Проверить серу"}]),
        execute=execute,
    )
    assert [name for name, _ in calls] == ["web_search", "report_search"]
    assert calls[1][1] == {"query": query}
    assert result["diagnostics"]["steps"][1]["argument_repair"]["mode"] == "plan_rag_query"
    assert result["status"] == "ok" and result["rejected_actions"] == 0
    assert REPORT["text"] in result["answer"] and quote["text"] in result["answer"]


def test_report_query_is_not_fabricated_when_both_action_and_plan_are_missing():
    calls = []

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        assert arguments["query"] == "Sokol sulfur"
        return [REPORT], None, event(name)

    client = Scripted(
        [
            {"action": "report_search", "purpose": "Find facts"},
            {"action": "report_search", "purpose": "Specify query", "query": "Sokol sulfur"},
        ]
    )
    result = ask(
        "Сера Sokol?", client=client, execute=execute, planner=lambda *_: plan(rag_query=" ")
    )
    # This transport fixture disables semantic review; no verified completeness is claimed.
    assert result["diagnostics"]["semantic_review_enabled"] is False
    assert result["status"] == "partial" and result["rejected_actions"] == 1
    assert len(calls) == 1
    assert result["decisions"][0]["query"] == ""


def test_missing_query_on_followup_does_not_reuse_initial_broad_plan():
    calls = []

    def execute(registry, name, **arguments):
        calls.append(arguments)
        return [REPORT], None, event(name)

    client = Scripted(
        [
            {"action": "report_search", "purpose": "Find sulfur", "query": "sulfur"},
            {"action": "report_search", "purpose": "Find the missing density"},
            {"action": "finish", "purpose": "Only partial evidence available"},
        ],
        complete=False,
    )
    result = ask(
        "Сера и плотность Sokol?",
        client=client,
        execute=execute,
        planner=lambda *_: plan(rag_query="broad properties"),
    )
    assert calls == [{"query": "sulfur"}]
    step = result["diagnostics"]["steps"][1]
    assert step["status"] == "rejected" and step.get("argument_repair") is None


def test_quotes_only_action_needs_no_text_search_query_in_real_registry(monkeypatch):
    called = []

    def quote_lookup(**arguments):
        called.append(arguments)
        return {"items": [{**WEB, "grade": "Brent", "price": 80.0}], "notices": []}

    monkeypatch.setitem(
        tools.FUNCTIONS, "web_search", (tools.WebArgs, quote_lookup, "Quote fixture")
    )
    client = Scripted([{"action": "web_search", "purpose": "Get the requested quote"}])
    result = ask(
        "Последняя цена Brent?",
        client=client,
        planner=lambda *_: plan(fresh=True, wants_context=False, quotes=["Brent"]),
    )
    assert result["status"] == "ok"
    assert len(called) == 1 and called[0]["query"] == "" and called[0]["search"] is False
    assert called[0]["quotes"] == ["Brent"]
    assert result["trace"][0]["status"] == "ok"


def test_minimal_finish_after_empty_web_returns_partial_without_invented_answer():
    client = Scripted(
        [
            {"action": "web_search", "purpose": "Find latest event", "query": "latest meeting"},
            {"action": "finish", "purpose": "Search did not return evidence"},
        ]
    )
    result = ask(
        "Последняя встреча?",
        client=client,
        planner=lambda *_: plan(fresh=True),
        execute=lambda *a, **kw: ({"items": [], "notices": []}, None, event("web_search")),
    )
    assert result["status"] == "partial" and result["claims"] == []
    assert result["sources"] == [] and len(result["trace"]) == 1
    assert result["diagnostics"]["failure"] == ""


def test_block_locator_can_be_used_without_pdf_page():
    action = NextAction.model_validate(
        {
            "action": "read_report",
            "purpose": "Read spreadsheet cells",
            "report": "document-id",
            "version_id": "immutable-version",
            "block_id": "sheet-block",
        }
    )
    assert action.page is None and action.query == ""
    assert action.block_id == "sheet-block"


def test_native_gigachat_minimal_decision_reaches_search_and_finishes_honestly():
    actions = iter(
        [
            {
                "action": "web_search",
                "purpose": "Find the latest meeting",
                "query": "latest OPEC+ meeting",
            },
            {"action": "finish", "purpose": "No usable evidence returned"},
        ]
    )
    calls = []

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        assert (
            json.loads(request.content)["response_format"]["schema"]["title"] == "AvailableAction"
        )
        return completion(next(actions))

    def execute(registry, name, **arguments):
        calls.append((name, arguments))
        return {"items": [], "notices": []}, None, event(name)

    client, native = gigachat_client(reply)
    try:
        result = ask(
            "Когда была последняя встреча ОПЕК+ и о чём договорились?",
            client=client,
            execute=execute,
            planner=lambda *_: plan(fresh=True),
        )
    finally:
        native.close()
    assert [name for name, _ in calls] == ["web_search"]
    assert calls[0][1]["query"] == "latest OPEC+ meeting"
    assert result["status"] == "partial" and result["claims"] == []
    assert not result["diagnostics"]["failure"]
    assert result["decisions"][0]["report"] == "" and result["decisions"][0]["page"] is None


@pytest.mark.parametrize("query", ["", " \n\t"])
def test_actual_web_search_contract_rejects_empty_search_text(query):
    with pytest.raises(ValidationError):
        tools.WebArgs.model_validate({"query": query, "fresh": True, "quotes": [], "search": True})
