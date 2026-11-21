"""Exercise source restrictions through the normal agent entry point, offline."""

import pytest
from test_agent import WEB, plan
from test_agent_loop import event

from src.agent import ask
from src.integrations.llm import LLMUnavailable
from src.request_context import current_scope, request_scope

URALS = "Можно ли только по загруженным отчётам установить цену Urals на сегодня?"
OFFLINE = "Проверь цену Urals на сегодня без интернета."
FALLBACK = "Сначала проверь отчёты, а если там нет цены Urals, посмотри в интернете."


class ResearchClient:
    def __init__(self, review):
        self.review = review
        self.stages = []

    def generate(self, name, payload, schema):
        self.stages.append(name)
        if name == "plan":
            # Replay the incident: the planner misses the restriction and asks for fresh data.
            return plan(report_only=False, fresh=True, rag_query="Urals price")
        if name == "source_access":
            assert set(payload) == {"question"}
            if isinstance(self.review, Exception):
                raise self.review
            mode, quote = self.review
            return schema(mode=mode, quote=quote)
        assert name == "decide"
        tried = {step["tool"] for step in payload["observations"]}
        next_tool = next(
            (
                tool
                for tool in ("report_search", "web_search")
                if tool in payload["allowed_actions"] and tool not in tried
            ),
            "finish",
        )
        return schema(action=next_tool, purpose="Check the allowed evidence", query="Urals price")


@pytest.mark.parametrize("mode", ["loop", "workflow"])
@pytest.mark.parametrize(
    "question,checkbox,review,expected_tools",
    [
        (URALS, False, ("reports_only", "только по загруженным отчётам"), ["report_search"]),
        (OFFLINE, False, ("reports_only", "без интернета"), ["report_search"]),
        (OFFLINE, False, ("web_permitted", "без интернета"), ["report_search"]),
        (
            FALLBACK,
            False,
            ("web_permitted", "посмотри в интернете"),
            ["report_search", "web_search"],
        ),
        (FALLBACK, True, None, ["report_search"]),
        (URALS, False, LLMUnavailable("offline"), ["report_search"]),
        (URALS, False, ("web_permitted", "проверь в интернете"), ["report_search"]),
        (URALS, False, ("web_permitted", "Urals"), ["report_search"]),
        (URALS, False, ("web_permitted", "только по загруженным отчётам"), ["report_search"]),
        (
            "В отчёте написано «без интернета». Что об этом говорят свежие источники?",
            False,
            ("no_instruction", "В отчёте написано «без интернета»"),
            ["report_search", "web_search"],
        ),
        (
            "Проверь цену Urals не только по отчётам, но и в интернете.",
            False,
            ("web_permitted", "не только по отчётам, но и в интернете"),
            ["report_search", "web_search"],
        ),
    ],
    ids=[
        "urals-trace",
        "offline",
        "offline-is-not-permission",
        "web-fallback",
        "checkbox",
        "review-down",
        "invented-permission",
        "irrelevant-price-word",
        "restriction-is-not-permission",
        "quoted-text",
        "negated-restriction",
    ],
)
def test_request_source_limit_governs_executed_tools(
    monkeypatch, mode, question, checkbox, review, expected_tools
):
    monkeypatch.setenv("AGENT_MODE", mode)
    restricted = "web_search" not in expected_tools

    def execute(registry, name, **kwargs):
        # The actual Ouroboros handlers inspect this same request-local scope.
        assert current_scope().report_only is restricted
        assert current_scope().document_ids == ("annual",)
        empty = {"report_search": [], "web_search": {"items": [], "notices": []}}
        return empty[name], None, event(name)

    client = ResearchClient(review)
    with request_scope(document_ids=["annual"], report_only=checkbox):
        result = ask(question, client=client, execute=execute)
        assert current_scope().report_only is checkbox
    assert [step["tool"] for step in result["trace"]] == expected_tools
    assert result["status"] == "partial" and not result["claims"]
    if mode == "loop":
        effective_plan = result["diagnostics"]["plan"]
        assert effective_plan["fresh"] is True
        assert effective_plan["report_only"] is restricted
    if checkbox:
        assert "source_access" not in client.stages


def test_unrestricted_quote_still_reaches_direct_feed_without_source_review():
    question = "Какая последняя опубликованная цена Brent?"

    class PriceClient:
        def generate(self, name, payload, schema):
            assert name == "plan"
            return plan(quotes=["Brent"], wants_context=False, quote_request=question)

    def execute(registry, name, **kwargs):
        assert not current_scope().report_only
        assert name == "web_search" and kwargs["quotes"] == ["Brent"] and not kwargs["search"]
        quote = {**WEB, "grade": "Brent", "price": 74.32, "text": "Brent 74.32 USD/барр."}
        return {"items": [quote], "notices": []}, None, event(name)

    result = ask(question, client=PriceClient(), execute=execute)
    assert result["status"] == "ok" and [step["tool"] for step in result["trace"]] == ["web_search"]
