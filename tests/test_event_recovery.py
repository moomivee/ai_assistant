"""Only an inconclusive repeated event search may be redirected."""

import pytest

from src.event_recovery import recover_event_search


def example():
    url = "https://forum.example/releases/meeting"
    context = {"followup": True, "publisher_pages": {url: ["W1"]}}
    args = {
        "query": "последнее заседание газового форума",
        "publisher_domain": "forum.example",
        "end_date": "2026-11-22",
    }
    state = {
        "answer_checks": [
            {
                "validation": {
                    "reason": "no_confirmed_event",
                    "event_relevance": {"kept_source_ids": ["W1"], "unverified_source_ids": []},
                }
            }
        ],
        "observations": [
            {"tool": "web_search", "arguments": {**args, "search": True}, "source_ids": ["W1"]}
        ],
        "sources": [{"id": "W1", "url": url, "content_type": "snippet", "provider": "tavily"}],
    }
    decision = {**args, "purpose": "Найти дату", "action": "web_search", "origin": "llm"}
    return decision, context, state


def test_repeated_event_query_opens_retained_snippet_without_extra_decision():
    decision, context, state = example()
    decision.update(query=decision["query"].upper(), search_depth="advanced")
    result = recover_event_search(decision, context, state, ["web_search", "read_web"])
    assert result["action"] == "read_web"
    assert result["url"] == state["sources"][0]["url"] and result["query"] == ""
    assert result["model_action"] == "web_search" and result["origin"] == "event_recovery"
    assert result["argument_repair"]["source_id"] == "W1"
    assert result["argument_repair"]["repeated_step"] == 1


@pytest.mark.parametrize("change", ["different_query", "publisher", "period", "reason", "failed"])
def test_novel_actions_and_unrelated_validation_are_unchanged(change):
    decision, context, state = example()
    if change == "different_query":
        decision["query"] = "Gas forum secretary official release"
    elif change == "publisher":
        decision["publisher_domain"] = "energy.example"
    elif change == "period":
        decision["end_date"] = "2026-11-20"
    elif change == "reason":
        state["answer_checks"][-1]["validation"]["reason"] = "citation_review_failed"
    else:
        state["observations"][0]["error"] = "network failure"
    assert recover_event_search(decision, context, state, ["web_search", "read_web"]) is decision


@pytest.mark.parametrize(
    "ineligible", ["opened", "redirected", "off_topic", "unchecked", "scope", "rss"]
)
def test_reformulation_keeps_filters_and_is_attempted_once(monkeypatch, ineligible):
    decision, context, state = example()
    source = state["sources"][0]
    relevance = state["answer_checks"][-1]["validation"]["event_relevance"]
    if ineligible == "opened":
        state["observations"].append({"tool": "read_web", "arguments": {"url": source["url"]}})
    elif ineligible == "redirected":
        state["sources"].append({**source, "id": "W2", "content_type": "extracted"})
    elif ineligible == "off_topic":
        relevance["kept_source_ids"] = []
    elif ineligible == "unchecked":
        relevance["unverified_source_ids"] = ["W1"]
    elif ineligible == "scope":
        source["url"] = "https://other.example/release"
        context["publisher_pages"] = {source["url"]: ["W1"]}
    else:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        source["provider"] = "rss"
    actions = ["web_search", "read_web"]
    retry = recover_event_search(decision, context, state, actions)
    assert retry["action"] == "web_search" and "коммюнике" in retry["query"]
    assert decision["query"] in retry["query"]
    assert retry["publisher_domain"] == decision["publisher_domain"]
    assert retry["end_date"] == decision["end_date"]
    state["observations"].append({"tool": "web_search", "arguments": {**retry, "search": True}})
    assert recover_event_search(decision, context, state, actions) is decision
    assert recover_event_search(retry, context, state, actions) is retry
