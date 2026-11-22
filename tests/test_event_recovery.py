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


def news_batch(items, *, primary="forum.example", publisher=""):
    decision, context, state = example()
    decision.update(publisher_domain=publisher, end_date=None)
    search = state["observations"][0]
    search["arguments"].update(publisher_domain=publisher, end_date=None)
    context.update(
        organizer_domain=primary,
        trusted_publishers=["forum.example", "wire.example", "press.example"],
    )
    sources = [
        dict(
            id=f"W{i}",
            url=f"https://{host}/news/{i}",
            date=day,
            provider="tavily",
            content_type="snippet",
        )
        for i, (host, day) in enumerate(items, 1)
    ]
    state["sources"] = sources
    context["publisher_pages"] = {s["url"]: [s["id"]] for s in sources}
    ids = [s["id"] for s in sources]
    search["source_ids"] = ids
    state["answer_checks"][-1]["validation"]["event_relevance"]["kept_source_ids"] = ids
    return decision, context, state


@pytest.mark.parametrize("proposal", ["repeat", "one_more_year", "explicit_read"])
def test_archive_recovery_targets_allowed_organizer_immediately(proposal, web_today):
    decision, context, state = news_batch(
        [("wire.example", "2009-02-11"), ("wire.example", "2015-05-01")]
    )
    if proposal == "one_more_year":
        decision["query"] += " 2026"
    elif proposal == "explicit_read":
        decision.update(action="read_web", url=state["sources"][0]["url"], query="")
    result = recover_event_search(decision, context, state, ["web_search", "read_web"])
    assert result["action"] == "web_search" and result["publisher_domain"] == "forum.example"
    assert "газового форума" in result["query"] and not result["start_date"]
    assert result["model_action"] == decision["action"]


@pytest.mark.parametrize("primary", ["", "untrusted.example"])
def test_newest_checked_hit_then_moves_to_another_publisher(primary, web_today):
    decision, context, state = news_batch(
        [
            ("wire.example", "2009-02-11"),
            ("wire.example", "2026-11-02"),
            ("press.example", "2026-11-01"),
            ("press.example", "2026-11-03"),
        ],
        primary=primary,
    )
    state["answer_checks"][-1]["validation"]["event_relevance"]["unverified_source_ids"] = ["W4"]
    actions = ["web_search", "read_web"]
    result = recover_event_search(decision, context, state, actions)
    assert result["argument_repair"]["source_id"] == "W2"
    state["observations"].append({"tool": "read_web", "arguments": {"url": result["url"]}})
    result = recover_event_search(decision, context, state, actions)
    assert result["argument_repair"]["source_id"] == "W3"


def test_journalist_publisher_filter_and_historical_period_are_preserved(web_today):
    decision, context, state = news_batch(
        [("wire.example", "2010-02-01")], publisher="wire.example"
    )
    actions = ["web_search", "read_web"]
    assert recover_event_search(decision, context, state, actions)["action"] == "read_web"

    decision, context, state = news_batch([("wire.example", "2010-02-01")])
    window = {"start_date": "2010-01-01", "end_date": "2010-12-31"}
    state["observations"][0]["arguments"].update(window)
    decision.update(window)
    assert recover_event_search(decision, context, state, actions)["action"] == "read_web"


@pytest.mark.parametrize("reason", ["no_relevant_event_sources", "no_evidence"])
def test_empty_checked_batch_seeks_official_source(reason):
    decision, context, state = news_batch([])
    state["answer_checks"][-1]["validation"]["reason"] = reason
    result = recover_event_search(decision, context, state, ["web_search", "read_web"])
    assert result["action"] == "web_search" and result["publisher_domain"] == "forum.example"


def test_uses_primary_lead_if_model_goes_back_to_broad_query(web_today):
    decision, context, state = news_batch([("wire.example", "2009-02-11"), ("forum.example", None)])
    actions = ["web_search", "read_web"]
    state["observations"][0]["source_ids"] = ["W1"]
    result = recover_event_search(decision, context, state, actions)
    assert result["publisher_domain"] == "forum.example"
    state["observations"].append(
        {"tool": "web_search", "arguments": {**result, "search": True}, "source_ids": ["W2"]}
    )
    result = recover_event_search(decision, context, state, actions)
    assert result["action"] == "read_web" and result["argument_repair"]["source_id"] == "W2"


def test_pivot_adds_language_of_international_primary_site_without_inventing_format():
    from src.event_recovery import publication_query

    question = "Когда была последняя встреча стран ОПЕК+?"
    query = publication_query(
        "последняя встреча ОПЕК+", organizer_domain="opec.org", question=question
    )
    assert "OPEC+" in query and "official communiqué" in query
    assert "последняя встреча ОПЕК+" in query and "Ministerial" not in query
    russian = publication_query(
        "заседание", organizer_domain="cbr.ru", question="Заседание Банка России", language="en"
    )
    assert "коммюнике" in russian and "official communiqué" not in russian


def test_replaces_model_read_of_old_article_with_a_fresher_checked_source(web_today):
    decision, context, state = news_batch(
        [("wire.example", "2009-02-11"), ("press.example", "2026-11-01")]
    )
    decision.update(action="read_web", url=state["sources"][0]["url"], query="")
    result = recover_event_search(decision, context, state, ["web_search", "read_web"])
    assert result["action"] == "read_web" and result["argument_repair"]["source_id"] == "W2"


def test_user_requested_publisher_applies_to_model_selected_page_and_ranking(web_today):
    decision, context, state = news_batch(
        [("wire.example", "2026-11-16"), ("press.example", "2026-11-09")]
    )
    context.update(requested_domain="press.example", source_limit="только Press")
    actions = ["web_search", "read_web"]
    result = recover_event_search(decision, context, state, actions)
    assert result["url"] == state["sources"][1]["url"]
    decision.update(action="read_web", url=state["sources"][0]["url"], query="")
    result = recover_event_search(decision, context, state, actions)
    assert result["action"] == "read_web" and result["url"] == state["sources"][1]["url"]


def test_ungrounded_publisher_restriction_preserves_the_original_search():
    decision, context, state = news_batch([("wire.example", "2009-02-11")])
    context["source_limit"] = "кроме отраслевых медиа"
    assert recover_event_search(decision, context, state, ["web_search", "read_web"]) is decision


def test_diversifies_monopoly_once_within_allowed_sites_and_reuses_new_lead(web_today):
    decision, context, state = news_batch(
        [("m.wire.example", "2009-05-01"), ("press.example", "2026-11-17")], primary=""
    )
    actions = ["web_search", "read_web"]
    state["observations"][0]["source_ids"] = ["W1"]
    result = recover_event_search(decision, context, state, actions)
    assert result["action"] == "web_search" and result["omit_publisher_domain"] == "wire.example"
    assert not result["publisher_domain"] and not result["start_date"]
    state["observations"].append(
        {"tool": "web_search", "arguments": {**result, "search": True}, "source_ids": ["W2"]}
    )
    result = recover_event_search(decision, context, state, actions)
    assert result["action"] == "read_web" and result["argument_repair"]["source_id"] == "W2"
    state["observations"][-1]["source_ids"] = []
    result = recover_event_search(decision, context, state, actions)
    assert result["action"] == "read_web" and result["argument_repair"]["source_id"] == "W1"


def test_empty_organizer_search_can_recover_via_other_publishers(web_today):
    decision, context, state = news_batch([("wire.example", "2009-05-01")])
    actions = ["web_search", "read_web"]
    primary = recover_event_search(decision, context, state, actions)
    assert primary["publisher_domain"] == "forum.example"
    state["observations"].append(
        {"tool": "web_search", "arguments": {**primary, "search": True}, "source_ids": []}
    )
    alternate = recover_event_search(decision, context, state, actions)
    assert (
        alternate["action"] == "web_search" and alternate["omit_publisher_domain"] == "wire.example"
    )


def test_single_publisher_or_date_restriction_remains_in_force(web_today):
    decision, context, state = news_batch([("wire.example", "2009-05-01")], primary="")
    actions = ["web_search", "read_web"]
    context["trusted_publishers"] = ["wire.example"]
    assert recover_event_search(decision, context, state, actions)["action"] == "read_web"
    context["trusted_publishers"].append("press.example")
    decision["end_date"] = "2010-01-01"
    state["observations"][0]["arguments"]["end_date"] = decision["end_date"]
    assert recover_event_search(decision, context, state, actions)["action"] == "read_web"


def test_a_new_committee_name_can_be_a_substantive_single_word_change(web_today):
    decision, context, state = news_batch([("wire.example", "2009-05-01")])
    decision["query"] += " JMMC"
    assert recover_event_search(decision, context, state, ["web_search", "read_web"]) is decision
