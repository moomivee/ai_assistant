import pytest
from test_gigachat_client import auth_reply, completion, empty_completion
from test_gigachat_client import client as gigachat

from src.event_relevance import select_event_sources
from src.integrations.llm import LLMUnavailable, StructuredOutputError


def publication(number, text):
    return {
        "id": f"W{number}",
        "kind": "web",
        "title": f"Article {number}",
        "url": f"https://example.org/{number}",
        "text": text,
    }


class Gate:
    def __init__(self, *replies):
        self.calls = []
        self.replies = iter(replies)

    def generate(self, name, payload, schema):
        assert name == "event_relevance"
        self.calls.append(payload)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate({"off_topic": reply})


def test_excludes_unrelated_publication_and_preserves_announcement_and_reports():
    announcement = publication(1, "The international gas board says its next meeting is planned.")
    unrelated = publication(2, "Revenue from the independent central bank's bond auction.")
    new_report = {"id": "R1", "kind": "report", "text": "Газовый совет"}
    direct_quote = {**publication(3, "market quote"), "price": 73}
    client = Gate([2])
    selected, check = select_event_sources(
        client,
        "Когда собирался газовый совет?",
        [new_report, announcement, unrelated, direct_quote],
    )
    assert selected == [new_report, announcement, direct_quote]
    assert check["excluded_source_ids"] == ["W2"]
    assert check["considered_source_ids"] == ["W1", "W2"]
    assert [row["index"] for row in client.calls[0]["publications"]] == [1, 2]


@pytest.mark.parametrize(
    "bad_reply",
    [[0, 1], [2, 2], [3], StructuredOutputError("schema", issues=[], response_content="{}")],
)
def test_malformed_verdict_keeps_all_and_can_be_retried_on_next_handoff(bad_reply):
    sources = [publication(1, "First"), publication(2, "Second")]
    client, cache = Gate(bad_reply, [2]), {}
    original, check = select_event_sources(client, "When?", sources, cache=cache)
    assert original == sources and check["excluded_source_ids"] == []
    assert check["status"] == "unverified" and check["unverified_source_ids"] == ["W1", "W2"]
    assert not cache
    selected, check = select_event_sources(client, "When?", sources, cache=cache)
    assert selected == [sources[0]] and check["cache_hits"] == 0


def test_valid_empty_verdict_means_keep_all_and_only_classify_changed_material():
    client, cache = Gate([], [1], [], []), {}
    sources = [publication(1, "opening paragraph"), publication(2, "link")]
    select_event_sources(client, "When?", sources, cache=cache)
    updated = [sources[0], {**sources[1], "text": "opened press release"}]
    selected, check = select_event_sources(client, "When?", updated, cache=cache)
    assert selected == [sources[0]] and check["checked_source_ids"] == ["W2"]
    assert check["cache_hits"] == 1
    selected, check = select_event_sources(client, "When?", updated, cache=cache)
    assert check["cache_hits"] == 2 and len(client.calls) == 2
    for question, previous in [("Other question", None), ("Other question", "Other organization")]:
        selected, check = select_event_sources(
            client, question, updated, previous=previous, cache=cache
        )
        assert selected == updated and check["cache_hits"] == 0


def test_batched_excerpts_are_bounded_and_indices_are_local_to_each_batch():
    sources = [publication(i, "description" * 800) for i in range(1, 12)]
    client = Gate([1], [1])
    _, check = select_event_sources(client, "When?", sources)
    assert len(client.calls) == 2
    assert check["excluded_source_ids"] == ["W1", "W9"]
    for call in client.calls:
        publications = call["publications"]
        assert len(publications) <= 8 and publications[0]["index"] == 1
        assert all(len(row["excerpt"]) <= 603 for row in publications)


def test_all_explicitly_off_topic_selects_no_sources_but_outage_propagates():
    sources = [publication(1, "unrelated title")]
    selected, check = select_event_sources(Gate([1]), "When?", sources)
    assert selected == [] and check["status"] == "ok"
    with pytest.raises(LLMUnavailable, match="refused"):
        select_event_sources(Gate(LLMUnavailable("refused")), "When?", sources)


@pytest.mark.parametrize("finish", ["empty", "length", "blacklist"])
def test_optional_screen_handles_only_unusable_completions_from_gigachat(finish):
    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        return empty_completion() if finish == "empty" else completion(finish=finish)

    sources = [publication(1, "Minutes of the gas council")]
    client, native = gigachat(reply)
    try:
        if finish == "blacklist":
            with pytest.raises(LLMUnavailable, match="отказался"):
                select_event_sources(client, "Когда состоялся совет?", sources)
        else:
            selected, check = select_event_sources(client, "Когда состоялся совет?", sources)
            assert selected == sources and check["status"] == "unverified"
    finally:
        native.close()
