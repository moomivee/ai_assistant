from datetime import timedelta

import httpx
import pytest

from src.config import settings
from src.integrations import web


def article(url, score=None, published=None):
    return {"url": url, "score": score, "published_date": published, "content": "Oil market news."}


@pytest.fixture
def tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv(
        "WEB_DOMAINS", "wire.example,press.wire.example,paper.example,official.example"
    )
    settings.cache_clear()
    calls, results = [], []

    def post(url, *, json, **kwargs):
        calls.append(json)
        data = {"results": results[: json["max_results"]]}
        return httpx.Response(200, request=httpx.Request("POST", url), json=data)

    monkeypatch.setattr(web.httpx, "post", post)
    return calls, results


@pytest.mark.parametrize("fresh,topic", [(True, "general"), (False, "news")])
def test_broad_search_keeps_close_independent_publishers(tavily, fresh, topic):
    calls, results = tavily
    results.extend(
        [
            article("https://wire.example/one", 0.99),
            article("https://press.wire.example/two", 0.98),
            article("https://www.wire.example/three", 0.97),
            article("https://paper.example/four", 0.96),
            article("https://official.example/five", 0.93),
        ]
    )
    rows = web.search_web("decisions at the meeting", fresh, topic=topic, max_results=3)
    assert [row["url"] for row in rows] == [results[i]["url"] for i in (0, 3, 4)]
    assert len(calls) == 1 and calls[0]["max_results"] == 6
    cached = web.search_web("decisions at the meeting", fresh, topic=topic, max_results=3)
    assert cached == [{**row, "cache_hit": True} for row in rows]
    two = web.search_web("decisions at the meeting", fresh, topic=topic, max_results=2)
    assert [row["url"] for row in two] == [results[i]["url"] for i in (0, 3)]
    assert len(calls) == 2


@pytest.mark.parametrize("score", [0.2, None, "0.99"])
def test_weaker_or_unscored_sources_do_not_displace_relevant_results(tavily, score):
    _, results = tavily
    results.extend(
        [
            article("https://wire.example/one", 0.99),
            article("https://wire.example/two", 0.97),
            article("https://paper.example/three", score),
        ]
    )
    rows = web.search_web("latest decisions", True, max_results=2)
    assert [row["url"] for row in rows] == [results[0]["url"], results[1]["url"]]


@pytest.mark.parametrize("options", [{"publisher_domain": "wire.example"}, {"fresh": False}])
def test_narrow_and_non_news_searches_keep_their_original_scope(tavily, options):
    calls, results = tavily
    results.extend(
        [
            article("https://wire.example/one", 0.99),
            article("https://press.wire.example/two", 0.98),
            article("https://paper.example/three", 0.98),
        ]
    )
    rows = web.search_web("OPEC", **{"fresh": True, "max_results": 2, **options})
    assert [row["url"] for row in rows] == [results[0]["url"], results[1]["url"]]
    assert calls[0]["max_results"] == 2
    if "publisher_domain" in options:
        assert calls[0]["include_domains"] == ["wire.example"]


def test_candidates_must_pass_existing_publisher_and_calendar_filters(tavily, web_today):
    _, results = tavily
    today = web_today.isoformat()
    stale = (web_today - timedelta(days=8)).isoformat()
    results.extend(
        [
            article("https://wire.example/one", 0.99, today),
            article("https://paper.example/old", 0.99, stale),
            article("https://untrusted.example/news", 0.98, today),
            article("https://wire.example/two", 0.97, today),
        ]
    )
    rows = web.search_web("OPEC", True, days=7, max_results=2)
    assert [row["url"] for row in rows] == [results[0]["url"], results[3]["url"]]


def test_independent_lookup_filters_at_provider_and_in_returned_material(tavily):
    calls, results = tavily
    results.extend(
        [
            article("https://press.wire.example/copy", 0.99),
            article("https://paper.example/report", 0.96),
        ]
    )
    reply = web.web_lookup(
        "meeting decisions", True, [], True, omit_publisher_domain="wire.example"
    )
    assert calls[0]["include_domains"] == ["official.example", "paper.example"]
    assert [row["url"] for row in reply["items"]] == [results[1]["url"]]


@pytest.mark.parametrize(
    "publisher,omit", [("paper.example", "wire.example"), ("", "other.example")]
)
def test_independent_lookup_cannot_replace_requested_or_untrusted_domain(tavily, publisher, omit):
    calls, _ = tavily
    with pytest.raises(ValueError):
        web.web_lookup(
            "meeting", True, [], True, publisher_domain=publisher, omit_publisher_domain=omit
        )
    assert not calls
