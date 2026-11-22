"""Only full, dated organizer releases from the requested meeting family can become evidence."""

from datetime import date
from time import monotonic

import pytest

from src.integrations import opec_publications as press

TODAY = date(2031, 4, 15)
INDEX = "https://www.opec.org/press-releases.html"


def card(title, day, path):
    return f'<div class="card pritem"><h1>{title}</h1><p>{day} | Online</p><p></p><a href="{path}">Read more</a></div>'


def article(title, text):
    return f'<article><h3 id="articleHeadline">{title}</h3><div id="articleContent"><p>{text}</p></div></article>'


@pytest.fixture
def newsroom(monkeypatch):
    articles = [
        (
            "Treasury, Banks, and Operators review market stability",
            "14 April 2031",
            "other",
            "Bankers met on 14 April 2031 to discuss lending.",
        ),
        (
            "A, B, and C reaffirm commitment to market stability",
            "12 April 2031",
            "group",
            "The OPEC+ countries met virtually on 11 April 2031. The next meeting is scheduled for May.",
        ),
        (
            "Meeting of the Joint Ministerial Monitoring Committee",
            "8 April 2031",
            "jmmc",
            "The committee held its latest review on 7 April 2031.",
        ),
        (
            "99th OPEC and non-OPEC Ministerial Meeting",
            "1 April 2031",
            "ministers",
            "The OPEC and non-OPEC Ministerial Meeting took place on 29 March 2031.",
        ),
        (
            "101st Meeting of the OPEC Conference",
            "13 April 2031",
            "members",
            "The OPEC Conference was held on 12 April 2031.",
        ),
        (
            "102nd Meeting of the OPEC Conference",
            "25 April 2031",
            "future",
            "A future release must not be read.",
        ),
    ]
    pages, calls = {}, []
    cards = []
    for title, day, slug, text in articles:
        path = f"/pr-detail/{slug}.html"
        cards.append(card(title, day, path))
        pages["https://www.opec.org" + path] = article(title, text)
    cards.append(
        card("OPEC Conference", "13 April 2031", "https://elsewhere.example/pr-detail/bad.html")
    )
    pages[INDEX] = "".join(cards)

    async def read(client, url, deadline):
        calls.append(url)
        return pages.get(url, "").encode()

    monkeypatch.setattr(press, "_fetch_html", read)
    return pages, calls


def search(query, **kwargs):
    options = dict(limit=6, today=TODAY, start=None, end=None, deadline=monotonic() + 10)
    return press.search_opec_releases(query, **{**options, **kwargs})


def test_broad_query_checks_actual_bodies_and_preserves_attribution(newsroom):
    _, calls = newsroom
    hits = search("Когда была последняя встреча опэк+?")
    assert "https://www.opec.org/pr-detail/group.html" in {h["url"] for h in hits}
    assert all(
        item["provider"] == "opec_press" and item["content_type"] == "extracted" for item in hits
    )
    group = next(h for h in hits if h["url"].endswith("group.html"))
    assert group["date"] == "2031-04-12" and "11 April 2031" in group["text"]
    assert not any(h["url"].endswith(("other.html", "members.html", "future.html")) for h in hits)
    assert all(u.startswith("https://www.opec.org/") for u in calls)
    assert len(calls) <= 5


@pytest.mark.parametrize(
    ("query", "slug"),
    [
        ("Когда была последняя министерская встреча ОПЕК+?", "ministers"),
        ("Когда собирался мониторинговый комитет ОПЕК+?", "jmmc"),
        ("Когда была последняя встреча ОПЕК?", "members"),
        ("Когда собиралась добровольная группа ОПЕК+?", "group"),
    ],
)
def test_official_search_keeps_requested_meeting_body(newsroom, query, slug):
    assert [r["url"].rsplit("/", 1)[1] for r in search(query)] == [slug + ".html"]


def test_only_explicitly_linked_annual_index_and_requested_days_are_used(newsroom):
    pages, calls = newsroom
    title = "Meeting of the OPEC Conference"
    pages[INDEX] += '<a href="./press-releases-2029.html">2029</a>'
    pages["https://www.opec.org/press-releases-2029.html"] = card(
        title, "1 May 2029", "./pr-detail/a.html"
    )
    pages["https://www.opec.org/pr-detail/a.html"] = article(
        title, "The Conference met on 30 April 2029."
    )
    hits = search("Последняя конференция ОПЕК", start=date(2029, 5, 1), end=date(2029, 6, 1))
    assert [h["date"] for h in hits] == ["2029-05-01"]
    assert len(calls) == 3


def test_empty_blocked_official_or_changed_markup_leaves_provider_fallback_available(newsroom):
    pages, calls = newsroom
    assert not search("Цены на нефть")
    assert not calls
    pages[INDEX] = '<a href="https://www.opec.org/pr-detail/a.html">Unscoped link</a>'
    assert not search("Последняя встреча ОПЕК+")
    assert calls == [INDEX]


@pytest.mark.parametrize(
    "question",
    [
        "Когда была последняя встреча ОПЕК и Китая?",
        "When did OPEC last meet with India?",
        "Когда была последняя встреча ОПЕК на форуме в Эр-Рияде?",
        "Когда было последнее изменение состава ОПЕК?",
    ],
)
def test_external_or_nonmeeting_scope_keeps_open_search(newsroom, question):
    _, calls = newsroom
    assert not search(question)
    assert not calls
