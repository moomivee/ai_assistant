"""Only direct, attributable excerpts may be returned outside verified claims."""

import pytest

from src.event_chronology import build_event_chronology
from src.evidence import catalog
from src.preliminary import preliminary_event_excerpts

QUESTION = "Когда участники газового форума встретились в последний раз?"
ARTICLE = "Участники газового форума встретились для обсуждения экспорта."
SCHEDULE = "Следующая встреча газового форума пройдет 5 октября."


def article(text, **overrides):
    return {
        "id": "W1",
        "kind": "web",
        "url": "https://news.example/forum",
        "text": text,
        "date": "2025-09-07",
        "title": "Встреча газового форума",
        "content_type": "extracted",
        **overrides,
    }


def choose(sources, *, ledger=None, relevance=None, **kwargs):
    evidence = catalog(sources)
    return preliminary_event_excerpts(
        sources=sources,
        question=QUESTION + " Используй публикации сентября 2025.",
        evidence=evidence,
        validation={
            "event_chronology": {
                "as_of": "2026-11-22",
                "latest_held_date": None,
                "unresolved_evidence_ids": list(evidence),
                **(ledger or {}),
            },
            "event_relevance": {"kept_source_ids": [s["id"] for s in sources], **(relevance or {})},
        },
        **kwargs,
    )


def test_returns_direct_attributed_occurrence_and_archival_schedule_instead_of_related_headlines():
    sources = [
        article("Участники газового форума встретились 1 ноября.", id="W2", content_type="snippet"),
        article(ARTICLE + "\n" + SCHEDULE + "\nНовости по теме\n" + SCHEDULE.replace("5", "9")),
        article(ARTICLE, id="W3", date="2025-08-01"),
    ]
    result = choose(sources)
    assert [(item["source_id"], item["kind"], item["quote"]) for item in result] == [
        ("W1", "held_report", ARTICLE),
        ("W1", "schedule_report", SCHEDULE),
    ]
    assert all(item["publication_date"] == "2025-09-07" for item in result)


@pytest.mark.parametrize(
    "changes",
    [
        {"ledger": {"latest_held_date": "2026-11-01"}},
        {"ledger": {"other_scope_blocks": ["W1:B1"]}},
        {"ledger": {"extraction": {"unparsed_evidence_ids": ["W1:B1"]}}},
        {"relevance": {"unverified_source_ids": ["W1"]}},
        {"relevance": {"kept_source_ids": []}},
        {"exclude_ids": {"W1"}},
        {"exclude_ids": {"W1:B1"}},
        {"publication_window": {"quote": "публикации сентября 2025", "end_date": "2025-09-06"}},
    ],
)
def test_dates_scopes_and_citation_constraints_are_preserved(changes):
    assert not choose([article(ARTICLE)], **changes)


def test_navigation_rejected_even_on_another_evidence_block():
    # Place the related headline in a different catalog block from the divider.
    body = "Новости по теме\n" + ("обзор " * 270) + "\n" + ARTICLE
    assert len(catalog([article(body)])) > 1
    assert not choose([article(body)])


@pytest.mark.parametrize("day", [None, "н/д", "2026-11-23"])
def test_unclassified_fallback_needs_dated_publication(day):
    assert not choose([article(ARTICLE, date=day)])


def test_temporal_annotation_uses_exact_original_even_without_year_or_publication_date():
    quote = "Газовый совет возобновил работу после перерыва."
    text = "Заметка.\n" + quote
    sources = [article(text, date=None)]
    evidence = catalog(sources)

    class Reader:
        def generate(self, name, payload, schema):
            return schema.model_validate(
                {
                    "blocks": [
                        {
                            "evidence_id": "W1:B1",
                            "status": "uncertain",
                            "events": [],
                            "reported_excerpt": {"kind": "held_report", "quote": quote},
                        }
                    ]
                }
            )

    ledger = build_event_chronology(Reader(), QUESTION, evidence)
    assert not ledger["held_events"] and ledger["latest_held_date"] is None
    excerpt = choose(sources, ledger=ledger)
    assert excerpt[0]["quote"] == quote and excerpt[0]["publication_date"] is None
    assert not choose(
        sources,
        ledger=ledger,
        publication_window={"quote": "публикации сентября 2025", "start_date": "2025-09-01"},
    )


@pytest.mark.parametrize(
    "reported",
    [
        {"kind": "held_report", "quote": "An invented quote."},
        {"kind": "incorrect", "quote": "A malformed role."},
        {"kind": "held_report", "quote": "too long " * 100},
    ],
)
def test_optional_excerpt_cannot_introduce_text_or_invalidate_a_verified_row(reported):
    text = "The council met on 7 September 2025."

    class Reader:
        def generate(self, name, payload, schema):
            return schema.model_validate(
                {
                    "blocks": [
                        dict(
                            evidence_id="W1:B1",
                            status="assessed",
                            events=[
                                dict(
                                    kind="held",
                                    scope="council",
                                    date="2025-09-07",
                                    quote=text,
                                    match="in_scope",
                                )
                            ],
                            reported_excerpt=reported,
                        )
                    ]
                }
            )

    ledger = build_event_chronology(
        Reader(), "When did the council meet?", catalog([article(text)])
    )
    assert ledger["latest_held_date"] == "2025-09-07"
    assert not ledger["reported_excerpts"]


@pytest.mark.parametrize("background", ["", "The archive contains several related articles. " * 9])
def test_selected_clause_retains_negation_and_the_attribution(background):
    quote = "the council will convene on Monday"
    sentence = f"A ministry official denied that {quote}."
    text = background + sentence
    start = text.index(quote)
    ledger = {
        "reported_excerpts": {
            "W1:B1": {"kind": "schedule_report", "start": start, "end": start + len(quote)}
        }
    }
    result = choose([article(text, date=None)], ledger=ledger)
    assert result[0]["quote"] == sentence


def test_selected_excerpt_crossing_a_long_original_line_is_omitted():
    quote = "the council will convene on Monday"
    note = "The source says " + " 'maybe' " * 220 + quote + "."
    sources = [article(note, date=None)]
    evidence = catalog(sources)
    key, block = next((k, b) for k, b in evidence.items() if quote in b["text"])
    start = block["text"].index(quote)
    ledger = {
        "reported_excerpts": {
            key: {"kind": "schedule_report", "start": start, "end": start + len(quote)}
        }
    }
    assert not choose(sources, ledger=ledger)
