import pytest

from src.event_dates import open_event_query
from src.event_relevance import select_event_sources
from src.event_search import event_publication_search, event_search_context
from src.event_subject import organizer_domain, organizer_search_terms, requested_event_publisher

TRUSTED = ("opec.org", "iea.org", "cbr.ru", "rbc.ru", "reuters.com", "forum.example")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Когда была последняя встреча опэк?", "opec.org"),
        ("Когда заседала ОПЕК+?", "opec.org"),
        ("Какое последнее сообщение выпустило МЭА?", "iea.org"),
        ("Когда прошла встреча ОПЕК и МЭА?", ""),
        ("Когда заседал отраслевой совет?", ""),
    ],
)
def test_known_organizer_without_planner_metadata(question, expected):
    assert organizer_domain(question, {}, TRUSTED) == expected
    assert not organizer_domain(question, {}, ("reuters.com",))


def test_planner_suggestion_must_be_trusted_and_grounded():
    event = {"quote": "forum.example", "domain": "forum.example"}
    assert (
        organizer_domain("Когда заседал forum.example?", {"event_organizer": event}, TRUSTED)
        == "forum.example"
    )
    assert not organizer_domain("Когда заседал совет?", {"event_organizer": event}, TRUSTED)
    event = {"quote": "совет", "domain": "reuters.com"}
    assert not organizer_domain("Когда заседал совет?", {"event_organizer": event}, TRUSTED)
    assert (
        organizer_domain(
            "А когда они собирались?", {"rag_query": "IEA meeting"}, TRUSTED, "Меня интересует МЭА."
        )
        == "iea.org"
    )


@pytest.mark.parametrize(
    ("question", "terms"),
    [
        ("Когда прошла встреча опэк?", "OPEC"),
        ("Когда была конференция ОПЕК?", "OPEC Conference"),
        ("Когда была последняя встреча ОПЕК+?", "OPEC+"),
        ("Когда прошла министерская встреча ОПЕК+?", "OPEC and non-OPEC Ministerial Meeting"),
        (
            "Когда встречались страны ОПЕК+ с добровольными сокращениями?",
            "OPEC+ voluntary production adjustments",
        ),
    ],
)
def test_archive_search_terms_keep_the_requested_group(question, terms):
    assert organizer_search_terms(question, "opec.org") == terms
    assert not organizer_search_terms(question, "iea.org")


class PermissiveModel:
    def generate(self, name, payload, schema):
        assert name == "event_relevance"
        return schema(off_topic=[])


def test_giga_accepting_everything_cannot_turn_an_alliance_item_into_members_only_event():
    titles = [
        "Восемь стран ОПЕК+ провели встречу",
        "Meeting of OPEC and non-OPEC ministers",
        "OPEC Conference",
        "OPEC+ meeting and OPEC Conference",
        "Future meeting",
    ]
    articles = [
        dict(id=f"W{i}", kind="web", title=title, text=title) for i, title in enumerate(titles, 1)
    ]
    member_items, check = select_event_sources(
        PermissiveModel(), "Когда встречалась опэк?", articles
    )
    assert check["other_event_body_source_ids"] == ["W1", "W2"]
    assert [s["id"] for s in member_items] == ["W3", "W4", "W5"]
    plus_items, _ = select_event_sources(PermissiveModel(), "Когда заседала опэк+?", articles)
    assert [s["id"] for s in plus_items] == ["W1", "W2", "W4", "W5"]
    # A joint-comparison question still needs all the evidence, including both official bodies.
    all_items, _ = select_event_sources(PermissiveModel(), "Сравни встречи ОПЕК и ОПЕК+", articles)
    assert all_items == articles


def test_explicit_or_freeform_source_limits_take_precedence(monkeypatch):
    monkeypatch.setenv("WEB_DOMAINS", ",".join(TRUSTED))
    state = dict(
        question="Когда была встреча опэк? Только по данным РБК.",
        plan=dict(rag_query="ОПЕК встреча", latest_completed_event=True),
        observations=[],
        sources=[],
        answer_checks=[],
        draft={"complete": False},
    )
    context = event_search_context(state)
    assert context["requested_domain"] == "rbc.ru" and not context["organizer_domain"]
    decision = event_publication_search(
        dict(query="опэк 2026", publisher_domain="opec.org"), context, state
    )
    assert (decision["query"], decision["publisher_domain"]) == ("ОПЕК", "rbc.ru")
    for limit in ("Кроме РБК", "Только РБК и Reuters"):
        state["question"] = "Когда прошла встреча ОПЕК? " + limit
        state["plan"]["event_source_limit"] = limit
        context = event_search_context(state)
        assert (
            context["source_limit"] == limit
            and not context["requested_domain"]
            and not context["organizer_domain"]
        )
    assert requested_event_publisher(
        "Когда заседала ОПЕК?", {"event_source_limit": "только РБК"}, TRUSTED
    ) == ("", "")


def test_calendar_year_supplied_by_model_does_not_limit_open_search():
    assert (
        open_event_query("IEA latest ministerial 2026", question="Когда был последний совет?")
        == "IEA latest ministerial"
    )
    assert (
        open_event_query("IEA 2026 ministerial", question="Когда был последний совет в 2026?")
        == "IEA 2026 ministerial"
    )
    assert (
        open_event_query(
            "IEA 2025 meeting", question="Когда прошла встреча?", lead_date="2025-02-02"
        )
        == "IEA 2025 meeting"
    )


@pytest.mark.parametrize("text", ["Не только РБК", "Только РБК, Reuters и FT"])
def test_a_named_publisher_is_not_necessarily_exclusive(text):
    assert requested_event_publisher(text, {}, TRUSTED)[1] == ""


def test_adjacent_energy_agency_meeting_is_screened_for_its_named_participants():
    from src.event_subject import other_event_body

    question = "Когда прошла последняя встреча МЭА?"
    assert other_event_body(question, "EIA meeting", "Minutes of the EIA meeting.")
    assert not other_event_body(question, "EIA meeting", "EIA and IEA met to discuss data.")
    assert not other_event_body(question, "New release", "A brief announcement.")
