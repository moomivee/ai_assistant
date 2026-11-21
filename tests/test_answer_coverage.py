"""Coverage contracts through the GigaChat adapter; not live answer-quality scores."""

import json

import pytest
from test_gigachat_client import auth_reply, completion
from test_gigachat_client import client as gigachat_client

from src.config import settings
from src.evidence import generate_answer
from src.models.schemas import CitationReview

SOURCE = {
    "id": "W4",
    "kind": "web",
    "title": "Справочный фрагмент",
    "date": None,
    "date_basis": "unknown",
    "retrieved_at": "2026-11-17T18:42:32+00:00",
    "url": "https://www.eia.gov/example",
    "text": "Содержание серы — 1,3%.",
    "content_type": "snippet",
    "content_scope": "search_result",
    "content_truncated": False,
}


@pytest.fixture(autouse=True)
def enable_review(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()


def generate(
    review,
    *,
    question="Какое содержание серы указано во фрагменте?",
    fresh=False,
    allow_research_clarification=False,
):
    requests, diagnostics = [], {}

    def reply(request):
        if request.url.path.endswith("oauth"):
            return auth_reply()
        payload = json.loads(request.content)
        title = payload["response_format"]["schema"]["title"]
        requests.append((title, payload))
        if title == "CitedAnswer":
            return completion(
                {
                    "claims": [{"text": SOURCE["text"], "evidence_ids": ["W4:B1"]}],
                    "complete": True,
                }
            )
        assert title == "CitationReview"
        return completion(review)

    client, native = gigachat_client(reply)
    try:
        result = generate_answer(
            client,
            question,
            [SOURCE],
            fresh=fresh,
            diagnostics=diagnostics,
            allow_research_clarification=allow_research_clarification,
        )
    finally:
        native.close()
    return result, diagnostics, requests


@pytest.mark.parametrize(
    "allow, options, supported, expect_choices",
    [
        (True, ["Место по добыче", "Место по экспорту"], [], True),
        (True, ["Добыча", "добыча."], [], False),
        (True, ["Место по экспорту"], [], False),
        (True, ["Добыча", "Экспорт"], [0], False),
        (False, ["Добыча", "Экспорт"], [], False),
    ],
)
def test_unclear_research_goal_uses_existing_review(allow, options, supported, expect_choices):
    (draft, _, _), diagnostics, requests = generate(
        {
            "supported": supported,
            "complete": False,
            "feedback": "Не указан критерий сравнения.",
            "coverage": "unverified",
            "research_options": options,
        },
        question="Какой международный рейтинг у этой нефти?",
        allow_research_clarification=allow,
    )
    assert [title for title, _ in requests] == ["CitedAnswer", "CitationReview"]
    assert bool(draft.claims) == bool(supported)
    assert not draft.complete
    review_input = json.loads(requests[-1][1]["messages"][-1]["content"])
    assert review_input["allow_research_clarification"] is allow
    if expect_choices:
        clarification = diagnostics["research_clarification"]
        assert clarification["research_options"] == options
        assert all(value in clarification["clarify"] for value in options)
    else:
        assert "research_clarification" not in diagnostics


def test_supported_property_is_not_complete_coverage_of_a_trade_question():
    gap = "Не установлено, какими сортами торгуют: характеристика сорта не доказывает поставки."
    (draft, _, feedback), diagnostic, requests = generate(
        {
            "supported": [0],
            "complete": False,
            "feedback": "",
            "coverage": "partial",
            "missing_aspects": [gap],
        },
        question="Какой нефтью торгует страна?",
    )
    assert draft.claims and not draft.complete
    assert gap in feedback
    assert diagnostic["accepted_claims"] == 1
    assert diagnostic["coverage"] == "partial"
    assert diagnostic["missing_aspects"] == [gap]
    assert diagnostic["reason"] == "coverage_incomplete"
    assert diagnostic["review_enabled"] and diagnostic["review_executed"]
    assert [title for title, _ in requests] == ["CitedAnswer", "CitationReview"]


@pytest.mark.parametrize(
    "coverage,missing_aspects",
    [
        ("partial", []),
        ("unverified", []),
        ("sufficient", ["Требуется подтверждение периода поставок."]),
    ],
)
def test_complete_flag_cannot_override_incomplete_or_unverified_coverage(coverage, missing_aspects):
    (draft, _, feedback), diagnostic, _ = generate(
        {
            "supported": [0],
            "complete": True,
            "feedback": "",
            "coverage": coverage,
            "missing_aspects": missing_aspects,
        }
    )
    assert draft.claims and not draft.complete and feedback
    assert diagnostic["coverage"] == coverage


def test_omitted_coverage_is_unverified_instead_of_a_schema_error_or_full_answer():
    (draft, _, feedback), diagnostic, requests = generate(
        {"supported": [0], "complete": True, "feedback": ""}
    )
    assert draft.claims and not draft.complete
    assert "Coverage of the question was not verified" in feedback
    assert diagnostic["reason"] == "coverage_unverified"
    assert len(requests) == 2


def test_short_undated_snippet_can_cover_a_narrow_source_bounded_question():
    (draft, evidence, feedback), diagnostic, requests = generate(
        {
            "supported": [0],
            "complete": True,
            "feedback": "",
            "coverage": "sufficient",
            "missing_aspects": [],
        }
    )
    assert draft.complete and not feedback
    assert diagnostic["reason"] == "complete"
    assert diagnostic["coverage"] == "sufficient"
    assert evidence["W4:B1"]["content_type"] == "snippet"
    assert evidence["W4:B1"]["content_truncated"] is False
    # Both generations receive the excerpt metadata; no forced page-read call
    # or minimum source count is required for a fully covered narrow question.
    for _, payload in requests:
        data = json.loads(payload["messages"][-1]["content"])
        supplied = data["sources"][0] if "sources" in data else data["claims"][0]["evidence"][0]
        assert supplied["content_type"] == "snippet"
        assert supplied["content_truncated"] is False
        assert supplied["date"] is None


def test_disabled_review_is_visible_and_does_not_claim_coverage_was_checked(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "false")
    settings.cache_clear()
    (_, _, _), diagnostic, requests = generate({})
    assert diagnostic["review_enabled"] is False
    assert diagnostic["review_executed"] is False
    assert diagnostic["candidate_complete"] is True
    assert diagnostic["reason"] == "semantic_review_disabled"
    assert diagnostic["coverage"] == "unverified"
    assert diagnostic["missing_aspects"] == []
    assert [title for title, _ in requests] == ["CitedAnswer"]


@pytest.mark.parametrize("aspects", [["x" * 251], ["gap"] * 7])
def test_missing_aspects_remain_bounded_for_the_research_feedback(aspects):
    with pytest.raises(ValueError):
        CitationReview(
            supported=[],
            complete=False,
            feedback="",
            coverage="partial",
            missing_aspects=aspects,
        )
