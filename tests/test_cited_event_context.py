"""Focused coverage of scoped dates, independent outcomes and temporal gaps."""

import pytest
from test_agent import WEB

from src.evidence import generate_answer


def answer(monkeypatch, chronology, claims, *, review=True):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true" if review else "false")
    monkeypatch.setattr("src.evidence.build_event_chronology", lambda *a, **k: chronology)
    sources = [
        {
            **WEB,
            "id": "W1",
            "date": None,
            "text": "At the Delta summit on 1 November 2019, delegates met.",
        },
        {**WEB, "id": "W2", "text": "The delegates of the Delta summit voted to suspend the fee."},
        {**WEB, "id": "W3", "text": "The next summit will take place on 29 November 2026."},
    ]

    class PermissiveReviewer:
        def generate(self, name, payload, schema):
            if name == "event_relevance":
                return schema(off_topic=[])
            if name == "cited_answer":
                return schema.model_validate({"claims": claims, "complete": True})
            assert name == "citation_review"
            assert payload["event_chronology"] == chronology
            return schema(
                supported=list(range(len(payload["claims"]))),
                complete=True,
                coverage="sufficient",
                feedback="",
            )

    checks = {}
    draft, _, feedback = generate_answer(
        PermissiveReviewer(),
        "Когда прошла последняя встреча?",
        sources,
        latest_completed_event=True,
        fresh=True,
        diagnostics=checks,
    )
    return draft, feedback, checks


def ledger(*, next_day="2026-11-29", requested=(), unresolved=()):
    latest = "2019-11-01"
    return {
        "as_of": "2026-11-21",
        "latest_held_date": latest,
        "latest_evidence_ids": ["W1:B1"],
        "held_events": [{"date": latest, "evidence_id": "W1:B1"}],
        "scheduled_events": [{"date": next_day, "evidence_id": "W3:B1"}],
        "requested_context": list(requested),
        "unresolved_evidence_ids": list(unresolved),
    }


HELD = {"text": "Делегаты собрались на саммит Delta 01.11.2019.", "evidence_ids": ["W1:B1"]}
DECISION = {"text": "Делегаты решили приостановить сбор.", "evidence_ids": ["W1:B1", "W2:B1"]}


@pytest.mark.parametrize(
    "extra",
    [[], [{"text": "Последняя встреча прошла 29 ноября 2026 года.", "evidence_ids": ["W3:B1"]}]],
)
def test_separate_requested_schedule_is_labeled_from_verified_role(monkeypatch, extra):
    draft, _, _ = answer(
        monkeypatch, ledger(requested=["next_scheduled"]), [HELD, DECISION, *extra]
    )
    assert draft.complete
    assert [c.text for c in draft.claims[:2]] == [HELD["text"], DECISION["text"]]
    assert len(draft.claims) == 3
    assert "объявленная дата" in draft.claims[2].text and "29.11.2026" in draft.claims[2].text
    assert draft.claims[2].evidence_ids == ["W3:B1"]


def test_past_record_without_newer_signal_has_no_fixed_expiry(monkeypatch):
    draft, _, _ = answer(monkeypatch, ledger(), [HELD])
    assert draft.complete and len(draft.claims) == 1


@pytest.mark.parametrize("review", [True, False])
@pytest.mark.parametrize(
    "chronology",
    [
        ledger(next_day="2020-03-01"),
        ledger(unresolved=["W3:B1"]),
        ledger(requested=["previous_held"]),
    ],
)
def test_gaps_keep_the_checked_historical_fact_but_prevent_complete(
    monkeypatch, chronology, review
):
    draft, feedback, checks = answer(monkeypatch, chronology, [HELD], review=review)
    assert not draft.complete and draft.claims[0].text == HELD["text"]
    assert feedback and checks["missing_aspects"]


def test_requested_future_date_can_never_be_the_held_anchor(monkeypatch):
    fake = {"text": "Делегаты встретились 29 ноября 2026 года.", "evidence_ids": ["W3:B1"]}
    draft, _, checks = answer(monkeypatch, ledger(requested=["next_scheduled"]), [fake])
    assert not draft.complete and not draft.claims
    assert checks["reason"] == "event_chronology_mismatch"


def test_mixed_date_claim_needs_repair_even_if_both_dates_were_requested(monkeypatch):
    fake = {
        "text": "Встреча прошла 29 ноября 2026 года, а на 1 ноября 2019 года была запланирована.",
        "evidence_ids": ["W1:B1", "W3:B1"],
    }
    draft, _, checks = answer(monkeypatch, ledger(requested=["next_scheduled"]), [fake])
    assert not draft.complete and not draft.claims
    assert checks["reason"] == "event_chronology_mismatch"
