"""Review contracts and observed failure shapes; not a live-model quality score."""

import json
from types import SimpleNamespace

import pytest

from src.config import settings
from src.evidence import generate_answer, render_claims
from src.integrations.llm import LLMClient, LLMUnavailable

CLOCK = {
    "today": "2026-11-17",
    "current_time": "2026-11-17T20:00:00+03:00",
    "timezone": "Europe/Moscow",
    "current_year": 2026,
}
REPORT = {
    "id": "R7",
    "kind": "report",
    "title": "EIA Russia analysis",
    "date": "2025-07-24",
    "date_basis": "publication",
    "retrieved_at": "2026-11-17T17:00:00+00:00",
    "page": 24,
    "url": "https://www.eia.gov/international/content/analysis/countries_long/Russia/",
    "text": (
        "References\nReuters, EU's new Russia sanctions aim for more effective oil price cap, "
        "July 18, 2025.\nOPEC, eight countries adjust production upward, April 3, 2025."
    ),
}


class ScriptedReview:
    def __init__(self, claim, review, *, complete=True):
        self.responses = {
            "cited_answer": {"claims": [claim], "complete": complete},
            "citation_review": review,
        }
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append((name, payload))
        return schema.model_validate(self.responses[name])


def claim(text="Последняя встреча состоялась в июле 2025 года."):
    return {"text": text, "evidence_ids": ["R7:B1"]}


@pytest.fixture(autouse=True)
def semantic_review(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    monkeypatch.setattr("src.evidence.clock_payload", lambda: CLOCK.copy())
    settings.cache_clear()


def test_bibliography_numbers_are_not_enough_to_accept_event():
    client = ScriptedReview(
        claim(),
        {
            "coverage": "partial",
            "supported": [],
            "complete": False,
            "feedback": "Библиография не подтверждает дату встречи. Откройте первичный источник.",
        },
    )
    draft, evidence, feedback = generate_answer(
        client, "Когда была последняя встреча и что решили?", [REPORT], fresh=True
    )
    # The old numeric/ID check passes, so the rejection must reach the reviewer.
    assert [name for name, _ in client.calls] == ["cited_answer", "citation_review"]
    review_payload = client.calls[-1][1]
    assert review_payload["claims"][0]["evidence"][0]["text"] == REPORT["text"]
    assert not draft.claims and not draft.complete
    assert "Библиография" in feedback
    assert evidence["R7:B1"]["source_text_end"] == len(REPORT["text"])


def test_reviewer_receives_clock_scope_and_only_claims_cited_evidence():
    unrelated = {**REPORT, "id": "W1", "kind": "web", "text": "Unrelated new publication."}
    client = ScriptedReview(
        claim(),
        {
            "coverage": "partial",
            "supported": [],
            "complete": False,
            "feedback": "Currency is unverified.",
        },
    )
    generate_answer(client, "Последняя встреча?", [REPORT, unrelated], fresh=True, report_only=True)
    payload = client.calls[-1][1]
    assert all(payload[k] == v for k, v in CLOCK.items())
    assert payload["temporal_requirements"] == {"fresh": True, "report_only": True}
    evidence = payload["claims"][0]["evidence"]
    assert len(evidence) == 1 and evidence[0]["source"] == "R7"
    for field in ("title", "kind", "date", "date_basis", "retrieved_at", "page", "url"):
        assert evidence[0][field] == REPORT[field]
    assert "user_scope" in payload


def test_historical_fact_can_pass_and_keep_exact_source_quote():
    source = {**REPORT, "text": "Встреча состоялась 3 апреля 2025 года."}
    client = ScriptedReview(
        claim(source["text"]),
        {"coverage": "sufficient", "supported": [0], "complete": True, "feedback": ""},
    )
    draft, evidence, feedback = generate_answer(
        client, "Когда была встреча в апреле 2025?", [source]
    )
    assert draft.complete and not feedback
    blocks, claims, sources = render_claims(draft, evidence, [source])
    assert blocks == [source["text"] + " [R7]"]
    assert claims[0]["quote"] == source["text"]
    assert sources[0]["evidence"] == {"R7:B1": source["text"]}
    assert sources[0]["page"] == 24


@pytest.mark.parametrize(
    "raw_complete,review",
    [
        (False, {"coverage": "sufficient", "supported": [0], "complete": True, "feedback": ""}),
        (True, {"coverage": "partial", "supported": [0], "complete": False, "feedback": ""}),
        (
            True,
            {
                "coverage": "sufficient",
                "supported": [0],
                "complete": True,
                "feedback": "Дата не проверена.",
            },
        ),
    ],
)
def test_disagreement_never_upgrades_answer_to_complete(raw_complete, review):
    client = ScriptedReview(claim(), review, complete=raw_complete)
    draft, _, feedback = generate_answer(client, "Последняя встреча?", [REPORT], fresh=True)
    assert draft.claims and not draft.complete and feedback
    assert len(client.calls) == 2  # One review, no recursive self-repair loop.


@pytest.mark.parametrize("indexes", [[0, 0], [1], [-1]])
def test_invalid_review_indexes_fail_closed(indexes):
    client = ScriptedReview(
        claim(), {"coverage": "sufficient", "supported": indexes, "complete": True, "feedback": ""}
    )
    draft, _, feedback = generate_answer(client, "Последняя встреча?", [REPORT])
    assert not draft.claims and not draft.complete
    assert "invalid or duplicate" in feedback


def test_current_answer_cannot_finish_with_semantic_review_disabled(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "false")
    settings.cache_clear()
    client = ScriptedReview(claim(), {})
    draft, _, feedback = generate_answer(client, "Последняя встреча?", [REPORT], fresh=True)
    assert draft.claims and not draft.complete
    assert "semantic citation review is disabled" in feedback
    assert [name for name, _ in client.calls] == ["cited_answer"]


def test_malformed_review_through_real_client_is_not_an_accepted_draft():
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            # Observed structured-output failure: model echoes a schema instead
            # of returning a data instance. Exercise real Pydantic validation.
            data = {"claims": [claim()], "complete": True} if self.calls == 1 else kwargs["schema"]
            return SimpleNamespace(content=json.dumps(data), model="fake", usage={}, attempts=[])

    provider = FakeProvider()
    client = LLMClient(provider, provider="gigachat")
    with pytest.raises(LLMUnavailable, match="описание JSON-схемы"):
        generate_answer(client, "Последняя встреча?", [REPORT], fresh=True)
    assert provider.calls == 2
