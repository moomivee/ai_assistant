"""Distinguish empty candidate generation from rejected claims without logging text."""

import json

import pytest
from test_agent import REPORT, plan
from test_agent_loop import action, event

from src.agent import ask
from src.config import settings
from src.evidence import generate_answer
from src.integrations.llm import LLMUnavailable


class Responses:
    def __init__(self, claims, *, complete=True, review=None):
        self.claims = claims
        self.complete = complete
        self.review = review
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append(name)
        if name == "cited_answer":
            return schema(claims=self.claims, complete=self.complete)
        if isinstance(self.review, Exception):
            raise self.review
        assert name == "citation_review"
        return schema.model_validate(self.review)


def claim(text="Сера 0.2%.", evidence_ids=None):
    return {"text": text, "evidence_ids": evidence_ids or ["R1:B1"]}


def test_empty_generation_is_not_reported_as_invalid_citations_or_numbers():
    diagnostic = {}
    client = Responses([], complete=True)
    draft, _, feedback = generate_answer(
        client, "Нет подходящего факта?", [REPORT], diagnostics=diagnostic
    )
    assert not draft.complete and not draft.claims
    assert diagnostic["candidate_claims"] == 0
    assert diagnostic["reason"] == "no_candidate_claims"
    assert all(
        diagnostic[field] == 0
        for field in (
            "citation_rejected",
            "numeric_rejected",
            "semantic_rejected",
            "accepted_claims",
        )
    )
    assert "no supported claims" in feedback and "do not repeat" in feedback
    assert "No claims passed citation and numeric checks" not in feedback
    assert client.calls == ["cited_answer"]


def test_local_rejections_count_claims_at_the_first_failed_check_only():
    diagnostic = {}
    secret = "Untrusted rejected text should never enter diagnostics"
    client = Responses(
        [
            claim(secret + " 99%", ["unknown-id"]),
            claim("Сера 99%."),
            claim(),
        ]
    )
    draft, _, _ = generate_answer(client, "Сера?", [REPORT], diagnostics=diagnostic)
    assert diagnostic["candidate_claims"] == 3
    assert diagnostic["citation_rejected"] == 1
    assert diagnostic["numeric_rejected"] == 1
    assert diagnostic["semantic_rejected"] == 0
    assert diagnostic["accepted_claims"] == 1
    assert diagnostic["reason"] == "semantic_review_disabled" and not draft.complete
    assert diagnostic["coverage"] == "unverified" and diagnostic["review_executed"] is False
    assert len(draft.claims) == 1
    encoded = json.dumps(diagnostic)
    assert secret not in encoded and "unknown-id" not in encoded and "99" not in encoded


def test_all_invalid_candidates_are_distinct_from_empty_generation():
    diagnostic = {}
    draft, _, feedback = generate_answer(
        Responses([claim("Сера 99%.")]), "Сера?", [REPORT], diagnostics=diagnostic
    )
    assert not draft.claims
    assert diagnostic["candidate_claims"] == 1
    assert diagnostic["numeric_rejected"] == 1
    assert diagnostic["reason"] == "local_validation_rejected"
    assert "No claims passed citation and numeric checks" in feedback


def test_semantic_rejections_have_separate_counters(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    diagnostic = {}
    client = Responses(
        [claim()],
        review={
            "coverage": "partial",
            "supported": [],
            "complete": False,
            "feedback": "Not relevant to the requested object.",
        },
    )
    draft, _, _ = generate_answer(client, "Иной объект?", [REPORT], diagnostics=diagnostic)
    assert not draft.claims and not draft.complete
    assert diagnostic["candidate_claims"] == diagnostic["semantic_rejected"] == 1
    assert diagnostic["citation_rejected"] == diagnostic["numeric_rejected"] == 0
    assert diagnostic["accepted_claims"] == 0
    assert diagnostic["reason"] == "semantic_validation_rejected"
    assert client.calls == ["cited_answer", "citation_review"]


def test_no_evidence_does_not_claim_generation_was_attempted():
    diagnostic = {}
    client = Responses([])
    generate_answer(client, "Факт?", [], diagnostics=diagnostic)
    assert client.calls == []
    assert diagnostic["candidate_claims"] is None
    assert diagnostic["reason"] == "no_evidence"


def test_review_exception_preserves_stage_without_exposing_error_text(monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    settings.cache_clear()
    diagnostic = {}
    client = Responses([claim()], review=LLMUnavailable("provider private details"))
    with pytest.raises(LLMUnavailable):
        generate_answer(client, "Сера?", [REPORT], diagnostics=diagnostic)
    assert diagnostic["candidate_claims"] == 1
    assert diagnostic["stage"] == "semantic_review"
    assert diagnostic["review_attempted"] is True
    assert diagnostic["review_executed"] is False
    assert diagnostic["accepted_claims"] == 0
    assert "provider private details" not in json.dumps(diagnostic)


def test_empty_candidate_diagnostics_reach_the_agent_response():
    class EmptyAnswer(Responses):
        def generate(self, name, payload, schema):
            if name == "decide":
                return schema.model_validate(action("finish"))
            return super().generate(name, payload, schema)

    client = EmptyAnswer([], complete=False)
    response = ask(
        "Что в отчёте сказано о качестве ESPO?",
        client=client,
        planner=lambda *_: plan(report_only=True, rag_query="ESPO quality"),
        execute=lambda *a, **kw: ([REPORT], None, event("report_search")),
    )
    assert response["status"] == "partial" and not response["claims"]
    checks = response["diagnostics"]["answer_checks"]
    assert len(checks) == 1 and checks[0]["source_ids"] == ["R1"]
    assert checks[0]["validation"]["candidate_claims"] == 0
    assert checks[0]["validation"]["reason"] == "no_candidate_claims"
    assert "do not repeat" in checks[0]["feedback"]
    assert client.calls == ["cited_answer"]
