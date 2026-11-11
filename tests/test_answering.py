"""Replay the observed failure with explicitly scripted review/repair responses."""

import json
from pathlib import Path

import pytest
from test_agent import FakeLLM, executor

from src.answering import checked_review, verified_answer
from src.integrations.llm import LLMUnavailable
from src.models.schemas import AnswerReview, EvidenceAnswer, Plan
from src.validation import validate_claims
from src.workflow import ask

FIXTURE = json.loads((Path(__file__).parent / "fixtures/urals_espo_trace.json").read_text())
URALS = "Urals 30 to 32 1.3% to 1.5% Western Siberia and Volga Urals"
ESPO = "ESPO 34 to 37 0.4% to 0.6% Eastern Siberia"


def good_answer():
    return EvidenceAnswer(
        complete=True,
        claims=[
            {"source": "R1", "quote": URALS, "text": "Urals: 30–32 °API; сера 1,3–1,5%."},
            {"source": "R1", "quote": ESPO, "text": "ESPO: 34–37 °API; сера 0,4–0,6%."},
        ],
        limitations=[],
    )


def verdict(draft, *, complete=True, label="supported", evidence=()):
    return AnswerReview(
        checks=[{"index": i, "verdict": label} for i, _ in enumerate(draft["claims"])],
        complete=complete,
        repair_evidence=list(evidence),
    )


EVIDENCE = [{"source": "R1", "quote": q} for q in (URALS, ESPO)]


class ReplayClient:
    def __init__(self, *, repair=None, review_failure=False):
        self.calls = []
        self.repair = repair
        self.review_failure = review_failure

    def generate(self, name, payload, schema):
        self.calls.append((name, payload))
        if name == "plan":
            return Plan.model_validate(FIXTURE["plan"])
        if name == "answer":
            if payload["phase"] == "draft":
                return EvidenceAnswer.model_validate(FIXTURE["draft"])
            assert [s["id"] for s in payload["sources"]] == ["R1"]
            assert payload["sources"][0]["text"] == URALS + "\n\n" + ESPO
            return self.repair or good_answer()
        assert name == "answer_review"
        if self.review_failure:
            raise LLMUnavailable("review offline")
        if payload["phase"] == "draft":
            assert len(payload["draft"]["claims"]) == 1  # Wrong source ID already rejected.
            return verdict(payload["draft"], complete=False, label="irrelevant", evidence=EVIDENCE)
        return verdict(payload["draft"])


def replay_executor(registry, name, **kwargs):
    assert name == "report_search", "The report already contains both requested rows"
    return FIXTURE["sources"], None, {"tool": name, "engine": "fixture", "status": "ok", "ms": 0}


def test_observed_failure_is_repaired_from_relevant_evidence_before_web():
    client = ReplayClient()
    result = ask(FIXTURE["question"], client=client, execute=replay_executor)
    assert result["status"] == "ok"
    assert [t["tool"] for t in result["trace"]] == ["report_search"]
    assert {c["source"] for c in result["claims"]} == {"R1"}
    assert "30–32" in result["answer"] and "34–37" in result["answer"]
    assert "1,3–1,5%" in result["answer"] and "0,4–0,6%" in result["answer"]
    assert "8.978" not in result["answer"] and "Отсутствуют данные" not in result["answer"]
    assert [name for name, _ in client.calls] == [
        "plan",
        "answer",
        "answer_review",
        "answer",
        "answer_review",
    ]
    for name, payload in client.calls:
        if name.startswith("answer"):
            assert all(
                "local_url" not in s and "score" not in s and "url" not in s
                for s in payload["sources"]
            )


def test_review_failure_never_publishes_unchecked_claims():
    client = ReplayClient(review_failure=True)

    def execute(registry, name, **kwargs):
        if name == "report_search":
            return replay_executor(registry, name, **kwargs)
        return (
            {"items": [], "notices": []},
            None,
            {"tool": name, "engine": "fixture", "status": "ok", "ms": 0},
        )

    result = ask(FIXTURE["question"], client=client, execute=execute)
    assert result["status"] == "partial" and result["claims"] == []
    assert "8.978" not in result["answer"]
    assert [name for name, _ in client.calls].count("answer") == 1


def test_empty_web_skips_second_generation_even_for_incomplete_answer():
    calls = []

    class Client(FakeLLM):
        def generate(self, name, payload, schema):
            calls.append(name)
            return super().generate(name, payload, schema)

    def empty_web(registry, name, **kwargs):
        if name == "report_search":
            return executor(registry, name, **kwargs)
        return (
            {"items": [], "notices": ["Tavily не настроен."]},
            None,
            {"tool": name, "engine": "fixture", "status": "ok", "ms": 0},
        )

    result = ask("Какова сернистость Sokol?", client=Client(complete=False), execute=empty_web)
    assert result["status"] == "partial" and "Tavily" in result["answer"]
    assert calls == ["plan", "answer", "answer_review"]


def test_duplicate_web_content_also_skips_generation():
    calls = []

    class Client(FakeLLM):
        def generate(self, name, payload, schema):
            calls.append(name)
            return super().generate(name, payload, schema)

    def duplicate(registry, name, **kwargs):
        reports, _, _ = executor(registry, "report_search")
        data = (
            reports
            if name == "report_search"
            else {"items": [{**reports[0], "id": "W1", "kind": "web"}], "notices": []}
        )
        return data, None, {"tool": name, "engine": "fixture", "status": "ok", "ms": 0}

    ask("Sokol", client=Client(complete=False), execute=duplicate)
    assert calls == ["plan", "answer", "answer_review"]


@pytest.mark.parametrize(
    "checks",
    [[], [{"index": 1, "verdict": "supported"}], [{"index": 0, "verdict": "supported"}] * 2],
)
def test_invalid_review_indices_fail_closed(checks):
    draft = EvidenceAnswer(complete=True, claims=good_answer().claims[:1], limitations=[])
    review = AnswerReview(checks=checks, complete=True, repair_evidence=[])
    checked, _ = checked_review(draft, review, FIXTURE["sources"])
    assert not checked.complete and not checked.claims


def test_invented_repair_evidence_is_rejected():
    raw = validate_claims(EvidenceAnswer.model_validate(FIXTURE["draft"]), FIXTURE["sources"])
    review = verdict(
        raw.model_dump(),
        complete=False,
        label="irrelevant",
        evidence=[
            {"source": "R1", "quote": "Urals 99 to 100"},
            {"source": "R99", "quote": URALS},
            {"source": "R1", "quote": "      "},
        ],
    )
    checked, evidence = checked_review(raw, review, FIXTURE["sources"])
    assert not checked.claims and evidence == []


def test_semantic_review_rejects_to_vs_by_even_with_matching_numbers():
    draft = validate_claims(EvidenceAnswer.model_validate(FIXTURE["draft"]), FIXTURE["sources"])
    assert len(draft.claims) == 1  # Quote/numeric checks alone cannot detect this.
    checked, _ = checked_review(
        draft, verdict(draft.model_dump(), complete=False, label="contradicted"), FIXTURE["sources"]
    )
    assert not checked.claims and not checked.complete


def test_repair_cannot_join_noncontiguous_quotes_into_a_citation():
    joined = EvidenceAnswer(
        complete=True,
        claims=[
            {
                "source": "R1",
                "quote": URALS + "\n\n" + ESPO,
                "text": "Urals и ESPO имеют разные свойства.",
            }
        ],
        limitations=[],
    )
    # Insert a real gap so joining the excerpts is no longer an original quote.
    sources = [
        {**FIXTURE["sources"][0], "text": URALS + " omitted content " + ESPO},
        FIXTURE["sources"][1],
    ]
    client = ReplayClient(repair=joined)
    answer, repaired = verified_answer(
        client, {"question": FIXTURE["question"], "sources": sources}
    )
    assert repaired and not answer.claims


def test_pdf_chunk_preserves_table_rows_and_stable_word_ids():
    from src.config import manifest, settings
    from src.rag.index import pdf_chunks
    from src.validation import normalized

    report = manifest()[0]
    row = next(
        c
        for c in pdf_chunks(report, settings().data / "reports" / report["file"])
        if c["chunk_id"] == "eia-russia-2025:p6:w0"
    )
    assert "\n" in row["text"]
    assert normalized(URALS) in normalized(row["text"]) and normalized(ESPO) in normalized(
        row["text"]
    )


def test_failed_repair_does_not_create_an_unbounded_loop():
    client = ReplayClient(repair=EvidenceAnswer(complete=False, claims=[], limitations=[]))
    answer, repaired = verified_answer(
        client, {"question": FIXTURE["question"], "sources": FIXTURE["sources"]}
    )
    assert repaired and not answer.complete and not answer.claims
    assert [name for name, _ in client.calls] == ["answer", "answer_review", "answer"]


def test_live_smoke_checks_answer_facts_not_only_success_status():
    from eval.smoke import check_result

    cases = [
        {
            "id": "comparison",
            "expected": {
                "answer_facts": [
                    {"entity": "Urals", "values": [30, 32, 1.3, 1.5]},
                    {"entity": "ESPO", "values": [34, 37, 0.4, 0.6]},
                ]
            },
        }
    ]
    metrics = {
        key: 1.0
        for key in (
            "tool_sequence_exact",
            "intent_correct",
            "status_correct",
            "plan_fields_correct",
        )
    }
    metrics["forbidden_tool_violation"] = 0.0
    row = {
        "case_id": "comparison",
        "metrics": metrics,
        "response": {"claims": good_answer().model_dump()["claims"]},
    }
    assert check_result({"rows": [row]}, cases)[0]["passed"]
    row["response"]["claims"] = FIXTURE["draft"]["claims"]
    assert not check_result({"rows": [row]}, cases)[0]["passed"]


def test_missing_data_smoke_rejects_invented_claims_even_with_partial_status():
    from eval.smoke import check_result

    case = {"id": "missing", "expected": {"no_claims": True}}
    row = {
        "case_id": "missing",
        "metrics": dict(
            tool_policy_pass=1,
            intent_correct=1,
            status_correct=1,
            plan_fields_correct=1,
            forbidden_tool_violation=0,
        ),
        "response": {"claims": [{"text": "Unseen grade has 42 API."}]},
    }
    assert not check_result({"rows": [row]}, [case])[0]["passed"]
    row["response"]["claims"] = []
    assert check_result({"rows": [row]}, [case])[0]["passed"]


def test_repair_budget_is_shared_with_web_assessment():
    class Client(ReplayClient):
        def generate(self, name, payload, schema):
            self.calls.append((name, payload))
            if name == "plan":
                return Plan.model_validate(FIXTURE["plan"])
            if name == "answer":
                if payload["phase"] == "repair":
                    return EvidenceAnswer(complete=False, claims=[], limitations=[])
                return EvidenceAnswer.model_validate(FIXTURE["draft"])
            return verdict(payload["draft"], complete=False, label="irrelevant", evidence=EVIDENCE)

    def execute(registry, name, **kwargs):
        if name == "report_search":
            return replay_executor(registry, name, **kwargs)
        return executor(registry, name, **kwargs)

    client = Client()
    result = ask(FIXTURE["question"], client=client, execute=execute)
    phases = [payload["phase"] for name, payload in client.calls if name == "answer"]
    assert phases == ["draft", "repair", "draft"]
    assert result["status"] == "partial" and result["claims"] == []
