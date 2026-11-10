"""Regressions found in the real Qwen run, including percentage ranges."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.answering import review_schema, verified_answer
from src.models.schemas import AnswerReview, EvidenceAnswer
from src.validation import numbers, validate_claims

FIXTURE = json.loads((Path(__file__).parent / "fixtures/qwen-review-failure.json").read_text())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30-32; 1.3%-1.5%", ["30", "32", "1.3", "1.5"]),
        ("30 – 32; 0,4% – 0,6%", ["30", "32", "0.4", "0.6"]),
        ("1,3 % - 1,5 %", ["1.3", "1.5"]),
        ("изменение -1.5%; затем −0.6", ["-1.5", "-0.6"]),
        ("−3%–−1%", ["-3", "-1"]),
        ("−3%–1%", ["-3", "1"]),
    ],
)
def test_ranges_do_not_become_negative_numbers(text, expected):
    assert numbers(text) == {Decimal(value) for value in expected}


def test_real_qwen_percentage_ranges_survive_quote_validation():
    draft = EvidenceAnswer.model_validate(FIXTURE["draft"])
    checked = validate_claims(draft, FIXTURE["sources"])
    assert checked.claims == draft.claims
    changed = draft.model_copy(deep=True)
    changed.claims[0].text = "Содержание серы изменилось на -1.5%."
    assert validate_claims(changed, FIXTURE["sources"]).claims == []


@pytest.mark.parametrize("count", [0, 1, 2, 10])
def test_guided_review_schema_has_exact_count_and_valid_indexes(count):
    schema = review_schema(count)
    properties = schema.model_json_schema()["properties"]["checks"]
    assert properties["minItems"] == properties["maxItems"] == count
    good = {
        "checks": [{"index": i, "verdict": "supported"} for i in range(count)],
        "complete": bool(count),
        "repair_evidence": [],
    }
    schema.model_validate(good)
    extra = {**good, "checks": good["checks"] + [{"index": 0, "verdict": "supported"}]}
    with pytest.raises(ValidationError):
        schema.model_validate(extra)
    if count:
        bad_index = {
            **good,
            "checks": [
                *good["checks"][:-1],
                {"index": count, "verdict": "supported"},
            ],
        }
        with pytest.raises(ValidationError):
            schema.model_validate(bad_index)


def test_real_draft_reaches_review_without_unrequested_limitations():
    class Client:
        def generate(self, name, payload, schema):
            if name == "answer":
                return EvidenceAnswer.model_validate(FIXTURE["draft"])
            assert name == "answer_review"
            assert payload["claim_indexes"] == [0]
            assert set(payload["draft"]) == {"claims"}
            assert payload["draft"]["claims"][0]["index"] == 0
            assert schema.model_json_schema()["properties"]["checks"]["maxItems"] == 1
            return AnswerReview(
                checks=[{"index": 0, "verdict": "supported"}],
                complete=True,
                repair_evidence=[],
            )

    answer, repaired = verified_answer(
        Client(), {"question": FIXTURE["question"], "sources": FIXTURE["sources"]}
    )
    assert answer.complete and len(answer.claims) == 1
    assert not repaired and answer.limitations == []
