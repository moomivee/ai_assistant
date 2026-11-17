"""Wire contracts and local recovery for citations in a request-specific catalog."""

import json

import httpx
import pytest
from openai import OpenAI
from test_agent import WEB
from test_gigachat_client import auth_reply, completion, empty_completion
from test_gigachat_client import client as gigachat_client

from src.budget import RequestBudgetExceeded
from src.evidence import generate_answer
from src.integrations.llm import LLMClient, LLMUnavailable

FACT = "Порт отгружает сорт Orion."
SOURCE = {**WEB, "id": "W5", "text": "abc " * 400 + FACT}
QUESTION = "Какие сорта отгружает порт?"


def answer(*ids, text=FACT):
    return {"claims": [{"text": text, "evidence_ids": list(ids)}], "complete": True}


def block_ids(schema):
    return schema["$defs"]["CitedClaim"]["properties"]["evidence_ids"]["items"]["enum"]


@pytest.mark.parametrize("fallback", [False, True])
def test_gigachat_repairs_on_original_catalog_and_reviews_exact_block(fallback, monkeypatch):
    monkeypatch.setenv("ANSWER_SELF_REVIEW", "true")
    requests = []
    replies = iter(
        ([empty_completion()] if fallback else [])
        + [
            completion(answer("W5")),
            completion(answer("W5:B2")),
            completion(
                {"feedback": "", "supported": [0], "coverage": "sufficient", "complete": True}
            ),
        ]
    )

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        requests.append(json.loads(req.content))
        return next(replies)

    client, native = gigachat_client(reply)
    checks = {}
    try:
        draft, evidence, _ = generate_answer(client, QUESTION, [SOURCE], diagnostics=checks)
    finally:
        native.close()
    assert draft.complete and draft.claims[0].evidence_ids == ["W5:B2"]
    assert list(evidence) == ["W5:B1", "W5:B2"]
    if fallback:
        assert requests[1]["response_format"]["type"] == "text"
        assert all(k in requests[1]["messages"][0]["content"] for k in evidence)
    for req in (requests[0], requests[-2]):
        schema = req["response_format"]["schema"]
        assert schema["title"] == "CitedAnswer"
        assert block_ids(schema) == list(evidence)
    first, repaired, reviewed = [
        json.loads(r["messages"][-1]["content"]) for r in (requests[0], requests[-2], requests[-1])
    ]
    assert repaired.pop("citation_repair") and repaired == first
    assert reviewed["claims"][0]["evidence"][0]["text"] == FACT
    assert checks["initial_citation_rejected"] == 1
    assert checks["citation_repair_outcome"] == "ids_valid" and checks["review_executed"]


def test_openai_parse_receives_only_this_requests_ids():
    sent_ids = []

    def reply(req):
        data = json.loads(req.content)
        fmt = data["response_format"]["json_schema"]
        ids = block_ids(fmt["schema"])
        sent_ids.append(ids)
        assert fmt["name"] == "CitedAnswer" and fmt["strict"]
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(answer(ids[-1]))},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="fixture",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as sdk:
        for source_id in ["W5", "R1"]:
            draft, _, _ = generate_answer(LLMClient(sdk), QUESTION, [{**SOURCE, "id": source_id}])
            assert draft.claims[0].evidence_ids == [f"{source_id}:B2"]
    assert sent_ids == [["W5:B1", "W5:B2"], ["R1:B1", "R1:B2"]]


class Scripted:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, name, payload, schema):
        self.calls.append(name)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)


@pytest.mark.parametrize(
    "second,outcome",
    [
        (answer("W5"), "invalid_ids"),
        (answer("W5:B1", "W5:B9"), "invalid_ids"),
        ({"claims": [], "complete": False}, "empty"),
        (answer("W5:B2", text="Отгружено 970 тонн."), "ids_valid"),
    ],
)
def test_repair_fails_closed_without_guessing_or_third_generation(second, outcome):
    checks = {}
    client = Scripted(answer("W5"), second)
    draft, _, _ = generate_answer(client, QUESTION, [SOURCE], diagnostics=checks)
    assert not draft.claims and not draft.complete
    assert checks["citation_repair_attempted"] and checks["citation_repair_outcome"] == outcome
    assert checks["initial_citation_rejected"] == 1 and not checks["review_attempted"]
    assert client.calls == ["cited_answer", "cited_answer"]


@pytest.mark.parametrize("error", [LLMUnavailable("schema"), RequestBudgetExceeded()])
def test_technical_or_deadline_error_during_repair_is_not_hidden(error):
    checks = {}
    client = Scripted(answer("bad reference"), error)
    with pytest.raises(type(error)):
        generate_answer(client, QUESTION, [SOURCE], diagnostics=checks)
    assert checks["stage"] == "citation_repair" and checks["citation_repair_attempted"]
    assert "bad reference" not in json.dumps(checks)


def test_original_valid_claim_is_retained_when_other_ids_are_invalid():
    mixed = answer("W5:B2")
    mixed["claims"].extend(answer("W5")["claims"])
    checks = {}
    client = Scripted(mixed)
    draft, _, _ = generate_answer(client, QUESTION, [SOURCE], diagnostics=checks)
    assert len(draft.claims) == 1 and draft.claims[0].evidence_ids == ["W5:B2"]
    assert not draft.complete and not checks["citation_repair_attempted"]
    assert client.calls == ["cited_answer"]
