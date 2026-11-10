import json
import subprocess
import sys

import httpx
import pytest
from openai import OpenAI
from test_agent import REPORT, plan

from src.integrations.llm import LLMClient, LLMUnavailable
from src.models.schemas import EvidenceAnswer, JudgeResult
from src.validation import validate_claims, validate_plan


@pytest.mark.parametrize("valid", [True, False])
def test_openai_real_sdk_sends_schema(valid):
    seen = []

    def reply(req):
        seen.append(json.loads(req.content))
        payload = (
            dict(groundedness=4, relevance=3, tone=4, completeness=2, explanation="ok")
            if valid
            else {"wrong": "shape"}
        )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    client = LLMClient(
        OpenAI(
            api_key="local",
            base_url="http://test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(reply)),
        )
    )
    if valid:
        assert client.generate("judge", {}, JudgeResult).groundedness == 4
    else:
        with pytest.raises(LLMUnavailable):
            client.generate("judge", {}, JudgeResult)
    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_claim_validation():
    good = {"source": "R1", "quote": REPORT["text"], "text": "Сера 0.2%–0.3%."}
    draft = EvidenceAnswer(
        complete=True,
        claims=[good, {**good, "source": "W999"}, {**good, "text": "Сера 99%."}],
        limitations=[],
    )
    result = validate_claims(draft, [REPORT])
    assert len(result.claims) == 1 and not result.complete


def test_scenario_evidence_and_percent_discount():
    p = plan(
        intent="forecast",
        benchmark="Urals",
        months=3,
        discount=15,
        parameters=[{"field": "discount", "quote": "дисконт 15 USD"}],
    )
    result = validate_plan(p, "Прогноз Urals на 3 месяца")
    assert result.discount is None and result.clarify
    p = plan(
        intent="forecast",
        benchmark="Urals",
        months=3,
        discount=5,
        parameters=[{"field": "discount", "quote": "дисконт 5%"}],
    )
    result = validate_plan(p, "Urals на 3 месяца дисконт 5%")
    assert result.discount is None


def test_ort_optout_before_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import os,src; assert os.environ["ORT_DISABLE_TELEMETRY"]=="1"',
        ],
        capture_output=True,
    )
    assert result.returncode == 0


def test_slow_llm_timeout_does_not_increase_web_timeout(monkeypatch):
    from src.config import settings

    monkeypatch.setenv("HTTP_TIMEOUT", "25")
    monkeypatch.setenv("LLM_HTTP_TIMEOUT", "600")
    monkeypatch.setenv("LLM_API_KEY", "local")
    settings.cache_clear()
    client = LLMClient()
    assert settings().timeout == 25
    assert httpx.Timeout(client.client.timeout).read == 600
    client.client.close()
