import importlib

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eval.run import make_judge
from src.api import app
from src.config import settings


def configure_gigachat(monkeypatch, credentials="test-credential"):
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", credentials)
    monkeypatch.setenv("GIGACHAT_MODEL", "available-personal-model")
    monkeypatch.setenv("LLM_API_KEY", "old-local-placeholder")
    settings.cache_clear()


def test_status_uses_selected_provider_without_exposing_credentials(monkeypatch):
    configure_gigachat(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/status")
    data = response.json()
    assert data["model"] == "available-personal-model"
    assert data["provider"] == "gigachat" and data["llm_configured"]
    assert "test-credential" not in response.text
    assert "old-local-placeholder" not in response.text


def test_old_ollama_key_does_not_configure_gigachat(monkeypatch):
    configure_gigachat(monkeypatch, credentials="")
    with TestClient(app) as client:
        data = client.get("/api/status").json()
    assert data["model"] is None and not data["llm_configured"]


def test_unknown_provider_is_not_silently_routed_to_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "giga-chat-typo")
    settings.cache_clear()
    with pytest.raises(ValidationError):
        settings()


def test_default_judge_uses_main_gigachat_provider(monkeypatch):
    configure_gigachat(monkeypatch)
    judge = make_judge()
    assert judge.provider == "gigachat"
    assert judge.model == "available-personal-model"
    judge.client.close()


def test_explicit_judge_does_not_inherit_gigachat_model_or_credentials(monkeypatch):
    configure_gigachat(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "judge-only")
    with pytest.raises(ValueError, match="JUDGE_BASE_URL"):
        make_judge()
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example.invalid/v1")
    monkeypatch.setenv("JUDGE_MODEL", "independent-model")
    judge = make_judge()
    assert judge.provider == "openai"
    assert judge.model == "independent-model"
    assert judge.client.api_key == "judge-only"
    judge.client.close()


def test_readiness_uses_provider_contract(monkeypatch):
    llm_module = importlib.import_module("src.integrations.llm")
    rag_module = importlib.import_module("src.rag")
    checks = []

    class Provider:
        def check_ready(self, *, timeout):
            checks.append(timeout)

    monkeypatch.setattr(llm_module, "llm", lambda: Provider())
    monkeypatch.setattr(rag_module, "index", lambda: object())
    with TestClient(app) as client:
        assert client.get("/ready").json() == {"status": "ready"}
    assert checks == [3]


def test_ask_exposes_safe_provider_initialization_error(monkeypatch):
    api_module = importlib.import_module("src.api")
    from src.integrations.llm import LLMUnavailable

    def unavailable(*args):
        raise LLMUnavailable("Проверьте GIGACHAT_CA_BUNDLE.")

    monkeypatch.setattr(api_module, "ask", unavailable)
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"question": "Что такое Brent?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Проверьте GIGACHAT_CA_BUNDLE."


@pytest.mark.parametrize("available", [True, False])
def test_check_llm_prints_ids_without_generating(monkeypatch, capsys, available):
    from src import cli

    llm_module = importlib.import_module("src.integrations.llm")

    class Provider:
        provider = "gigachat"
        model = "requested"

        def list_models(self, *, timeout):
            return ["requested"] if available else ["other"]

        def generate(self, *args, **kwargs):
            pytest.fail("check-llm must not generate an answer")

    monkeypatch.setattr(llm_module, "llm", lambda: Provider())
    monkeypatch.setattr("sys.argv", ["cli", "check-llm"])
    if available:
        cli.main()
    else:
        with pytest.raises(SystemExit) as result:
            cli.main()
        assert result.value.code == 1
    captured = capsys.readouterr()
    assert "available_models" in captured.out
    assert ("requested" if available else "other") in captured.out
