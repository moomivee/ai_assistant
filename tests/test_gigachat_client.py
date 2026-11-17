import json
import threading
from concurrent.futures import ThreadPoolExecutor
from time import time
from urllib.parse import parse_qs
from uuid import UUID

import httpx
import pytest
from test_agent import plan

from src import budget
from src.agent import action_schema
from src.config import settings
from src.integrations import gigachat
from src.integrations.gigachat import GigaChatClient
from src.integrations.llm import LLMClient, LLMUnavailable
from src.models.schemas import CitedAnswer, JudgeResult, Plan, WebTimeFilterChoice

KEY = "test-only-authorization-key"
TOKEN = "test-only-access-token"
JUDGE = dict(groundedness=4, relevance=3, tone=4, completeness=2, explanation="ok")


def auth_reply(*, token=TOKEN, expires_at=None):
    return httpx.Response(
        200,
        json={
            "access_token": token,
            "expires_at": expires_at if expires_at is not None else int((time() + 1800) * 1000),
        },
    )


def completion(payload=None, *, finish="stop", usage=None):
    return httpx.Response(
        200,
        json={
            "model": "GigaChat-2:test-revision",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(JUDGE if payload is None else payload),
                    },
                }
            ],
            "usage": usage
            if usage is not None
            else {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        },
    )


def empty_completion():
    # Observed GigaChat-2:2.0.30.01 response: stop, no role, two output tokens.
    return httpx.Response(
        200,
        json={
            "model": "GigaChat-2:2.0.30.01",
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
            "usage": {"prompt_tokens": 3363, "completion_tokens": 2, "total_tokens": 3365},
        },
    )


def client(handler):
    native = GigaChatClient(KEY, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return LLMClient(native, provider="gigachat"), native


def test_native_schema_oauth_cache_and_safe_trace():
    requests = []

    def reply(req):
        requests.append(req)
        return auth_reply() if req.url.path.endswith("oauth") else completion()

    wrapper, native = client(reply)
    for _ in range(2):
        assert wrapper.generate("judge", {"question": "test"}, JudgeResult).groundedness == 4
    assert len(requests) == 3
    auth, chat, second = requests
    assert str(auth.url) == gigachat.AUTH_URL
    assert auth.headers["Authorization"] == "Basic " + KEY
    assert auth.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert UUID(auth.headers["RqUID"]).version == 4
    assert parse_qs(auth.content.decode()) == {"scope": ["GIGACHAT_API_PERS"]}
    assert str(chat.url) == "https://api.giga.chat/v1/chat/completions"
    assert chat.headers["Authorization"] == second.headers["Authorization"] == "Bearer " + TOKEN
    payload = json.loads(chat.content)
    assert payload["response_format"] == {
        "type": "json_schema",
        "schema": JudgeResult.model_json_schema(),
        "strict": True,
    }
    assert payload["stream"] is False
    events = settings().trace_file.read_text()
    assert KEY not in events and TOKEN not in events
    trace = json.loads(events.splitlines()[-1])
    assert trace["metadata"]["provider"] == "gigachat"
    assert trace["metadata"]["response_model"] == "GigaChat-2:test-revision"
    assert trace["metadata"]["usage"] == {"input": 10, "output": 20, "total": 30}
    native.close()


@pytest.mark.parametrize(
    "schema,payload",
    [
        (Plan, plan().model_dump()),
        (WebTimeFilterChoice, {"mode": "unbounded"}),
        (
            action_schema(("report_search", "finish")),
            {
                "purpose": "Нужны сведения",
                "action": "report_search",
                "query": "нефть",
                "report": "",
                "page": None,
            },
        ),
        (CitedAnswer, {"claims": [{"text": "Факт", "evidence_ids": ["R1:B1"]}], "complete": True}),
    ],
)
def test_agent_schemas_use_native_transport_and_local_validation(schema, payload):
    sent = []

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        sent.append(json.loads(req.content))
        return completion(payload)

    wrapper, native = client(reply)
    result = wrapper.generate("judge", {}, schema)
    assert result.model_dump() == schema.model_validate(payload).model_dump()
    assert sent[0]["response_format"]["schema"] == schema.model_json_schema()
    native.close()


@pytest.mark.parametrize(
    "finish,expected",
    [
        ("blacklist", "отказался"),
        ("length", "оборван"),
        ("function_call", "не завершил"),
    ],
)
def test_refusal_and_truncation_are_not_parsed_as_success(finish, expected):
    wrapper, native = client(
        lambda req: auth_reply() if req.url.path.endswith("oauth") else completion(finish=finish)
    )
    with pytest.raises(LLMUnavailable, match=expected):
        wrapper.generate("judge", {}, JudgeResult)
    assert json.loads(settings().trace_file.read_text().splitlines()[-1])["status"] == "error"
    native.close()


def test_invalid_structured_content_is_rejected_without_rule_fallback():
    calls = []

    def reply(req):
        calls.append(req)
        return auth_reply() if req.url.path.endswith("oauth") else completion({"wrong": "shape"})

    wrapper, native = client(reply)
    with pytest.raises(LLMUnavailable, match="схеме"):
        wrapper.generate("judge", {}, JudgeResult)
    assert len(calls) == 2
    native.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"claims": [], "complete": False},
        {"claims": [{"text": "Факт", "evidence_ids": ["R1:B1"]}], "complete": True},
    ],
)
def test_empty_native_answer_retries_once_with_same_schema_locally_validated(payload, monkeypatch):
    monkeypatch.setenv("TRACE_CONTENT", "false")
    settings.cache_clear()
    sent = []

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        sent.append(json.loads(req.content))
        return empty_completion() if len(sent) == 1 else completion(payload)

    wrapper, native = client(reply)
    answer = wrapper.generate("cited_answer", {"question": "PRIVATE-QUESTION"}, CitedAnswer)
    assert answer.model_dump() == payload
    assert len(sent) == 2
    assert sent[0]["response_format"]["type"] == "json_schema"
    assert sent[1]["response_format"] == {"type": "text"}
    assert sent[1]["model"] == sent[0]["model"]
    assert sent[1]["messages"][1:] == sent[0]["messages"][1:]
    assert "Предыдущая попытка" not in sent[0]["messages"][0]["content"]
    assert (
        json.dumps(CitedAnswer.model_json_schema(), ensure_ascii=False)
        in sent[1]["messages"][0]["content"]
    )
    events = settings().trace_file.read_text()
    assert all(secret not in events for secret in (KEY, TOKEN, "PRIVATE-QUESTION"))
    metadata = json.loads(events.splitlines()[-1])["metadata"]
    assert [a["mode"] for a in metadata["generation_attempts"]] == ["json_schema", "validated_json"]
    assert metadata["generation_attempts"][0]["content_length"] == 0
    assert metadata["usage"] == {"input": 3373, "output": 22, "total": 3395}
    native.close()


def test_two_empty_answers_stop_with_safe_diagnostics_and_no_third_attempt():
    calls = []

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        calls.append(req)
        return empty_completion()

    wrapper, native = client(reply)
    with pytest.raises(LLMUnavailable, match="после двух попыток"):
        wrapper.generate("cited_answer", {}, CitedAnswer)
    assert len(calls) == 2
    trace = json.loads(settings().trace_file.read_text().splitlines()[-1])
    assert trace["status"] == "error"
    assert trace["metadata"]["provider_error"] == "empty_response"
    assert len(trace["metadata"]["generation_attempts"]) == 2
    assert trace["metadata"]["usage"]["output"] == 4
    native.close()


def test_retry_cannot_bypass_pydantic_validation():
    calls = []

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        calls.append(req)
        return (
            empty_completion()
            if len(calls) == 1
            else completion({"claims": [{"text": "Факт", "evidence_ids": []}], "complete": True})
        )

    wrapper, native = client(reply)
    with pytest.raises(LLMUnavailable, match="схеме"):
        wrapper.generate("cited_answer", {}, CitedAnswer)
    assert len(calls) == 2
    trace = json.loads(settings().trace_file.read_text().splitlines()[-1])
    assert trace["status"] == "error"
    assert len(trace["metadata"]["generation_attempts"]) == 2
    native.close()


@pytest.mark.parametrize("elapsed,expected_calls", [(4, 2), (11, 1)])
def test_retry_shares_original_generation_timeout(monkeypatch, elapsed, expected_calls):
    clock, timeouts = [0.0], []
    monkeypatch.setattr(gigachat, "monotonic", lambda: clock[0])

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        timeouts.append(req.extensions["timeout"]["read"])
        clock[0] = elapsed
        return empty_completion() if len(timeouts) == 1 else completion()

    _, native = client(reply)
    kwargs = dict(model="GigaChat-2", messages=[], schema={}, max_tokens=128, timeout=10)
    if elapsed > 10:
        with pytest.raises(gigachat.GigaChatError, match="время ожидания"):
            native.generate(**kwargs)
    else:
        native.generate(**kwargs)
        assert timeouts == [10, 6]
    assert len(timeouts) == expected_calls
    native.close()


def test_retry_does_not_start_after_agent_budget_expires(monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr(gigachat, "monotonic", lambda: clock[0])
    monkeypatch.setattr(budget, "monotonic", lambda: clock[0])

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        calls.append(req)
        clock[0] = 6
        return empty_completion()

    wrapper, native = client(reply)
    with budget.request_budget(5), pytest.raises(budget.RequestBudgetExceeded):
        wrapper.generate("cited_answer", {}, CitedAnswer)
    assert len(calls) == 1
    native.close()


def test_recovered_empty_answer_allows_model_to_choose_web_search():
    from test_agent import REPORT, WEB

    from src.agent import ask

    replies = iter(
        [
            empty_completion(),
            completion({"claims": [], "complete": False}),
            completion(
                {
                    "action": "web_search",
                    "query": "latest oil news",
                    "purpose": "Fresh evidence",
                    "report": "",
                    "page": None,
                }
            ),
            completion(
                {
                    "claims": [{"text": WEB["text"], "evidence_ids": ["W1:B1"]}],
                    "complete": True,
                }
            ),
        ]
    )
    calls = []

    def reply(req):
        return auth_reply() if req.url.path.endswith("oauth") else next(replies)

    def execute(registry, name, **kwargs):
        calls.append(name)
        assert name in {"report_search", "web_search"}
        data = [REPORT] if name == "report_search" else {"items": [WEB], "notices": []}
        return data, None, {"tool": name, "engine": "fixture", "status": "ok", "ms": 1}

    wrapper, native = client(reply)
    result = ask(
        "Последние новости нефти?",
        client=wrapper,
        planner=lambda *_: plan(fresh=True),
        execute=execute,
    )
    assert calls == ["report_search", "web_search"]
    assert result["status"] == "ok"
    assert not result["diagnostics"]["failure"]
    assert result["claims"][0]["evidence_ids"] == ["W1:B1"]
    assert result["decisions"][1]["action"] == "web_search"
    assert next(replies, None) is None
    native.close()


@pytest.mark.parametrize("milliseconds", [True, False])
def test_expiration_supports_milliseconds_and_seconds(monkeypatch, milliseconds):
    clock, auth_count = [1000.0], []
    monkeypatch.setattr(gigachat, "time", lambda: clock[0])

    def reply(req):
        if req.url.path.endswith("oauth"):
            auth_count.append(req)
            return auth_reply(expires_at=(clock[0] + 100) * (1000 if milliseconds else 1))
        return completion()

    # Use actual-looking Unix epoch; small values are seconds by design.
    clock[0] = 1_800_000_000.0
    wrapper, native = client(reply)
    wrapper.generate("judge", {}, JudgeResult)
    clock[0] += 60
    wrapper.generate("judge", {}, JudgeResult)
    assert len(auth_count) == 1
    clock[0] += 15  # Within the refresh safety margin.
    wrapper.generate("judge", {}, JudgeResult)
    assert len(auth_count) == 2
    assert auth_count[0].headers["RqUID"] != auth_count[1].headers["RqUID"]
    native.close()


@pytest.mark.parametrize("always_fail", [False, True])
def test_401_forces_only_one_refresh(always_fail):
    oauth_calls, chat_calls = [], []

    def reply(req):
        if req.url.path.endswith("oauth"):
            oauth_calls.append(req)
            return auth_reply(token=TOKEN + str(len(oauth_calls)))
        chat_calls.append(req)
        if len(chat_calls) == 1 or always_fail:
            return httpx.Response(401, json={"secret": KEY})
        return completion()

    wrapper, native = client(reply)
    if always_fail:
        with pytest.raises(LLMUnavailable, match="ключ авторизации"):
            wrapper.generate("judge", {}, JudgeResult)
    else:
        wrapper.generate("judge", {}, JudgeResult)
    assert len(oauth_calls) == len(chat_calls) == 2
    assert chat_calls[0].headers["Authorization"] != chat_calls[1].headers["Authorization"]
    assert KEY not in settings().trace_file.read_text()
    native.close()


@pytest.mark.parametrize(
    "status,expected",
    [(403, "нет доступа"), (429, "лимит запросов"), (422, "JSON-схему"), (500, "HTTP 500")],
)
def test_http_errors_are_safe_and_not_retried(status, expected):
    seen = []

    def reply(req):
        seen.append(req)
        return (
            auth_reply()
            if req.url.path.endswith("oauth")
            else httpx.Response(status, json={"message": KEY + TOKEN})
        )

    wrapper, native = client(reply)
    with pytest.raises(LLMUnavailable, match=expected) as caught:
        wrapper.generate("judge", {}, JudgeResult)
    assert KEY not in str(caught.value) and TOKEN not in str(caught.value)
    assert len(seen) == 2
    assert KEY not in settings().trace_file.read_text()
    native.close()


def test_request_budget_includes_oauth(monkeypatch):
    clock, seen = [0.0], []
    monkeypatch.setattr(budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gigachat, "monotonic", lambda: clock[0])

    def reply(req):
        seen.append(req.extensions["timeout"]["read"])
        if req.url.path.endswith("oauth"):
            clock[0] = 3.0
            return auth_reply()
        return completion()

    wrapper, native = client(reply)
    with budget.request_budget(10):
        wrapper.generate("judge", {}, JudgeResult)
    assert seen == [10, 7]
    native.close()


def test_expired_budget_after_oauth_does_not_start_generation(monkeypatch):
    clock, seen = [0.0], []
    monkeypatch.setattr(budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gigachat, "monotonic", lambda: clock[0])

    def reply(req):
        seen.append(req)
        clock[0] = 11
        return auth_reply()

    wrapper, native = client(reply)
    with budget.request_budget(10), pytest.raises(budget.RequestBudgetExceeded):
        wrapper.generate("judge", {}, JudgeResult)
    assert len(seen) == 1
    native.close()


def test_budget_is_rechecked_after_waiting_for_shared_gate(monkeypatch):
    clock, released = [0.0], []
    monkeypatch.setattr(budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gigachat, "monotonic", lambda: clock[0])

    class DelayedGate:
        def acquire(self, timeout):
            assert timeout == 10
            clock[0] = 11
            return True

        def release(self):
            released.append(True)

    monkeypatch.setattr(gigachat, "_generation_lock", DelayedGate())
    wrapper, native = client(lambda req: pytest.fail("Budget expired before HTTP request"))
    with budget.request_budget(10), pytest.raises(budget.RequestBudgetExceeded):
        wrapper.generate("judge", {}, JudgeResult)
    assert released == [True]
    native.close()


def test_personal_generations_are_serialized_across_clients():
    first_started, second_started, unblock = threading.Event(), threading.Event(), threading.Event()
    active, maximum = [0], [0]
    guard = threading.Lock()

    def reply(req):
        if req.url.path.endswith("oauth"):
            return auth_reply()
        with guard:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        first_started.set()
        assert unblock.wait(2)
        with guard:
            active[0] -= 1
        return completion()

    a, native_a = client(reply)
    b, native_b = client(reply)

    def second():
        second_started.set()
        return b.generate("judge", {}, JudgeResult)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(a.generate, "judge", {}, JudgeResult)
        assert first_started.wait(1)
        two = pool.submit(second)
        assert second_started.wait(1)
        unblock.set()
        assert one.result().groundedness == two.result().groundedness == 4
    assert maximum[0] == 1
    native_a.close()
    native_b.close()


def test_models_check_uses_oauth_without_generation():
    seen = []

    def reply(req):
        seen.append(req)
        if req.url.path.endswith("oauth"):
            return auth_reply()
        assert req.method == "GET" and req.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "GigaChat-2"}]})

    wrapper, native = client(reply)
    assert wrapper.list_models() == ["GigaChat-2"]
    wrapper.check_ready()
    wrapper.model = "unavailable-model"
    with pytest.raises(LLMUnavailable, match="отсутствует"):
        wrapper.check_ready()
    assert len(seen) == 4
    native.close()


def test_missing_key_is_an_actionable_configuration_error(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "")
    settings.cache_clear()
    with pytest.raises(LLMUnavailable, match="GIGACHAT_CREDENTIALS"):
        LLMClient(provider="gigachat").generate("judge", {}, JudgeResult)


def test_missing_ca_bundle_is_reported_without_traceback_details(monkeypatch, tmp_path):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", KEY)
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", str(tmp_path / "missing.pem"))
    settings.cache_clear()
    with pytest.raises(LLMUnavailable, match="GIGACHAT_CA_BUNDLE") as caught:
        LLMClient(provider="gigachat")
    assert KEY not in str(caught.value)


def test_malformed_url_is_an_actionable_configuration_error(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", KEY)
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://api.giga.chat:invalid/v1")
    settings.cache_clear()
    with pytest.raises(LLMUnavailable, match="GIGACHAT_BASE_URL"):
        LLMClient(provider="gigachat")


@pytest.mark.parametrize("url", ["http://api.giga.chat/v1", "https://user:pass@api.giga.chat/v1"])
def test_insecure_base_url_is_rejected(url):
    with pytest.raises(ValueError, match="HTTPS"):
        GigaChatClient(KEY, base_url=url)


def test_tls_ca_adds_trust_without_disabling_public_roots(monkeypatch):
    loaded, clients = [], []

    class Context:
        def load_verify_locations(self, *, cafile):
            loaded.append(cafile)

    context = Context()
    monkeypatch.setattr(gigachat.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(gigachat.httpx, "Client", lambda **kw: clients.append(kw))
    GigaChatClient(KEY, ca_bundle="custom-root.pem")
    assert loaded == ["custom-root.pem"]
    assert clients == [{"verify": context, "follow_redirects": False}]


def test_network_exception_does_not_leak_header_or_body():
    def reply(req):
        raise httpx.ConnectError(KEY + TOKEN, request=req)

    wrapper, native = client(reply)
    with pytest.raises(LLMUnavailable, match="сертификаты TLS") as caught:
        wrapper.generate("judge", {}, JudgeResult)
    assert KEY not in str(caught.value) and TOKEN not in settings().trace_file.read_text()
    native.close()
