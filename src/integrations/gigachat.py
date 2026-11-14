"""Native GigaChat v1 transport with OAuth and bounded, serialized requests.

Only public provider contracts live here. The agent still validates every
structured response with its own Pydantic schema in ``llm.py``.
"""

import math
import ssl
import threading
from dataclasses import dataclass
from time import monotonic, time
from uuid import uuid4

import httpx

from ..budget import bounded_timeout, check_budget

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
# The personal API allows one stream. This gate covers this Python process;
# deployments with multiple workers need an external/shared limiter.
_generation_lock = threading.Lock()


class GigaChatError(RuntimeError):
    """A safe public message/code, never a provider response body or credential."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GigaChatResponse:
    content: str
    usage: dict | None
    model: str | None


def _timeout(deadline):
    check_budget()
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise GigaChatError("timeout", "Истекло время ожидания GigaChat.")
    return bounded_timeout(remaining)


def _check_status(response, *, auth=False):
    status = response.status_code
    if 200 <= status < 300:
        return
    if status == 401 or (auth and status in {400, 403}):
        code, message = "auth", "GigaChat: проверьте ключ авторизации и GIGACHAT_SCOPE."
    elif status == 402:
        code, message = "quota", "GigaChat: исчерпана доступная квота токенов."
    elif status == 403:
        code, message = "access", "GigaChat: нет доступа к выбранной модели или операции."
    elif status == 404:
        code, message = "model", "GigaChat: модель или адрес API не найдены."
    elif status == 429:
        code, message = "rate_limit", "GigaChat: превышен лимит запросов; повторите позднее."
    elif status in {400, 422}:
        code, message = "request", "GigaChat не принял параметры запроса или JSON-схему."
    else:
        code, message = "http", f"GigaChat временно недоступен (HTTP {status})."
    raise GigaChatError(code, message)


def _json_object(response):
    try:
        data = response.json()
    except ValueError:
        raise GigaChatError("response", "GigaChat вернул некорректный JSON-ответ.") from None
    if not isinstance(data, dict):
        raise GigaChatError("response", "GigaChat вернул ответ неожиданного формата.")
    return data


class GigaChatClient:
    def __init__(
        self,
        credentials,
        *,
        scope="GIGACHAT_API_PERS",
        base_url="https://api.giga.chat/v1",
        ca_bundle="",
        http_client=None,
    ):
        self._credentials = credentials
        self.scope = scope
        self.base_url = base_url.rstrip("/")
        try:
            parsed = httpx.URL(self.base_url)
        except httpx.InvalidURL:
            raise ValueError("GIGACHAT_BASE_URL содержит некорректный адрес.") from None
        if (
            parsed.scheme != "https"
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "GIGACHAT_BASE_URL должен быть HTTPS-адресом без учётных данных и параметров."
            )
        if http_client is None:
            context = ssl.create_default_context()
            if ca_bundle:
                context.load_verify_locations(cafile=ca_bundle)
            self.http = httpx.Client(verify=context, follow_redirects=False)
        else:
            self.http = http_client
        self._access_token = None
        self._expires_at = 0.0

    def close(self):
        self.http.close()

    def _token(self, deadline, *, refresh=False):
        if not refresh and self._access_token and time() < self._expires_at - 30:
            return self._access_token
        self._access_token, self._expires_at = None, 0.0
        response = self.http.post(
            AUTH_URL,
            headers={
                "Authorization": "Basic " + self._credentials,
                "RqUID": str(uuid4()),
                "Accept": "application/json",
            },
            data={"scope": self.scope},
            timeout=_timeout(deadline),
        )
        _timeout(deadline)
        _check_status(response, auth=True)
        data = _json_object(response)
        token, expiry = data.get("access_token"), data.get("expires_at")
        if (
            not isinstance(token, str)
            or not token.strip()
            or isinstance(expiry, bool)
            or not isinstance(expiry, (int, float))
            or not math.isfinite(expiry)
            or expiry <= 0
        ):
            raise GigaChatError("auth_response", "GigaChat не вернул действующий токен доступа.")
        # Milliseconds are canonical; accept seconds used by some API backends.
        expires_at = expiry if expiry < 10_000_000_000 else expiry / 1000
        if expires_at <= time():
            raise GigaChatError("auth_response", "GigaChat вернул просроченный токен доступа.")
        self._access_token, self._expires_at = token, expires_at
        return token

    def _request(self, method, path, timeout, *, payload=None):
        """Queueing, OAuth and one 401 recovery share one call deadline."""
        check_budget()
        deadline = monotonic() + timeout
        acquired = _generation_lock.acquire(timeout=_timeout(deadline))
        if not acquired:
            check_budget()
            raise GigaChatError("busy", "GigaChat занят другим запросом; повторите позднее.")
        try:
            _timeout(deadline)
            token = self._token(deadline)
            for attempt in range(2):
                response = self.http.request(
                    method,
                    self.base_url + path,
                    headers={"Authorization": "Bearer " + token},
                    **({"json": payload} if payload is not None else {}),
                    timeout=_timeout(deadline),
                )
                _timeout(deadline)
                if response.status_code == 401 and attempt == 0:
                    token = self._token(deadline, refresh=True)
                    continue
                _check_status(response)
                return _json_object(response)
        except httpx.TimeoutException:
            check_budget()
            raise GigaChatError("timeout", "Истекло время ожидания GigaChat.") from None
        except httpx.HTTPError:
            check_budget()
            raise GigaChatError(
                "connection",
                "Не удалось подключиться к GigaChat. Проверьте сеть и сертификаты TLS.",
            ) from None
        finally:
            _generation_lock.release()

    def list_models(self, timeout=3):
        data = self._request("GET", "/models", timeout)
        models = data.get("data")
        if not isinstance(models, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in models
        ):
            raise GigaChatError("response", "GigaChat вернул некорректный список моделей.")
        return [item["id"] for item in models]

    def generate(self, *, model, messages, schema, max_tokens, timeout):
        data = self._request(
            "POST",
            "/chat/completions",
            timeout,
            payload={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {"type": "json_schema", "schema": schema, "strict": True},
            },
        )
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            raise GigaChatError("response", "GigaChat не вернул вариант ответа.")
        finish = choice.get("finish_reason")
        if finish == "blacklist":
            raise GigaChatError("refusal", "GigaChat отказался обрабатывать этот запрос.")
        if finish == "length":
            raise GigaChatError("length", "Ответ GigaChat оборван по лимиту токенов.")
        message = choice.get("message")
        if finish != "stop" or not isinstance(message, dict):
            raise GigaChatError("response", "GigaChat не завершил структурированный ответ.")
        if message.get("refusal"):
            raise GigaChatError("refusal", "GigaChat отказался обрабатывать этот запрос.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GigaChatError("response", "GigaChat вернул пустой структурированный ответ.")
        usage = data.get("usage")
        usage = {
            dest: usage[key]
            for key, dest in (
                ("prompt_tokens", "input"),
                ("completion_tokens", "output"),
                ("total_tokens", "total"),
            )
            if isinstance(usage, dict)
            and isinstance(usage.get(key), int)
            and not isinstance(usage[key], bool)
            and usage[key] >= 0
        }
        actual_model = data.get("model")
        return GigaChatResponse(
            content, usage or None, actual_model if isinstance(actual_model, str) else None
        )
