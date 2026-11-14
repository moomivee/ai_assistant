"""Provider-neutral structured generation; no JSON/regex fallback."""

import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from ..budget import bounded_timeout, check_budget
from ..config import settings
from ..logging import log
from .gigachat import GigaChatClient, GigaChatError
from .prompts import get_prompt
from .telemetry import span


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, client=None, *, model=None, provider=None):
        cfg = settings()
        self.provider = provider or cfg.llm_provider
        self.model = model or (cfg.gigachat_model if self.provider == "gigachat" else cfg.llm_model)
        if self.provider == "gigachat":
            try:
                self.client = (
                    client
                    if client is not None
                    else (
                        GigaChatClient(
                            cfg.gigachat_credentials,
                            scope=cfg.gigachat_scope,
                            base_url=cfg.gigachat_base_url,
                            ca_bundle=cfg.gigachat_ca_bundle,
                        )
                        if cfg.gigachat_credentials
                        else None
                    )
                )
            except (OSError, ValueError):
                raise LLMUnavailable(
                    "Не удалось настроить GigaChat: проверьте GIGACHAT_BASE_URL и GIGACHAT_CA_BUNDLE."
                ) from None
        elif self.provider == "openai":
            self.client = (
                client
                if client is not None
                else (
                    OpenAI(
                        api_key=cfg.llm_key,
                        base_url=cfg.llm_base,
                        timeout=cfg.llm_timeout,
                        max_retries=0,
                    )
                    if cfg.llm_key
                    else None
                )
            )
        else:
            raise ValueError("Неизвестный LLM_PROVIDER; доступны openai и gigachat.")

    def _check_configured(self):
        if self.client is None:
            if self.provider == "gigachat":
                raise LLMUnavailable("GigaChat не настроен: задайте GIGACHAT_CREDENTIALS.")
            raise LLMUnavailable("LLM не настроена: задайте LLM_API_KEY и LLM_BASE_URL.")

    def list_models(self, timeout=3):
        self._check_configured()
        try:
            if self.provider == "gigachat":
                return self.client.list_models(timeout=bounded_timeout(timeout))
            return [
                item.id
                for item in self.client.with_options(
                    timeout=bounded_timeout(timeout), max_retries=0
                ).models.list()
            ]
        except GigaChatError as exc:
            check_budget()
            raise LLMUnavailable(str(exc)) from None
        except OpenAIError:
            check_budget()
            raise LLMUnavailable("Не удалось получить список моделей.") from None

    def check_ready(self, timeout=3):
        models = self.list_models(timeout=timeout)
        if self.model not in models:
            raise LLMUnavailable("Выбранная модель отсутствует в списке доступных моделей.")

    def generate(self, prompt_name: str, payload: dict, schema: type[BaseModel]):
        self._check_configured()
        prompt = get_prompt(prompt_name)
        with span(
            "llm." + prompt_name,
            "generation",
            input=payload,
            model=self.model,
            prompt=prompt.remote,
            metadata={
                "prompt_version": prompt.version,
                "schema": schema.__name__,
                "provider": self.provider,
                "model": self.model,
            },
        ) as obs:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": prompt.text
                        + "\nOutput JSON Schema:\n"
                        + json.dumps(schema.model_json_schema(), ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
                if self.provider == "gigachat":
                    response = self.client.generate(
                        model=self.model,
                        messages=messages,
                        schema=schema.model_json_schema(),
                        max_tokens=settings().llm_max_tokens,
                        timeout=bounded_timeout(settings().llm_timeout),
                    )
                    check_budget()
                    result = schema.model_validate_json(response.content)
                    obs.update(
                        output=result.model_dump(),
                        metadata={"response_model": response.model, "usage": response.usage},
                        usage_details=response.usage,
                    )
                    return result
                response = self.client.with_options(
                    timeout=bounded_timeout(settings().llm_timeout), max_retries=0
                ).chat.completions.parse(
                    model=self.model,
                    temperature=0,
                    max_tokens=settings().llm_max_tokens,
                    messages=messages,
                    response_format=schema,
                )
                message = response.choices[0].message
                if message.refusal or message.parsed is None:
                    raise ValueError("Missing structured response")
                result = schema.model_validate(message.parsed)
                usage = response.usage
                obs.update(
                    output=result.model_dump(),
                    usage_details=(
                        {
                            "input": usage.prompt_tokens,
                            "output": usage.completion_tokens,
                            "total": usage.total_tokens,
                        }
                        if usage
                        else None
                    ),
                )
                return result
            except GigaChatError as exc:
                check_budget()
                obs.update(status="error", metadata={"provider_error": exc.code})
                log(
                    "llm.failed",
                    logging.ERROR,
                    stage=prompt_name,
                    provider=self.provider,
                    error_type=type(exc).__name__,
                    error_code=exc.code,
                )
                raise LLMUnavailable(str(exc)) from None
            except (OpenAIError, ValidationError, ValueError, IndexError) as exc:
                check_budget()
                log(
                    "llm.failed",
                    logging.ERROR,
                    stage=prompt_name,
                    provider=self.provider,
                    error_type=type(exc).__name__,
                )
                raise LLMUnavailable(
                    "Модель недоступна или не вернула ответ по заданной схеме."
                ) from exc


@lru_cache
def llm():
    return LLMClient()
