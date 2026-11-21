"""Provider-neutral schema-validated generation; no rule-based answer fallback."""

import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from ..budget import bounded_timeout, check_budget
from ..clock import clock_instruction, clock_payload
from ..config import settings
from ..logging import log
from .agent_skills import current_skill_context
from .gigachat import GigaChatClient, GigaChatError
from .prompts import get_prompt
from .structured_output import is_schema_echo, schema_contract, validation_issues
from .telemetry import span


class LLMUnavailable(RuntimeError):
    pass


class InvalidModelOutput(LLMUnavailable):
    """The provider completed generation without usable structured data."""


class StructuredOutputError(InvalidModelOutput):
    """A provider answered, but the data could not be parsed as the requested model."""

    def __init__(self, message, *, issues, response_content):
        super().__init__(message)
        self.issues = issues
        # Consumers can recover independently valid records. Never log or pass
        # this text to a repair prompt; it may contain source or provider data.
        self.response_content = response_content


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
        skill = current_skill_context()
        output_schema = schema.model_json_schema()
        clock = clock_payload()
        # Runtime values win over caller-supplied fields and are repeated in the
        # system instruction; retrieved text must never set the present date.
        payload = {**payload, **clock}
        raw_content = None
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
                "skill": skill["metadata"],
                "clock": clock,
            },
        ) as obs:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": prompt.text
                        + ("\n\n" + skill["text"] if skill["text"] else "")
                        + clock_instruction(clock)
                        + schema_contract(output_schema),
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
                        schema=output_schema,
                        max_tokens=settings().llm_max_tokens,
                        timeout=bounded_timeout(settings().llm_timeout),
                    )
                    check_budget()
                    obs.update(
                        metadata={
                            "response_model": response.model,
                            "usage": response.usage,
                            "generation_attempts": list(response.attempts),
                        },
                        usage_details=response.usage,
                    )
                    raw_content = response.content
                    result = schema.model_validate_json(response.content)
                    obs.update(output=result.model_dump())
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
                usage = {}
                for attempt in exc.attempts:
                    for key, value in attempt["usage"].items():
                        usage[key] = usage.get(key, 0) + value
                obs.update(
                    status="error",
                    metadata={
                        "provider_error": exc.code,
                        "generation_attempts": list(exc.attempts),
                        "usage": usage or None,
                    },
                    usage_details=usage or None,
                )
                log(
                    "llm.failed",
                    logging.ERROR,
                    stage=prompt_name,
                    provider=self.provider,
                    error_type=type(exc).__name__,
                    error_code=exc.code,
                )
                if exc.code in {"empty_response", "length"}:
                    raise InvalidModelOutput(str(exc)) from None
                raise LLMUnavailable(str(exc)) from None
            except ValidationError as exc:
                check_budget()
                issues = validation_issues(exc, output_schema)
                echo = is_schema_echo(raw_content, output_schema)
                error_code = "schema_echo" if echo else "schema_validation"
                obs.update(
                    status="error",
                    metadata={
                        "validation_errors": issues,
                        "error_code": error_code,
                    },
                )
                log(
                    "llm.failed",
                    logging.ERROR,
                    stage=prompt_name,
                    provider=self.provider,
                    error_type=type(exc).__name__,
                    error_code=error_code,
                    validation_errors=issues,
                )
                if echo:
                    expected = ", ".join(output_schema.get("required", []))
                    message = (
                        "Модель вернула описание JSON-схемы вместо данных ответа. "
                        f"Ожидается объект с заполненными полями: {expected}."
                    )
                else:
                    details = "; ".join(
                        f"{'.'.join(map(str, item['path'])) or '<root>'}: {item['type']}"
                        for item in issues[:5]
                    )
                    message = (
                        f"Ответ модели не прошёл проверку по схеме {schema.__name__}: {details}."
                    )
                raise StructuredOutputError(
                    message, issues=issues, response_content=raw_content
                ) from exc
            except (OpenAIError, ValueError, IndexError) as exc:
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
