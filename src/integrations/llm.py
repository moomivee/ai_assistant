"""The only application module calling an LLM provider; no JSON/regex fallback."""

import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..logging import log
from .prompts import get_prompt
from .telemetry import span


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, client=None, *, model=None):
        cfg = settings()
        self.model = model or cfg.llm_model
        self.client = client or (
            OpenAI(
                api_key=cfg.llm_key,
                base_url=cfg.llm_base,
                timeout=cfg.timeout,
                max_retries=0,
            )
            if cfg.llm_key
            else None
        )

    def generate(self, prompt_name: str, payload: dict, schema: type[BaseModel]):
        if self.client is None:
            raise LLMUnavailable("LLM не настроена: задайте LLM_API_KEY и LLM_BASE_URL.")
        prompt = get_prompt(prompt_name)
        with span(
            "llm." + prompt_name,
            "generation",
            input=payload,
            model=self.model,
            prompt=prompt.remote,
            metadata={"prompt_version": prompt.version, "schema": schema.__name__},
        ) as obs:
            try:
                response = self.client.chat.completions.parse(
                    model=self.model,
                    temperature=0,
                    max_tokens=settings().llm_max_tokens,
                    messages=[
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
                    ],
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
            except (OpenAIError, ValidationError, ValueError, IndexError) as exc:
                log(
                    "llm.failed",
                    logging.ERROR,
                    stage=prompt_name,
                    error_type=type(exc).__name__,
                )
                raise LLMUnavailable(
                    "Модель недоступна или не вернула ответ по заданной схеме."
                ) from exc


@lru_cache
def llm():
    return LLMClient()
