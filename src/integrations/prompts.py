import hashlib
import logging
from dataclasses import dataclass

from ..config import ROOT, settings
from ..logging import log
from .telemetry import langfuse_client

PROMPT_NAMES = (
    "plan",
    "forecast",
    "forecast_scope",
    "answer",
    "answer_review",
    "judge",
    "decide",
    "cited_answer",
    "citation_review",
)


@dataclass
class Prompt:
    text: str
    version: str
    remote: object | None = None


def get_prompt(name):
    if name not in PROMPT_NAMES:
        raise ValueError("Unknown prompt")
    text = (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
    if settings().prompt_source == "langfuse":
        client = langfuse_client()
        if client:
            try:
                remote = client.get_prompt(
                    f"oil-gas/{name}",
                    label=settings().prompt_label,
                    type="text",
                    fetch_timeout_seconds=3,
                    max_retries=0,
                )
                return Prompt(remote.prompt, f"langfuse:{remote.version}", remote)
            except Exception as exc:
                log(
                    "prompt.local_fallback",
                    logging.WARNING,
                    prompt=name,
                    error_type=type(exc).__name__,
                )
    return Prompt(text, "sha256:" + hashlib.sha256(text.encode()).hexdigest())


def publish_prompts(label="staging"):
    client = langfuse_client()
    if client is None:
        raise ValueError("Включите LANGFUSE_ENABLED и задайте ключи Langfuse.")
    for name in PROMPT_NAMES:
        text = (ROOT / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
        prompt = client.create_prompt(
            name=f"oil-gas/{name}",
            type="text",
            prompt=text,
            labels=[label],
            commit_message="Publish repository prompt",
        )
        log("prompt.published", prompt=name, version=prompt.version, label=label)
