"""Screen web publications for the organization and event before extracting dates."""

import json
from hashlib import sha256

from pydantic import StrictInt

from .budget import check_budget
from .integrations.llm import InvalidModelOutput
from .models.schemas import StrictModel


class EventRelevance(StrictModel):
    off_topic: list[StrictInt]


_BATCH_SIZE = 8


def _excerpt(text):
    body = " ".join(text.split())
    return body if len(body) <= 600 else body[:400] + " … " + body[-200:]


def select_event_sources(client, question, sources, *, previous=None, cache=None):
    """Exclude only explicit off-topic judgments. A malformed judgment retains the batch."""
    cache = {} if cache is None else cache
    publications = [
        source for source in sources if source.get("kind") == "web" and "price" not in source
    ]
    excluded = set()
    pending = []
    for source in publications:
        key = sha256(
            json.dumps(
                [
                    question,
                    previous,
                    source["id"],
                    source.get("url"),
                    source.get("title"),
                    source["text"],
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if key in cache:
            if cache[key]:
                excluded.add(source["id"])
        else:
            pending.append((key, source))

    unverified = []
    for start in range(0, len(pending), _BATCH_SIZE):
        batch = pending[start : start + _BATCH_SIZE]
        payload = {
            "question": question,
            **({"previous": previous} if previous else {}),
            "publications": [
                {
                    "index": index,
                    "source_id": source["id"],
                    "title": (source.get("title") or "")[:300],
                    "url": (source.get("url") or "")[:350],
                    "excerpt": _excerpt(source["text"]),
                }
                for index, (_, source) in enumerate(batch, 1)
            ],
        }
        check_budget()
        try:
            indices = client.generate("event_relevance", payload, EventRelevance).off_topic
        except InvalidModelOutput:
            indices = None
        if (
            indices is None
            or len(indices) != len(set(indices))
            or any(index < 1 or index > len(batch) for index in indices)
        ):
            unverified.extend(source["id"] for _, source in batch)
            continue
        for index, (key, source) in enumerate(batch, 1):
            cache[key] = index in indices
            if cache[key]:
                excluded.add(source["id"])

    selected = [source for source in sources if source["id"] not in excluded]
    return selected, {
        "status": "unverified" if unverified else "ok" if publications else "skipped",
        "considered_source_ids": [source["id"] for source in publications],
        "checked_source_ids": [source["id"] for _, source in pending],
        "cache_hits": len(publications) - len(pending),
        "excluded_source_ids": [
            source["id"] for source in publications if source["id"] in excluded
        ],
        "kept_source_ids": [source["id"] for source in selected],
        "unverified_source_ids": unverified,
    }
