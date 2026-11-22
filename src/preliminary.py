"""Show attributed event reports when the latest occurrence could not be established."""

import re

from .event_dates import mentioned_dates
from .event_search import requested_publication_window
from .event_subject import canonical_event_question
from .integrations.web import web_date

# Optional model-selected quotations work in any language. For older or incomplete
# model outputs, only use complete passages with explicit occurrence/announcement
# wording from a dated, opened article that passed the event relevance review.
_REPORTED = re.compile(
    r"\b(?:начал(?:а|и|ось)?\s+(?:встреч|заседан|совещан)|состоял(?:ся|ась|ись|ось)\b|"
    r"собрались|встретил[а-яё]*|convened\b|met\b|gathered\b|took place\b|was held\b)",
    re.I,
)
_ANNOUNCED = re.compile(
    r"\b(?:пройд[её]т|планируется|запланирован[а-яё]*|намечен[а-яё]*|"
    r"назначен[а-яё]*|состоится|scheduled|will meet|will convene|will take place)\b",
    re.I,
)
# Anchoring to a standalone editorial divider avoids quoting headlines from a
# related-stories sidebar as statements made in the article body.
_RELATED = re.compile(
    r"^\s*(?:новости по теме|читайте также|материалы по теме|related (?:articles|news|stories)|"
    r"read (?:also|more)|see also)\s*[:：]?\s*$",
    re.I | re.M,
)


def _topic_words(text):
    return {word[:5] for word in re.findall(r"[^\W\d_]{4,}", text.casefold())}


def _attributed_passage(text, start, end, block_start, block_end):
    """Keep the enclosing line or complete sentence, including its attribution."""
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    right = len(text) if right == -1 else right
    if right - left > 360:
        # Extractors may flatten several sentences into a long paragraph.
        # Find boundaries in the source itself before checking the visible block.
        sentence_start = left
        for boundary in re.finditer(r'(?<=[.!?])\s+(?=[А-ЯЁA-Z«“"])', text[left:right]):
            before, after = left + boundary.start(), left + boundary.end()
            if after <= start:
                sentence_start = after
            elif before >= end:
                right = before
                break
        left = sentence_start
    if left < block_start or right > block_end:
        return ""
    line = text[left:right].strip()
    return line if 0 < len(line) <= 360 else ""


def _archive_excerpts(passage, title, query_words, published):
    for line in passage.splitlines():
        line = line.strip()
        if not 20 <= len(line) <= 360 or len(query_words.intersection(_topic_words(line))) < 2:
            continue
        # A headline can be a line on its own. For body text require a whole
        # sentence so a cropped chunk does not change its meaning.
        if line != title and (not line[:1].isupper() or line[-1] not in '.!?»"'):
            continue
        if _ANNOUNCED.search(line) and mentioned_dates(line, default_year=int(published[:4])):
            # The year above only recognizes a day and month; never append or deduce a year.
            yield "schedule_report", line
        elif _REPORTED.search(line):
            yield "held_report", line


def preliminary_event_excerpts(
    *, question, sources, evidence, validation, exclude_ids=(), publication_window=None
):
    """Return verbatim, attributable leads, separate from verified date claims."""
    chronology = validation.get("event_chronology", {})
    if not chronology or chronology.get("latest_held_date"):
        return []
    relevance = validation.get("event_relevance", {})
    kept = set(relevance.get("kept_source_ids", [])) - set(
        relevance.get("unverified_source_ids", [])
    )
    by_id = {s["id"]: s for s in sources if s["id"] in kept}
    model_quotes = chronology.get("reported_excerpts", {})
    unresolved = set(chronology.get("unresolved_evidence_ids", []))
    unparsed = set(chronology.get("extraction", {}).get("unparsed_evidence_ids", []))
    unrelated = set(chronology.get("other_scope_blocks", []))
    legacy_blocks = unresolved - unparsed
    query_words = _topic_words(canonical_event_question(question))
    start_date = end_date = None
    if window := requested_publication_window(publication_window, question):
        start_date = window.get("start_date")
        end_date = window.get("end_date")
    candidates = []
    for key, block in evidence.items():
        source_id = block["source"]
        source = by_id.get(source_id)
        if not source or source_id in exclude_ids or key in exclude_ids or key in unrelated:
            continue
        day = web_date(source.get("date"))
        published = day.isoformat() if day else None
        if (start_date or end_date) and (
            not published
            or (start_date and published < start_date)
            or (end_date and published > end_date)
        ):
            continue
        if published and published > chronology["as_of"]:
            continue
        # Work with the original source body, even when the divider falls in an earlier block.
        text = source["text"]
        divider = _RELATED.search(text)
        end = divider.start() if divider else len(text)
        start = block.get("source_text_start")
        if start is None:
            start = text.find(block["text"])
        if start < 0 or not text.startswith(block["text"], start):
            continue
        passage = block["text"][: max(0, end - start)]
        lead = model_quotes.get(key)
        rows = []
        block_end = start + len(passage)
        if lead and 0 <= lead["start"] < lead["end"] <= len(passage):
            quote = _attributed_passage(
                text, start + lead["start"], start + lead["end"], start, block_end
            )
            if quote:
                rows.append((lead["kind"], quote, True))
        if (
            published
            and source.get("content_type") == "extracted"
            and source.get("kind") == "web"
            and key in legacy_blocks
        ):
            # A block may start or end mid-line. Only full original lines qualify.
            body = passage
            if start and text[start - 1] != "\n":
                body = body.partition("\n")[2]
            if block_end < len(text) and text[block_end] != "\n" and not body.endswith("\n"):
                body = body.rpartition("\n")[0]
            rows.extend(
                (kind, quote, False)
                for kind, quote in _archive_excerpts(
                    body, source.get("title"), query_words, published
                )
            )
        for kind, quote, selected in rows:
            rank = (
                published or "",
                selected,
                source.get("content_type") == "extracted",
                quote != source.get("title"),
            )
            item = {
                "source_id": source_id,
                "evidence_id": key,
                "quote": quote,
                "publication_date": published,
                "title": source.get("title", ""),
                "kind": kind,
            }
            candidates.append((rank, item))
    # One excerpt per role avoids repeated snippets from the same publication
    # while allowing one article to document both the meeting and its announcement.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    result = []
    for kind in ("held_report", "schedule_report"):
        for _, item in candidates:
            if item["kind"] == kind and all(item["quote"] != old["quote"] for old in result):
                result.append(item)
                break
    return result
