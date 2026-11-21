"""Address original evidence by IDs: models select blocks; code supplies quotations."""

from datetime import date
from typing import Annotated

from pydantic import Field, create_model

from .budget import check_budget
from .clock import clock_payload
from .config import settings
from .event_chronology import build_event_chronology, mentioned_dates
from .models.schemas import MAX_CITED_CLAIMS, CitationReview, CitedAnswer, CitedClaim
from .progress import context_blocks
from .request_context import scope_payload
from .validation import normalized, numbers


def _answer_schema(evidence):
    # Advertise only this call's block IDs to both transports. Keep the local
    # wire parser string-typed so an invalid ID can be diagnosed per claim.
    # Actual membership is checked below before any claim reaches the reviewer.
    block_id = Annotated[str, Field(json_schema_extra={"enum": list(evidence)})]
    claim = create_model(
        "CitedClaim",
        __base__=CitedClaim,
        evidence_ids=(list[block_id], CitedClaim.model_fields["evidence_ids"]),
    )
    return create_model(
        "CitedAnswer",
        __base__=CitedAnswer,
        claims=(list[claim], CitedAnswer.model_fields["claims"]),
    )


def _references_exist(claim, evidence):
    return bool(claim.evidence_ids) and all(key in evidence for key in claim.evidence_ids)


def _requested_event_context(chronology):
    """Render requested comparison dates from verified occurrences, preserving their roles."""
    result, gaps = [], []
    latest = chronology["latest_held_date"]
    for role in chronology.get("requested_context", []):
        if role == "next_scheduled":
            candidates = [
                row
                for row in chronology["scheduled_events"]
                if row["date"] >= chronology["as_of"] and (not latest or row["date"] > latest)
            ]
            if not candidates:
                gaps.append("Не подтверждена дата следующего запланированного события.")
                continue
            event = min(candidates, key=lambda row: row["date"])
            label = "Ближайшая объявленная дата по найденным источникам"
        else:
            candidates = [
                row for row in chronology["held_events"] if latest and row["date"] < latest
            ]
            if not candidates:
                gaps.append("Не установлена дата предыдущего состоявшегося события.")
                continue
            event = max(candidates, key=lambda row: row["date"])
            label = "Предыдущее состоявшееся событие по найденным источникам"
        day = date.fromisoformat(event["date"])
        result.append(
            CitedClaim(text=f"{label} — {day:%d.%m.%Y}.", evidence_ids=[event["evidence_id"]])
        )
    return result, gaps


def _attach_event_context(draft, related, chronology):
    if not related:
        return draft
    latest = date.fromisoformat(chronology["latest_held_date"])
    context_dates = {day for claim in related for day in mentioned_dates(claim.text)}
    # The verified announcement/history becomes its own claim. A draft labelling
    # another date as the latest cannot stand in for a separate valid dated anchor.
    primary = []
    for claim in draft.claims:
        if latest not in mentioned_dates(claim.text) and any(
            day in mentioned_dates(claim.text, default_year=day.year) for day in context_dates
        ):
            continue
        primary.append(claim)
    room = MAX_CITED_CLAIMS - len(related)
    return draft.model_copy(
        update={
            "claims": primary[:room] + related,
            "complete": draft.complete and len(primary) <= room,
        }
    )


def _matches_latest_event(claims, evidence, chronology, related):
    """A semantic approval cannot override a documented, later completed event."""
    newest = date.fromisoformat(chronology["latest_held_date"])
    anchors = set(chronology["latest_evidence_ids"])
    winner_sources = {evidence[key]["source"] for key in anchors}
    other_sources = {
        evidence[event["evidence_id"]]["source"]
        for event in chronology["held_events"] + chronology["scheduled_events"]
        if event["date"] != newest.isoformat()
    }
    unresolved = set(chronology["unresolved_evidence_ids"])
    competing_dates = {
        date.fromisoformat(event["date"])
        for event in chronology["held_events"] + chronology["scheduled_events"]
        if event["date"] != newest.isoformat()
    }
    dated_anchor = False
    for claim in claims:
        if not _references_exist(claim, evidence):
            return False
        if any(
            claim.text == context.text and claim.evidence_ids == context.evidence_ids
            for context in related
        ):
            continue
        supporting = [
            key for key in claim.evidence_ids if evidence[key]["source"] not in winner_sources
        ]
        # An independent source can establish a decision when jointly cited with
        # a block identifying this occurrence. Other editions cannot corroborate it.
        if supporting and (
            not anchors.intersection(claim.evidence_ids)
            or any(
                key in unresolved or evidence[key]["source"] in other_sources for key in supporting
            )
        ):
            return False
        # Keep the answer focused on the selected occurrence. A competing meeting
        # date (including a proposed next meeting) cannot stand in for that date.
        if any(
            day in mentioned_dates(claim.text, default_year=day.year) for day in competing_dates
        ):
            return False
        if anchors.intersection(claim.evidence_ids) and newest in mentioned_dates(claim.text):
            dated_anchor = True
    return dated_anchor


def catalog(sources, limit=18000, *, priority_source_ids=()):
    evidence = {}
    for source, i, start, end, block in context_blocks(
        sources, limit, priority_source_ids=priority_source_ids
    ):
        key = f"{source['id']}:B{i}"
        evidence[key] = {
            "source": source["id"],
            "text": block,
            "date": source.get("date"),
            "source_text_start": start,
            "source_text_end": end,
            **{
                k: source.get(k)
                for k in (
                    "title",
                    "kind",
                    "locator",
                    "page",
                    "url",
                    "local_url",
                    "document_id",
                    "version_id",
                    "block_id",
                    "method",
                    "content_type",
                    "content_scope",
                    "content_truncated",
                    "date_basis",
                    "retrieved_at",
                )
            },
        }
    return evidence


def generate_answer(
    client,
    question,
    sources,
    *,
    feedback="",
    previous=None,
    fresh=False,
    report_only=False,
    spot_grades=(),
    priority_source_ids=(),
    diagnostics: dict | None = None,
    allow_citation_repair=True,
    allow_research_clarification=False,
    latest_completed_event=False,
    allow_chronology_repair=True,
    allow_format_repair=True,
):
    # These counters describe validation, not model quality. Keep raw text and
    # rejected values out of diagnostics; None means generation did not finish.
    checks = diagnostics if diagnostics is not None else {}
    review_enabled = settings().answer_review
    checks.update(
        candidate_claims=None,
        candidate_complete=None,
        citation_rejected=0,
        numeric_rejected=0,
        semantic_rejected=0,
        accepted_claims=0,
        initial_citation_rejected=0,
        citation_repair_attempted=False,
        citation_repair_outcome="not_attempted",
        review_enabled=review_enabled,
        review_attempted=False,
        review_executed=False,
        coverage="unverified",
        missing_aspects=[],
        stage="evidence",
        reason="pending",
    )
    evidence = catalog(sources, priority_source_ids=priority_source_ids)
    if not evidence:
        checks["reason"] = "no_evidence"
        return CitedAnswer(claims=[], complete=False), evidence, "No evidence available"
    temporal_requirements = {"fresh": fresh, "report_only": report_only}
    if spot_grades:
        temporal_requirements["required_spot_grades"] = list(spot_grades)
    chronology = None
    chronology_gap = ""
    related, related_gaps = [], []
    if latest_completed_event:
        checks["stage"] = "event_chronology"
        chronology = build_event_chronology(
            client, question, evidence, previous=previous, allow_format_repair=allow_format_repair
        )
        latest = chronology["latest_held_date"]
        overdue = sorted(
            {
                item["date"]
                for item in chronology["scheduled_events"]
                if (not latest or latest < item["date"]) and item["date"] <= chronology["as_of"]
            }
        )
        checks["event_chronology"] = {
            key: chronology[key]
            for key in (
                "as_of",
                "latest_held_date",
                "latest_evidence_ids",
                "unresolved_evidence_ids",
            )
        }
        checks["event_chronology"]["overdue_scheduled_dates"] = overdue
        checks["event_chronology"]["scheduled_leads"] = [
            {key: event[key] for key in ("date", "evidence_id", "scope") if key in event}
            for event in chronology["scheduled_events"]
        ]
        extraction = chronology.get("extraction", {})
        if extraction:
            checks["event_chronology"]["extraction"] = extraction
        format_incomplete = extraction.get("status") in {"incomplete", "invalid_format"}
        related, related_gaps = _requested_event_context(chronology)
        checks["event_chronology"]["missing_requested_context"] = related_gaps
        if not latest:
            checks.update(
                reason="chronology_format_incomplete"
                if format_incomplete
                else "no_confirmed_event",
                accepted_claims=len(related),
            )
            return (
                CitedAnswer(claims=related, complete=False),
                evidence,
                "Не удалось завершить сопоставление дат в полученных материалах."
                if format_incomplete
                else (
                    "Не удалось подтвердить дату состоявшегося события. Проверьте итоговое "
                    "сообщение организатора или публикацию после назначенной даты."
                ),
            )
        if overdue:
            chronology_gap = (
                "После подтверждённого события было запланировано другое. Требуется "
                "проверить его итоги или отмену с учётом дат: " + ", ".join(overdue) + "."
            )
        elif format_incomplete:
            chronology_gap = "Проверку хронологии удалось завершить только для части фрагментов."
        elif chronology["unresolved_evidence_ids"]:
            chronology_gap = (
                "Остались фрагменты с неустановленной хронологией. Проверьте, есть ли "
                "в них более позднее состоявшееся событие, либо уточните поиск."
            )
        chronology_gap = " ".join(value for value in [chronology_gap, *related_gaps] if value)
    payload = {
        "question": question,
        **({"previous": previous} if previous else {}),
        **clock_payload(),
        "temporal_requirements": temporal_requirements,
        "user_scope": scope_payload(),
        "sources": [
            {
                k: s.get(k)
                for k in (
                    "id",
                    "title",
                    "date",
                    "url",
                    "page",
                    "kind",
                    "locator",
                    "version_id",
                    "method",
                    "content_type",
                    "content_scope",
                    "content_truncated",
                    "date_basis",
                    "retrieved_at",
                )
            }
            for s in sources
        ],
        "evidence": {key: row["text"] for key, row in evidence.items()},
        "feedback": feedback,
    }
    checks["stage"] = "answer_generation"
    if chronology is not None:
        payload["event_chronology"] = chronology
        if related:
            payload["verified_context_claims"] = [claim.model_dump() for claim in related]
    schema = _answer_schema(evidence)
    raw = client.generate("cited_answer", payload, schema)
    if related:
        raw = _attach_event_context(raw, related, chronology)
    if chronology is not None and not _matches_latest_event(
        raw.claims, evidence, chronology, related
    ):
        if allow_chronology_repair:
            check_budget()
            checks.update(stage="chronology_repair", chronology_repair_attempted=True)
            raw = client.generate(
                "cited_answer",
                {
                    **payload,
                    "chronology_repair": (
                        "Укажи полную дату latest_held_date и процитируй её блок. Все выводы "
                        "должны относиться к участникам и решениям именно этого события. "
                        "Опиши его как последнее подтверждённое в найденных материалах. "
                        "Пропусти даты других заседаний, запланированные события и чужие решения."
                    ),
                },
                schema,
            )
            if related:
                raw = _attach_event_context(raw, related, chronology)
        if not _matches_latest_event(raw.claims, evidence, chronology, related):
            checks.update(
                stage="finished",
                reason="event_chronology_mismatch",
                candidate_claims=len(raw.claims),
                candidate_complete=raw.complete,
                accepted_claims=0,
            )
            return (
                CitedAnswer(claims=[], complete=False),
                evidence,
                (
                    "Черновик не привязал дату и выводы к самому позднему подтверждённому "
                    "событию. Нужен источник с его датой и итогами."
                ),
            )
    if chronology is not None:
        checks["chronology_matches_draft"] = True
    invalid = sum(not _references_exist(claim, evidence) for claim in raw.claims)
    checks["initial_citation_rejected"] = invalid
    if allow_citation_repair and invalid and invalid == len(raw.claims):
        check_budget()
        checks.update(
            stage="citation_repair",
            candidate_claims=len(raw.claims),
            candidate_complete=raw.complete,
            citation_rejected=invalid,
            citation_repair_attempted=True,
            citation_repair_outcome="pending",
        )
        corrected_payload = dict(
            payload,
            citation_repair=(
                "Все предыдущие утверждения содержали хотя бы один неизвестный адрес блока. "
                "Повторно прочитай те же фрагменты. Для каждого подтверждённого утверждения "
                "скопируй в evidence_ids точные ключи объекта evidence. Поле id в sources "
                "обозначает документ или страницу и не адресует текстовый блок. "
                "Если ни один блок не подтверждает утверждение, пропусти его."
            ),
        )
        raw = client.generate("cited_answer", corrected_payload, schema)
        invalid = sum(not _references_exist(claim, evidence) for claim in raw.claims)
        if not raw.claims:
            outcome = "empty"
        elif invalid == 0:
            outcome = "ids_valid"
        elif invalid == len(raw.claims):
            outcome = "invalid_ids"
        else:
            outcome = "partially_valid_ids"
        checks["citation_repair_outcome"] = outcome
    checks["citation_rejected"] = invalid
    checks["candidate_claims"] = len(raw.claims)
    checks["candidate_complete"] = raw.complete
    if not raw.claims:
        checks["reason"] = "no_candidate_claims"
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "The model proposed no supported claims from the supplied evidence. "
            "Reassess whether the sources answer the question. Change the query or source "
            "to address a specific gap, or finish with the limitation; do not repeat the same search.",
        )
    checks["stage"] = "local_validation"
    valid = []
    for claim in raw.claims:
        if not _references_exist(claim, evidence):
            continue
        text = " ".join(
            evidence[key]["text"] + " " + (evidence[key]["date"] or "")
            for key in claim.evidence_ids
        )
        if chronology is not None:
            winner = date.fromisoformat(chronology["latest_held_date"])
            if set(claim.evidence_ids).intersection(
                chronology["latest_evidence_ids"]
            ) and winner in mentioned_dates(claim.text):
                text += f" {winner.isoformat()} {winner:%d.%m.%Y}"
        if numbers(claim.text) <= numbers(text) or claim in related:
            valid.append(claim)
        else:
            checks["numeric_rejected"] += 1
    if not valid:
        checks["reason"] = "local_validation_rejected"
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "The proposed citations contained unknown block IDs. No claim had all its citations "
            "in the supplied evidence. Do not repeat the same search; use a specific different "
            "source or page if it addresses an evidence gap, otherwise finish with this limitation."
            if invalid == len(raw.claims)
            else "No claims passed citation and numeric checks.",
        )
    if chronology is not None and not _matches_latest_event(valid, evidence, chronology, related):
        checks.update(stage="finished", reason="event_chronology_mismatch", accepted_claims=0)
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            ("После проверки чисел не осталось утверждения с подтверждённой датой события."),
        )
    if not review_enabled:
        # Valid IDs and matching numbers cannot establish currency. A fetched
        # web result may itself be old or unrelated to the claim.
        complete = (
            raw.complete and len(valid) == len(raw.claims) and not fresh and chronology is None
        )
        checks.update(
            accepted_claims=len(valid),
            stage="finished",
            reason="semantic_review_disabled",
        )
        if chronology_gap:
            checks["missing_aspects"] = [chronology_gap]
            return CitedAnswer(claims=valid, complete=False), evidence, chronology_gap
        return (
            CitedAnswer(claims=valid, complete=complete),
            evidence,
            "Freshness was not verified because semantic citation review is disabled. "
            "Do not present historical evidence as the latest state."
            if fresh
            else ""
            if complete
            else "Some requested facts remain unsupported. Search for the missing evidence or open the relevant page.",
        )
    checks["stage"] = "semantic_review"
    checks["review_attempted"] = True
    review_payload = {
        "question": question,
        **({"previous": previous} if previous else {}),
        **clock_payload(),
        "temporal_requirements": temporal_requirements,
        "allow_research_clarification": allow_research_clarification,
        "user_scope": scope_payload(),
        "claims": [
            {"index": i, "text": c.text, "evidence": [evidence[key] for key in c.evidence_ids]}
            for i, c in enumerate(valid)
        ],
    }
    if chronology is not None:
        cited = {key for claim in valid for key in claim.evidence_ids}
        review_payload["event_chronology"] = chronology
        review_payload["other_evidence"] = {
            key: row["text"] for key, row in evidence.items() if key not in cited
        }
    review = client.generate("citation_review", review_payload, CitationReview)
    checks.update(
        review_executed=True,
        coverage=review.coverage,
        missing_aspects=review.missing_aspects,
    )
    indexes = review.supported
    well_formed = len(set(indexes)) == len(indexes) and all(0 <= i < len(valid) for i in indexes)
    if not well_formed:
        checks.update(semantic_rejected=len(valid), reason="invalid_review_indexes")
        return (
            CitedAnswer(claims=[], complete=False),
            evidence,
            "Citation review returned invalid or duplicate claim indexes. "
            "No claims passed semantic validation.",
        )
    supported = [c for i, c in enumerate(valid) if i in indexes]
    if (
        chronology is not None
        and supported
        and not _matches_latest_event(supported, evidence, chronology, related)
    ):
        # Returning just the accepted outcome after its date claim failed review
        # would associate it with an unconfirmed event.
        supported = []
        chronology_gap = (
            "При смысловой проверке не подтверждена дата события и его связь с выводами."
        )
    if chronology is not None and any(item not in supported for item in related):
        chronology_gap += " Требуется проверить запрошенные даты связанных событий."
    if chronology_gap:
        review = review.model_copy(
            update={
                "missing_aspects": [*review.missing_aspects, chronology_gap],
                "coverage": "partial",
                "complete": False,
            }
        )
        checks.update(coverage=review.coverage, missing_aspects=review.missing_aspects)
    if (
        allow_research_clarification
        and not supported
        and not review.complete
        and review.coverage != "sufficient"
    ):
        options = [" ".join(value.split()) for value in review.research_options]
        labels = {normalized(value).strip(" .!?…") for value in options}
        if (
            2 <= len(options) <= 3
            and len(labels) == len(options)
            and "" not in labels
            and all(len(value) <= 140 for value in options)
        ):
            choices = "\n".join(f"{i}. {option}" for i, option in enumerate(options, 1))
            checks["research_clarification"] = {
                "clarify": f"Что именно вы хотите узнать?\n\n{choices}",
                "research_options": options,
            }
    complete = (
        raw.complete
        and review.complete
        and bool(supported)
        and len(supported) == len(valid) == len(raw.claims)
        and not review.feedback.strip()
        and review.coverage == "sufficient"
        and not review.missing_aspects
    )
    checks.update(
        semantic_rejected=len(valid) - len(supported),
        accepted_claims=len(supported),
        stage="finished",
        reason="complete"
        if complete
        else "semantic_validation_rejected"
        if not supported
        else "coverage_unverified"
        if review.coverage == "unverified"
        else "coverage_incomplete"
        if review.coverage != "sufficient" or review.missing_aspects
        else "incomplete",
    )
    feedback_parts = [review.feedback.strip()]
    if review.missing_aspects:
        feedback_parts.append("Missing requested aspects: " + "; ".join(review.missing_aspects))
    if review.coverage == "unverified":
        feedback_parts.append(
            "Coverage of the question was not verified. Check the requested relation, scope "
            "and time period before treating the answer as complete."
        )
    elif review.coverage == "partial" and not review.missing_aspects:
        feedback_parts.append(
            "The review found only partial coverage of the question. "
            "Identify the missing requested aspect and obtain direct evidence."
        )
    return (
        CitedAnswer(claims=supported, complete=complete),
        evidence,
        " ".join(part for part in feedback_parts if part)
        or (
            ""
            if complete
            else "Some requested facts or their currency remain unverified. "
            "Find direct evidence for the missing facts before finishing."
        ),
    )


def render_claims(draft, evidence, sources):
    lookup = {s["id"]: s for s in sources}
    blocks, claims, used = [], [], {}
    for claim in draft.claims:
        ids = list(dict.fromkeys(evidence[key]["source"] for key in claim.evidence_ids))
        blocks.append(claim.text + " " + " ".join(f"[{key}]" for key in ids))
        # Keep the public compatibility field quote as one original block. All
        # additional citations are exposed by evidence_ids and source.evidence.
        first = evidence[claim.evidence_ids[0]]
        claims.append(
            {
                "source": first["source"],
                "quote": first["text"][:1800],
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
                "citations": [
                    {
                        "evidence_id": eid,
                        "quote": evidence[eid]["text"],
                        **{k: v for k, v in evidence[eid].items() if k not in {"text", "date"}},
                    }
                    for eid in claim.evidence_ids
                ],
            }
        )
        for key in ids:
            if key not in used:
                used[key] = {**lookup[key], "evidence": {}}
            used[key]["evidence"].update(
                {
                    eid: evidence[eid]["text"]
                    for eid in claim.evidence_ids
                    if evidence[eid]["source"] == key
                }
            )
    return blocks, claims, list(used.values())
