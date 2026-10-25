"""Find contradictions between timeline events and derive content-free alerts."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from pipeline.contracts import (
    Alert,
    AlertType,
    Certainty,
    ConflictField,
    Contradiction,
    ContradictionStatus,
    Event,
    EventStatus,
)
from pipeline.merge import actor_set, date_gap_days, description_similarity

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0
MAX_CANDIDATES = 50

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|without|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)

_FALLBACK_SYSTEM = (
    "You review pairs of events extracted from legal case documents. Decide whether the two "
    "statements genuinely contradict each other (they cannot both be true), not merely differ in "
    "detail or describe different occurrences. Answer only with JSON matching the schema: "
    '{"is_contradiction": bool, "field": "date" | "actor" | "fact", "explanation": str}. '
    "Keep the explanation to one or two sentences and cite the event labels A and B."
)
_FALLBACK_USER = (
    "Suspected conflict type: {field}\n\n"
    "Event A:\n{event_a}\n\n"
    "Event B:\n{event_b}\n\n"
    "Do A and B contradict each other?"
)


class ConflictVerdict(BaseModel):
    """LLM confirmation of a candidate contradiction."""

    is_contradiction: bool
    field: ConflictField
    explanation: str


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Event
    b: Event
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


def _prompts() -> tuple[str, str]:
    try:
        from pipeline.prompts import CONFLICT_SYSTEM, CONFLICT_USER

        return CONFLICT_SYSTEM, CONFLICT_USER
    except ImportError:
        return _FALLBACK_SYSTEM, _FALLBACK_USER


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def contradiction_id(event_ids: list[str], field: ConflictField | str) -> str:
    """Deterministic id from the sorted event ids and the conflict field."""
    key = "|".join(sorted(event_ids)) + "#" + str(field)
    return "ctr_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _alert_id(case_id: str, kind: AlertType, key: str) -> str:
    raw = f"{case_id}|{kind}|{key}"
    return "alr_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _has_negation(event: Event) -> bool:
    return bool(NEGATION_CUES.search(event.description) or NEGATION_CUES.search(event.source.quote))


def _classify(a: Event, b: Event) -> Candidate | None:
    """Apply the DATE / ACTOR / FACT candidate rules to one pair."""
    gap = date_gap_days(a, b)
    if gap is None:
        return None
    sa, sb = actor_set(a), actor_set(b)
    sim = description_similarity(a, b)
    both_exact = a.certainty == Certainty.EXACT and b.certainty == Certainty.EXACT
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("March 11" vs "12").
        if sa & sb and sim >= SIMILAR_THRESHOLD and a.source.doc_id != b.source.doc_id:
            return Candidate(a, b, ConflictField.DATE, sim + min(gap, 30))
        return None
    if gap != 0:
        return None
    if sim >= SIMILAR_THRESHOLD and sa and sb and not (sa & sb):
        return Candidate(a, b, ConflictField.ACTOR, sim)
    if sa & sb and sim < SIMILAR_THRESHOLD and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, ConflictField.FACT, 100.0 - sim)
    return None


def candidate_pairs(
    events: list[Event],
    max_candidates: int = MAX_CANDIDATES,
    *,
    focus_ids: set[str] | None = None,
) -> list[Candidate]:
    """Generate candidate contradictions with cheap rules, capped at ``max_candidates``.

    With ``focus_ids``, only pairs involving at least one of those events are considered.
    """
    live = [e for e in events if e.status != EventStatus.REJECTED]
    found: list[Candidate] = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.id == b.id:
                continue
            if focus_ids is not None and a.id not in focus_ids and b.id not in focus_ids:
                continue
            cand = _classify(a, b)
            if cand is not None:
                found.append(cand)
    found.sort(key=lambda c: (-c.score, c.a.id, c.b.id))
    if len(found) > max_candidates:
        dropped = found[max_candidates:]
        log.warning(
            "conflict candidates capped at %d; dropped %d: %s",
            max_candidates,
            len(dropped),
            [(c.a.id, c.b.id, str(c.field)) for c in dropped],
        )
        found = found[:max_candidates]
    return found


def _render_event(e: Event) -> str:
    return (
        f"id: {e.id}\ndocument: {e.source.doc_id}\ndate as written: {e.date_text}\n"
        f"resolved date: {_resolved(e)}\nactors: {', '.join(e.actors) or 'none'}\n"
        f"description: {e.description}\nquote: {e.source.quote}"
    )


def _resolved(e: Event) -> str:
    if e.date_start is None:
        return "unresolved"
    if e.date_end and e.date_end != e.date_start:
        return f"{e.date_start.isoformat()} to {e.date_end.isoformat()}"
    return e.date_start.isoformat()


def _event_fields(e: Event, suffix: str) -> dict[str, str]:
    """Per-event placeholders used by ``prompts.CONFLICT_USER`` (``doc_a``, ``quote_b``, ...)."""
    return {
        f"doc_{suffix}": e.source.doc_id,
        f"date_{suffix}": e.date_text,
        f"resolved_{suffix}": _resolved(e),
        f"actors_{suffix}": ", ".join(e.actors) or "none",
        f"description_{suffix}": e.description,
        f"quote_{suffix}": e.source.quote,
    }


def _confirm(cand: Candidate) -> ConflictVerdict | None:
    """Ask the local LLM whether a candidate is a real contradiction."""
    from pipeline import llm

    system, user_tpl = _prompts()
    ea, eb = _render_event(cand.a), _render_event(cand.b)
    values: dict[str, Any] = _SafeDict(
        field=str(cand.field),
        conflict_type=str(cand.field),
        event_a=ea,
        event_b=eb,
        a=ea,
        b=eb,
        **_event_fields(cand.a, "a"),
        **_event_fields(cand.b, "b"),
    )
    user = user_tpl.format_map(values)
    result = llm.chat_json(system, user, ConflictVerdict)
    if isinstance(result, ConflictVerdict):
        return result
    if isinstance(result, BaseModel):
        return ConflictVerdict.model_validate(result.model_dump())
    return ConflictVerdict.model_validate(result)


def find_contradictions(
    events: list[Event],
    case_id: str,
    *,
    max_candidates: int = MAX_CANDIDATES,
    skip_ids: set[str] | None = None,
    focus_ids: set[str] | None = None,
) -> list[Contradiction]:
    """Generate candidate pairs with cheap rules, then confirm each with the local LLM.

    ``skip_ids`` holds contradiction ids already stored; those candidates skip the LLM call.
    ``focus_ids`` restricts candidates to pairs involving at least one of those event ids.
    Candidates whose confirmation call fails are logged and skipped.
    """
    skip = skip_ids or set()
    out: dict[str, Contradiction] = {}
    for cand in candidate_pairs(events, max_candidates, focus_ids=focus_ids):
        ids = sorted([cand.a.id, cand.b.id])
        if contradiction_id(ids, cand.field) in skip:
            continue
        try:
            verdict = _confirm(cand)
        except Exception as exc:
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, exc)
            continue
        if verdict is None or not verdict.is_contradiction:
            continue
        cid = contradiction_id(ids, verdict.field)
        if cid in skip or cid in out:
            continue
        out[cid] = Contradiction(
            id=cid,
            case_id=case_id,
            event_ids=ids,
            field=verdict.field,
            explanation=verdict.explanation.strip(),
        )
    log.info("contradictions confirmed: %d", len(out))
    return list(out.values())


def alerts_for(
    contradictions: list[Contradiction],
    events: list[Event],
    case_id: str,
    base_url: str,
    *,
    now: datetime | None = None,
) -> list[Alert]:
    """Build messaging-safe alerts: types and ids only, never case content."""
    created = now or datetime.now(UTC)
    url = f"{base_url.rstrip('/')}/alerts"
    by_id = {e.id: e for e in events}
    alerts: list[Alert] = []

    def add(kind: AlertType, key: str, refs: list[str]) -> None:
        alerts.append(
            Alert(
                id=_alert_id(case_id, kind, key),
                case_id=case_id,
                type=kind,
                created_at=created,
                dashboard_url=url,
                ref_ids=refs,
            )
        )

    active = [c for c in contradictions if c.status != ContradictionStatus.DISMISSED]
    for c in active:
        add(AlertType.NEW_CONTRADICTION, c.id, [c.id])

    for e in events:
        if e.deadline_relevant and e.status != EventStatus.REJECTED:
            add(AlertType.DEADLINE_DATE, e.id, [e.id])

    involvement: dict[str, set[str]] = defaultdict(set)
    for c in active:
        actors: set[str] = set()
        for eid in c.event_ids:
            if eid in by_id:
                actors |= actor_set(by_id[eid])
        for actor in actors:
            involvement[actor].add(c.id)
    for actor in sorted(involvement):
        cids = involvement[actor]
        if len(cids) >= 2:
            add(AlertType.FLAGGED_WITNESS, actor, sorted(cids))
    return alerts
