"""Find contradictions between timeline events and derive content-free alerts."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
from pipeline.merge import (
    actor_set,
    actors_overlap,
    date_gap_days,
    description_similarity,
    is_organization,
    name_tokens,
    names_match,
    unmatched_people,
)
from pipeline.prompts import CONFLICT_SYSTEM, CONFLICT_USER

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0  # legacy description-similarity rule (token_set_ratio)
MAX_CANDIDATES = 60
MAX_GAP_DAYS = 62  # events further apart than this are rarely one misdated occurrence
TOPIC_MIN = 0.3  # topic-word overlap for resolved-date pairs further apart than NEAR_DAYS
TOPIC_MIN_UNRESOLVED = 0.4  # stricter when one date is unresolved (no date evidence)
NEAR_DAYS = 14  # within this many days, one shared topic word is enough for an LLM check
CONTEXT_BEFORE = 300  # characters of source text shown around each quote
CONTEXT_AFTER = 400
VERDICT_MAX_TOKENS = 800

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)

# A disagreement about *who* was there needs one source to exclude someone affirmatively;
# silence is not exclusion. One of these cues must appear in either event's quote or context.
EXCLUSION_CUES = re.compile(
    r"\b(only|nobody|no\s+one|no-one|none|alone|nor|neither|never|not|no|absent|"
    r"wasn't|weren't|didn't|denie[sd]|deny|except|besides|other\s+than|else)\b",
    re.IGNORECASE,
)
# Words marking a plan or schedule ("set for", "due by"). A plan compared with what actually
# happened is not a date contradiction, so only plan-vs-plan or fact-vs-fact pairs qualify.
PLAN_CUES = re.compile(
    r"\b(schedul\w*|reschedul\w*|set\s+for|due\s+(?:by|on)|deadline|planned|plans?\s+to|"
    r"expected\s+to|will|postpon\w*|adjourn\w*|moved\s+(?:from|to))\b",
    re.IGNORECASE,
)
# Words marking someone's knowledge or awareness ("first learned", "became aware"). When one
# side is about when someone knew of a matter, a timing disagreement is a FACT (notice)
# conflict about that knowledge, not two dates for one occurrence.
KNOWLEDGE_CUES = re.compile(
    r"\b(learn(?:ed|t|s|ing)?|became\s+aware|been\s+aware|was\s+aware|were\s+aware|unaware|"
    r"aware\s+of|knew|know(?:s|n)?\s+(?:of|about)|first\s+heard|found\s+out|discover(?:ed|s|y)|"
    r"realiz(?:ed|es)|realis(?:ed|es))\b",
    re.IGNORECASE,
)


def _all_required(schema: dict[str, Any]) -> None:
    """Mark every property required so constrained decoding always emits each field."""
    schema["required"] = list(schema.get("properties", {}))


class ConflictVerdict(BaseModel):
    """LLM confirmation of a candidate contradiction."""

    model_config = ConfigDict(json_schema_extra=_all_required)

    comparison: str = ""
    same_subject: bool = True
    reconciliation: str = ""
    is_contradiction: bool
    field: ConflictField
    other_fields: list[ConflictField] = Field(default_factory=list)
    explanation: str = ""

    def fields(self) -> list[ConflictField]:
        """Primary field first, then distinct additional date/actor fields.

        "fact" is the catch-all, so as a secondary field it almost always restates the primary
        disagreement; it is only kept as the primary field.
        """
        out = [self.field]
        out += [f for f in self.other_fields if f not in out and f != ConflictField.FACT]
        return out


def _final_fields(
    verdict: ConflictVerdict, a: Event, b: Event, context: str = ""
) -> list[ConflictField]:
    """Fields to report after deterministic sanity checks on the LLM's answer.

    - DATE needs two resolved dates; when either side has none the disagreement is reported as
      FACT (e.g. "learned of it after X" vs a dated notice). Two resolved dates that overlap
      cannot be a date contradiction, so DATE is dropped then. When either event is about
      someone's knowledge ("first learned of it in mid-June"), DATE becomes FACT too.
    - When the events share a person, ACTOR needs an exclusion cue ("only", "no one else",
      "was not there") in either event's quote or surrounding text: a source that merely omits
      a person is not a conflict. (The same act attributed to different people needs none.)
    """
    out: list[ConflictField] = []
    for f in verdict.fields():
        if f == ConflictField.DATE:
            gap = date_gap_days(a, b)
            if gap is None or _is_knowledge(a) or _is_knowledge(b):
                f = ConflictField.FACT
            elif gap == 0:
                continue
        if (
            f == ConflictField.ACTOR
            and linked(a, b)
            and not EXCLUSION_CUES.search(" ".join([a.source.quote, b.source.quote, context]))
        ):
            continue
        if f not in out:
            out.append(f)
    return out


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Event
    b: Event
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


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


def _is_plan(event: Event) -> bool:
    return bool(PLAN_CUES.search(event.description) or PLAN_CUES.search(event.source.quote))


def _is_knowledge(event: Event) -> bool:
    return bool(KNOWLEDGE_CUES.search(f"{event.description} {event.source.quote}"))


def _has_negation(event: Event) -> bool:
    return bool(NEGATION_CUES.search(event.description) or NEGATION_CUES.search(event.source.quote))


# --------------------------------------------------------------------------- topic overlap

_STOPWORDS = frozenset(
    """a an the of to in on at for with by from and or but if was were is are be been being
    has had have having do does did that this these those his her hers their theirs its as
    into onto about after before during between over under than then there here also any all
    some each other another such same one two three first second he she they it i me my we
    our you your him them who whom whose which what when where while how why not no so very
    just only both either neither said says stated stating state told tells tell asked
    according approximately around about later earlier sometime time day days week weeks
    month months year years today yesterday morning afternoon evening night am pm
    mr mrs ms dr take took taken taking place held hold holds holding believe believed
    kept keep keeps went go goes going came come comes made make makes get got gave give
    given put way thing things matter re regarding concerning""".split()
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WEEKDAYS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thurs fri "
    "sat sun".split()
)
_DATE_NUMBERS = re.compile(
    rf"\b(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})"
    rf"|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}:\d{{2}}"
    rf"|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_MONTH_SET = frozenset(_MONTHS.split("|"))


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def topic_tokens(event: Event, names: set[str] | None = None) -> set[str]:
    """Content words of the description and quote, minus dates, names and stopwords."""
    text = _DATE_NUMBERS.sub(" ", f"{event.description} {event.source.quote}".casefold())
    skip = names if names is not None else _actor_name_tokens(event)
    out: set[str] = set()
    for word in _WORD.findall(text):
        if word in _STOPWORDS or word in _MONTH_SET or word in _WEEKDAYS or word in skip:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        out.add(_stem(word))
    return out


def _actor_name_tokens(*events: Event) -> set[str]:
    return {t for e in events for name in e.actors for t in name_tokens(name)}


def topic_overlap(a: Event, b: Event) -> tuple[float, int]:
    """Overlap coefficient (0..1) of the two events' topic words, and the shared-word count.

    Actor names and date expressions are excluded so that the score reflects *what* happened
    (meeting, dismissal, injury...) rather than who or when.
    """
    names = _actor_name_tokens(a, b)
    ta, tb = topic_tokens(a, names), topic_tokens(b, names)
    if not ta or not tb:
        return 0.0, 0
    shared = len(ta & tb)
    return shared / min(len(ta), len(tb)), shared


# --------------------------------------------------------------------------- candidates


def _people(e: Event) -> list[str]:
    return [x for x in e.actors if x.strip() and name_tokens(x) and not is_organization(x)]


def linked(a: Event, b: Event) -> bool:
    """The events share an actor; a shared person is required when both name people, so a
    common employer or agency alone does not link two events."""
    pa, pb = _people(a), _people(b)
    if pa and pb:
        return any(names_match(x, y) for x in pa for y in pb)
    return actors_overlap(a, b)


def _classify(a: Event, b: Event) -> Candidate | None:
    """Apply the DATE / ACTOR / FACT candidate rules to one pair."""
    cross_doc = a.source.doc_id != b.source.doc_id
    gap = date_gap_days(a, b)
    sim = description_similarity(a, b)
    link = linked(a, b)
    topic, shared = topic_overlap(a, b) if cross_doc and link else (0.0, 0)

    if gap is None:
        # One side has no usable date: only a strong same-subject signal is worth an LLM call
        # (e.g. "learned of it only after the hearing" vs a dated written notice).
        if cross_doc and link and topic >= TOPIC_MIN_UNRESOLVED:
            return Candidate(a, b, ConflictField.FACT, 100.0 * topic)
        return None

    both_exact = a.certainty == Certainty.EXACT and b.certainty == Certainty.EXACT
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("June 4" vs "June 5").
        if not cross_doc or not link or _is_plan(a) != _is_plan(b):
            return None
        near = gap <= NEAR_DAYS and shared >= 1
        if sim >= SIMILAR_THRESHOLD or near or (gap <= MAX_GAP_DAYS and topic >= TOPIC_MIN):
            closeness = max(0.0, 1.0 - gap / MAX_GAP_DAYS)
            kind = (
                ConflictField.FACT if _is_knowledge(a) or _is_knowledge(b) else ConflictField.DATE
            )
            return Candidate(a, b, kind, 100.0 * max(topic, sim / 100) + closeness)
        return None
    if gap != 0:
        return None

    sa, sb = actor_set(a), actor_set(b)
    if cross_doc and sa and sb and not link and sim >= SIMILAR_THRESHOLD:
        return Candidate(a, b, ConflictField.ACTOR, sim)
    if cross_doc and link and shared >= 1:
        differing = unmatched_people(a, b) + unmatched_people(b, a)
        if differing:
            # Same occurrence, different people named: an omission or a real disagreement
            # about who was there; the LLM decides using the surrounding source text.
            return Candidate(a, b, ConflictField.ACTOR, 100.0 * topic)
    if link and sim < SIMILAR_THRESHOLD and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, ConflictField.FACT, 100.0 - sim)
    if cross_doc and link and topic >= TOPIC_MIN and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, ConflictField.FACT, 100.0 * topic)
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
    log.info("conflict candidates: %d", len(found))
    return found


# --------------------------------------------------------------------------- source context

SourceText = Callable[[str], str | None]

_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n")


def _locate(text: str, e: Event) -> tuple[int, int] | None:
    s, t = e.source.char_start, e.source.char_end
    quote = e.source.quote
    if 0 <= s < t <= len(text):
        if _WS.sub(" ", text[s:t]).strip() == _WS.sub(" ", quote).strip() or not quote.strip():
            return s, t
    if quote.strip():
        i = text.find(quote)
        if i >= 0:
            return i, i + len(quote)
    if 0 <= s < t <= len(text):
        return s, t
    return None


def quote_context(text: str | None, e: Event) -> str:
    """Up to a few sentences of source text around the event's quote (whitespace collapsed)."""
    if not text:
        return ""
    span = _locate(text, e)
    if span is None:
        return ""
    s, t = span
    lo, hi = max(0, s - CONTEXT_BEFORE), min(len(text), t + CONTEXT_AFTER)
    if lo > 0:  # start at a sentence/line boundary inside the window
        m = _BOUNDARY.search(text, lo, s)
        if m:
            lo = m.end()
    if hi < len(text):  # end at the last boundary inside the window
        last = None
        for m in _BOUNDARY.finditer(text, t, hi):
            last = m
        if last is not None:
            hi = last.start()
    return _WS.sub(" ", text[lo:hi]).strip()


class _StoreDocs:
    """Best-effort read access to stored document text and filenames (cached per doc)."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._text: dict[str, str | None] = {}
        self._name: dict[str, str] = {}

    @classmethod
    def open_default(cls) -> _StoreDocs | None:
        try:
            from pipeline.config import get_settings
            from pipeline.store import Store

            db_path = Path(get_settings().db_path)
            if not db_path.exists():
                return None
            return cls(Store(db_path))
        except Exception as exc:
            log.debug("no stored document text for conflict context: %s", exc)
            return None

    def text(self, doc_id: str) -> str | None:
        if doc_id not in self._text:
            try:
                parsed = self._store.get_parsed_text(doc_id)
                self._text[doc_id] = parsed.text if parsed is not None else None
            except Exception as exc:
                log.debug("could not read text of %s: %s", doc_id, exc)
                self._text[doc_id] = None
        return self._text[doc_id]

    def name(self, doc_id: str) -> str:
        if doc_id not in self._name:
            try:
                doc = self._store.get_document(doc_id)
                self._name[doc_id] = doc.filename if doc is not None else doc_id
            except Exception:
                self._name[doc_id] = doc_id
        return self._name[doc_id]

    def linked_duplicates(self, case_id: str) -> list[Event]:
        """Events hidden from the timeline as duplicates of another event."""
        try:
            every = self._store.list_events(case_id, include_duplicates=True)
            shown = {e.id for e in self._store.list_events(case_id)}
        except Exception as exc:
            log.debug("could not list duplicate events: %s", exc)
            return []
        return [e for e in every if e.id not in shown]

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:
            pass


def with_linked_duplicates(
    events: list[Event], duplicates: list[Event], focus_ids: set[str] | None
) -> tuple[list[Event], set[str] | None]:
    """Add linked duplicate events to the pool checked for contradictions.

    A duplicate is another document's own account of an occurrence; its quote and context can
    disagree with a third document even when the canonical event's do not. With ``focus_ids``,
    duplicates from the documents being scanned (those of the focus events) join the focus.
    """
    known = {e.id for e in events}
    extra = [d for d in duplicates if d.id not in known]
    if not extra:
        return events, focus_ids
    if focus_ids is not None:
        focus_docs = {e.source.doc_id for e in events if e.id in focus_ids}
        focus_ids = focus_ids | {d.id for d in extra if d.source.doc_id in focus_docs}
    return [*events, *extra], focus_ids


# --------------------------------------------------------------------------- confirmation


def _resolved(e: Event) -> str:
    if e.date_start is None:
        return "unresolved"
    if e.date_end and e.date_end != e.date_start:
        return f"{e.date_start.isoformat()} to {e.date_end.isoformat()}"
    return e.date_start.isoformat()


def _event_fields(e: Event, suffix: str, context: str, doc_name: str) -> dict[str, str]:
    """Per-event placeholders used by ``CONFLICT_USER`` (``doc_a``, ``quote_b``, ...)."""
    return {
        f"doc_{suffix}": doc_name,
        f"date_{suffix}": e.date_text,
        f"resolved_{suffix}": _resolved(e),
        f"actors_{suffix}": ", ".join(e.actors) or "none",
        f"description_{suffix}": e.description,
        f"quote_{suffix}": e.source.quote,
        f"context_{suffix}": context or "(not available)",
    }


def _context(e: Event, source_text: SourceText | None) -> str:
    if source_text is None:
        return ""
    try:
        return quote_context(source_text(e.source.doc_id), e)
    except Exception as exc:
        log.debug("context lookup failed for %s: %s", e.id, exc)
        return ""


def _confirm(
    cand: Candidate,
    source_text: SourceText | None = None,
    doc_name: Callable[[str], str] | None = None,
    contexts: tuple[str, str] | None = None,
) -> ConflictVerdict | None:
    """Ask the local LLM whether a candidate is a real contradiction."""
    from pipeline import llm

    ctx_a, ctx_b = contexts or (_context(cand.a, source_text), _context(cand.b, source_text))

    def name(e: Event) -> str:
        return doc_name(e.source.doc_id) if doc_name is not None else e.source.doc_id

    values: dict[str, Any] = _SafeDict(
        field=str(cand.field),
        **_event_fields(cand.a, "a", ctx_a, name(cand.a)),
        **_event_fields(cand.b, "b", ctx_b, name(cand.b)),
    )
    user = CONFLICT_USER.format_map(values)
    result = llm.chat_json(CONFLICT_SYSTEM, user, ConflictVerdict, max_tokens=VERDICT_MAX_TOKENS)
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
    source_text: SourceText | None = None,
    doc_name: Callable[[str], str] | None = None,
) -> list[Contradiction]:
    """Generate candidate pairs with cheap rules, then confirm each with the local LLM.

    ``skip_ids`` holds contradiction ids already stored; those candidates skip the LLM call.
    ``focus_ids`` restricts candidates to pairs involving at least one of those event ids.
    ``source_text(doc_id)`` returns a document's normalized text so the LLM sees the sentences
    around each quote and ``doc_name(doc_id)`` a display name; by default both are read from
    the configured store when one exists. In that case the store's linked duplicates (other
    documents' accounts of an occurrence, hidden from the timeline) are checked as well.
    A confirmed pair yields one contradiction per conflicting field.
    Candidates whose confirmation call fails are logged and skipped.
    """
    skip = skip_ids or set()
    out: dict[str, Contradiction] = {}
    docs = _StoreDocs.open_default() if source_text is None and events else None
    try:
        if docs is not None:
            source_text = docs.text
            doc_name = doc_name or docs.name
            events, focus_ids = with_linked_duplicates(
                events, docs.linked_duplicates(case_id), focus_ids
            )
        cands = candidate_pairs(events, max_candidates, focus_ids=focus_ids)
        _confirm_all(cands, case_id, skip, out, source_text, doc_name)
    finally:
        if docs is not None:
            docs.close()
    log.info("contradictions confirmed: %d (from %d candidates)", len(out), len(cands))
    return list(out.values())


def _redundant(cand: Candidate, f: ConflictField, covered: set[tuple[str, str, str]]) -> bool:
    return (cand.a.id, cand.b.source.doc_id, str(f)) in covered or (
        cand.b.id,
        cand.a.source.doc_id,
        str(f),
    ) in covered


def _confirm_all(
    cands: list[Candidate],
    case_id: str,
    skip: set[str],
    out: dict[str, Contradiction],
    source_text: SourceText | None,
    doc_name: Callable[[str], str] | None,
) -> None:
    # (event id, other document id, field) already reported in this run. Candidates arrive best
    # first, so a weaker pair repeating the same event-vs-document disagreement is dropped.
    covered: set[tuple[str, str, str]] = set()
    for cand in cands:
        ids = sorted([cand.a.id, cand.b.id])
        if contradiction_id(ids, cand.field) in skip:
            continue
        contexts = (_context(cand.a, source_text), _context(cand.b, source_text))
        try:
            verdict = _confirm(cand, source_text, doc_name, contexts)
        except Exception as exc:
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, exc)
            continue
        if verdict is None:
            continue
        confirmed = verdict.is_contradiction and verdict.same_subject
        log.info(
            "conflict verdict %s suggested=%s score=%.1f same_subject=%s contradiction=%s "
            "fields=%s: %s",
            ids,
            cand.field,
            cand.score,
            verdict.same_subject,
            verdict.is_contradiction,
            [str(f) for f in verdict.fields()],
            (verdict.explanation or verdict.comparison).strip()[:300],
        )
        if not confirmed:
            continue
        explanation = (verdict.explanation or verdict.comparison).strip()
        for f in _final_fields(verdict, cand.a, cand.b, " ".join(contexts)):
            cid = contradiction_id(ids, f)
            if cid in skip or cid in out:
                continue
            if _redundant(cand, f, covered):
                log.info("conflict %s (%s) already covered for these documents; skipped", ids, f)
                continue
            covered.add((cand.a.id, cand.b.source.doc_id, str(f)))
            covered.add((cand.b.id, cand.a.source.doc_id, str(f)))
            out[cid] = Contradiction(
                id=cid, case_id=case_id, event_ids=ids, field=f, explanation=explanation
            )


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
