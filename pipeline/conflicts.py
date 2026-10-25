"""Open contradictions between facts (SPEC §6.4): cheap candidate rules, then LLM confirmation.

A fact can cite several documents (merged evidence), so the rules compare documents, not just
facts: a pair is checked only when the facts' sources differ (a document is not checked against
itself). Candidate rules, over non-rejected facts, for pairs with at least one fact the job
touched; "linked" means the facts share a person (names compare loosely; a shared organization
alone doesn't link facts) and "topic" is the overlap of content words (names and dates removed):

- **date**: linked facts (or, when one names nobody, near-identical ones) more than a day apart
  (a day apart when both dates are exact) that look like one occurrence: similar wording, a
  shared topic word within two weeks, or a strong topic overlap within two months. A plan or
  schedule ("set for", "rescheduled") is never compared with an occurrence: things get moved.
  When either fact is about someone's knowledge ("first learned"), the candidate is "fact".
- **actor**: facts on overlapping dates that are similar but name disjoint people, or that share
  a person and a topic word but name different people;
- **fact**: linked facts on overlapping dates where one negates ("never", "denied", ...) and
  the wording differs or the topic is shared; and linked facts where one is undated but the topic
  overlap is strong.

Each candidate is confirmed by the local LLM, which sees every quote of both facts with the
surrounding source text; deterministic checks then veto impossible fields (a "date" conflict
needs two non-overlapping resolved dates; an "actor" conflict between linked facts needs someone
to exclude a person). A disagreement already opened (or dismissed) through another pair of facts
about the same occurrences is not opened again. A reviewer then confirms (facts become disputed)
or dismisses each contradiction.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline import prompts
from pipeline.merge import (
    Person,
    fact_similarity,
    fact_text,
    gap_days,
    interval,
    overlap,
    people_of,
    proper_words,
    shares_person,
    topic_tokens,
    unmatched,
)
from pipeline.schema import ConflictField, Contradiction, Fact
from pipeline.store import Store

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0
STRONG_SIMILAR = 85.0  # date rule when one side names nobody
MAX_CANDIDATES = 50
MAX_QUOTES = 3  # evidence quotes per fact shown to the model
MAX_GAP_DAYS = 62  # facts further apart than this are rarely one misdated occurrence
NEAR_DAYS = 14  # within this many days, one shared topic word is enough for an LLM check
TOPIC_MIN = 0.3  # topic overlap for dated pairs further apart than NEAR_DAYS
TOPIC_MIN_UNDATED = 0.4  # stricter when one fact is undated (no date evidence)
SAME_OCCURRENCE_TOPIC = 0.5  # two facts on one day this alike are one occurrence (dedupe)
CONTEXT_BEFORE = 300  # characters of source text shown around each quote
CONTEXT_AFTER = 400
VERDICT_MAX_TOKENS = 1200

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|without|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)
# A disagreement about *who* was there needs one source to exclude someone affirmatively;
# silence is not exclusion. One of these cues must appear in either fact's quotes or context.
EXCLUSION_CUES = re.compile(
    r"\b(only|nobody|no\s+one|no-one|none|alone|nor|neither|never|not|no|absent|"
    r"wasn't|weren't|didn't|denie[sd]|deny|except|besides|other\s+than|else)\b",
    re.IGNORECASE,
)
# A plan or schedule ("set for", "due by"). A plan compared with what actually happened (or
# with a rescheduling) is not a date contradiction, so only plan-vs-plan or fact-vs-fact pairs
# qualify.
PLAN_CUES = re.compile(
    r"\b(schedul\w*|reschedul\w*|set\s+for|due\s+(?:by|on)|deadline|planned|plans?\s+to|"
    r"expected\s+to|will|postpon\w*|adjourn\w*|moved\s+(?:from|to))\b",
    re.IGNORECASE,
)
# Someone's knowledge or awareness ("first learned", "became aware"). When one side is about
# when someone knew of a matter, a timing disagreement is a FACT (notice) conflict about that
# knowledge, not two dates for one occurrence.
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
        out: list[ConflictField] = [self.field]
        out += [f for f in self.other_fields if f not in out and f != "fact"]
        return out


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Fact
    b: Fact
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


# --------------------------------------------------------------------------- fact features


def _evidence(f: Fact) -> list[Any]:
    return [e for e in f.evidence if e.review != "rejected"]


def _docs(f: Fact) -> frozenset[str]:
    return frozenset(e.doc_id for e in _evidence(f) if e.doc_id)


def _quotes(f: Fact) -> list[str]:
    return [e.quote for e in _evidence(f) if e.quote]


def _text(f: Fact) -> str:
    return " ".join([f.title, f.details or "", *_quotes(f)])


def _is_plan(f: Fact) -> bool:
    return bool(PLAN_CUES.search(_text(f)))


def _is_knowledge(f: Fact) -> bool:
    return bool(KNOWLEDGE_CUES.search(_text(f)))


def _has_negation(f: Fact) -> bool:
    return bool(NEGATION_CUES.search(_text(f)))


def _persons(f: Fact, witness_names: Mapping[str, Sequence[str]] | None) -> list[Person]:
    return people_of(f.witness_ids, witness_names)


def _people_only(ps: Sequence[Person]) -> list[Person]:
    return [p for p in ps if not p.is_org]


def linked(pa: Sequence[Person], pb: Sequence[Person]) -> bool:
    """The facts share someone; a shared person is required when both name people, so a
    common employer or agency alone does not link two facts."""
    qa, qb = _people_only(pa), _people_only(pb)
    if qa and qb:
        return shares_person(qa, qb)
    return shares_person(pa, pb)


def topic_overlap(
    a: Fact, b: Fact, witness_names: Mapping[str, Sequence[str]] | None = None
) -> tuple[float, int]:
    """Overlap coefficient (0..1) of the two facts' topic words, and the shared-word count.

    Names and date expressions are excluded so that the score reflects *what* happened
    (meeting, dismissal, injury...) rather than who or when.
    """
    texts = (_text(a), _text(b))
    names: list[str] = [*proper_words(*texts)]
    for w in {*a.witness_ids, *b.witness_ids}:
        names += list((witness_names or {}).get(w, ()))
    return overlap(topic_tokens(texts[0], names), topic_tokens(texts[1], names))


# --------------------------------------------------------------------------- candidates


def classify(
    a: Fact, b: Fact, witness_names: Mapping[str, Sequence[str]] | None = None
) -> Candidate | None:
    """Apply the date / actor / fact candidate rules to one pair."""
    da, db = _docs(a), _docs(b)
    if da == db:
        return None  # one source is not checked against itself
    gap = gap_days(interval(a.date_start, a.date_end), interval(b.date_start, b.date_end))
    pa, pb = _persons(a, witness_names), _persons(b, witness_names)
    link = linked(pa, pb)
    sim = fact_similarity(a.title, a.details, b.title, b.details)
    topic, shared = topic_overlap(a, b, witness_names)

    if gap is None:
        # One side is undated: only a strong same-subject signal is worth an LLM call
        # (e.g. "learned of it only after the hearing" vs a dated written notice).
        if link and shared >= 2 and topic >= TOPIC_MIN_UNDATED and (a.date_start or b.date_start):
            return Candidate(a, b, "fact", 100.0 * topic)
        return None

    both_exact = a.date_certainty == "exact" and b.date_certainty == "exact"
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("March 11" vs "12").
        strong = (not pa or not pb) and sim >= STRONG_SIMILAR
        if not (link or strong) or _is_plan(a) != _is_plan(b):
            return None
        near = gap <= NEAR_DAYS and shared >= 1
        if sim >= SIMILAR_THRESHOLD or near or (gap <= MAX_GAP_DAYS and topic >= TOPIC_MIN):
            closeness = max(0.0, 1.0 - gap / MAX_GAP_DAYS)
            kind: ConflictField = "fact" if _is_knowledge(a) or _is_knowledge(b) else "date"
            return Candidate(a, b, kind, 100.0 * max(topic, sim / 100) + closeness)
        return None
    if gap != 0:
        return None

    qa, qb = _people_only(pa), _people_only(pb)
    if qa and qb and not link and sim >= SIMILAR_THRESHOLD:
        return Candidate(a, b, "actor", sim)
    if link and shared >= 1 and (unmatched(qa, qb) or unmatched(qb, qa)):
        # Same occurrence, different people named: an omission or a real disagreement about
        # who was there; the LLM decides using the surrounding source text.
        return Candidate(a, b, "actor", 100.0 * topic)
    negation = _has_negation(a) or _has_negation(b)
    if link and negation and (sim < SIMILAR_THRESHOLD or topic >= TOPIC_MIN):
        return Candidate(a, b, "fact", max(100.0 - sim, 100.0 * topic))
    return None


def candidate_pairs(
    facts: Sequence[Fact],
    focus_ids: set[str],
    *,
    max_candidates: int = MAX_CANDIDATES,
    skip_pairs: set[frozenset[str]] | None = None,
    witness_names: Mapping[str, Sequence[str]] | None = None,
) -> list[Candidate]:
    """Candidates involving at least one fact in ``focus_ids``, capped at ``max_candidates``."""
    skip = skip_pairs or set()
    live = [f for f in facts if f.review != "rejected"]
    found: list[Candidate] = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.fact_id == b.fact_id:
                continue
            if a.fact_id not in focus_ids and b.fact_id not in focus_ids:
                continue
            if frozenset((a.fact_id, b.fact_id)) in skip:
                continue
            if cand := classify(a, b, witness_names):
                found.append(cand)
    found.sort(key=lambda c: (-c.score, c.a.fact_id, c.b.fact_id))
    if len(found) > max_candidates:
        dropped = found[max_candidates:]
        log.warning(
            "conflict candidates capped at %d; dropped %d: %s",
            max_candidates,
            len(dropped),
            [(c.a.fact_id, c.b.fact_id, c.field) for c in dropped],
        )
        found = found[:max_candidates]
    log.info("conflict candidates: %d", len(found))
    return found


# --------------------------------------------------------------------------- same disagreement


def same_occurrence(x: Fact, y: Fact) -> bool:
    """The same fact, or two facts on overlapping dates that read as one occurrence."""
    if x.fact_id == y.fact_id:
        return True
    gap = gap_days(interval(x.date_start, x.date_end), interval(y.date_start, y.date_end))
    if gap != 0:
        return False
    if fact_similarity(x.title, x.details, y.title, y.details) >= SIMILAR_THRESHOLD:
        return True
    names = proper_words(fact_text(x.title, x.details), fact_text(y.title, y.details))
    ta = topic_tokens(fact_text(x.title, x.details), names)
    tb = topic_tokens(fact_text(y.title, y.details), names)
    coeff, shared = overlap(ta, tb)
    return shared >= 1 and coeff >= SAME_OCCURRENCE_TOPIC


def same_disagreement(
    field: str, a: Fact, b: Fact, opened: Sequence[tuple[str, Fact, Fact]]
) -> bool:
    """``a`` vs ``b`` restates a disagreement already opened on ``field`` through other facts:
    each side is the same occurrence as one side of it (e.g. one account dated March 11 against
    a second March 12 account, after the first March 12 account was already flagged)."""
    for f, x, y in opened:
        if f != field or {a.fact_id, b.fact_id} == {x.fact_id, y.fact_id}:
            continue
        if (same_occurrence(a, x) and same_occurrence(b, y)) or (
            same_occurrence(a, y) and same_occurrence(b, x)
        ):
            return True
    return False


# --------------------------------------------------------------------------- source context

_WS = re.compile(r"\s+")
_BOUNDARY = re.compile(r"(?<=[.!?:;])\s+|\n")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _locate(text: str, quote: str, start: int | None, end: int | None) -> tuple[int, int] | None:
    span = (start, end) if start is not None and end is not None else None
    if span is not None and not 0 <= span[0] < span[1] <= len(text):
        span = None
    if span is not None and (not quote.strip() or _norm(text[span[0] : span[1]]) == _norm(quote)):
        return span
    if quote.strip():
        i = text.find(quote)
        if i >= 0:
            return i, i + len(quote)
    return span


def quote_context(
    text: str | None, quote: str, start: int | None = None, end: int | None = None
) -> str:
    """A few sentences of source text around a quote (whitespace collapsed)."""
    if not text:
        return ""
    span = _locate(text, quote, start, end)
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


class _Docs:
    """Cached document text and filenames from the store (best effort)."""

    def __init__(self, store: Store | None) -> None:
        self._store = store
        self._text: dict[str, str | None] = {}
        self._name: dict[str, str] = {}

    def text(self, doc_id: str) -> str | None:
        if doc_id not in self._text:
            parsed = None
            if self._store is not None:
                try:
                    parsed = self._store.get_parsed(doc_id)
                except Exception as exc:
                    log.debug("could not read text of %s: %s", doc_id, exc)
            self._text[doc_id] = parsed.text if parsed is not None else None
        return self._text[doc_id]

    def name(self, doc_id: str) -> str:
        if doc_id not in self._name:
            doc = None
            if self._store is not None:
                try:
                    doc = self._store.get_document(doc_id)
                except Exception:
                    doc = None
            self._name[doc_id] = f"{doc.filename} ({doc_id})" if doc is not None else doc_id
        return self._name[doc_id]


def _shown_evidence(f: Fact) -> list[Any]:
    """Up to MAX_QUOTES quoted evidence links, one per document first."""
    quoted = [e for e in _evidence(f) if e.quote]
    first: list[Any] = []
    seen: set[str | None] = set()
    for e in quoted:
        if e.doc_id not in seen:
            seen.add(e.doc_id)
            first.append(e)
    rest = [e for e in quoted if e not in first]
    return (first + rest)[:MAX_QUOTES]


def _evidence_block(f: Fact, docs: _Docs) -> tuple[str, str]:
    """The prompt's evidence list for one fact, and the raw context text (for cue checks)."""
    lines: list[str] = []
    contexts: list[str] = []
    for e in _shown_evidence(f):
        ctx = quote_context(docs.text(e.doc_id), e.quote, e.char_start, e.char_end)
        contexts.append(ctx)
        lines.append(f'  - document {docs.name(e.doc_id) if e.doc_id else "unknown"}: "{e.quote}"')
        lines.append(f"    surrounding text: <context>{ctx or '(not available)'}</context>")
    if not lines:
        names = ", ".join(docs.name(d) for d in sorted(_docs(f))) or "no document"
        lines.append(f"  - (no quote; cites {names})")
    return "\n".join(lines), " ".join(contexts)


def _resolved(f: Fact) -> str:
    if f.date_start is None:
        return "undated"
    if f.date_end and f.date_end != f.date_start:
        return f"{f.date_start.isoformat()} to {f.date_end.isoformat()} ({f.date_certainty})"
    return f"{f.date_start.isoformat()} ({f.date_certainty})"


def _fields(f: Fact, suffix: str, names: Mapping[str, str], evidence: str) -> dict[str, str]:
    return {
        f"fact_{suffix}": f.fact_id,
        f"title_{suffix}": f.title,
        f"details_{suffix}": f.details or "none",
        f"date_{suffix}": f.date_text or "none",
        f"resolved_{suffix}": _resolved(f),
        f"people_{suffix}": ", ".join(names.get(w, w) for w in f.witness_ids) or "none",
        f"evidence_{suffix}": evidence,
    }


# --------------------------------------------------------------------------- confirmation


def confirm(
    cand: Candidate, names: Mapping[str, str], docs: _Docs | None = None
) -> tuple[ConflictVerdict, str]:
    """Ask the local LLM whether a candidate is a real contradiction.

    Returns the verdict and the surrounding source text both facts were shown with.
    """
    from pipeline import llm

    docs = docs or _Docs(None)
    ev_a, ctx_a = _evidence_block(cand.a, docs)
    ev_b, ctx_b = _evidence_block(cand.b, docs)
    user = prompts.CONFLICT_USER.format(
        field=cand.field, **_fields(cand.a, "a", names, ev_a), **_fields(cand.b, "b", names, ev_b)
    )
    result = llm.chat_json(
        prompts.CONFLICT_SYSTEM, user, ConflictVerdict, max_tokens=VERDICT_MAX_TOKENS
    )
    if not isinstance(result, ConflictVerdict):
        data = result.model_dump() if isinstance(result, BaseModel) else result
        result = ConflictVerdict.model_validate(data)
    return result, f"{ctx_a} {ctx_b}"


def final_fields(
    verdict: ConflictVerdict,
    a: Fact,
    b: Fact,
    context: str = "",
    witness_names: Mapping[str, Sequence[str]] | None = None,
) -> list[ConflictField]:
    """Fields to open after deterministic sanity checks on the model's answer.

    - DATE needs two resolved dates; when either side has none the disagreement is reported as
      FACT. Two resolved dates that overlap cannot be a date contradiction, so DATE is dropped
      then. When either fact is about someone's knowledge ("first learned of it in mid-March"),
      DATE becomes FACT.
    - When the facts share a person, ACTOR needs an exclusion cue ("only", "no one else", "was
      not there") in either fact's quotes or surrounding text: a source that merely omits a
      person is not a conflict. (The same act attributed to different people needs none.)
    """
    if not (verdict.is_contradiction and verdict.same_subject):
        return []
    out: list[ConflictField] = []
    for f in verdict.fields():
        if f == "date":
            gap = gap_days(interval(a.date_start, a.date_end), interval(b.date_start, b.date_end))
            if gap is None or _is_knowledge(a) or _is_knowledge(b):
                f = "fact"
            elif gap == 0:
                continue
        if f == "actor":
            pa, pb = _persons(a, witness_names), _persons(b, witness_names)
            cues = " ".join([*_quotes(a), *_quotes(b), context])
            if linked(pa, pb) and not EXCLUSION_CUES.search(cues):
                continue
        if f not in out:
            out.append(f)
    return out


def detect(
    store: Store, case_id: str, fact_ids: Sequence[str], *, max_candidates: int = MAX_CANDIDATES
) -> list[Contradiction]:
    """Find, confirm and open contradictions touching ``fact_ids``; returns the new ones.

    Pairs that already have a contradiction (any status, any field) are not re-checked, so a
    dismissed conflict stays dismissed, and a pair that restates an existing contradiction's
    disagreement through other facts about the same occurrences is skipped. A failed
    confirmation call skips that candidate.
    """
    focus = set(fact_ids)
    if not focus:
        return []
    facts = store.list_facts(case_id)  # drafts and accepted; rejected facts are excluded
    by_id = {f.fact_id: f for f in facts}
    existing = store.list_contradictions(case_id)
    known = {frozenset(c.fact_ids) for c in existing}
    witnesses = store.list_witnesses(case_id)
    names = {w.witness_id: w.name for w in witnesses}
    witness_names = {w.witness_id: [w.name, *w.aliases] for w in witnesses}
    # Disagreements already on record (any status), as (field, fact, fact).
    opened_pairs: list[tuple[str, Fact, Fact]] = []
    for c in existing:
        fs = [by_id[i] for i in c.fact_ids if i in by_id]
        opened_pairs += [(c.field, x, y) for i, x in enumerate(fs) for y in fs[i + 1 :]]
    docs = _Docs(store)
    opened: list[Contradiction] = []
    cands = candidate_pairs(
        facts, focus, max_candidates=max_candidates, skip_pairs=known, witness_names=witness_names
    )
    for cand in cands:
        ids = [cand.a.fact_id, cand.b.fact_id]
        if same_disagreement(cand.field, cand.a, cand.b, opened_pairs):
            log.info("conflict %s (%s) restates an open disagreement; skipped", ids, cand.field)
            continue
        try:
            verdict, context = confirm(cand, names, docs)
        except Exception as exc:
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, exc)
            continue
        log.info(
            "conflict verdict %s suggested=%s score=%.1f same_subject=%s contradiction=%s "
            "fields=%s: %s",
            ids,
            cand.field,
            cand.score,
            verdict.same_subject,
            verdict.is_contradiction,
            verdict.fields(),
            (verdict.explanation or verdict.comparison).strip()[:300],
        )
        explanation = (verdict.explanation or verdict.comparison).strip()
        for field in final_fields(verdict, cand.a, cand.b, context, witness_names):
            if same_disagreement(field, cand.a, cand.b, opened_pairs):
                log.info("conflict %s (%s) restates an open disagreement; skipped", ids, field)
                continue
            contradiction, created = store.add_contradiction(
                case_id, ids, field, explanation or f"Possible {field} conflict."
            )
            if created:
                opened.append(contradiction)
                opened_pairs.append((field, cand.a, cand.b))
    log.info("contradictions opened: %d (from %d candidates)", len(opened), len(cands))
    return opened
