"""Open contradictions between facts (SPEC §6.4): cheap candidate rules, then LLM confirmation.

Candidate rules, over non-rejected facts, for pairs with at least one fact the job touched:
- **date**: similar facts sharing people (or one side naming nobody) whose evidence comes from
  different documents but whose dates don't overlap (more than a day apart, or a day apart
  when both dates are exact);
- **actor**: similar facts on overlapping dates whose people are disjoint;
- **fact**: facts on overlapping dates sharing people, worded differently, where one negates
  ("never", "denied", "did not", ...).
Each candidate is confirmed with the local LLM; confirmed ones are opened in the store. A
reviewer then confirms (facts become disputed) or dismisses them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from pipeline import prompts
from pipeline.merge import fact_similarity, gap_days, interval
from pipeline.schema import ConflictField, Contradiction, Fact
from pipeline.store import Store

log = logging.getLogger(__name__)

SIMILAR_THRESHOLD = 70.0
STRONG_SIMILAR = 85.0  # date rule when one side names nobody
MAX_CANDIDATES = 50
MAX_QUOTES = 3

NEGATION_CUES = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|without|denie[sd]|deny|denying|"
    r"refused?|refuses|absent|failed|cannot|can't|didn't|did\s+not|wasn't|weren't|isn't|"
    r"aren't|hasn't|haven't|hadn't|won't|wouldn't|couldn't|don't|doesn't)\b",
    re.IGNORECASE,
)


class ConflictVerdict(BaseModel):
    """LLM confirmation of a candidate contradiction."""

    is_contradiction: bool
    field: ConflictField
    explanation: str


@dataclass(frozen=True)
class Candidate:
    """A cheap-rule candidate pair awaiting LLM confirmation."""

    a: Fact
    b: Fact
    field: ConflictField
    score: float  # higher = more suspicious; used for capping


def _docs(f: Fact) -> frozenset[str]:
    return frozenset(e.doc_id for e in f.evidence if e.doc_id)


def _quotes(f: Fact) -> list[str]:
    return [e.quote for e in f.evidence if e.quote][:MAX_QUOTES]


def _has_negation(f: Fact) -> bool:
    return any(NEGATION_CUES.search(t) for t in [f.title, f.details or "", *_quotes(f)])


def classify(a: Fact, b: Fact) -> Candidate | None:
    """Apply the date / actor / fact candidate rules to one pair."""
    gap = gap_days(interval(a.date_start, a.date_end), interval(b.date_start, b.date_end))
    if gap is None:
        return None
    pa, pb = set(a.witness_ids), set(b.witness_ids)
    sim = fact_similarity(a.title, a.details, b.title, b.details)
    both_exact = a.date_certainty == "exact" and b.date_certainty == "exact"
    if gap > 1 or (gap == 1 and both_exact):
        # A one-day gap only counts when both dates are stated exactly ("March 11" vs "12").
        shared = bool(pa & pb) or ((not pa or not pb) and sim >= STRONG_SIMILAR)
        if shared and sim >= SIMILAR_THRESHOLD and _docs(a) != _docs(b):
            return Candidate(a, b, "date", sim + min(gap, 30))
        return None
    if gap != 0:
        return None
    if sim >= SIMILAR_THRESHOLD and pa and pb and not (pa & pb):
        return Candidate(a, b, "actor", sim)
    if pa & pb and sim < SIMILAR_THRESHOLD and (_has_negation(a) or _has_negation(b)):
        return Candidate(a, b, "fact", 100.0 - sim)
    return None


def candidate_pairs(
    facts: Sequence[Fact],
    focus_ids: set[str],
    *,
    max_candidates: int = MAX_CANDIDATES,
    skip_pairs: set[frozenset[str]] | None = None,
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
            if cand := classify(a, b):
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
    return found


def _resolved(f: Fact) -> str:
    if f.date_start is None:
        return "undated"
    if f.date_end and f.date_end != f.date_start:
        return f"{f.date_start.isoformat()} to {f.date_end.isoformat()} ({f.date_certainty})"
    return f"{f.date_start.isoformat()} ({f.date_certainty})"


def _fields(f: Fact, suffix: str, names: dict[str, str]) -> dict[str, str]:
    quotes = "; ".join(f'"{q}"' for q in _quotes(f)) or "none"
    return {
        f"fact_{suffix}": f.fact_id,
        f"docs_{suffix}": ", ".join(sorted(_docs(f))) or "no document",
        f"title_{suffix}": f.title,
        f"details_{suffix}": f.details or "none",
        f"date_{suffix}": f.date_text or "none",
        f"resolved_{suffix}": _resolved(f),
        f"people_{suffix}": ", ".join(names.get(w, w) for w in f.witness_ids) or "none",
        f"quotes_{suffix}": quotes,
    }


def confirm(cand: Candidate, names: dict[str, str]) -> ConflictVerdict:
    """Ask the local LLM whether a candidate is a real contradiction."""
    from pipeline import llm

    user = prompts.CONFLICT_USER.format(
        field=cand.field, **_fields(cand.a, "a", names), **_fields(cand.b, "b", names)
    )
    result = llm.chat_json(prompts.CONFLICT_SYSTEM, user, ConflictVerdict)
    if isinstance(result, ConflictVerdict):
        return result
    if isinstance(result, BaseModel):
        return ConflictVerdict.model_validate(result.model_dump())
    return ConflictVerdict.model_validate(result)


def detect(
    store: Store, case_id: str, fact_ids: Sequence[str], *, max_candidates: int = MAX_CANDIDATES
) -> list[Contradiction]:
    """Find, confirm and open contradictions touching ``fact_ids``; returns the new ones.

    Pairs that already have a contradiction (any status, any field) are not re-checked, so a
    dismissed conflict stays dismissed. A failed confirmation call skips that candidate.
    """
    focus = set(fact_ids)
    if not focus:
        return []
    facts = store.list_facts(case_id)  # drafts and accepted; rejected facts are excluded
    known = {frozenset(c.fact_ids) for c in store.list_contradictions(case_id)}
    names = {w.witness_id: w.name for w in store.list_witnesses(case_id)}
    opened: list[Contradiction] = []
    for cand in candidate_pairs(facts, focus, max_candidates=max_candidates, skip_pairs=known):
        ids = [cand.a.fact_id, cand.b.fact_id]
        try:
            verdict = confirm(cand, names)
        except Exception as exc:
            log.warning("conflict check failed for %s (%s): %s", ids, cand.field, exc)
            continue
        if not verdict.is_contradiction:
            continue
        explanation = verdict.explanation.strip() or f"Possible {verdict.field} conflict."
        contradiction, created = store.add_contradiction(case_id, ids, verdict.field, explanation)
        if created:
            opened.append(contradiction)
    log.info("contradictions opened: %d", len(opened))
    return opened
