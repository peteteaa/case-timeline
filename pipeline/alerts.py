"""Alert rules (SPEC §8.4), computed at the end of each job without the LLM.

- ``contradiction``: one alert per contradiction the job opened.
- ``flagged_witness``: one alert per flagged witness mentioned by a document the job processed
  (name or alias in its text, or linked to a fact drafted from it).
- ``possible_deadline``: one alert per new fact dated in the future or whose text uses deadline
  language. The UI always labels these "needs attorney review".

Alerts carry only ids (facts, documents), never case content, so the agent can announce them
outside the UI.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date

from pipeline.schema import Alert, Contradiction, Fact
from pipeline.store import Store

log = logging.getLogger(__name__)

DEADLINE_RE = re.compile(
    r"\b(?:no|not) later than\b"
    r"|\bdue\b(?!\s+(?:to|process)\b)"
    r"|\bbar date\b"
    r"|\bdeadlines?\b"
    r"|\bon or before\b"
    r"|\bmust be (?:filed|received|served|submitted|delivered|postmarked)\b"
    r"|\bstatutes? of limitations?\b"
    r"|\btime[- ]barred\b",
    re.IGNORECASE,
)


def has_deadline_language(text: str) -> bool:
    return bool(DEADLINE_RE.search(text))


def _fact_texts(f: Fact) -> list[str]:
    return [f.title, f.details or "", *(e.quote for e in f.evidence if e.quote)]


def _fact_docs(facts: Sequence[Fact]) -> list[str]:
    out: list[str] = []
    for f in facts:
        for e in f.evidence:
            if e.doc_id and e.doc_id not in out:
                out.append(e.doc_id)
    return out


def _mentions(text: str, names: Sequence[str]) -> bool:
    low = " ".join(text.split()).casefold()
    for name in names:
        n = " ".join(name.split()).casefold()
        if len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", low):
            return True
    return False


def compute(
    store: Store,
    case_id: str,
    *,
    job_id: str | None,
    new_fact_ids: Sequence[str],
    contradictions: Sequence[Contradiction],
    doc_ids: Sequence[str],
    today: date | None = None,
) -> list[Alert]:
    """Write and return this job's alerts (contradiction, flagged_witness, possible_deadline)."""
    today = today or date.today()
    made: list[Alert] = []

    for c in contradictions:
        made.append(
            store.add_alert(
                case_id,
                "contradiction",
                fact_ids=c.fact_ids,
                doc_ids=_fact_docs(store.get_facts(c.fact_ids)),
                job_id=job_id,
            )
        )

    flagged = [w for w in store.list_witnesses(case_id) if w.flagged]
    if flagged and doc_ids:
        texts = {d: (p.text if (p := store.get_parsed(d)) else "") for d in doc_ids}
        citing = {d: store.facts_citing_document(d) for d in doc_ids}
        for w in flagged:
            docs = []
            facts: list[str] = []
            for d in doc_ids:
                linked = [f.fact_id for f in citing[d] if w.witness_id in f.witness_ids]
                if linked or _mentions(texts[d], [w.name, *w.aliases]):
                    docs.append(d)
                    facts += [f for f in linked if f not in facts]
            if docs:
                made.append(
                    store.add_alert(
                        case_id, "flagged_witness", fact_ids=facts, doc_ids=docs, job_id=job_id
                    )
                )

    for f in store.get_facts(list(dict.fromkeys(new_fact_ids))):
        if f.review == "rejected":
            continue
        future = f.date_start is not None and f.date_start > today
        if future or any(has_deadline_language(t) for t in _fact_texts(f)):
            made.append(
                store.add_alert(
                    case_id,
                    "possible_deadline",
                    fact_ids=[f.fact_id],
                    doc_ids=_fact_docs([f]),
                    job_id=job_id,
                )
            )
    if made:
        log.info("job %s: %d alert(s) for case %s", job_id, len(made), case_id)
    return made
