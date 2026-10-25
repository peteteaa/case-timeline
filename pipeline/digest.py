"""Daily digest: the day's changes as text with fact IDs (MCP ``daily_digest`` and the UI's
"Today" panel share this function).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pipeline.store import Store

WINDOW = timedelta(hours=24)
MAX_IDS = 12  # ids listed per line before "and N more"

_ALERT_LABELS = {
    "contradiction": "possible conflict",
    "flagged_witness": "flagged witness mentioned",
    "possible_deadline": "possible deadline (needs attorney review)",
}


def _ids(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:MAX_IDS])
    more = len(ids) - MAX_IDS
    return f"{shown} and {more} more" if more > 0 else shown


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def daily_digest(store: Store, case_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    """``{"case_id", "since", "text", "counts": {...}, "fact_ids": [...]}`` for the last 24h."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    since = now - WINDOW
    act = store.activity_since(case_id, since)

    new_facts = act["facts"]
    drafts = [f.fact_id for f in new_facts if f.review == "draft"]
    by_people = [f.fact_id for f in new_facts if f.origin == "user"]
    reviewed: dict[str, list[str]] = {"accept": [], "edit": [], "reject": []}
    for ev in store.list_review_events(case_id):
        at = ev.at if ev.at.tzinfo else ev.at.replace(tzinfo=UTC)
        if since < at <= now and ev.fact_id not in reviewed[ev.action]:
            reviewed[ev.action].append(ev.fact_id)
    contradictions = act["contradictions"]
    alerts = act["alerts"]
    documents = act["documents"]
    pending_total = store.review_counts(case_id).get("draft", 0)

    lines: list[str] = []
    if drafts:
        lines.append(f"- {_plural(len(drafts), 'new AI draft')} awaiting review: {_ids(drafts)}.")
    if by_people:
        lines.append(f"- {_plural(len(by_people), 'fact')} added by people: {_ids(by_people)}.")
    for action, label in (("accept", "accepted"), ("edit", "edited"), ("reject", "rejected")):
        if reviewed[action]:
            lines.append(
                f"- {_plural(len(reviewed[action]), 'fact')} {label}: {_ids(reviewed[action])}."
            )
    for c in contradictions:
        lines.append(
            f"- New possible conflict {c.contradiction_id} ({c.field}): "
            f"{' vs '.join(c.fact_ids)} [{c.status}]."
        )
    if alerts:
        kinds: dict[str, int] = {}
        for a in alerts:
            kinds[a.type] = kinds.get(a.type, 0) + 1
        parts = ", ".join(f"{n} {_ALERT_LABELS.get(t, t)}" for t, n in sorted(kinds.items()))
        cited = list(dict.fromkeys(fid for a in alerts for fid in a.fact_ids))
        tail = f" (facts {_ids(cited)})" if cited else ""
        lines.append(f"- {_plural(len(alerts), 'alert')}: {parts}{tail}.")
    if documents:
        docs = ", ".join(f"{d.doc_id} {d.filename}" for d in documents[:MAX_IDS])
        more = len(documents) - MAX_IDS
        docs += f" and {more} more" if more > 0 else ""
        lines.append(f"- {_plural(len(documents), 'document')} added: {docs}.")

    header = f"Case {case_id}, last 24 hours (since {since:%Y-%m-%d %H:%M} UTC):"
    if not lines:
        lines.append("- No changes.")
    lines.append(f"{_plural(pending_total, 'draft')} in total awaiting review.")
    text = "\n".join([header, *lines])

    fact_ids = list(
        dict.fromkeys(
            [
                *drafts,
                *by_people,
                *(fid for ids in reviewed.values() for fid in ids),
                *(fid for c in contradictions for fid in c.fact_ids),
                *(fid for a in alerts for fid in a.fact_ids),
            ]
        )
    )
    counts = {
        "new_facts": len(new_facts),
        "new_drafts": len(drafts),
        "new_user_facts": len(by_people),
        "accepted": len(reviewed["accept"]),
        "edited": len(reviewed["edit"]),
        "rejected": len(reviewed["reject"]),
        "contradictions": len(contradictions),
        "alerts": len(alerts),
        "documents": len(documents),
        "drafts_pending": pending_total,
    }
    return {
        "case_id": case_id,
        "since": since.isoformat(),
        "text": text,
        "counts": counts,
        "fact_ids": fact_ids,
    }
