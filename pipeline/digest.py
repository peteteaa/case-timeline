"""Daily digest: the day's changes as text with fact IDs (MCP ``daily_digest`` and the UI's
"Today" panel share this function).

STUB — signature is the contract; the pipeline owner fills in the body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pipeline.store import Store


def daily_digest(store: Store, case_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    """``{"case_id", "since", "text", "counts": {...}, "fact_ids": [...]}`` for the last 24h."""
    raise NotImplementedError
