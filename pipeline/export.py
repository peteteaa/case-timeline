"""Exports (SPEC §7): XLSX/CSV of the current list, SVG/PNG of the visual timeline.

STUB — signatures are the contract; the export owner fills in the bodies. Every function takes
the facts already filtered/sorted by the caller (the view's query), so exports respect it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.schema import Fact
from pipeline.store import Store

# Column key -> header, in the list view's default on-screen order.
COLUMNS: dict[str, str] = {
    "date": "Date",
    "title": "Title",
    "statuses": "Statuses",
    "review": "Review",
    "evidence": "Evidence",
    "tags": "Tags",
    "witnesses": "Witnesses",
    "notes": "Notes",
}
DEFAULT_COLUMNS: tuple[str, ...] = tuple(COLUMNS)
ZOOMS = ("day", "week", "month", "year")


def to_csv(store: Store, case_id: str, facts: Sequence[Fact],
           columns: Sequence[str] = DEFAULT_COLUMNS) -> bytes:
    raise NotImplementedError


def to_xlsx(store: Store, case_id: str, facts: Sequence[Fact],
            columns: Sequence[str] = DEFAULT_COLUMNS) -> bytes:
    raise NotImplementedError


def to_svg(store: Store, case_id: str, facts: Sequence[Fact], zoom: str = "month") -> str:
    raise NotImplementedError


def to_png(store: Store, case_id: str, facts: Sequence[Fact], zoom: str = "month") -> bytes:
    raise NotImplementedError
