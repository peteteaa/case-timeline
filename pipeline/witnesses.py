"""Match people named in drafts to witness records; unmatched names become suggestions.

STUB — signature is the contract; the pipeline owner fills in the body.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.store import Store


def match_people(store: Store, case_id: str, names: Sequence[str]) -> list[str]:
    """Witness ids for ``names`` (by name/alias); creates suggested witnesses as needed."""
    raise NotImplementedError
