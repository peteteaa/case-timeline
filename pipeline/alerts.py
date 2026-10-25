"""Alert rules (SPEC §8.4), computed at the end of each job without the LLM.

STUB — signature is the contract; the pipeline owner fills in the body.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.schema import Alert, Contradiction
from pipeline.store import Store


def compute(
    store: Store,
    case_id: str,
    *,
    job_id: str | None,
    new_fact_ids: Sequence[str],
    contradictions: Sequence[Contradiction],
    doc_ids: Sequence[str],
) -> list[Alert]:
    """Write and return this job's alerts (contradiction, flagged_witness, possible_deadline)."""
    raise NotImplementedError
