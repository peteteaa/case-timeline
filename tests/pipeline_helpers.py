"""Builders shared by the pipeline stage tests (no model, no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from pipeline.llm import hashed_embedding
from pipeline.schema import (
    Document,
    EvidenceLink,
    Fact,
    Line,
    Page,
    ParsedDocument,
)
from pipeline.store import Store

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_doc(doc_id: str = "D-0001", filename: str = "memo.txt", case_id: str = "demo") -> Document:
    return Document(
        doc_id=doc_id,
        case_id=case_id,
        filename=filename,
        sha256="0" * 64,
        location="evidence",
        kind="other",
        pages=None,
        parse_status="parsed",
        added_at=NOW,
    )


def make_parsed(
    text: str,
    pages: list[Page] | None = None,
    lines: list[Line] | None = None,
    *,
    doc_id: str = "D-0001",
    filename: str = "memo.txt",
) -> ParsedDocument:
    return ParsedDocument(
        doc=make_doc(doc_id, filename), text=text, pages=pages or [], lines=lines or []
    )


def make_fact(
    fact_id: str,
    title: str,
    d: date | None = date(2024, 3, 3),
    *,
    end: date | None = None,
    certainty: str | None = None,
    witnesses: list[str] | None = None,
    docs: list[str] | None = None,
    quotes: list[str] | None = None,
    details: str | None = None,
    review: str = "draft",
    date_text: str | None = None,
) -> Fact:
    quotes = quotes or []
    docs = docs or ["D-0001"]
    evidence = [
        EvidenceLink(
            evidence_id=f"E-{fact_id}-{i}",
            kind="span",
            doc_id=docs[i % len(docs)],
            char_start=0,
            char_end=len(q),
            quote=q,
            origin="ai",
            review="draft",
        )
        for i, q in enumerate(quotes)
    ] or [
        EvidenceLink(
            evidence_id=f"E-{fact_id}-{i}", kind="document", doc_id=doc, origin="ai", review="draft"
        )
        for i, doc in enumerate(docs)
    ]
    return Fact(
        fact_id=fact_id,
        case_id="demo",
        title=title,
        details=details,
        date_text=date_text,
        date_start=d,
        date_end=end or d,
        date_certainty=certainty or ("exact" if d else "unknown"),
        witness_ids=witnesses if witnesses is not None else [],
        evidence=evidence,
        origin="ai",
        review=review,
        confidence=0.9,
        created_by="agent",
        created_at=NOW,
        updated_by="agent",
        updated_at=NOW,
    )


def fake_embedder(texts: list[str]) -> list[list[float]]:
    return [hashed_embedding(t) for t in texts]


def new_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "case.db", embedder=fake_embedder)


def add_doc(
    store: Store,
    text: str,
    *,
    case_id: str = "demo",
    filename: str = "memo.txt",
    pages: list[Page] | None = None,
    lines: list[Line] | None = None,
) -> str:
    """Register a parsed document with ``text``; returns its doc id."""
    import hashlib

    sha = hashlib.sha256(f"{filename}\x00{text}".encode()).hexdigest()
    doc, _ = store.register_document(
        case_id, filename=filename, rel_path=filename, sha256=sha, location="evidence"
    )
    store.save_parsed_text(doc.doc_id, text, pages or [], lines or [])
    store.set_parse_result(doc.doc_id, "parsed", pages=len(pages or []) or None)
    return doc.doc_id
