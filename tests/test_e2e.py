"""End-to-end: scan the demo fixtures with a deterministic fake LLM and check the whole pipeline.

The fake model returns canned drafts per document whose quotes are real sentences from the
fixtures (plus one invented quote that verification must reject).
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import pytest

from pipeline import jobs, llm
from pipeline.config import Settings
from pipeline.digest import daily_digest
from pipeline.extract import ExtractedFact, ExtractionResult
from pipeline.store import Store
from tests.pipeline_helpers import fake_embedder

FIXTURES = Path(__file__).parent / "fixtures" / "case_demo"
MEETING = "Meeting in Mark Ellison's office"


def F(title, date_text, people, *quotes, deadline=False, confidence=0.8):  # noqa: N802
    return ExtractedFact(
        title=title,
        date_text=date_text,
        people=people,
        quotes=list(quotes),
        deadline_language=deadline,
        confidence=confidence,
    )


CANNED: dict[str, list[ExtractedFact]] = {
    "deposition_excerpt_ellison.txt": [
        F(
            MEETING,
            "March 12, 2024",
            ["Mark Ellison", "Dana Rivera", "Priya Nand"],
            "Q. When did your meeting with Ms. Rivera take place?",
            "A. March 12, 2024. In the afternoon, I believe.",
        ),
    ],
    "email_thread.eml": [
        F(
            "Rivera reports failing forklift brakes to Ellison",
            "March 4, 2024",
            ["Dana Rivera", "Mark Ellison"],
            "On March 4, 2024 I reported in writing to Mark Ellison that the brakes on forklift "
            "#7 were failing.",
        ),
    ],
    "hr_memo.md": [
        F(
            MEETING,
            "March 12, 2024",
            ["Priya Nand", "Mark Ellison", "Dana Rivera"],
            "On March 12, 2024, I attended a meeting between Mark Ellison and Dana Rivera "
            "regarding attendance concerns.",
        ),
        F(
            "Rivera placed on unpaid administrative leave",
            "March 14, 2024",
            ["Ms. Rivera"],
            "Ms. Rivera was placed on unpaid administrative leave on March 14, 2024 pending "
            "review.",
        ),
        F(
            "Deadline to appeal termination",
            "May 6, 2024",
            ["Ms. Rivera"],
            "The appeal must be received by May 6, 2024.",
            deadline=True,
        ),
        F(
            "Company position statement due",
            "July 1, 2024",
            [],
            "The company's position statement is due by July 1, 2024.",
            deadline=True,
        ),
    ],
    "incident_report.pdf": [
        F(
            MEETING,
            "March 12, 2024",
            ["Ms. Rivera", "Mr. Ellison"],
            "Mr. Ellison stated that the March 12, 2024 meeting concerned Ms. Rivera's attendance",
        ),
        F(
            "Police respond to parking lot altercation",
            "March 14, 2024",
            ["Officer Jamie Ortiz", "Dana Rivera", "Mark Ellison"],
            "On March 14, 2024 at approximately 5:40 PM, Officer Jamie Ortiz responded",
        ),
        F(
            "Police interview Becker by telephone",
            "March 15, 2024",
            ["Tom Becker", "Officer Ortiz"],
            "On March 15, 2024, Officer Ortiz conducted a follow-up telephone interview with Tom "
            "Becker",
        ),
        F(
            "Ellison apologizes to Rivera",
            "March 20, 2024",
            ["Mark Ellison"],
            "Mr. Ellison apologized to Ms. Rivera on March 20, 2024.",
        ),  # invented: rejected
    ],
    "witness_statement_becker.docx": [
        F(
            MEETING,
            "March 11, 2024",
            ["Tom Becker", "Mark Ellison", "Dana Rivera"],
            "The meeting in Mark Ellison's office took place on March 11, 2024.",
        ),
        F(
            "Rivera sent home without pay",
            "Two days after the meeting",
            ["Dana"],
            "Two days after the meeting, Dana was sent home without pay.",
        ),
        F(
            "Becker affirms his statement",
            "",
            ["Tom Becker"],
            "I make this statement freely and believe it to be true.",
        ),
    ],
    "note_from_upload.txt": [
        F(
            "Becker calls HR",
            "April 3, 2024",
            ["Tom Becker"],
            "Tom Becker called HR on April 3, 2024.",
        ),
    ],
}


def fake_chat_json(system, user, schema, **kw):
    if schema is ExtractionResult:
        filename = re.search(r"^Document: (.+)$", user, re.M).group(1)
        return ExtractionResult(facts=CANNED.get(filename, []))
    m = re.search(r"Suspected conflict type: (\w+)", user)
    return schema(is_contradiction=True, field=m.group(1), explanation="Sources disagree.")


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("CT_DISABLE_DOCLING", "1")
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(llm, "embed", fake_embedder)
    return Settings(evidence_dir=FIXTURES, data_dir=tmp_path / "data", case_id="demo")


def _snapshot(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_full_scan_end_to_end(settings: Settings) -> None:
    before = _snapshot(FIXTURES)
    job = jobs.run_now("demo", settings=settings)
    assert job.status == "done", job.error
    c = job.counts
    assert (c["docs_seen"], c["docs_new"], c["docs_parsed"], c["docs_failed"]) == (5, 5, 5, 0)
    assert c["drafts"] == 13
    assert (c["quotes_checked"], c["quotes_rejected"], c["drafts_discarded"]) == (14, 1, 1)
    assert c["facts_created"] == 10 and c["evidence_attached"] == 2
    assert c["contradictions"] == 1
    assert c["alerts"] == 3  # 1 contradiction + 2 possible deadlines
    assert job.error is None
    assert _snapshot(FIXTURES) == before  # evidence is read-only

    with Store(settings.db_path) as store:
        docs = {d.filename: d for d in store.list_documents("demo")}
        assert {d.kind for d in docs.values()} >= {"transcript", "email", "other"}
        assert docs["deposition_excerpt_ellison.txt"].kind == "transcript"
        assert docs["deposition_excerpt_ellison.txt"].pages == 2
        facts = store.list_facts("demo")
        assert len(facts) == 10
        assert all(f.review == "draft" and f.origin == "ai" for f in facts)
        by_title: dict[str, list] = {}
        for f in facts:
            by_title.setdefault(f.title, []).append(f)

        # Every piece of evidence is a verbatim span of its document.
        for f in facts:
            assert f.evidence
            for e in f.evidence:
                text = store.get_parsed(e.doc_id).text
                assert text[e.char_start : e.char_end] == e.quote
                assert e.origin == "ai" and e.review == "draft"

        # One March 12 meeting fact with quotes from three documents; Becker's March 11 apart.
        meetings = sorted(by_title[MEETING], key=lambda f: f.date_start)
        becker_meeting, meeting = meetings
        assert becker_meeting.date_start == date(2024, 3, 11)
        assert meeting.date_start == date(2024, 3, 12) and meeting.date_certainty == "exact"
        cited = {e.doc_id for e in meeting.evidence}
        assert cited == {
            docs[n].doc_id
            for n in ("deposition_excerpt_ellison.txt", "hr_memo.md", "incident_report.pdf")
        }
        depo = [
            e for e in meeting.evidence if e.doc_id == docs["deposition_excerpt_ellison.txt"].doc_id
        ]
        assert [(e.page, e.line_start, e.line_end) for e in depo] == [(14, 1, 1), (14, 2, 2)]
        pdf = [e for e in meeting.evidence if e.doc_id == docs["incident_report.pdf"].doc_id]
        assert [e.page for e in pdf] == [1]

        # Relative date anchored on Becker's own meeting date.
        (home,) = by_title["Rivera sent home without pay"]
        assert home.date_start == date(2024, 3, 13) and home.date_certainty == "relative"
        assert home.anchor_fact_id == becker_meeting.fact_id
        (undated,) = by_title["Becker affirms his statement"]
        assert undated.date_start is None and undated.date_certainty == "unknown"
        assert "Ellison apologizes to Rivera" not in by_title

        # Witnesses: suggestions, matched across name variants, no duplicates.
        wits = {w.name: w for w in store.list_witnesses("demo")}
        assert set(wits) == {
            "Mark Ellison",
            "Dana Rivera",
            "Priya Nand",
            "Jamie Ortiz",
            "Tom Becker",
        }
        assert not any(w.confirmed for w in wits.values())
        assert set(meeting.witness_ids) == {
            wits[n].witness_id for n in ("Mark Ellison", "Dana Rivera", "Priya Nand")
        }
        assert home.witness_ids == [wits["Dana Rivera"].witness_id]

        # The planted date contradiction, open for a reviewer.
        (contra,) = store.list_contradictions("demo")
        assert contra.field == "date" and contra.status == "open"
        assert contra.fact_ids == sorted([meeting.fact_id, becker_meeting.fact_id])

        alerts = store.list_alerts("demo", pending_only=True)
        types = sorted(a.type for a in alerts)
        assert types == ["contradiction", "possible_deadline", "possible_deadline"]
        deadline_facts = {
            fid for a in alerts if a.type == "possible_deadline" for fid in a.fact_ids
        }
        assert deadline_facts == {
            by_title["Deadline to appeal termination"][0].fact_id,
            by_title["Company position statement due"][0].fact_id,
        }

        digest = daily_digest(store, "demo")
        assert digest["counts"]["new_drafts"] == 10 and digest["counts"]["contradictions"] == 1
        assert meeting.fact_id in digest["text"]

    # A second scan finds nothing new.
    again = jobs.run_now("demo", settings=settings)
    assert again.status == "done"
    assert again.counts["docs_seen"] == 5 and again.counts["docs_new"] == 0
    assert again.counts["drafts"] == 0 and again.counts["facts_created"] == 0
    assert again.counts["alerts"] == 0

    # A renamed copy uploaded to the inbox is recognized by content and not processed again.
    shutil.copy(FIXTURES / "hr_memo.md", settings.inbox / "memo (copy).md")
    copied = jobs.run_now("demo", settings=settings, trigger="upload")
    assert copied.trigger == "upload"
    assert copied.counts["docs_new"] == 0 and copied.counts["drafts"] == 0

    # A person flags Becker; a new upload that mentions him raises a flagged-witness alert.
    with Store(settings.db_path) as store:
        becker = store.find_witness("demo", "Tom Becker")
        store.update_witness(becker.witness_id, {"flagged": True, "confirmed": True}, by="alice")
    (settings.inbox / "note_from_upload.txt").write_text("Tom Becker called HR on April 3, 2024.\n")
    third = jobs.run_now("demo", settings=settings, trigger="upload")
    assert third.counts["docs_new"] == 1 and third.counts["facts_created"] == 1
    with Store(settings.db_path) as store:
        new_doc = store.get_document_by_sha(
            "demo", hashlib.sha256(b"Tom Becker called HR on April 3, 2024.\n").hexdigest()
        )
        assert new_doc.location == "inbox"
        flagged = [a for a in store.list_alerts("demo") if a.type == "flagged_witness"]
        assert len(flagged) == 1 and flagged[0].doc_ids == [new_doc.doc_id]
        (note_fact,) = store.facts_citing_document(new_doc.doc_id)
        assert note_fact.witness_ids == [becker.witness_id]
        assert flagged[0].fact_ids == [note_fact.fact_id]


@pytest.mark.llm
def test_live_model_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real vLLM: every stored quote is verbatim and at least some facts are drafted."""
    monkeypatch.setattr(llm, "embed", fake_embedder)
    settings = Settings(evidence_dir=FIXTURES, data_dir=tmp_path / "data", case_id="demo")
    job = jobs.run_now("demo", settings=settings)
    assert job.status == "done", job.error
    assert job.counts["facts_created"] > 0
    with Store(settings.db_path) as store:
        for f in store.list_facts("demo"):
            for e in f.evidence:
                assert store.get_parsed(e.doc_id).text[e.char_start : e.char_end] == e.quote
