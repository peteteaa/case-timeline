"""Tests for eval.run_eval: SPEC §13 metrics computed from a real temp Store (no LLM)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import run_eval as ev
from pipeline.schema import AGENT, Job, Page, SpanIn
from pipeline.store import Store

CASE = "t"
PERSON = "Ana"
A_TXT = (
    "Smith signed the supply contract on March 3, 2024. "
    "In April 2024 the Acme warehouse flooded. Weather was mild."
)
B_TXT = (
    "Confirming that Smith signed the supply contract on March 3, 2024. "
    "Lee terminated the agreement by letter. Lunch was served."
)
GOLD = [
    {
        "id": "G1",
        "title": "John Smith signed the supply contract",
        "date": "2024-03-03",
        "date_certainty": "exact",
        "doc": "a.txt",
        "event_key": "sign",
        "quote": "Smith signed the supply contract on March 3, 2024",
    },
    {
        "id": "G2",
        "title": "Smith signed the supply contract",
        "date": "2024-03-03",
        "doc": "b.txt",
        "event_key": "sign",
        "quote": "Smith signed the supply contract",
    },
    {
        "id": "G3",
        "title": "Acme warehouse flooded",
        "date": "2024-04-01",
        "date_end": "2024-04-30",
        "date_certainty": "approximate",
        "doc": "a.txt",
        "event_key": "flood",
    },
    {
        "id": "G4",
        "title": "Bob Lee terminated the agreement by letter",
        "date": "2024-05-01",
        "doc": "b.txt",
        "event_key": "terminate",
    },
]
DISPUTES = [
    {
        "id": "D1",
        "label": "disputed",
        "field": "fact",
        "sides": [
            {"doc": "a.txt", "quote": "signed the supply contract on March 3"},
            {"doc": "b.txt", "quote": "Lee terminated the agreement"},
        ],
    },
    {
        "id": "D2",
        "label": "undisputed",
        "sides": [
            {"doc": "a.txt", "quote": "Weather was mild"},
            {"doc": "b.txt", "quote": "Lunch was served"},
        ],
    },
]


def _doc(store: Store, name: str, text: str) -> str:
    doc, _ = store.register_document(
        CASE,
        filename=name,
        rel_path=name,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        location="evidence",
    )
    store.save_parsed_text(doc.doc_id, text, [Page(number=1, char_start=0, char_end=len(text))])
    store.set_parse_result(doc.doc_id, "parsed", pages=1, kind="exhibit")
    return doc.doc_id


def _span(doc_id: str, text: str, quote: str) -> SpanIn:
    i = text.index(quote)
    return SpanIn(doc_id=doc_id, char_start=i, char_end=i + len(quote), quote=quote)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "case.db", embedder=lambda t: [[0.0] * 1024 for _ in t])
    a, b = _doc(s, "a.txt", A_TXT), _doc(s, "b.txt", B_TXT)
    sign = s.add_fact(
        CASE,
        "Smith signed the supply contract",
        by=AGENT,
        date_start="2024-03-03",
        date_certainty="exact",
        spans=[
            _span(a, A_TXT, "Smith signed the supply contract on March 3, 2024."),
            _span(b, B_TXT, "Smith signed the supply contract on March 3, 2024."),
        ],
    )
    flood = s.add_fact(
        CASE,
        "The Acme warehouse flooded",
        by=AGENT,
        date_start="2024-04-15",
        date_certainty="approximate",
        spans=[_span(a, A_TXT, "In April 2024 the Acme warehouse flooded.")],
    )
    term = s.add_fact(  # matches G4, wrong date
        CASE,
        "Lee terminated the agreement",
        by=AGENT,
        date_start="2024-06-01",
        spans=[_span(b, B_TXT, "Lee terminated the agreement by letter.")],
    )
    noise = s.add_fact(
        CASE, "Lunch was served", by=AGENT, spans=[_span(b, B_TXT, "Lunch was served.")]
    )
    s.add_fact(CASE, "Person-made fact about weather", by=PERSON)  # not an AI draft
    s.add_contradiction(CASE, [sign.fact_id, term.fact_id], "fact", "signed vs terminated")
    s.add_contradiction(CASE, [flood.fact_id, noise.fact_id], "date", "noise")

    s.review_fact(sign.fact_id, "accept", reviewer=PERSON)
    s.review_fact(flood.fact_id, "edit", reviewer=PERSON, changes={"details": "basement"})
    s.review_fact(noise.fact_id, "reject", reviewer=PERSON, reason="irrelevant")
    evidence_id = s.get_fact(term.fact_id).evidence[0].evidence_id
    s.review_evidence(evidence_id, "accept", reviewer=PERSON)  # evidence-level: not counted

    job, _ = s.request_job(CASE, "agent")
    s.claim_next_job()
    s.finish_job(
        job.job_id, "done", counts={"quotes_checked": 10, "quotes_rejected": 2, "docs_parsed": 2}
    )
    job2, _ = s.request_job(CASE, "ui")
    s.claim_next_job()
    s.finish_job(job2.job_id, "done", counts={"quotes_checked": 5, "quotes_rejected": 1})
    s.fx = SimpleNamespace(
        sign=sign.fact_id, flood=flood.fact_id, term=term.fact_id, noise=noise.fact_id
    )  # type: ignore[attr-defined]
    yield s
    s.close()


def _gold(tmp_path: Path) -> tuple[Path, Path]:
    g, d = tmp_path / "gold.json", tmp_path / "disputes.json"
    g.write_text(json.dumps(GOLD))
    d.write_text(json.dumps(DISPUTES))
    return g, d


def test_evaluate_all_metrics(store: Store, tmp_path: Path) -> None:
    g, d = _gold(tmp_path)
    res = ev.evaluate(store, CASE, ev.load_gold(g), ev.load_disputes(d))
    fx = store.fx  # type: ignore[attr-defined]

    f = res["facts"]
    assert f["n_pred"] == 4  # AI drafts in every review state; person-made fact excluded
    assert f["n_gold_matched"] == 4 and f["recall"] == 1.0
    assert f["precision"] == 0.75 and f["unmatched_facts"] == [fx.noise]
    assert f["f1"] == pytest.approx(2 * 0.75 / 1.75)
    by_gold = {m["gold_id"]: m for m in f["matches"]}
    assert by_gold["G1"]["fact_id"] == by_gold["G2"]["fact_id"] == fx.sign
    assert by_gold["G3"]["date_correct"] is True  # 2024-04-15 overlaps April
    assert by_gold["G4"]["date_correct"] is False
    assert (f["date_correct"], f["date_scored"]) == (3, 4) and f["date_accuracy"] == 0.75
    assert f["strict_recall"] == 0.75 and f["strict_precision"] == 0.5

    assert res["merge"]["merge_accuracy"] == 1.0 and res["merge"]["scored"] == 1
    assert res["merge"]["events"]["sign"]["facts"] == [fx.sign]

    c = res["contradictions"]
    assert c["n_gold"] == 1 and c["n_pred"] == 2  # undisputed labels don't count
    assert c["precision"] == 0.5 and c["recall"] == 1.0 and c["missed"] == []

    j = res["jobs"]
    assert (j["quotes_checked"], j["quotes_rejected"]) == (15, 3)
    assert j["quote_verification_rate"] == pytest.approx(12 / 15)
    assert j["docs_parsed"] == 2 and j["jobs"] == 2

    r = res["review"]
    assert (r["accept"], r["edit"], r["reject"], r["events"]) == (1, 1, 1, 3)
    assert r["acceptance_rate"] == pytest.approx(2 / 3)


def _pred(fid: str, doc: str, quote: str, title: str = "", d: date | None = None) -> ev.Pred:
    return ev.Pred(
        fact_id=fid,
        title=title or quote,
        details="",
        date_start=d,
        date_end=None,
        review="draft",
        quotes=[(doc, quote)],
    )


def test_unmerged_duplicates_cost_merge_accuracy_and_precision() -> None:
    gold = ev.load_gold_items(
        [
            {"id": "G1", "title": "Smith signed the contract", "doc": "a.txt", "event_key": "k"},
            {"id": "G2", "title": "Smith signed the contract", "doc": "b.txt", "event_key": "k"},
            {"id": "G3", "title": "Only one doc", "doc": "a.txt", "event_key": "solo"},
        ]
    )
    preds = [
        _pred("F1", "a.txt", "Smith signed the contract"),
        _pred("F2", "b.txt", "Smith signed the contract"),
        _pred("F3", "a.txt", "Smith signed the contract"),  # an unmerged duplicate
    ]
    facts = ev.score_facts(gold, preds)
    merges = ev.score_merges(gold, preds, facts.pop("_matches"))
    assert merges["multi_doc_events"] == 1 and merges["merge_accuracy"] == 0.0
    assert merges["events"]["k"]["facts"] == ["F1", "F2"]
    assert facts["precision"] == pytest.approx(2 / 3)  # one of the duplicates is unused


def test_date_rules() -> None:
    exact = ev.GoldFact(id="g", title="x", date=date(2024, 3, 3), date_certainty="exact")
    rng = ev.GoldFact(
        id="g",
        title="x",
        date=date(2024, 3, 1),
        date_end=date(2024, 3, 31),
        date_certainty="approximate",
    )
    undated = ev.GoldFact(id="g", title="x")
    assert ev.date_correct(exact, _pred("F", "", "", d=date(2024, 3, 3))) is True
    assert ev.date_correct(exact, _pred("F", "", "", d=date(2024, 3, 4))) is False
    assert ev.date_correct(exact, _pred("F", "", "")) is False
    span = ev.Pred(
        fact_id="F",
        title="x",
        details="",
        date_start=date(2024, 2, 20),
        date_end=date(2024, 3, 2),
        review="draft",
    )
    assert ev.date_correct(rng, span) is True
    assert ev.date_correct(exact, span) is False  # a range is not an exact date
    assert ev.date_correct(undated, span) is None


def test_contradiction_needs_two_facts_on_two_sides() -> None:
    g = ev.GoldDispute(id="D", sides=(("a.txt", "met on May 1"), ("b.txt", "met on May 2")))
    f1 = _pred("F1", "a.txt", "They met on May 1, 2024.")
    f2 = _pred("F2", "b.txt", "They met on May 2, 2024.")
    f3 = _pred("F3", "a.txt", "We met on May 1 at noon.")
    assert ev.contradiction_matches([f1, f2], g)
    assert not ev.contradiction_matches([f1, f3], g)  # both on the same side
    assert not ev.contradiction_matches([f1], g)


def test_contradiction_side_matches_through_event_key_and_needs_same_field() -> None:
    g = ev.GoldDispute(
        id="D",
        sides=(("a.txt", "met on May 1"), ("b.txt", "met on May 2")),
        field="date",
        event_key="meeting",
    )
    f1 = _pred("F1", "a.txt", "Ellison threatened Rivera.")  # other sentence, same event
    f2 = _pred("F2", "b.txt", "They met on May 2, 2024.")
    assert not ev.contradiction_matches([f1, f2], g)
    events = {("meeting", "a.txt"): {"F1"}}
    assert ev.contradiction_matches([f1, f2], g, events, "date")
    assert not ev.contradiction_matches([f1, f2], g, events, "actor")


def test_contradiction_recall_counts_planted_and_precision_any(store: Store) -> None:
    a = store.add_fact("demo", "A", by=AGENT)
    b = store.add_fact("demo", "B", by=AGENT)
    c = store.add_fact("demo", "C", by=AGENT)
    store.add_contradiction("demo", [a.fact_id, b.fact_id], "date", "derived one")
    dismissed, _ = store.add_contradiction("demo", [a.fact_id, c.fact_id], "date", "noise")
    store.review_contradiction(dismissed.contradiction_id, "dismiss", reviewer="qa")
    planted = ev.GoldDispute(id="P", sides=(("x.txt", "q"), ("y.txt", "r")), field="date")
    derived = ev.GoldDispute(
        id="D", sides=(("x.txt", "q"), ("y.txt", "r")), field="date", planted=False
    )
    by_id = {
        a.fact_id: _pred(a.fact_id, "x.txt", "q"),
        b.fact_id: _pred(b.fact_id, "y.txt", "r"),
        c.fact_id: _pred(c.fact_id, "y.txt", "r"),
    }
    res = ev.score_contradictions(store, "demo", [derived], by_id)
    assert res["n_pred"] == 1 and res["precision"] == 1.0  # the dismissed one doesn't count
    res = ev.score_contradictions(store, "demo", [planted, derived], by_id)
    assert res["n_gold"] == 1 and res["recall"] == 1.0


def test_throughput_from_job_times() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)

    def job(i: int, parsed: int, minutes: int, status: str = "done") -> Job:
        return Job(
            job_id=f"J-{i}",
            case_id=CASE,
            trigger="ui",
            status=status,
            counts={"docs_parsed": parsed},
            error=None,
            queued_at=t0,
            started_at=t0,
            finished_at=t0 + timedelta(minutes=minutes),
        )

    fake = SimpleNamespace(list_jobs=lambda case_id, limit=20: [job(1, 10, 30), job(2, 20, 30)])
    out = ev.score_jobs(fake, CASE)
    assert out["docs_per_hour"] == 30.0 and out["quote_verification_rate"] is None


def test_legacy_gold_fields_load(tmp_path: Path) -> None:
    p = tmp_path / "legacy.json"
    p.write_text(
        json.dumps(
            [
                {
                    "date": None,
                    "actors": ["Dana Rivera"],
                    "description": "Rivera reports brakes",
                    "doc": "x.docx",
                    "certainty": "approximate",
                    "date_range": ["2024-03-01", "2024-03-10"],
                    "event_key": "k",
                    "quote": "q",
                },
            ]
        )
    )
    (g,) = ev.load_gold(p)
    assert g.title == "Rivera reports brakes" and g.people == ("Dana Rivera",)
    assert (g.date, g.date_end) == (date(2024, 3, 1), date(2024, 3, 10)) and g.is_range


def test_demo_gold_files_are_well_formed() -> None:
    gold = ev.load_gold(ev.GOLD_DIR / "demo_timeline.json")
    disputes = ev.load_disputes(ev.GOLD_DIR / "demo_disputes.json")
    assert len(gold) == 29 and all(g.title and g.doc and g.event_key for g in gold)
    assert len({g.id for g in gold}) == len(gold)
    assert len(disputes) == 6 and all(len(d.sides) >= 2 for d in disputes)
    assert [d.id for d in disputes if d.planted] == ["C1", "C2", "C3"]
    for name in ("gold_timeline.json", "gold_disputes.json"):  # public set: hand-made later
        assert json.loads((ev.EVAL_DIR / name).read_text()) == []


def test_main_prints_table_and_writes_report(store: Store, tmp_path: Path, capsys) -> None:
    g, d = _gold(tmp_path)
    out = tmp_path / "report.json"
    rc = ev.main(
        [
            "--case",
            CASE,
            "--gold",
            str(g),
            "--disputes",
            str(d),
            "--db",
            str(store.db_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    table = capsys.readouterr().out
    for label in (
        "Fact precision",
        "Date accuracy",
        "Quote verification rate",
        "Merge accuracy",
        "Contradiction recall",
        "Review acceptance rate",
        "Throughput",
    ):
        assert label in table
    report = json.loads(out.read_text())
    assert report["case_id"] == CASE and report["facts"]["recall"] == 1.0
    assert report["gold"]["timeline"] == str(g)


def _fetch_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_eval.py"
    spec = importlib.util.spec_from_file_location("fetch_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_fetch_eval_fills_and_verifies_sha256(tmp_path: Path) -> None:
    fe = _fetch_module()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.htm").write_text("<html><style>x{}</style><p>Dear&nbsp;Sir,</p><p>Bye</p></html>")
    (src / "b.txt").write_text("plain")
    docs = tmp_path / "docs"
    docs.mkdir()
    manifest = docs / "manifest.json"
    entries = [
        {
            "id": "a",
            "group": "g1",
            "url": (src / "a.htm").as_uri(),
            "filename": "a.htm",
            "text_filename": "a.txt",
            "sha256": None,
            "retrieved_at": None,
        },
        {
            "id": "b",
            "group": "g2",
            "url": (src / "b.txt").as_uri(),
            "filename": "b.txt",
            "sha256": None,
            "retrieved_at": None,
        },
    ]
    manifest.write_text(json.dumps({"documents": entries, "todo": []}))

    assert fe.fetch_all(manifest, dry_run=True)["fetched"] == []
    assert not (docs / "g1").exists()
    res = fe.fetch_all(manifest, pause=0)
    assert res["fetched"] == ["a", "b"]
    saved = json.loads(manifest.read_text())["documents"]
    assert saved[1]["sha256"] == hashlib.sha256(b"plain").hexdigest()
    assert all(d["retrieved_at"] for d in saved)
    assert (docs / "g1" / "a.txt").read_text() == "Dear Sir,\n\nBye\n"

    assert fe.fetch_all(manifest, pause=0)["present"] == ["a", "b"]  # nothing re-downloaded
    (docs / "g2" / "b.txt").write_text("tampered")
    assert fe.fetch_all(manifest, pause=0, only=["g2"])["mismatch"] == ["b"]

    bad = {
        "documents": [{"id": "x", "group": "..", "url": "file:///etc/hosts", "filename": "hosts"}]
    }
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        fe.fetch_all(manifest)


def test_default_gold_paths_accept_the_bare_case_file_and_warn_on_fallback(caplog):
    from eval import run_eval

    timeline, disputes = run_eval.default_paths("ntsb")
    assert timeline == run_eval.GOLD_DIR / "ntsb.json"  # eval/gold/ntsb.json, not the demo set
    assert not disputes.exists()  # never the demo set's disputes
    caplog.clear()
    with caplog.at_level("WARNING"):
        timeline, disputes = run_eval.default_paths("no-such-case")
    assert timeline == run_eval.EVAL_DIR / "gold_timeline.json"
    assert disputes == run_eval.EVAL_DIR / "gold_disputes.json"
    assert "no gold timeline" in caplog.text
