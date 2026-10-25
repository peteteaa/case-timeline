"""Tests for eval.run_eval scoring with --pred-json (no DB, no LLM)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from eval.run_eval import (
    GoldContradiction,
    GoldEvent,
    Pred,
    contradictions_to_preds,
    load_gold,
    load_gold_contradictions,
    main,
    match,
    run,
    score,
    score_contradictions,
)
from pipeline.contracts import Contradiction, Event, SourceRef

GOLD = [
    {
        "date": "2024-03-03",
        "actors": ["John Smith", "Acme Corp"],
        "description": "John Smith signed the supply contract with Acme Corp",
        "doc": "contract.pdf",
    },
    {
        "date": "2024-03-10",
        "actors": ["Jane Doe"],
        "description": "Jane Doe emailed Smith about the late delivery",
        "doc": "emails.eml",
    },
    {
        "date": None,
        "actors": ["Acme Corp"],
        "description": "Acme Corp warehouse flooded",
        "doc": "emails.eml",
    },
    {
        "date": "2024-04-01",
        "actors": ["Bob Lee"],
        "description": "Bob Lee terminated the agreement by letter",
        "doc": "letter.docx",
    },
]


def _event(i: int, desc: str, actors: list[str], d: date | None, doc: str) -> Event:
    return Event(
        id=f"e{i}",
        case_id="t",
        date_text=str(d) if d else "unknown",
        date_start=d,
        date_end=d,
        actors=actors,
        description=desc,
        source=SourceRef(doc_id=doc, char_start=0, char_end=1, quote="x"),
        confidence=0.9,
    )


PREDS = [
    # match gold 0, correct date
    _event(0, "Smith signed supply contract with Acme", ["Smith", "Acme Corp"], date(2024, 3, 3),
           "contract.pdf"),
    # match gold 1, wrong date
    _event(1, "Jane Doe sent an email to Smith about late delivery", ["Jane Doe"],
           date(2024, 3, 11), "emails.eml"),
    # match gold 2 (gold date null -> not date-scored)
    _event(2, "The Acme Corp warehouse was flooded", ["Acme Corp"], None, "emails.eml"),
    # spurious
    _event(3, "Weather report mentioned heavy rain in Ohio", [], None, "emails.eml"),
]  # fmt: skip


@pytest.fixture()
def files(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(GOLD))
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps([e.model_dump(mode="json") for e in PREDS]))
    return gold, pred


def test_score_pred_json(files: tuple[Path, Path], tmp_path: Path) -> None:
    gold, pred = files
    out = tmp_path / "out" / "eval_t.json"
    res = run("t", gold_path=gold, pred_json=pred, out_path=out)

    assert res.n_gold == 4 and res.n_pred == 4
    assert {(m["gold_idx"], m["pred_idx"]) for m in res.matches} == {(0, 0), (1, 1), (2, 2)}
    assert res.precision == pytest.approx(0.75)
    assert res.recall == pytest.approx(0.75)
    assert res.f1 == pytest.approx(0.75)
    assert res.date_scored == 2 and res.date_correct == 1
    assert res.date_accuracy == pytest.approx(0.5)

    assert res.per_doc["contract.pdf"]["recall"] == pytest.approx(1.0)
    assert res.per_doc["emails.eml"]["gold"] == 2
    assert res.per_doc["emails.eml"]["pred"] == 3
    assert res.per_doc["emails.eml"]["precision"] == pytest.approx(2 / 3)
    assert res.per_doc["letter.docx"]["recall"] == 0.0

    report = json.loads(out.read_text())
    assert report["n_matched"] == 3 and report["unmatched_gold"] == [3]


def test_greedy_prefers_actor_overlap() -> None:
    gold = [GoldEvent(None, ["Alice"], "meeting held at the office", "a")]
    preds = [
        Pred(None, ["Bob"], "meeting held at the office", "a"),
        Pred(None, ["Alice"], "meeting held at the office", "a"),
    ]
    assert match(gold, preds)[0][:2] == (0, 1)


def test_one_to_one_and_threshold() -> None:
    gold = [GoldEvent(None, [], "contract signed", "a")]
    preds = [
        Pred(None, [], "contract signed", "a"),
        Pred(None, [], "contract signed", "a"),
        Pred(None, [], "totally unrelated text about weather", "a"),
    ]
    res = score("x", gold, preds)
    assert len(res.matches) == 1
    assert res.recall == 1.0 and res.precision == pytest.approx(1 / 3)
    assert res.date_accuracy is None


def test_empty_inputs() -> None:
    res = score("x", [], [])
    assert res.precision == res.recall == res.f1 == 0.0


def test_load_gold_parses_dates(files: tuple[Path, Path]) -> None:
    gold = load_gold(files[0])
    assert gold[0].date == date(2024, 3, 3) and gold[2].date is None


def test_main_exits_zero(files: tuple[Path, Path], tmp_path: Path, capsys) -> None:
    gold, pred = files
    out = tmp_path / "r.json"
    rc = main(["--case", "t", "--gold", str(gold), "--pred-json", str(pred), "--out", str(out)])
    assert rc == 0 and out.exists()
    assert "precision=0.750" in capsys.readouterr().out
    # Missing gold file still exits 0.
    assert main(["--case", "t", "--gold", str(tmp_path / "nope.json"), "--out", str(out)]) == 0


GOLD_C = [
    {
        "id": "C1",
        "field": "date",
        "planted": True,
        "sides": [{"doc": "contract.pdf"}, {"doc": "emails.eml"}, {"doc": "letter.docx"}],
    },
    {
        "id": "C2",
        "field": "actor",
        "planted": True,
        "sides": [{"doc": "contract.pdf"}, {"doc": "emails.eml"}],
    },
    {"id": "C3", "field": "fact", "planted": False,
     "sides": [{"doc": "emails.eml"}, {"doc": "letter.docx"}]},
]  # fmt: skip


def _contra(cid: str, ids: list[str], field: str, **kw) -> Contradiction:
    return Contradiction(id=cid, case_id="t", event_ids=ids, field=field, explanation="x", **kw)


def test_score_contradictions(tmp_path: Path) -> None:
    path = tmp_path / "t_contradictions.json"
    path.write_text(json.dumps(GOLD_C))
    gold = load_gold_contradictions(path)
    assert gold[0].docs == {"contract.pdf", "emails.eml", "letter.docx"} and not gold[2].planted
    contras = [
        _contra("p1", ["e0", "e1"], "date"),  # C1
        _contra("p2", ["e1", "e0"], "fact"),  # right docs, wrong field -> FP (docs-only hit)
        _contra("p3", ["e1", "e2"], "fact"),  # same document -> FP
        _contra("p4", ["e1", "e9"], "fact"),  # unknown event -> one doc -> FP
        _contra("p5", ["e0", "e1"], "actor", status="dismissed"),  # ignored
    ]
    preds = contradictions_to_preds(contras, PREDS)
    assert [p.id for p in preds] == ["p1", "p2", "p3", "p4"]
    res = score_contradictions(gold, preds)
    assert res.n_gold == 2 and res.n_gold_all == 3 and res.n_pred == 4
    assert res.found == ["C1"] and res.missed == ["C2"]
    assert res.recall == pytest.approx(0.5) and res.precision == pytest.approx(0.25)
    assert res.false_positives == ["p2", "p3", "p4"]
    assert res.docs_only_recall == pytest.approx(1.0)
    assert res.docs_only_precision == pytest.approx(0.5)


def test_derived_gold_counts_for_precision_not_recall() -> None:
    gold = [GoldContradiction("D", "fact", frozenset({"a", "b"}), planted=False)]
    preds = contradictions_to_preds(
        [_contra("p", ["x", "y"], "fact")],
        [
            _event(7, "d", [], None, "a").model_copy(update={"id": "x"}),
            _event(8, "d", [], None, "b").model_copy(update={"id": "y"}),
        ],
    )
    res = score_contradictions(gold, preds)
    # No planted entries: every gold entry is scored for recall.
    assert res.n_gold == 1 and res.recall == 1.0 and res.precision == 1.0


def test_run_reports_contradictions_from_pred_json(tmp_path: Path, capsys) -> None:
    gold = tmp_path / "t.json"
    gold.write_text(json.dumps(GOLD))
    (tmp_path / "t_contradictions.json").write_text(json.dumps(GOLD_C))
    pred = tmp_path / "pred.json"
    pred.write_text(
        json.dumps(
            {
                "events": [e.model_dump(mode="json") for e in PREDS],
                "contradictions": [_contra("p1", ["e0", "e1"], "date").model_dump(mode="json")],
            }
        )
    )
    out = tmp_path / "r.json"
    assert (
        main(["--case", "t", "--gold", str(gold), "--pred-json", str(pred), "--out", str(out)]) == 0
    )
    text = capsys.readouterr().out
    assert "contra_recall=0.500" in text and "contra_precision=1.000" in text
    report = json.loads(out.read_text())
    assert report["contradictions"]["found"] == ["C1"]
    assert report["n_matched"] == 3


def test_no_contradiction_gold_means_no_section(files: tuple[Path, Path], tmp_path: Path) -> None:
    gold, pred = files
    res = run("t", gold_path=gold, pred_json=pred, out_path=tmp_path / "o.json")
    assert res.contradictions is None
