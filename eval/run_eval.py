"""Score extracted events against a hand-labelled gold chronology.

Gold file format (``eval/gold/<case>.json``): a list of
``{"date": "YYYY-MM-DD" | null, "actors": [...], "description": str, "doc": str}``.

Predictions come from the Store (``settings.db_path``) or, with ``--pred-json``,
from a JSON file holding a list of ``Event`` objects (or ``{"events": [...]}``).

Precision/recall are also reported for key events only (``importance >=
KEY_EVENT_THRESHOLD``, the timeline's default view), so noise reduction is measurable; the
gold chronology is scored in full against that subset.

Contradictions are scored too when ``eval/gold/<case>_contradictions.json`` exists: a list of
``{"id", "field": "date"|"actor"|"fact", "planted": bool, "sides": [{"doc": ...}, ...]}``.
A predicted contradiction matches a gold one when the fields agree and the documents of its
events are all among the gold entry's side documents (at least two distinct documents).
Recall is over planted entries (all entries if none is marked); precision counts a prediction
as correct when it matches any gold entry, planted or derived.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from pipeline.contracts import Contradiction, ContradictionStatus, Event

log = logging.getLogger(__name__)

GOLD_DIR = Path(__file__).resolve().parent / "gold"
DESC_THRESHOLD = 60.0
ACTOR_BONUS = 20.0  # bonus added at full actor overlap
KEY_EVENT_THRESHOLD = 0.6  # same as pipeline.store.KEY_EVENT_THRESHOLD (checked by a test)


@dataclass(frozen=True)
class GoldEvent:
    date: date | None
    actors: list[str]
    description: str
    doc: str = ""


@dataclass(frozen=True)
class Pred:
    date: date | None
    actors: list[str]
    description: str
    doc: str
    event_id: str = ""
    importance: float = 0.5
    duplicate: bool = False  # linked duplicate (hidden on the timeline)


@dataclass
class DocStats:
    gold: int = 0
    pred: int = 0
    matched_gold: int = 0
    matched_pred: int = 0

    def as_dict(self) -> dict[str, Any]:
        p = _ratio(self.matched_pred, self.pred)
        r = _ratio(self.matched_gold, self.gold)
        return {
            "gold": self.gold,
            "pred": self.pred,
            "matched": self.matched_gold,
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
        }


@dataclass(frozen=True)
class GoldContradiction:
    id: str
    field: str
    docs: frozenset[str]
    planted: bool = True
    explanation: str = ""


@dataclass(frozen=True)
class PredContradiction:
    id: str
    field: str
    docs: frozenset[str]
    explanation: str = ""


@dataclass
class ContradictionResult:
    n_gold: int  # gold entries scored for recall (planted ones, or all if none is marked)
    n_gold_all: int
    n_pred: int
    found: list[str] = field(default_factory=list)  # scored gold ids matched by >= 1 pred
    missed: list[str] = field(default_factory=list)
    pred_matches: dict[str, list[str]] = field(default_factory=dict)  # pred id -> gold ids
    false_positives: list[str] = field(default_factory=list)
    recall: float = 0.0
    precision: float = 0.0
    docs_only_recall: float = 0.0  # same, ignoring the field
    docs_only_precision: float = 0.0
    preds: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_gold": self.n_gold,
            "n_gold_all": self.n_gold_all,
            "n_pred": self.n_pred,
            "recall": self.recall,
            "precision": self.precision,
            "docs_only_recall": self.docs_only_recall,
            "docs_only_precision": self.docs_only_precision,
            "found": self.found,
            "missed": self.missed,
            "pred_matches": self.pred_matches,
            "false_positives": self.false_positives,
            "preds": self.preds,
        }


@dataclass
class EvalResult:
    case_id: str
    n_gold: int
    n_pred: int
    matches: list[dict[str, Any]] = field(default_factory=list)
    unmatched_gold: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    date_accuracy: float | None = None
    date_scored: int = 0
    date_correct: int = 0
    per_doc: dict[str, dict[str, Any]] = field(default_factory=dict)
    contradictions: ContradictionResult | None = None
    n_duplicates: int = 0  # predictions that are linked duplicates (hidden on the timeline)
    key_events: dict[str, Any] | None = None  # the same scores over key events only

    def summary(self) -> dict[str, Any]:
        """Headline numbers (no per-match details)."""
        return {
            "n_pred": self.n_pred,
            "n_matched": len(self.matches),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "date_accuracy": self.date_accuracy,
            "date_scored": self.date_scored,
            "date_correct": self.date_correct,
            "unmatched_gold": self.unmatched_gold,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "n_duplicates": self.n_duplicates,
            "n_timeline": self.n_pred - self.n_duplicates,
            "n_matched": len(self.matches),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "date_accuracy": self.date_accuracy,
            "date_scored": self.date_scored,
            "date_correct": self.date_correct,
            "key_events": self.key_events,
            "per_doc": self.per_doc,
            "matches": self.matches,
            "unmatched_gold": self.unmatched_gold,
            "unmatched_pred": self.unmatched_pred,
            "contradictions": self.contradictions.as_dict() if self.contradictions else None,
        }


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _norm_doc(name: str) -> str:
    return Path(name).name.strip().lower() if name else ""


def load_gold(path: Path) -> list[GoldEvent]:
    """Load a gold chronology file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("events", [])
    return [
        GoldEvent(
            date=_parse_date(item.get("date")),
            actors=list(item.get("actors") or []),
            description=str(item.get("description", "")),
            doc=str(item.get("doc") or ""),
        )
        for item in raw
    ]


def events_to_preds(
    events: Sequence[Event],
    doc_names: dict[str, str] | None = None,
    duplicate_ids: set[str] | frozenset[str] = frozenset(),
) -> list[Pred]:
    """Convert Events to scoring records; ``doc_names`` maps doc_id -> filename and
    ``duplicate_ids`` are the linked duplicates among them."""
    names = doc_names or {}
    return [
        Pred(
            date=ev.date_start,
            actors=list(ev.actors),
            description=ev.description,
            doc=names.get(ev.source.doc_id, ev.source.doc_id),
            event_id=ev.id,
            importance=ev.importance,
            duplicate=ev.id in duplicate_ids,
        )
        for ev in events
    ]


def load_pred_json_contradictions(path: Path) -> list[Contradiction]:
    """Contradictions from a ``{"events": [...], "contradictions": [...]}`` JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    return [Contradiction.model_validate(item) for item in raw.get("contradictions") or []]


def load_pred_json(path: Path) -> list[Event]:
    """Load Events from a JSON file (a list, or an object with an ``events`` key)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("events", [])
    return [Event.model_validate(item) for item in raw]


def load_store_events(
    case_id: str, *, include_duplicates: bool = False
) -> tuple[list[Event], dict[str, str]]:
    """Load events (and a doc_id -> filename map, if available) from the local Store.

    Linked duplicates are hidden like in the timeline unless ``include_duplicates``.
    """
    from pipeline.config import get_settings
    from pipeline.store import Store

    settings = get_settings()
    db_path = Path(settings.db_path)
    if not db_path.exists():
        log.warning("database %s does not exist; scoring zero predictions", db_path)
        return [], {}
    store = Store(db_path)
    if hasattr(store, "init"):
        store.init()
    if include_duplicates:
        events = list(store.list_events(case_id, include_duplicates=True))
    else:
        events = list(store.list_events(case_id))
    return events, _doc_names(store, case_id)


def load_store_duplicate_ids(case_id: str) -> set[str]:
    """Ids of the stored events that are linked duplicates (hidden on the timeline)."""
    from pipeline.config import get_settings
    from pipeline.store import Store

    db_path = Path(get_settings().db_path)
    if not db_path.exists():
        return set()
    store = Store(db_path)
    every = {e.id for e in store.list_events(case_id, include_duplicates=True)}
    return every - {e.id for e in store.list_events(case_id)}


def load_store_contradictions(case_id: str) -> tuple[list[Contradiction], list[Event]]:
    """Stored contradictions plus every event (duplicates included) they may reference."""
    from pipeline.config import get_settings
    from pipeline.store import Store

    db_path = Path(get_settings().db_path)
    if not db_path.exists():
        return [], []
    store = Store(db_path)
    return list(store.list_contradictions(case_id)), list(
        store.list_events(case_id, include_duplicates=True)
    )


def _doc_names(store: Any, case_id: str) -> dict[str, str]:
    """Best-effort doc_id -> filename lookup; Store's document API is not frozen."""
    for attr in ("list_documents", "documents"):
        fn = getattr(store, attr, None)
        if callable(fn):
            try:
                return {d.id: d.filename for d in fn(case_id)}
            except Exception:
                log.debug("store.%s failed; falling back to doc ids", attr, exc_info=True)
    return {}


def gold_contradictions_path(case_id: str, gold_path: Path | None = None) -> Path:
    """``<gold dir>/<case>_contradictions.json`` next to the event gold file."""
    base = gold_path.parent if gold_path is not None else GOLD_DIR
    return base / f"{case_id}_contradictions.json"


def load_gold_contradictions(path: Path) -> list[GoldContradiction]:
    """Load a gold contradictions file (a list, or an object with a ``contradictions`` key)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("contradictions", [])
    out = []
    for i, item in enumerate(raw):
        docs = frozenset(
            _norm_doc(str(side.get("doc") or "")) for side in item.get("sides") or []
        ) - {""}
        out.append(
            GoldContradiction(
                id=str(item.get("id") or f"G{i + 1}"),
                field=str(item.get("field", "")).strip().lower(),
                docs=docs,
                planted=bool(item.get("planted", True)),
                explanation=str(item.get("explanation", "")),
            )
        )
    return out


def contradictions_to_preds(
    contradictions: Sequence[Contradiction],
    events: Sequence[Event],
    doc_names: dict[str, str] | None = None,
) -> list[PredContradiction]:
    """Map contradictions to the (file) names of their events' documents; dismissed ones and
    event ids that cannot be resolved are skipped."""
    names = doc_names or {}
    doc_of = {e.id: e.source.doc_id for e in events}
    out = []
    for c in contradictions:
        if c.status == ContradictionStatus.DISMISSED:
            continue
        docs = {
            _norm_doc(names.get(doc_of[eid], doc_of[eid])) for eid in c.event_ids if eid in doc_of
        }
        out.append(
            PredContradiction(
                id=c.id, field=str(c.field), docs=frozenset(docs), explanation=c.explanation
            )
        )
    return out


def _contradiction_matches(p: PredContradiction, g: GoldContradiction, use_field: bool) -> bool:
    if use_field and p.field != g.field:
        return False
    return len(p.docs) >= 2 and p.docs <= g.docs


def score_contradictions(
    gold: Sequence[GoldContradiction], preds: Sequence[PredContradiction]
) -> ContradictionResult:
    """Contradiction recall (over planted gold) and precision (vs any gold entry)."""
    scored = [g for g in gold if g.planted] or list(gold)
    res = ContradictionResult(n_gold=len(scored), n_gold_all=len(gold), n_pred=len(preds))

    def rates(use_field: bool) -> tuple[float, float, dict[str, list[str]]]:
        hits = {
            p.id: [g.id for g in gold if _contradiction_matches(p, g, use_field)] for p in preds
        }
        found = {gid for gids in hits.values() for gid in gids}
        recall = _ratio(sum(1 for g in scored if g.id in found), len(scored))
        precision = _ratio(sum(1 for gids in hits.values() if gids), len(preds))
        return recall, precision, hits

    res.recall, res.precision, hits = rates(True)
    res.docs_only_recall, res.docs_only_precision, _ = rates(False)
    found = {gid for gids in hits.values() for gid in gids}
    res.found = [g.id for g in scored if g.id in found]
    res.missed = [g.id for g in scored if g.id not in found]
    res.pred_matches = {pid: gids for pid, gids in hits.items() if gids}
    res.false_positives = [p.id for p in preds if not hits[p.id]]
    res.preds = [
        {
            "id": p.id,
            "field": p.field,
            "docs": sorted(p.docs),
            "matches": hits[p.id],
            "explanation": p.explanation,
        }
        for p in preds
    ]
    return res


def _actor_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {x.strip().lower() for x in a if x.strip()}
    sb = {x.strip().lower() for x in b if x.strip()}
    if not sa or not sb:
        return 0.0
    # Fuzzy hits so "Smith" matches "John Smith".
    hits = sum(1 for x in sa if any(fuzz.token_set_ratio(x, y) >= 90 for y in sb))
    return hits / max(len(sa), len(sb))


def match(
    gold: Sequence[GoldEvent], preds: Sequence[Pred], threshold: float = DESC_THRESHOLD
) -> list[tuple[int, int, float, float]]:
    """Greedy one-to-one matching. Returns (gold_idx, pred_idx, desc_score, total_score).

    Descriptions are compared case- and punctuation-insensitively. Pairs from the same
    document are matched first, so that two documents' accounts of one occurrence are not
    swapped; cross-document pairs only fill what is left.
    """
    candidates: list[tuple[bool, float, float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(preds):
            desc = fuzz.token_set_ratio(g.description, p.description, processor=default_process)
            if desc < threshold:
                continue
            total = desc + ACTOR_BONUS * _actor_overlap(g.actors, p.actors)
            same_doc = bool(g.doc) and _norm_doc(g.doc) == _norm_doc(p.doc)
            candidates.append((not same_doc, total, desc, gi, pi))
    candidates.sort(key=lambda c: (c[0], -c[1], -c[2], c[3], c[4]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int, float, float]] = []
    for _, total, desc, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, desc, total))
    return out


def score(
    case_id: str,
    gold: Sequence[GoldEvent],
    preds: Sequence[Pred],
    *,
    key_threshold: float | None = KEY_EVENT_THRESHOLD,
) -> EvalResult:
    """Compute precision/recall/F1, date accuracy and a per-doc breakdown.

    With ``key_threshold``, ``key_events`` holds the same headline scores computed over the
    predictions whose importance reaches it (against the full gold list).
    """
    res = _score(case_id, gold, preds)
    if key_threshold is not None:
        key_preds = [p for p in preds if p.importance >= key_threshold]
        key = _score(case_id, gold, key_preds)
        res.key_events = {
            "threshold": key_threshold,
            **key.summary(),
            "n_duplicates": sum(p.duplicate for p in key_preds),
            "n_timeline": sum(not p.duplicate for p in key_preds),
        }
    return res


def _score(case_id: str, gold: Sequence[GoldEvent], preds: Sequence[Pred]) -> EvalResult:
    pairs = match(gold, preds)
    res = EvalResult(case_id=case_id, n_gold=len(gold), n_pred=len(preds))
    res.n_duplicates = sum(p.duplicate for p in preds)
    res.precision = _ratio(len(pairs), len(preds))
    res.recall = _ratio(len(pairs), len(gold))
    res.f1 = _f1(res.precision, res.recall)

    per_doc: dict[str, DocStats] = {}
    for g in gold:
        per_doc.setdefault(_norm_doc(g.doc), DocStats()).gold += 1
    for p in preds:
        per_doc.setdefault(_norm_doc(p.doc), DocStats()).pred += 1

    for gi, pi, desc, total in sorted(pairs):
        g, p = gold[gi], preds[pi]
        date_ok: bool | None = None
        if g.date is not None:
            date_ok = p.date == g.date
            res.date_scored += 1
            res.date_correct += int(date_ok)
        per_doc[_norm_doc(g.doc)].matched_gold += 1
        per_doc[_norm_doc(p.doc)].matched_pred += 1
        res.matches.append(
            {
                "gold_idx": gi,
                "pred_idx": pi,
                "event_id": p.event_id,
                "desc_score": round(desc, 1),
                "score": round(total, 1),
                "gold_date": g.date.isoformat() if g.date else None,
                "pred_date": p.date.isoformat() if p.date else None,
                "date_correct": date_ok,
            }
        )
    matched_g = {gi for gi, *_ in pairs}
    matched_p = {pi for _, pi, *_ in pairs}
    res.unmatched_gold = [i for i in range(len(gold)) if i not in matched_g]
    res.unmatched_pred = [i for i in range(len(preds)) if i not in matched_p]
    res.date_accuracy = _ratio(res.date_correct, res.date_scored) if res.date_scored else None
    res.per_doc = {(k or "(none)"): v.as_dict() for k, v in sorted(per_doc.items())}
    return res


def format_table(res: EvalResult) -> str:
    """Render a plain-text summary table."""
    da = "n/a" if res.date_accuracy is None else f"{res.date_accuracy:.3f}"
    lines = [
        f"Case: {res.case_id}   gold={res.n_gold}  pred={res.n_pred} "
        f"(timeline {res.n_pred - res.n_duplicates}, linked duplicates {res.n_duplicates})  "
        f"matched={len(res.matches)}",
        f"all events: precision={res.precision:.3f}  recall={res.recall:.3f}  f1={res.f1:.3f}  "
        f"date_acc={da} ({res.date_correct}/{res.date_scored})",
    ]
    if res.key_events is not None:
        k = res.key_events
        kda = "n/a" if k["date_accuracy"] is None else f"{k['date_accuracy']:.3f}"
        lines.append(
            f"key events (importance >= {k['threshold']:g}): pred={k['n_pred']} "
            f"(timeline {k['n_timeline']})  matched={k['n_matched']}  "
            f"precision={k['precision']:.3f}  recall={k['recall']:.3f}  f1={k['f1']:.3f}  "
            f"date_acc={kda} ({k['date_correct']}/{k['date_scored']})"
        )
    lines += [
        "",
        f"{'doc':<40} {'gold':>5} {'pred':>5} {'match':>5} {'P':>6} {'R':>6} {'F1':>6}",
        "-" * 78,
    ]
    for doc, d in res.per_doc.items():
        name = doc if len(doc) <= 40 else doc[:37] + "..."
        lines.append(
            f"{name:<40} {d['gold']:>5} {d['pred']:>5} {d['matched']:>5} "
            f"{d['precision']:>6.3f} {d['recall']:>6.3f} {d['f1']:>6.3f}"
        )
    if res.contradictions is not None:
        lines += ["", *format_contradictions(res.contradictions)]
    return "\n".join(lines)


def format_contradictions(c: ContradictionResult) -> list[str]:
    """Plain-text contradiction summary lines."""
    lines = [
        f"Contradictions: gold={c.n_gold} (of {c.n_gold_all})  pred={c.n_pred}  "
        f"found={len(c.found)}  false_pos={len(c.false_positives)}",
        f"contra_recall={c.recall:.3f}  contra_precision={c.precision:.3f}  "
        f"(docs only: recall={c.docs_only_recall:.3f} precision={c.docs_only_precision:.3f})",
    ]
    if c.found:
        lines.append(f"found:  {', '.join(c.found)}")
    if c.missed:
        lines.append(f"missed: {', '.join(c.missed)}")
    for p in c.preds:
        tag = ",".join(p["matches"]) or "FP"
        lines.append(f"  [{tag:<6}] {p['field']:<5} {' vs '.join(p['docs'])}")
    return lines


def _data_dir() -> Path:
    try:
        from pipeline.config import get_settings

        return Path(get_settings().data_dir)
    except Exception:
        return Path("data")


def run(
    case_id: str,
    gold_path: Path | None = None,
    pred_json: Path | None = None,
    out_path: Path | None = None,
    *,
    contradictions_path: Path | None = None,
    include_duplicates: bool = True,
) -> EvalResult:
    """Load gold + predictions, score, and write the JSON report."""
    contradictions_path = contradictions_path or gold_contradictions_path(case_id, gold_path)
    gold_path = gold_path or GOLD_DIR / f"{case_id}.json"
    gold = load_gold(gold_path)
    names: dict[str, str] = {}
    dup_ids: set[str] = set()
    if pred_json is not None:
        events = load_pred_json(pred_json)
    else:
        events, names = load_store_events(case_id, include_duplicates=include_duplicates)
        if include_duplicates:
            dup_ids = load_store_duplicate_ids(case_id)
    preds = events_to_preds(events, names, dup_ids)
    res = score(case_id, gold, preds)
    if contradictions_path.exists():
        gold_c = load_gold_contradictions(contradictions_path)
        if pred_json is not None:
            pred_c, all_events = load_pred_json_contradictions(pred_json), events
        else:
            pred_c, all_events = load_store_contradictions(case_id)
        res.contradictions = score_contradictions(
            gold_c, contradictions_to_preds(pred_c, all_events, names)
        )
    out_path = out_path or _data_dir() / f"eval_{case_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res.as_dict(), indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Always returns 0 so eval never breaks a pipeline."""
    ap = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    ap.add_argument("--case", default=None, help="case id (default: settings.case_id or 'demo')")
    ap.add_argument("--gold", type=Path, default=None, help="gold file (default eval/gold/<case>)")
    ap.add_argument("--pred-json", type=Path, default=None, help="score Events from a JSON file")
    ap.add_argument("--out", type=Path, default=None, help="report path (default data/eval_<case>)")
    ap.add_argument(
        "--contradictions",
        type=Path,
        default=None,
        help="gold contradictions file (default <gold dir>/<case>_contradictions.json)",
    )
    ap.add_argument(
        "--exclude-duplicates",
        action="store_true",
        help="hide linked duplicate events like the timeline does (by default they are scored, "
        "because the gold lists each document's own account of an occurrence)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    case_id = args.case
    if not case_id:
        try:
            from pipeline.config import get_settings

            case_id = get_settings().case_id
        except Exception:
            case_id = "demo"
    try:
        res = run(
            case_id,
            args.gold,
            args.pred_json,
            args.out,
            contradictions_path=args.contradictions,
            include_duplicates=not args.exclude_duplicates,
        )
    except Exception as exc:
        log.error("eval failed: %s", exc)
        return 0
    sys.stdout.write(format_table(res) + "\n")
    return 0
