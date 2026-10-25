"""Score extracted events against a hand-labelled gold chronology.

Gold file format (``eval/gold/<case>.json``): a list of
``{"date": "YYYY-MM-DD" | null, "actors": [...], "description": str, "doc": str}``.

Predictions come from the Store (``settings.db_path``) or, with ``--pred-json``,
from a JSON file holding a list of ``Event`` objects (or ``{"events": [...]}``).
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

from pipeline.contracts import Event

log = logging.getLogger(__name__)

GOLD_DIR = Path(__file__).resolve().parent / "gold"
DESC_THRESHOLD = 60.0
ACTOR_BONUS = 20.0  # bonus added at full actor overlap


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
            "n_matched": len(self.matches),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "date_accuracy": self.date_accuracy,
            "date_scored": self.date_scored,
            "date_correct": self.date_correct,
            "per_doc": self.per_doc,
            "matches": self.matches,
            "unmatched_gold": self.unmatched_gold,
            "unmatched_pred": self.unmatched_pred,
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


def events_to_preds(events: Sequence[Event], doc_names: dict[str, str] | None = None) -> list[Pred]:
    """Convert Events to scoring records; ``doc_names`` maps doc_id -> filename."""
    names = doc_names or {}
    return [
        Pred(
            date=ev.date_start,
            actors=list(ev.actors),
            description=ev.description,
            doc=names.get(ev.source.doc_id, ev.source.doc_id),
            event_id=ev.id,
        )
        for ev in events
    ]


def load_pred_json(path: Path) -> list[Event]:
    """Load Events from a JSON file (a list, or an object with an ``events`` key)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("events", [])
    return [Event.model_validate(item) for item in raw]


def load_store_events(case_id: str) -> tuple[list[Event], dict[str, str]]:
    """Load events (and a doc_id -> filename map, if available) from the local Store."""
    from pipeline.store import Store

    from pipeline.config import get_settings

    settings = get_settings()
    db_path = Path(settings.db_path)
    if not db_path.exists():
        log.warning("database %s does not exist; scoring zero predictions", db_path)
        return [], {}
    store = Store(db_path)
    if hasattr(store, "init"):
        store.init()
    events = list(store.list_events(case_id))
    return events, _doc_names(store, case_id)


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
    """Greedy one-to-one matching. Returns (gold_idx, pred_idx, desc_score, total_score)."""
    candidates: list[tuple[float, float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(preds):
            desc = fuzz.token_set_ratio(g.description, p.description)
            if desc < threshold:
                continue
            total = desc + ACTOR_BONUS * _actor_overlap(g.actors, p.actors)
            candidates.append((total, desc, gi, pi))
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int, float, float]] = []
    for total, desc, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, desc, total))
    return out


def score(case_id: str, gold: Sequence[GoldEvent], preds: Sequence[Pred]) -> EvalResult:
    """Compute precision/recall/F1, date accuracy and a per-doc breakdown."""
    pairs = match(gold, preds)
    res = EvalResult(case_id=case_id, n_gold=len(gold), n_pred=len(preds))
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
        f"Case: {res.case_id}   gold={res.n_gold}  pred={res.n_pred}  matched={len(res.matches)}",
        f"precision={res.precision:.3f}  recall={res.recall:.3f}  f1={res.f1:.3f}  "
        f"date_acc={da} ({res.date_correct}/{res.date_scored})",
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
    return "\n".join(lines)


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
) -> EvalResult:
    """Load gold + predictions, score, and write the JSON report."""
    gold_path = gold_path or GOLD_DIR / f"{case_id}.json"
    gold = load_gold(gold_path)
    if pred_json is not None:
        preds = events_to_preds(load_pred_json(pred_json))
    else:
        events, names = load_store_events(case_id)
        preds = events_to_preds(events, names)
    res = score(case_id, gold, preds)
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
        res = run(case_id, args.gold, args.pred_json, args.out)
    except Exception as exc:
        log.error("eval failed: %s", exc)
        return 0
    sys.stdout.write(format_table(res) + "\n")
    return 0
