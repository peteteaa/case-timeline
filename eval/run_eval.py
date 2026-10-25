"""Score a scanned case against hand-made gold labels (SPEC §13).

Run a scan first (``make scan CASE=demo``), then ``make eval CASE=demo``. Metrics come from the
store, so they reflect exactly what the pipeline wrote:

- **Fact precision / recall / F1.** Predictions are the AI-drafted facts (any review state).
  A gold entry matches the best fact that cites the gold document and fuzzy-matches its title,
  details or quote; several gold entries may share one fact (merged evidence). A fact counts
  as correct when at least one gold entry chose it, so unmerged duplicates cost precision.
  ``strict_*`` also requires the date to be right.
- **Date accuracy.** Over matched gold entries with a date: exact match for exact dates, range
  overlap for approximate or ranged dates.
- **Quote verification rate.** ``1 - quotes_rejected / quotes_checked`` summed over the case's
  jobs (the verbatim check in ``verify.py``).
- **Merge accuracy.** Gold entries sharing an ``event_key`` across documents describe one
  event; the fraction of such events whose matched entries all landed on a single fact.
- **Contradiction precision / recall.** A predicted contradiction matches a gold dispute when
  the conflicting field agrees (if both name one) and two of its facts match two different
  sides of it. A fact matches a side when it quotes the side's text, or when it is the fact
  matched to the gold timeline entry for the dispute's ``event_key`` in the side's document
  (the model may quote a different sentence about the same event).
- **Review acceptance rate.** (accept + edit) / all fact-level review events.
- **Throughput.** Documents parsed per hour of job run time.

Gold timeline (``eval/gold/<case>_timeline.json``, else ``eval/gold_timeline.json``), a list of
``{"id", "title", "details"?, "date": "YYYY-MM-DD" | null, "date_end"?, "date_certainty"?,
"doc", "event_key"?, "quote"?, "people"?}``. Gold disputes (``eval/gold/<case>_disputes.json``,
else ``eval/gold_disputes.json``): ``{"id", "label": "disputed" | "undisputed", "field"?,
"sides": [{"doc", "quote"}, ...]}``; only ``disputed`` entries count.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
GOLD_DIR = EVAL_DIR / "gold"
TEXT_THRESHOLD = 60.0  # token_set_ratio on title/details
QUOTE_THRESHOLD = 85.0  # partial_ratio of gold quote vs a fact quote from the same document
SIDE_THRESHOLD = 80.0  # partial_ratio for matching a dispute side
DATE_BONUS = 15.0


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class GoldFact:
    id: str
    title: str
    doc: str = ""
    details: str = ""
    date: date | None = None
    date_end: date | None = None
    date_certainty: str = "unknown"
    event_key: str = ""
    quote: str = ""
    people: tuple[str, ...] = ()

    @property
    def is_range(self) -> bool:
        return self.date_certainty == "approximate" or (
            self.date_end is not None and self.date_end != self.date
        )


@dataclass(frozen=True)
class GoldDispute:
    id: str
    sides: tuple[tuple[str, str], ...]  # (doc, quote)
    field: str = ""
    event_key: str = ""


@dataclass
class Pred:
    """One stored fact, flattened for scoring."""

    fact_id: str
    title: str
    details: str
    date_start: date | None
    date_end: date | None
    review: str
    quotes: list[tuple[str, str]] = field(default_factory=list)  # (doc filename, quote)

    @property
    def docs(self) -> set[str]:
        return {d for d, _ in self.quotes}


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 2 * p * r / (p + r) if p + r else 0.0


def _date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _norm_doc(name: str | None) -> str:
    return Path(name).name.strip().casefold() if name else ""


# --------------------------------------------------------------------------- loading


def load_gold(path: Path) -> list[GoldFact]:
    """Load a gold timeline. Also accepts the v0.2 field names (description, actors,
    certainty, date_range) that ``scripts/make_fixtures.py`` writes."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("facts") or raw.get("events") or []
    return load_gold_items(raw)


def load_gold_items(raw: Sequence[dict[str, Any]]) -> list[GoldFact]:
    """Gold facts from already-parsed JSON items."""
    out = []
    for i, item in enumerate(raw, start=1):
        rng = item.get("date_range") or [None, None]
        start = _date(item.get("date")) or _date(rng[0])
        out.append(
            GoldFact(
                id=str(item.get("id") or f"G{i:02d}"),
                title=str(item.get("title") or item.get("description") or ""),
                doc=str(item.get("doc") or ""),
                details=str(item.get("details") or ""),
                date=start,
                date_end=_date(item.get("date_end")) or _date(rng[1]),
                date_certainty=str(
                    item.get("date_certainty") or item.get("certainty") or "unknown"
                ),
                event_key=str(item.get("event_key") or ""),
                quote=str(item.get("quote") or ""),
                people=tuple(item.get("people") or item.get("actors") or ()),
            )
        )
    return out


def load_disputes(path: Path) -> list[GoldDispute]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, item in enumerate(raw, start=1):
        if str(item.get("label", "disputed")) != "disputed":
            continue
        sides = tuple(
            (str(s.get("doc") or ""), str(s.get("quote") or "")) for s in item.get("sides", [])
        )
        if len(sides) >= 2:
            out.append(
                GoldDispute(
                    id=str(item.get("id") or f"C{i}"),
                    sides=sides,
                    field=str(item.get("field") or ""),
                    event_key=str(item.get("event_key") or ""),
                )
            )
    return out


def default_paths(case_id: str) -> tuple[Path, Path]:
    """Per-case gold under eval/gold/, else the public set's files at eval/."""
    timeline = GOLD_DIR / f"{case_id}_timeline.json"
    disputes = GOLD_DIR / f"{case_id}_disputes.json"
    return (
        timeline if timeline.exists() else EVAL_DIR / "gold_timeline.json",
        disputes if disputes.exists() else EVAL_DIR / "gold_disputes.json",
    )


def load_preds(store: Any, case_id: str) -> list[Pred]:
    """AI-drafted facts in every review state, with their evidence quotes."""
    from pipeline.schema import FactQuery

    names = {d.doc_id: d.filename for d in store.list_documents(case_id)}
    facts = store.list_facts(case_id, FactQuery(reviews=[], origins=["ai"]))
    return [_pred(f, names) for f in facts]


def _pred(fact: Any, names: dict[str, str]) -> Pred:
    quotes = [
        (names.get(e.doc_id, e.doc_id or ""), e.quote or "")
        for e in fact.evidence
        if e.doc_id is not None
    ]
    return Pred(
        fact_id=fact.fact_id,
        title=fact.title,
        details=fact.details or "",
        date_start=fact.date_start,
        date_end=fact.date_end,
        review=fact.review,
        quotes=quotes,
    )


# --------------------------------------------------------------------------- matching


def date_correct(g: GoldFact, p: Pred) -> bool | None:
    """None when the gold entry is undated; else exact or range-overlap agreement."""
    if g.date is None:
        return None
    if p.date_start is None:
        return False
    p_end = p.date_end or p.date_start
    if g.is_range:
        g_end = g.date_end or g.date
        return p.date_start <= g_end and p_end >= g.date
    return p.date_start == g.date and p_end == g.date


def text_score(g: GoldFact, p: Pred) -> float:
    """Best fuzzy agreement of title/details, or of the gold quote with a quote from its doc."""
    best = 0.0
    for gold_text in filter(None, (g.title, g.details)):
        for pred_text in filter(None, (p.title, p.details)):
            best = max(best, fuzz.token_set_ratio(gold_text, pred_text))
    if best < TEXT_THRESHOLD:
        best = 0.0
    if g.quote:
        doc = _norm_doc(g.doc)
        for d, q in p.quotes:
            if q and (not doc or _norm_doc(d) == doc):
                if fuzz.partial_ratio(g.quote, q) >= QUOTE_THRESHOLD:
                    best = max(best, 100.0)
    return best


def best_matches(gold: Sequence[GoldFact], preds: Sequence[Pred]) -> dict[int, tuple[int, float]]:
    """gold index -> (pred index, score). A fact must cite the gold entry's document."""
    out: dict[int, tuple[int, float]] = {}
    for gi, g in enumerate(gold):
        doc = _norm_doc(g.doc)
        best: tuple[int, float] | None = None
        for pi, p in enumerate(preds):
            if doc and doc not in {_norm_doc(d) for d in p.docs}:
                continue
            score = text_score(g, p)
            if score <= 0:
                continue
            score += DATE_BONUS * bool(date_correct(g, p))
            if best is None or score > best[1]:
                best = (pi, score)
        if best is not None:
            out[gi] = best
    return out


EventFacts = dict[tuple[str, str], set[str]]  # (event_key, normalized doc) -> fact ids


def event_facts(
    gold: Sequence[GoldFact], preds: Sequence[Pred], matches: dict[int, tuple[int, float]]
) -> EventFacts:
    """Which fact each gold timeline entry landed on, keyed by its event and document."""
    out: EventFacts = defaultdict(set)
    for gi, (pi, _) in matches.items():
        g = gold[gi]
        if g.event_key:
            out[(g.event_key, _norm_doc(g.doc))].add(preds[pi].fact_id)
    return out


def side_matches(
    side: tuple[str, str], p: Pred, event_key: str = "", events: EventFacts | None = None
) -> bool:
    doc, quote = _norm_doc(side[0]), side[1]
    if event_key and events and p.fact_id in events.get((event_key, doc), ()):
        return True
    for d, q in p.quotes:
        if doc and _norm_doc(d) != doc:
            continue
        if not quote or (q and fuzz.partial_ratio(quote, q) >= SIDE_THRESHOLD):
            return True
    return False


def contradiction_matches(
    fact_preds: Sequence[Pred],
    g: GoldDispute,
    events: EventFacts | None = None,
    field: str = "",
) -> bool:
    """Same field (when both are known) and two different facts match two different sides."""
    if field and g.field and field != g.field:
        return False
    hits = [
        {i for i, s in enumerate(g.sides) if side_matches(s, p, g.event_key, events)}
        for p in fact_preds
    ]
    for a in range(len(hits)):
        for b in range(a + 1, len(hits)):
            if any(i != j for i in hits[a] for j in hits[b]):
                return True
    return False


# --------------------------------------------------------------------------- scoring


def score_facts(gold: Sequence[GoldFact], preds: Sequence[Pred]) -> dict[str, Any]:
    matches = best_matches(gold, preds)
    used = {pi for pi, _ in matches.values()}
    strict = {
        gi: m for gi, m in matches.items() if date_correct(gold[gi], preds[m[0]]) is not False
    }
    strict_used = {pi for pi, _ in strict.values()}
    p = _ratio(len(used), len(preds))
    r = _ratio(len(matches), len(gold))
    sp = _ratio(len(strict_used), len(preds))
    sr = _ratio(len(strict), len(gold))

    scored = correct = 0
    rows = []
    for gi, g in enumerate(gold):
        m = matches.get(gi)
        ok = date_correct(g, preds[m[0]]) if m else None
        if m and ok is not None:
            scored += 1
            correct += int(ok)
        rows.append(
            {
                "gold_id": g.id,
                "doc": g.doc,
                "fact_id": preds[m[0]].fact_id if m else None,
                "score": round(m[1], 1) if m else None,
                "gold_date": g.date.isoformat() if g.date else None,
                "pred_date": preds[m[0]].date_start.isoformat()
                if m and preds[m[0]].date_start
                else None,
                "date_correct": ok,
            }
        )
    return {
        "n_gold": len(gold),
        "n_pred": len(preds),
        "n_gold_matched": len(matches),
        "n_pred_matched": len(used),
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
        "strict_precision": sp,
        "strict_recall": sr,
        "strict_f1": _f1(sp, sr),
        "date_accuracy": _ratio(correct, scored),
        "date_scored": scored,
        "date_correct": correct,
        "unmatched_facts": [preds[i].fact_id for i in range(len(preds)) if i not in used],
        "matches": rows,
        "_matches": matches,
    }


def score_merges(
    gold: Sequence[GoldFact], preds: Sequence[Pred], matches: dict[int, tuple[int, float]]
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for gi, g in enumerate(gold):
        if g.event_key:
            groups[g.event_key].append(gi)
    multi = {k: v for k, v in groups.items() if len({_norm_doc(gold[i].doc) for i in v}) >= 2}
    scored = correct = 0
    detail = {}
    for key, members in sorted(multi.items()):
        matched = [i for i in members if i in matches]
        docs = {_norm_doc(gold[i].doc) for i in matched}
        facts = sorted({preds[matches[i][0]].fact_id for i in matched})
        judged = len(docs) >= 2
        ok = judged and len(facts) == 1
        scored += int(judged)
        correct += int(ok)
        detail[key] = {
            "gold": len(members),
            "matched": len(matched),
            "facts": facts,
            "merged": ok if judged else None,
        }
    return {
        "multi_doc_events": len(multi),
        "scored": scored,
        "merged": correct,
        "merge_accuracy": _ratio(correct, scored),
        "events": detail,
    }


def score_contradictions(
    store: Any,
    case_id: str,
    disputes: Sequence[GoldDispute],
    by_id: dict[str, Pred],
    events: EventFacts | None = None,
) -> dict[str, Any]:
    predicted = store.list_contradictions(case_id)
    pred_hits: dict[str, list[str]] = {}
    gold_hit: set[str] = set()
    for c in predicted:
        facts = [by_id[i] for i in c.fact_ids if i in by_id]
        hits = [g.id for g in disputes if contradiction_matches(facts, g, events, c.field)]
        pred_hits[c.contradiction_id] = hits
        gold_hit.update(hits)
    p = _ratio(sum(1 for h in pred_hits.values() if h), len(predicted))
    r = _ratio(len(gold_hit), len(disputes))
    return {
        "n_gold": len(disputes),
        "n_pred": len(predicted),
        "precision": p,
        "recall": r,
        "f1": _f1(p, r),
        "matched": pred_hits,
        "missed": [g.id for g in disputes if g.id not in gold_hit],
    }


def score_jobs(store: Any, case_id: str) -> dict[str, Any]:
    jobs = store.list_jobs(case_id, limit=100_000)
    checked = sum(j.counts.get("quotes_checked", 0) for j in jobs)
    rejected = sum(j.counts.get("quotes_rejected", 0) for j in jobs)
    parsed = seconds = 0.0
    for j in jobs:
        if j.started_at and j.finished_at and j.status in ("done", "failed"):
            start, end = j.started_at, j.finished_at
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
            seconds += max((end - start).total_seconds(), 0.0)
            parsed += j.counts.get("docs_parsed", 0)
    return {
        "jobs": len(jobs),
        "quotes_checked": checked,
        "quotes_rejected": rejected,
        "quote_verification_rate": _ratio(checked - rejected, checked),
        "docs_parsed": int(parsed),
        "job_seconds": round(seconds, 3),
        "docs_per_hour": _ratio(parsed * 3600, seconds) if seconds > 0 else None,
    }


def score_reviews(store: Any, case_id: str) -> dict[str, Any]:
    events = [e for e in store.list_review_events(case_id) if e.evidence_id is None]
    counts = {a: sum(1 for e in events if e.action == a) for a in ("accept", "edit", "reject")}
    return {
        **counts,
        "events": len(events),
        "acceptance_rate": _ratio(counts["accept"] + counts["edit"], len(events)),
    }


def evaluate(
    store: Any, case_id: str, gold: Sequence[GoldFact], disputes: Sequence[GoldDispute]
) -> dict[str, Any]:
    """All SPEC §13 metrics for one case."""
    preds = load_preds(store, case_id)
    facts = score_facts(gold, preds)
    matches = facts.pop("_matches")
    return {
        "case_id": case_id,
        "facts": facts,
        "merge": score_merges(gold, preds, matches),
        "contradictions": score_contradictions(
            store,
            case_id,
            disputes,
            {p.fact_id: p for p in preds},
            event_facts(gold, preds, matches),
        ),
        "jobs": score_jobs(store, case_id),
        "review": score_reviews(store, case_id),
    }


# --------------------------------------------------------------------------- output


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def format_table(res: dict[str, Any]) -> str:
    f, m, c = res["facts"], res["merge"], res["contradictions"]
    j, r = res["jobs"], res["review"]
    rows = [
        ("Fact precision", f["precision"], f"{f['n_pred_matched']}/{f['n_pred']} facts"),
        ("Fact recall", f["recall"], f"{f['n_gold_matched']}/{f['n_gold']} gold"),
        ("Fact F1", f["f1"], f"strict (date too) {_fmt(f['strict_f1'])}"),
        ("Date accuracy", f["date_accuracy"], f"{f['date_correct']}/{f['date_scored']}"),
        (
            "Quote verification rate",
            j["quote_verification_rate"],
            f"{j['quotes_checked'] - j['quotes_rejected']}/{j['quotes_checked']} quotes",
        ),
        ("Merge accuracy", m["merge_accuracy"], f"{m['merged']}/{m['scored']} multi-doc events"),
        ("Contradiction precision", c["precision"], f"{c['n_pred']} predicted"),
        ("Contradiction recall", c["recall"], f"{c['n_gold'] - len(c['missed'])}/{c['n_gold']}"),
        ("Review acceptance rate", r["acceptance_rate"], f"{r['events']} review events"),
        (
            "Throughput (docs/hour)",
            j["docs_per_hour"],
            f"{j['docs_parsed']} docs in {j['job_seconds']:.0f}s",
        ),
    ]
    width = max(len(name) for name, *_ in rows)
    lines = [
        f"Case: {res['case_id']}",
        f"{'metric':<{width}}  {'value':>9}  detail",
        "-" * (width + 40),
    ]
    lines += [f"{name:<{width}}  {_fmt(v):>9}  {detail}" for name, v, detail in rows]
    return "\n".join(lines)


def run(
    case_id: str,
    *,
    gold_path: Path | None = None,
    disputes_path: Path | None = None,
    db_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Load gold and the store, score, write the JSON report; returns the report."""
    from pipeline.config import get_settings, valid_case_id
    from pipeline.store import Store

    case_id = valid_case_id(case_id)
    settings = get_settings()
    d_gold, d_disputes = default_paths(case_id)
    gold_path, disputes_path = gold_path or d_gold, disputes_path or d_disputes
    gold = load_gold(gold_path) if gold_path.exists() else []
    disputes = load_disputes(disputes_path) if disputes_path.exists() else []
    if not gold:
        log.warning("no gold facts in %s", gold_path)
    db_path = Path(db_path or settings.db_path)
    if not db_path.exists():
        log.warning("database %s does not exist; run `make scan CASE=%s` first", db_path, case_id)
    with Store(db_path) as store:
        res = evaluate(store, case_id, gold, disputes)
    res["gold"] = {"timeline": str(gold_path), "disputes": str(disputes_path)}
    out_path = out_path or Path(settings.data_dir) / f"eval_{case_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("wrote %s", out_path)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval", description="Score a case (SPEC §13).")
    ap.add_argument("--case", default=None, help="case id (default: CT_CASE_ID)")
    ap.add_argument("--gold", type=Path, default=None, help="gold timeline JSON")
    ap.add_argument("--disputes", type=Path, default=None, help="gold disputes JSON")
    ap.add_argument("--db", type=Path, default=None, help="SQLite database (default: settings)")
    ap.add_argument("--out", type=Path, default=None, help="report (default data/eval_<case>)")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.case is None:
        from pipeline.config import get_settings

        args.case = get_settings().case_id
    try:
        res = run(
            args.case,
            gold_path=args.gold,
            disputes_path=args.disputes,
            db_path=args.db,
            out_path=args.out,
        )
    except Exception as exc:
        log.error("eval failed: %s", exc)
        return 1
    sys.stdout.write(format_table(res) + "\n")
    return 0
