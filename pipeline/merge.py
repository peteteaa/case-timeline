"""Decide which drafts describe a fact already on the timeline (SPEC §6.1 step 4).

A draft merges into an existing fact (its quotes become new evidence on that fact) only when:
- **dates compatible**: both dated with overlapping ranges, or both undated with the same (or no)
  date text;
- **people agree**: when both sides name people they share at least one (names compare loosely:
  titles, initials and partial names, see :func:`names_match`; organizations don't count as
  people). Across documents a merge must never swallow a disagreement, so the two sides may not
  each name someone the other leaves out, and a one-sided extra person blocks the merge when
  either side says who was *not* there ("only", "nobody else", "was not present"). Silence about
  a person is not a disagreement;
- **similar**: near-identical title/details (rapidfuzz token_set *and* token_sort, so a short
  title that is a word subset of a long one doesn't match), a shared quote with a related title,
  enough shared topic words on the same day, or an embedding cosine for pairs passing the gates.
Drafts that look like a fact but disagree on the date or the people stay separate facts: that is
a possible contradiction, which ``conflicts`` checks. Rejected facts are never merge targets.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from rapidfuzz import fuzz

from pipeline.schema import DraftFact, Fact

log = logging.getLogger(__name__)

FUZZ_THRESHOLD = 85.0
# token_set_ratio scores 100 whenever one text's words are a subset of the other's ("Meeting
# held" vs "Meeting held with X about Y"), so a fuzzy hit also needs this much whole-string
# agreement (token_sort_ratio), which penalizes large length/content differences.
SORT_THRESHOLD = 55.0
QUOTE_THRESHOLD = 90.0  # a quote that matches one already cited by the fact
QUOTE_TITLE_THRESHOLD = 50.0  # ...still needs a related title
COSINE_THRESHOLD = 0.9
# Topic-word rule (same short date span only): shared content words of the titles, or of
# title + details, after dropping names, dates and stopwords.
TOPIC_MAX_SPAN_DAYS = 3
TOPIC_TITLE_MIN = 0.6  # overlap coefficient of title topic words (and >= 2 shared)
TOPIC_FULL_JACCARD = 0.4  # Jaccard of title + details topic words (and >= 3 shared)

_WS = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def _preprocess(text: str) -> str:
    return _WS.sub(" ", re.sub(r"[^\w\s]", " ", text)).strip().casefold()


def similarity(a: str, b: str) -> float:
    """rapidfuzz token_set_ratio (0-100) on punctuation-free, case-folded text."""
    return fuzz.token_set_ratio(a, b, processor=_preprocess)


def fact_text(title: str, details: str | None) -> str:
    return f"{title}. {details}" if details else title


def fact_similarity(
    title_a: str, details_a: str | None, title_b: str, details_b: str | None
) -> float:
    """Best of title-only and title-plus-details similarity (details often differ in length)."""
    return max(
        similarity(title_a, title_b),
        similarity(fact_text(title_a, details_a), fact_text(title_b, details_b)),
    )


def near_identical(a: str, b: str) -> bool:
    """High token_set_ratio *and* reasonable token_sort_ratio (no subset-only matches)."""
    if similarity(a, b) < FUZZ_THRESHOLD:
        return False
    return fuzz.token_sort_ratio(a, b, processor=_preprocess) >= SORT_THRESHOLD


def fuzzy_duplicate(
    title_a: str, details_a: str | None, title_b: str, details_b: str | None
) -> bool:
    """Titles, or titles plus details, are near-identical (see :func:`near_identical`)."""
    return near_identical(title_a, title_b) or near_identical(
        fact_text(title_a, details_a), fact_text(title_b, details_b)
    )


# --------------------------------------------------------------------------- names

# Titles and honorifics dropped before comparing names ("Dr. Lopez" ~ "Ana Lopez").
_HONORIFICS = frozenset(
    "mr mrs ms miss mx dr prof professor officer ofc sgt sergeant det detective cpl corporal "
    "lt lieutenant capt captain cmdr commander chief judge justice hon honorable atty attorney "
    "counsel esq sir madam dame rev father agent deputy sheriff trooper inspector investigator "
    "witness plaintiff defendant the jr sr ii iii iv".split()
)
# Words that mark an organization rather than a person.
_ORG_CUES = frozenset(
    "co company corp corporation inc incorporated llc llp lp ltd plc gmbh group holdings "
    "department dept division agency board bureau office administration authority commission "
    "council court police sheriff's university school hospital bank association union "
    "airlines airline services industries partners firm foundation trust committee state "
    "county city federal national hr freight".split()
)
_NAME_TOKEN = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?")


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-cased name tokens with honorifics, initials and possessives removed."""
    toks = []
    for raw in _NAME_TOKEN.findall(name.casefold()):
        tok = re.sub(r"['\u2019]s$", "", raw)
        if tok in _HONORIFICS or (len(tok) == 1 and tok.isalpha()):
            continue
        toks.append(tok)
    return tuple(toks)


def is_organization(name: str) -> bool:
    """Heuristic: the name contains an organization cue word ("Co.", "Department", ...)."""
    return any(t in _ORG_CUES for t in _NAME_TOKEN.findall(name.casefold()))


def names_match(x: str, y: str) -> bool:
    """Loose person/org name match: equal, one name's tokens contained in the other's
    ("Lopez" ~ "Ana Lopez", "Mr. Okafor" ~ "Sam Okafor"), or a near-identical spelling."""
    tx, ty = name_tokens(x), name_tokens(y)
    if not tx or not ty:
        return bool(_norm_text(x)) and _norm_text(x) == _norm_text(y)
    sx, sy = set(tx), set(ty)
    if sx <= sy or sy <= sx:
        return True
    return fuzz.token_sort_ratio(" ".join(tx), " ".join(ty)) >= 92


@dataclass(frozen=True)
class Person:
    """Someone named by a draft or a fact: a witness id and/or the names it goes by."""

    key: str  # witness id, or "name:<name>" for an unlinked name
    names: tuple[str, ...] = ()

    @property
    def is_org(self) -> bool:
        return any(is_organization(n) for n in self.names)

    def same(self, other: Person) -> bool:
        if self.key == other.key:
            return True
        return any(names_match(x, y) for x in self.names for y in other.names)


def people_of(
    witness_ids: Iterable[str],
    witness_names: Mapping[str, Sequence[str]] | None = None,
    extra_names: Iterable[str] = (),
) -> list[Person]:
    """People for witness ids (with their names, when known), plus unlinked ``extra_names``."""
    names = witness_names or {}
    out = [Person(w, tuple(names.get(w, ()))) for w in dict.fromkeys(witness_ids)]
    for n in extra_names:
        if n.strip() and not any(names_match(n, x) for p in out for x in p.names):
            out.append(Person(f"name:{_norm_text(n)}", (n,)))
    return out


def unmatched(a: Sequence[Person], b: Sequence[Person]) -> list[Person]:
    """People (not organizations) in ``a`` that nobody in ``b`` matches."""
    return [p for p in a if not p.is_org and not any(p.same(q) for q in b)]


def shares_person(a: Sequence[Person], b: Sequence[Person]) -> bool:
    return any(p.same(q) for p in a for q in b)


# Someone saying who was *not* there: a one-sided extra person is then a disagreement.
EXCLUSION_CUES = re.compile(
    r"\b(only|nobody|no\s+one|no-one|alone|absent|except|other\s+than|else|neither|nor|"
    r"(?:was|were|wasn't|weren't)\s+(?:not\s+)?(?:present|there|in\s+the\s+room))\b",
    re.IGNORECASE,
)


def people_agree(
    a: Sequence[Person], b: Sequence[Person], *, cross_doc: bool, texts: Iterable[str] = ()
) -> bool:
    """May two accounts naming ``a`` and ``b`` be merged into one fact?

    Either side naming nobody agrees with anything. Otherwise they must share someone. Within
    one document that is enough. Across documents the two sides may not each name a person the
    other leaves out (a different cast), and a person only one side names blocks the merge when
    either side affirmatively excludes people (``EXCLUSION_CUES`` in ``texts``).
    """
    if not a or not b:
        return True
    if not shares_person(a, b):
        return False
    if not cross_doc:
        return True
    extra_a, extra_b = unmatched(a, b), unmatched(b, a)
    if extra_a and extra_b:
        return False
    if extra_a or extra_b:
        return not any(EXCLUSION_CUES.search(t) for t in texts if t)
    return True


# --------------------------------------------------------------------------- topic words

_STOPWORDS = frozenset(
    """a an the of to in on at for with by from and or but if was were is are be been being
    has had have having do does did that this these those his her hers their theirs its as
    into onto about after before during between over under than then there here also any all
    some each other another such same one two three first second he she they it i me my we
    our you your him them who whom whose which what when where while how why not no so very
    just only both either neither said says stated stating state told tells tell asked
    according approximately around later earlier sometime time day days week weeks
    month months year years today yesterday morning afternoon evening night am pm
    mr mrs ms dr take took taken taking place held hold holds holding believe believed
    kept keep keeps went go goes going came come comes made make makes get got gave give
    given put way thing things matter re regarding concerning without against per via""".split()
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WEEKDAYS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thurs fri "
    "sat sun".split()
)
_MONTH_SET = frozenset(_MONTHS.split("|"))
_DATE_NUMBERS = re.compile(
    rf"\b(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})"
    rf"|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}:\d{{2}}"
    rf"|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ions", "ion", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude stem: strip one common suffix, then keep at most seven characters."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:7]


_CAPITALIZED = re.compile(r"(?<![.!?:;]\s)(?<!^)(?<![\n\"(])\b([A-Z][^\W\d_]+)")


def proper_words(*texts: str) -> set[str]:
    """Lower-cased words written capitalized mid-sentence (names, places, organizations)."""
    return {m.group(1).casefold() for t in texts for m in _CAPITALIZED.finditer(t.strip())}


def topic_tokens(text: str, names: Iterable[str] = ()) -> set[str]:
    """Stemmed content words of ``text``, minus dates, stopwords and the given names (full
    names or single lower-cased words)."""
    skip: set[str] = set()
    for n in names:
        skip.update(name_tokens(n))
        skip.update(w.casefold() for w in _WORD.findall(n))
    out: set[str] = set()
    for word in _WORD.findall(_DATE_NUMBERS.sub(" ", text.casefold())):
        if word in _STOPWORDS or word in _MONTH_SET or word in _WEEKDAYS or word in skip:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        out.add(stem(word))
    return out


def overlap(a: set[str], b: set[str]) -> tuple[float, int]:
    """Overlap coefficient (shared / smaller set) and the number of shared words."""
    if not a or not b:
        return 0.0, 0
    shared = len(a & b)
    return shared / min(len(a), len(b)), shared


def topic_match(
    title_a: str,
    details_a: str | None,
    title_b: str,
    details_b: str | None,
    names: Iterable[str] = (),
) -> float:
    """0..1 topic agreement of two facts, 0 unless the topic rule holds.

    Holds when the titles share at least two content words covering ``TOPIC_TITLE_MIN`` of the
    shorter title, or title + details share three or more words with Jaccard at least
    ``TOPIC_FULL_JACCARD``.
    """
    texts = (title_a, details_a or "", title_b, details_b or "")
    names = [*names, *proper_words(*texts)]
    ta, tb = topic_tokens(title_a, names), topic_tokens(title_b, names)
    coeff, shared = overlap(ta, tb)
    best = coeff if shared >= 2 and coeff >= TOPIC_TITLE_MIN else 0.0
    fa = topic_tokens(fact_text(title_a, details_a), names)
    fb = topic_tokens(fact_text(title_b, details_b), names)
    if fa and fb:
        both = len(fa & fb)
        jac = both / len(fa | fb)
        if both >= 3 and jac >= TOPIC_FULL_JACCARD:
            best = max(best, jac)
    return best


# --------------------------------------------------------------------------- dates

Interval = tuple[date, date]


def interval(start: date | None, end: date | None) -> Interval | None:
    if start is None:
        return None
    end = end or start
    return (min(start, end), max(start, end))


def gap_days(a: Interval | None, b: Interval | None) -> int | None:
    """Days between two date intervals (0 if they overlap); None if either is undated."""
    if a is None or b is None:
        return None
    if a[1] < b[0]:
        return (b[0] - a[1]).days
    if b[1] < a[0]:
        return (a[0] - b[1]).days
    return 0


def span_days(i: Interval | None) -> int | None:
    return None if i is None else (i[1] - i[0]).days


def dates_compatible(draft: DraftFact, fact: Fact) -> bool:
    gap = gap_days(
        interval(draft.date_start, draft.date_end), interval(fact.date_start, fact.date_end)
    )
    if gap is not None:
        return gap == 0
    if draft.date_start is None and fact.date_start is None:
        ta, tb = _norm_text(draft.date_text or ""), _norm_text(fact.date_text or "")
        return not ta or not tb or ta == tb
    return False  # one dated, one not: not enough to call them the same event


def _short_same_day(draft: DraftFact, fact: Fact) -> bool:
    """Both dated within a few days' span (the topic rule needs a precise date)."""
    sa = span_days(interval(draft.date_start, draft.date_end))
    sb = span_days(interval(fact.date_start, fact.date_end))
    return sa is not None and sb is not None and max(sa, sb) <= TOPIC_MAX_SPAN_DAYS


def people_compatible(draft_witnesses: Sequence[str], fact: Fact) -> bool:
    """True unless both sides name people and the sets are disjoint (ids only)."""
    a, b = set(draft_witnesses), set(fact.witness_ids)
    return not (a and b) or bool(a & b)


def fact_docs(fact: Fact) -> set[str]:
    return {e.doc_id for e in fact.evidence if e.doc_id and e.review != "rejected"}


def draft_docs(draft: DraftFact) -> set[str]:
    return {s.doc_id for s in draft.spans}


def _fact_texts(fact: Fact) -> list[str]:
    return [fact.title, fact.details or "", *(e.quote or "" for e in fact.evidence)]


def _draft_texts(draft: DraftFact) -> list[str]:
    return [draft.title, draft.details or "", *draft.quotes, *(s.quote for s in draft.spans)]


def may_merge(
    draft: DraftFact,
    fact: Fact,
    draft_people: Sequence[Person] | None,
    witness_names: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """Date and people gate for attaching ``draft`` to ``fact``.

    ``draft_people`` None means the draft's people are unknown (no witness matching): people
    don't block the merge then. Within one document (every source of the fact is the draft's
    document) sharing one person is enough; across documents :func:`people_agree` applies.
    """
    if not dates_compatible(draft, fact):
        return False
    if draft_people is None:
        return True
    docs = fact_docs(fact)
    cross_doc = not docs or not docs <= draft_docs(draft)
    fp = people_of(fact.witness_ids, witness_names)
    return people_agree(
        draft_people, fp, cross_doc=cross_doc, texts=[*_draft_texts(draft), *_fact_texts(fact)]
    )


def _quote_hit(draft: DraftFact, fact: Fact) -> bool:
    """The draft cites a span the fact already cites, or a near-identical quote."""
    for span in draft.spans:
        for ev in fact.evidence:
            if ev.kind != "span" or ev.quote is None:
                continue
            if (
                ev.doc_id == span.doc_id
                and ev.char_start is not None
                and ev.char_end is not None
                and ev.char_start < span.char_end
                and span.char_start < ev.char_end
            ):
                return True
            if fuzz.ratio(_preprocess(span.quote), _preprocess(ev.quote)) >= QUOTE_THRESHOLD:
                return True
    return False


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=False))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed via the local model; None when the embedder is unavailable."""
    try:
        from pipeline import llm

        return llm.embed(texts)
    except Exception as exc:  # embedder optional: fall back to fuzzy matching only
        log.warning("embedding unavailable, using fuzzy matching only: %s", exc)
        return None


def plan(
    drafts: Sequence[DraftFact],
    existing: Sequence[Fact],
    *,
    witness_ids: Mapping[str, Sequence[str]] | None = None,
    witness_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str | None]:
    """Map each draft key to the existing fact id it duplicates, or None for a new fact.

    ``witness_ids`` (draft key → matched witness ids) drives the people check; without it,
    people don't block a merge. ``witness_names`` (witness id → name and aliases) is optional:
    with it, names compare loosely across different witness records and organizations are
    told apart from people; without it, witness ids compare exactly.
    """
    live = [f for f in existing if f.review != "rejected"]
    # (draft key) -> [(fact, score, fuzzy/quote/topic hit)] for facts passing the gates
    candidates: dict[str, list[tuple[Fact, float, bool]]] = {}
    for d in drafts:
        dp = None
        if witness_ids is not None:
            dp = people_of(witness_ids.get(d.key, ()), witness_names)
        names = [*d.people, *(n for ns in (witness_names or {}).values() for n in ns)]
        hits: list[tuple[Fact, float, bool]] = []
        for f in live:
            if not may_merge(d, f, dp, witness_names):
                continue
            score = fact_similarity(d.title, d.details, f.title, f.details)
            hit = fuzzy_duplicate(d.title, d.details, f.title, f.details) or (
                score >= QUOTE_TITLE_THRESHOLD and _quote_hit(d, f)
            )
            if not hit and _short_same_day(d, f):
                topic = topic_match(d.title, d.details, f.title, f.details, names)
                if topic:
                    hit, score = True, max(score, 100 * topic)
            hits.append((f, score, hit))
        candidates[d.key] = hits

    vectors = _embed_needed(drafts, candidates)
    out: dict[str, str | None] = {}
    for d in drafts:
        dtext = fact_text(d.title, d.details)
        best: tuple[bool, float, float, str] | None = None
        for f, score, hit in candidates[d.key]:
            if not hit:
                cos = _cosine_of(vectors, f"d:{d.key}", f"f:{f.fact_id}")
                if cos is None or cos < COSINE_THRESHOLD:
                    continue
                score = max(score, cos * 100)
            # token_set_ratio ties at 100 for any subset; break ties on the plain ratio.
            tie = fuzz.ratio(_preprocess(dtext), _preprocess(fact_text(f.title, f.details)))
            rank = (hit, score, tie, f.fact_id)
            if best is None or rank[:3] > best[:3]:
                best = rank
        out[d.key] = best[3] if best else None
    n = sum(1 for v in out.values() if v)
    log.info("merge plan: %d new, %d attach to existing facts", len(out) - n, n)
    return out


def _embed_needed(
    drafts: Sequence[DraftFact], candidates: dict[str, list[tuple[Fact, float, bool]]]
) -> dict[str, list[float]]:
    """Embed only drafts (and facts) in candidate pairs that the fuzzy test didn't settle."""
    needed: dict[str, str] = {}
    for d in drafts:
        hits = candidates[d.key]
        if any(hit for _, _, hit in hits):
            continue
        for f, _, _ in hits:
            needed[f"d:{d.key}"] = fact_text(d.title, d.details)
            needed[f"f:{f.fact_id}"] = fact_text(f.title, f.details)
    if not needed:
        return {}
    ids = list(needed)
    vecs = _embed([needed[i] for i in ids])
    if not vecs or len(vecs) != len(ids):
        return {}
    return dict(zip(ids, vecs, strict=True))


def _cosine_of(vectors: dict[str, list[float]], a: str, b: str) -> float | None:
    va, vb = vectors.get(a), vectors.get(b)
    if va is None or vb is None:
        return None
    return _cosine(va, vb)
