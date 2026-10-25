"""Match people named in drafts to witness records (SPEC §6.1 step 3).

Matching, in order:
1. Exact name or alias, ignoring case, spacing, periods and titles ("Mr.", "Officer", "Dr.").
2. A unique variant match: a bare surname ("Mr. Smith" → "Smith"), an initial plus surname
   ("J. Smith"), or the same first and last name with a different middle part. A bare first name
   ("Dana") matches only when no surname does, and a full name ("Dana Rivera") falls back to a
   surname-only record ("Rivera") when no other full name shares that surname. When several
   witnesses fit, a single confirmed one wins; otherwise the name is ambiguous and is not
   linked to anyone.
Names that match nobody (or are ambiguous) become suggested witnesses (``confirmed=False``)
for a person to confirm, merge or dismiss. An existing suggestion is reused, never duplicated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pipeline.schema import AGENT, Witness
from pipeline.store import Store

log = logging.getLogger(__name__)

TITLES = frozenset(
    "mr mrs ms miss mx dr prof professor officer ofc sgt sergeant det detective lt lieutenant "
    "capt captain judge justice hon honorable atty attorney counsel rev sir madam dame".split()
)
_SUFFIXES = frozenset("jr sr ii iii iv esq md phd".split())
_PRONOUNS = frozenset("i me he him she her they them we us you it someone nobody".split())


def tokens(name: str) -> list[str]:
    """Lowercased name tokens without titles, suffixes or punctuation."""
    parts = [t for t in re.split(r"[^\w'-]+", name.casefold()) if t]
    while parts and parts[0].strip("'-") in TITLES:
        parts = parts[1:]
    while parts and parts[-1].strip("'-") in _SUFFIXES:
        parts = parts[:-1]
    return [t.strip("'-") for t in parts if t.strip("'-")]


def display_name(name: str) -> str:
    """The name as written, minus leading titles ("Officer Jamie Ortiz" → "Jamie Ortiz")."""
    words = " ".join(name.split()).strip(" ,;:").split(" ")
    while len(words) > 1 and words[0].rstrip(".").casefold() in TITLES:
        words = words[1:]
    return " ".join(words)


def _key(name: str) -> str:
    return " ".join(tokens(name))


def _names(w: Witness) -> list[list[str]]:
    return [t for t in (tokens(n) for n in [w.name, *w.aliases]) if t]


def _variant(query: list[str], cand: list[str]) -> bool:
    """Does ``query`` (from a draft) plausibly name the same person as ``cand`` (a record)?"""
    if len(query) == 1:
        return len(cand) >= 2 and query[0] == cand[-1]
    if len(cand) < 2 or query[-1] != cand[-1]:
        return False
    first, cfirst = query[0], cand[0]
    if len(first) == 1:  # "J. Smith"
        return cfirst.startswith(first)
    if len(cfirst) == 1:  # record "J. Smith", query "John Smith"
        return first.startswith(cfirst)
    return first == cfirst  # same first and last name, middle names may differ


def _pick(matches: list[Witness]) -> Witness | None:
    if len(matches) == 1:
        return matches[0]
    confirmed = [w for w in matches if w.confirmed]
    return confirmed[0] if len(confirmed) == 1 else None


def find(witnesses: Sequence[Witness], name: str) -> Witness | None:
    """The one witness ``name`` refers to, or None (no match, or ambiguous)."""
    key = _key(name)
    if not key:
        return None
    exact = [w for w in witnesses if key in {" ".join(t) for t in _names(w)}]
    if exact:
        return _pick(exact) or sorted(exact, key=lambda w: (not w.confirmed, w.witness_id))[0]
    query = key.split(" ")
    variants = [w for w in witnesses if any(_variant(query, t) for t in _names(w))]
    if variants:
        return _pick(variants)
    if len(query) >= 2:  # "Dana Rivera" when all we have so far is a "Rivera" record
        if any(len(t) >= 2 and t[-1] == query[-1] for w in witnesses for t in _names(w)):
            return None  # another full name shares the surname: don't guess
        return _pick([w for w in witnesses if [query[-1]] in _names(w)])
    if len(query) == 1:  # a bare first name ("Dana"), only if nobody has it as a surname
        firsts = [w for w in witnesses if any(len(t) >= 2 and t[0] == query[0] for t in _names(w))]
        return _pick(firsts)
    return None


def match_people(store: Store, case_id: str, names: Sequence[str]) -> list[str]:
    """Witness ids for ``names`` (by name/alias); creates suggested witnesses as needed."""
    cleaned = []
    for raw in names:
        name = display_name(raw or "")
        key = _key(name)
        if not key or key in _PRONOUNS or not any(ch.isalpha() for ch in key):
            continue
        cleaned.append((raw, name, key))
    if not cleaned:
        return []
    witnesses = store.list_witnesses(case_id)
    resolved: dict[str, str] = {}
    # Full names first, so "Rivera" can attach to a "Dana Rivera" created in this same call.
    for _raw, name, key in sorted(cleaned, key=lambda c: -len(c[2].split(" "))):
        if key in resolved:
            continue
        w = find(witnesses, name)
        ambiguous = w is None and _is_ambiguous(witnesses, name)
        if w is None:
            # Reuse a suggestion with this exact name rather than creating a duplicate.
            same = [x for x in witnesses if not x.confirmed and _key(x.name) == key]
            if same:
                w = same[0]
            else:
                w = store.add_witness(case_id, name, by=AGENT)
                witnesses.append(w)
                log.info(
                    "suggested witness %s (%s)%s",
                    w.witness_id,
                    name,
                    " (ambiguous name)" if ambiguous else "",
                )
        resolved[key] = w.witness_id
    out: list[str] = []
    for _raw, _name, key in cleaned:
        wid = resolved[key]
        if wid not in out:
            out.append(wid)
    return out


def _is_ambiguous(witnesses: Sequence[Witness], name: str) -> bool:
    query = _key(name).split(" ")
    return sum(any(_variant(query, t) for t in _names(w)) for w in witnesses) > 1
