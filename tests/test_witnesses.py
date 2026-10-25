"""Tests for pipeline.witnesses (store-backed, no model)."""

from __future__ import annotations

import pytest

from pipeline.witnesses import display_name, match_people, tokens
from tests.pipeline_helpers import new_store


@pytest.fixture
def store(tmp_path):
    s = new_store(tmp_path)
    yield s
    s.close()


def _names(store, ids):
    return [store.get_witness(i).name for i in ids]


def test_tokens_and_display_name():
    assert tokens("Mr. John  SMITH, Jr.") == ["john", "smith"]
    assert tokens("Officer Jamie Ortiz") == ["jamie", "ortiz"]
    assert display_name("Sgt.  Karen Whitfield") == "Karen Whitfield"
    assert display_name("Officer") == "Officer"


@pytest.mark.parametrize(
    "name",
    [
        "John Smith",
        "john  smith",
        "JOHN SMITH",
        "Mr. Smith",
        "J. Smith",
        "Smith",
        "John Q. Smith",
        "Mr. John Smith",
        "Johnny",
    ],
)
def test_variants_match_unambiguous_record(store, name):
    w = store.add_witness("demo", "John Smith", by="alice", aliases=["Johnny"])
    assert match_people(store, "demo", [name]) == [w.witness_id]
    assert len(store.list_witnesses("demo")) == 1


def test_first_name_alone_matches_when_unique(store):
    w = store.add_witness("demo", "Dana Rivera", by="alice")
    assert match_people(store, "demo", ["Dana", "Ms. Rivera"]) == [w.witness_id]


def test_ambiguous_surname_becomes_suggestion(store):
    store.add_witness("demo", "John Smith", by="alice")
    store.add_witness("demo", "Jane Smith", by="alice")
    (wid,) = match_people(store, "demo", ["Mr. Smith"])
    w = store.get_witness(wid)
    assert w.name == "Smith" and w.confirmed is False
    # J. Smith is ambiguous too; John Smith is not.
    assert _names(store, match_people(store, "demo", ["John Smith"])) == ["John Smith"]


def test_single_confirmed_wins_among_matches(store):
    confirmed = store.add_witness("demo", "Mark Ellison", by="alice")
    store.add_witness("demo", "M. Ellison", by="agent")  # an earlier AI suggestion
    assert match_people(store, "demo", ["Mr. Ellison"]) == [confirmed.witness_id]


def test_unmatched_names_become_suggestions_once(store):
    ids = match_people(store, "demo", ["Officer Jamie Ortiz", "Priya Nand", "Ortiz"])
    assert _names(store, ids) == ["Jamie Ortiz", "Priya Nand"]
    assert all(not store.get_witness(i).confirmed for i in ids)
    again = match_people(store, "demo", ["Jamie Ortiz", "Ms. Nand", "priya nand"])
    assert again == ids
    assert len(store.list_witnesses("demo")) == 2


def test_short_form_in_same_call_attaches_to_full_name(store):
    ids = match_people(store, "demo", ["Rivera", "Dana Rivera"])
    assert _names(store, ids) == ["Dana Rivera"]
    assert len(store.list_witnesses("demo")) == 1


def test_blank_and_pronouns_ignored(store):
    assert match_people(store, "demo", ["", "  ", "she", "Him"]) == []
    assert store.list_witnesses("demo") == []


def test_cases_are_separate(store):
    a = store.add_witness("a", "John Smith", by="alice")
    (wid,) = match_people(store, "b", ["John Smith"])
    assert wid != a.witness_id
    assert store.get_witness(wid).case_id == "b"


def test_full_name_falls_back_to_surname_only_record(store):
    (rivera,) = match_people(store, "demo", ["Ms. Rivera"])
    assert match_people(store, "demo", ["Dana Rivera"]) == [rivera]
    assert len(store.list_witnesses("demo")) == 1


def test_surname_fallback_not_used_when_surname_is_shared(store):
    store.add_witness("demo", "John Smith", by="alice")
    store.add_witness("demo", "Jane Smith", by="alice")
    (smith,) = match_people(store, "demo", ["Mr. Smith"])  # ambiguous: a "Smith" suggestion
    (bob,) = match_people(store, "demo", ["Bob Smith"])
    assert bob != smith and store.get_witness(bob).name == "Bob Smith"
