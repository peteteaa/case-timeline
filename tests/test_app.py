"""Web UI tests: a real temp Store, no model, worker or OpenClaw (SPEC §5.2, §6, §7)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import app as app_mod
from app.app import create_app, host_is_loopback, sanitize_filename
from app.helpers import source_segments
from pipeline import digest, export, search
from pipeline.config import Settings
from pipeline.schema import AGENT, Page, SpanIn
from pipeline.store import Store

XSS = "<script>alert(1)</script>"
PAGE1 = f"Intro.\nOn March 3, 2024 {XSS} Smith signed the lease.\n"
PAGE2 = 'Rent was due no later than May 1, 2024. <img src=x onerror="alert(2)">\n'
TEXT = PAGE1 + PAGE2
QUOTE = "Smith signed the lease"
Q_START = TEXT.index(QUOTE)
Q_END = Q_START + len(QUOTE)
EMAIL_TEXT = "From: Jones\nThe rent was paid late on May 3, 2024.\n"


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.evidence = tmp_path / "evidence"
        self.evidence.mkdir()
        self.settings = Settings(
            evidence_dir=self.evidence,
            data_dir=tmp_path / "data",
            case_id="demo",
            run_worker=False,
        )
        self.store = Store(self.settings.db_path, embedder=lambda t: [[0.0] * 1024 for _ in t])
        s = self.store
        self.doc, _ = s.register_document(
            "demo",
            filename="complaint.txt",
            rel_path="complaint.txt",
            sha256="a" * 64,
            location="evidence",
            kind="pleading",
        )
        s.save_parsed_text(
            self.doc.doc_id,
            TEXT,
            [
                Page(number=1, char_start=0, char_end=len(PAGE1)),
                Page(number=2, char_start=len(PAGE1), char_end=len(TEXT)),
            ],
        )
        s.set_parse_result(self.doc.doc_id, "parsed", pages=2)
        self.email, _ = s.register_document(
            "demo",
            filename="email.eml",
            rel_path="email.eml",
            sha256="b" * 64,
            location="evidence",
            kind="email",
        )
        s.save_parsed_text(
            self.email.doc_id, EMAIL_TEXT, [Page(number=1, char_start=0, char_end=len(EMAIL_TEXT))]
        )
        s.set_parse_result(self.email.doc_id, "parsed", pages=1)

        # F-0001: AI draft, dated, cites the pleading only (alleged).
        self.f1 = s.add_fact(
            "demo",
            "Smith signed the lease",
            by=AGENT,
            date_start="2024-03-03",
            date_certainty="exact",
            confidence=0.9,
            spans=[SpanIn(doc_id=self.doc.doc_id, char_start=Q_START, char_end=Q_END, quote=QUOTE)],
        )
        # F-0002: AI draft, undated, low confidence, cites the email.
        q2 = "The rent was paid late"
        a2 = EMAIL_TEXT.index(q2)
        self.f2 = s.add_fact(
            "demo",
            "Rent paid late",
            by=AGENT,
            confidence=0.2,
            spans=[
                SpanIn(doc_id=self.email.doc_id, char_start=a2, char_end=a2 + len(q2), quote=q2)
            ],
        )
        # F-0003: a person's fact, a date range, tagged and a key fact.
        self.tag = s.add_tag("demo", "Breach", by="Bob", color="#aa3300")
        self.f3 = s.add_fact(
            "demo",
            "Lease term",
            by="Bob",
            date_start="2024-01-01",
            date_end="2024-12-31",
            date_certainty="approximate",
            statuses=["key_fact"],
            tag_ids=[self.tag.tag_id],
        )
        self.witness = s.add_witness("demo", "John Smith", by="Bob", aliases=["J. Smith"])
        s.update_fact(self.f3.fact_id, {"witness_ids": [self.witness.witness_id]}, by="Bob")

        self.app = create_app(self.settings, store=self.store)
        self.client = self.new_client(user="Alice")

    def new_client(self, user: str | None = None) -> TestClient:
        client = TestClient(self.app, base_url="http://127.0.0.1", follow_redirects=False)
        if user:
            client.cookies.set("ct_user", user)
        return client


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    yield e
    e.store.close()


HTML = {"accept": "text/html"}
HX = {"hx-request": "true"}


def row_ids(html: str) -> list[str]:
    return re.findall(r'<tr id="(F-\d+)"', html)


# --------------------------------------------------------------------------- helpers, security


def test_helpers() -> None:
    assert host_is_loopback("127.0.0.1") and host_is_loopback("::1")
    assert host_is_loopback("localhost") and not host_is_loopback("0.0.0.0")
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..") is None


def test_host_and_origin_guard(env: Env) -> None:
    c = env.client
    assert c.get("/cases/demo/facts", headers={"host": "evil.example"}).status_code == 400
    r = c.post("/cases/demo/facts", data={"title": "x"}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = c.post("/cases/demo/facts", data={"title": "x"}, headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403
    r = c.post("/cases/demo/facts", data={"title": "ok"}, headers={"origin": "http://127.0.0.1"})
    assert r.status_code == 303


def test_security_headers_and_no_cdn(env: Env) -> None:
    for url in (
        "/cases/demo/facts",
        "/cases/demo/facts?view=visual",
        f"/documents/{env.doc.doc_id}/view",
    ):
        r = env.client.get(url)
        assert r.status_code == 200
        csp = r.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "unsafe-eval" not in csp
        assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
        assert r.headers["cache-control"] == "no-store"
        assert r.headers["x-content-type-options"] == "nosniff"
        # "no-referrer" would make browsers send "Origin: null" on our own POSTs (refused).
        assert r.headers["referrer-policy"] == "same-origin"
        srcs = re.findall(r'(?:src|href)="([^"]+)"', r.text)
        assert not [s for s in srcs if s.startswith(("http:", "https:", "//"))], srcs
        assert "<script>" not in r.text  # no inline scripts
        assert "DISCO" not in r.text and "Cecilia" not in r.text


def test_root_redirect_and_single_case(env: Env) -> None:
    r = env.client.get("/")
    assert r.status_code == 303 and r.headers["location"] == "/cases/demo/facts"
    assert env.client.get("/cases/other/facts").status_code == 404
    assert env.client.get("/cases/other").status_code == 404
    assert env.client.get("/facts/F-9999").status_code == 404
    assert env.client.get("/facts/F-9999", headers=HTML).text.count("Not found") >= 1


# --------------------------------------------------------------------------- display name


def test_writes_require_display_name(env: Env) -> None:
    anon = env.new_client()
    before = len(env.store.list_facts("demo"))
    r = anon.post("/cases/demo/facts", data={"title": "Anonymous"})
    assert r.status_code == 303 and r.headers["location"].startswith("/name")
    r = anon.patch(f"/facts/{env.f1.fact_id}", data={"title": "x"})
    assert r.status_code == 401
    r = anon.post(f"/facts/{env.f1.fact_id}/review", data={"action": "accept"}, headers=HX)
    assert r.status_code == 401 and r.headers["hx-redirect"] == "/name"
    assert len(env.store.list_facts("demo")) == before
    assert env.store.get_fact(env.f1.fact_id).review == "draft"
    # Reading works without a name, and the page asks for one.
    page = anon.get("/cases/demo/facts")
    assert page.status_code == 200 and 'action="/name"' in page.text


def test_set_display_name_cookie(env: Env) -> None:
    anon = env.new_client()
    r = anon.post("/name", data={"name": "  Jane   Doe ", "next": "/cases/demo/tags"})
    assert r.status_code == 303 and r.headers["location"] == "/cases/demo/tags"
    cookie = r.headers["set-cookie"]
    assert "ct_user=Jane%20Doe" in cookie and "samesite=strict" in cookie.lower()
    assert "httponly" in cookie.lower()
    r = anon.post("/cases/demo/facts", data={"title": "Named"})
    assert r.status_code == 303
    fid = r.headers["location"].rsplit("/", 1)[1]
    assert env.store.get_fact(fid).created_by == "Jane Doe"
    assert anon.post("/name", data={"name": "agent"}).status_code == 422
    r = anon.post("/name", data={"name": "Ann", "next": "//evil.example/x"})
    assert r.headers["location"] == "/cases/demo/facts"


# --------------------------------------------------------------------------- list view


def test_list_default_and_undated_group(env: Env) -> None:
    html = env.client.get("/cases/demo/facts").text
    ids = row_ids(html)
    assert ids == [env.f3.fact_id, env.f1.fact_id, env.f2.fact_id]  # undated last
    assert html.index("Undated <span") < html.index(f'<tr id="{env.f2.fact_id}"')
    assert html.index(f'<tr id="{env.f1.fact_id}"') < html.index("Undated <span")
    # badges: alleged (pleading-only evidence), draft, key fact, tag chip
    row1 = html[html.index(f'<tr id="{env.f1.fact_id}"') :]
    row1 = row1[: row1.index("</tr>")]
    assert "Alleged" in row1 and "Draft" in row1
    assert 'name="fact_ids"' in row1  # bulk checkbox on drafts
    row3 = html[html.index(f'<tr id="{env.f3.fact_id}"') :]
    row3 = row3[: row3.index("</tr>")]
    assert "Key fact" in row3 and "Breach" in row3 and 'name="fact_ids"' not in row3
    assert "~2024-01-01 – 2024-12-31" in row3


def test_list_filters_and_sort(env: Env) -> None:
    c = env.client
    get = lambda qs: row_ids(c.get("/cases/demo/facts?" + qs).text)  # noqa: E731
    assert get("status=key_fact") == [env.f3.fact_id]
    assert get(f"tag={env.tag.tag_id}") == [env.f3.fact_id]
    assert get(f"witness={env.witness.witness_id}") == [env.f3.fact_id]
    assert get(f"doc={env.email.doc_id}") == [env.f2.fact_id]
    assert get("kind=pleading") == [env.f1.fact_id]
    assert get("origin=user") == [env.f3.fact_id]
    assert get("review=accepted") == [env.f3.fact_id]
    assert get("date_from=2024-03-01&date_to=2024-03-31") == [env.f3.fact_id, env.f1.fact_id]
    # Title sort, then descending; undated facts still grouped at the end.
    assert get("sort=title&dir=desc") == [env.f1.fact_id, env.f3.fact_id, env.f2.fact_id]
    assert get("sort=title&dir=asc") == [env.f3.fact_id, env.f1.fact_id, env.f2.fact_id]
    # Filters combine and persist in links (sort headers, export form).
    html = c.get("/cases/demo/facts?kind=pleading&status=key_fact&dir=desc").text
    assert row_ids(html) == []
    assert 'name="kind" value="pleading"' in html
    assert "status=key_fact&amp;kind=pleading&amp;dir=desc" in html  # view toggle link
    # A bad date gives a friendly message, not a 500.
    r = c.get("/cases/demo/facts?date_from=yesterday")
    assert r.status_code == 200 and "From date must be a date" in r.text
    env.store.review_fact(env.f2.fact_id, "reject", reviewer="Bob")
    assert env.f2.fact_id not in get("")
    assert get("review=rejected") == [env.f2.fact_id]


def test_search_combines_with_filters(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(store: Store, case_id: str, query: str, limit: int = 1000) -> list[str]:
        seen.append(query)
        if query == "bad &":
            raise search.QueryError("dangling connector &")
        return [env.f1.fact_id, env.f3.fact_id]

    monkeypatch.setattr(search, "keyword_fact_ids", fake)
    html = env.client.get("/cases/demo/facts?q=lease").text
    assert row_ids(html) == [env.f3.fact_id, env.f1.fact_id]
    assert search.SYNTAX_HELP.split(" ")[0] in html
    assert row_ids(env.client.get("/cases/demo/facts?q=lease&origin=ai").text) == [env.f1.fact_id]
    html = env.client.get("/cases/demo/facts?q=bad+%26").text
    assert "Search not understood" in html and "dangling connector" in html
    assert row_ids(html) == []
    assert seen == ["lease", "lease", "bad &"]


def test_json_for_visual_view(env: Env) -> None:
    data = env.client.get("/cases/demo/facts?format=json").json()
    items = {i["id"]: i for i in data["items"]}
    assert set(items) == {env.f1.fact_id, env.f2.fact_id, env.f3.fact_id}
    f1, f2, f3 = items[env.f1.fact_id], items[env.f2.fact_id], items[env.f3.fact_id]
    assert f1["start"] == "2024-03-03" and f1["end"] is None and f1["type"] == "point"
    assert f1["alleged"] and f1["review"] == "draft" and "review-draft" in f1["class_name"]
    assert f2["undated"] and f2["start"] is None
    assert f3["type"] == "range" and f3["end"] == "2025-01-01"  # vis end is exclusive
    assert f3["key_fact"] and "st-key_fact" in f3["class_name"]
    assert f1["url"] == f"/facts/{env.f1.fact_id}"
    filtered = env.client.get("/cases/demo/facts?format=json&status=key_fact").json()
    assert [i["id"] for i in filtered["items"]] == [env.f3.fact_id]
    page = env.client.get("/cases/demo/facts?view=visual").text
    assert 'id="timeline"' in page and "format=json" in page and 'id="undated-lane"' in page
    assert "/static/vendor/vis-timeline-graph2d.min.js" in page


# --------------------------------------------------------------------------- facts


def test_create_fact(env: Env) -> None:
    r = env.client.post(
        "/cases/demo/facts",
        data={"title": "Meeting held", "date_start": "2024-02-02", "details": "At HQ"},
    )
    assert r.status_code == 303
    fid = r.headers["location"].rsplit("/", 1)[1]
    fact = env.store.get_fact(fid)
    assert fact.review == "accepted" and fact.origin == "user" and fact.created_by == "Alice"
    assert fact.date_start == date(2024, 2, 2) and fact.details == "At HQ"
    assert fact.date_certainty == "exact"
    assert env.client.post("/cases/demo/facts", data={"title": "  "}).status_code == 422
    r = env.client.post("/cases/demo/facts", json={"title": "Via JSON"})
    assert r.status_code == 201 and r.json()["created_by"] == "Alice"


def test_patch_edit_of_draft_accepts_it(env: Env) -> None:
    fid = env.f1.fact_id
    r = env.client.patch(
        f"/facts/{fid}",
        data={
            "title": "Smith signed the lease agreement",
            "has": ["statuses", "tag_ids"],
            "statuses": ["key_fact", "case_impact"],
            "new_tag": "Signing",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["review"] == "accepted" and body["title"] == "Smith signed the lease agreement"
    assert body["statuses"] == ["key_fact", "case_impact"]
    tag = next(t for t in env.store.list_tags("demo") if t.name == "Signing")
    assert body["tag_ids"] == [tag.tag_id]
    assert all(e.review == "accepted" for e in env.store.get_fact(fid).evidence)
    events = env.store.list_review_events("demo")
    assert events[-1].action == "edit" and events[-1].reviewer == "Alice"
    fields = {h["field"] for h in env.store.fact_history(fid)}
    assert {"title", "statuses", "tag_ids", "review"} <= fields
    # Plain HTML form: POST /facts/{id}, unchecked boxes clear statuses.
    r = env.client.post(f"/facts/{fid}", data={"title": "Signed", "has": "statuses"})
    assert r.status_code == 303 and r.headers["location"] == f"/facts/{fid}"
    assert env.store.get_fact(fid).statuses == []
    assert env.client.patch(f"/facts/{fid}", json={"date_start": "nope"}).status_code == 422
    assert env.client.patch(f"/facts/{fid}", json={"statuses": ["bogus"]}).status_code == 422
    # The card shows the history with the editor's name.
    card = env.client.get(f"/facts/{fid}").text
    assert "Alice" in card and "History" in card and "Signed" in card


def test_review_accept_reject_with_reason(env: Env) -> None:
    r = env.client.post(
        f"/facts/{env.f2.fact_id}/review",
        data={"action": "reject", "reason": "not a fact", "next": "/cases/demo/review"},
    )
    assert r.status_code == 303 and r.headers["location"] == "/cases/demo/review"
    assert env.store.get_fact(env.f2.fact_id).review == "rejected"
    ev = env.store.list_review_events("demo")[-1]
    assert (ev.fact_id, ev.action, ev.reason, ev.reviewer) == (
        env.f2.fact_id,
        "reject",
        "not a fact",
        "Alice",
    )
    r = env.client.post(f"/facts/{env.f1.fact_id}/review", data={"action": "accept"}, headers=HX)
    assert r.status_code == 200
    assert f'id="rv-{env.f1.fact_id}"' in r.text and "accepted" in r.text
    assert env.store.get_fact(env.f1.fact_id).review == "accepted"
    r = env.client.post(f"/facts/{env.f1.fact_id}/review", data={"action": "nope"})
    assert r.status_code == 422
    # Review with edits through the review route.
    f = env.store.add_fact("demo", "Drafty", by=AGENT)
    r = env.client.post(
        f"/facts/{f.fact_id}/review", data={"action": "edit", "title": "Fixed", "reason": "typo"}
    )
    assert r.status_code == 303
    assert env.store.get_fact(f.fact_id).title == "Fixed"
    assert env.store.list_review_events("demo")[-1].reason == "typo"


def test_needs_review_page_and_bulk(env: Env) -> None:
    html = env.client.get("/cases/demo/review").text
    assert html.index(env.f2.fact_id) < html.index(env.f1.fact_id)  # lowest confidence first
    assert env.f3.fact_id not in html
    assert f"/documents/{env.doc.doc_id}/view?evidence=" in html and QUOTE in html
    high = env.client.get("/cases/demo/review?order=high").text
    assert high.index(env.f1.fact_id) < high.index(env.f2.fact_id)
    by_doc = env.client.get(f"/cases/demo/review?doc={env.email.doc_id}").text
    assert env.f2.fact_id in by_doc and f'id="{env.f1.fact_id}"' not in by_doc
    r = env.client.post(
        "/cases/demo/review",
        data={
            "fact_ids": [env.f1.fact_id, env.f2.fact_id],
            "action": "reject",
            "reason": "duplicates",
        },
    )
    assert r.status_code == 303
    assert {env.store.get_fact(f).review for f in (env.f1.fact_id, env.f2.fact_id)} == {"rejected"}
    reasons = [e.reason for e in env.store.list_review_events("demo")]
    assert reasons[-2:] == ["duplicates", "duplicates"]
    assert env.client.post("/cases/demo/review", data={"action": "accept"}).status_code == 422
    r = env.client.post("/cases/demo/review", data={"fact_ids": "F-9999", "action": "accept"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- evidence, notes


def test_span_evidence_from_offsets(env: Env) -> None:
    fid = env.f3.fact_id
    start = TEXT.index("Rent was due")
    end = start + len("Rent was due no later than May 1, 2024.")
    r = env.client.post(
        f"/facts/{fid}/evidence",
        data={"kind": "span", "doc_id": env.doc.doc_id, "char_start": start, "char_end": end},
    )
    assert r.status_code == 303
    ev = env.store.get_fact(fid).evidence[-1]
    assert ev.quote == TEXT[start:end] and ev.page == 2 and ev.review == "accepted"
    assert r.headers["location"] == f"/documents/{env.doc.doc_id}/view?evidence={ev.evidence_id}"
    bad = [
        {"char_start": start, "char_end": end, "quote": "something else"},
        {"char_start": 0, "char_end": len(TEXT) + 5},
        {"char_start": 10, "char_end": 5},
        {"char_start": "x", "char_end": 5},
    ]
    for extra in bad:
        r = env.client.post(
            f"/facts/{fid}/evidence", data={"kind": "span", "doc_id": env.doc.doc_id, **extra}
        )
        assert r.status_code == 422, extra
    r = env.client.post(
        f"/facts/{fid}/evidence",
        data={"kind": "span", "doc_id": "D-9999", "char_start": 0, "char_end": 3},
    )
    assert r.status_code == 404
    # Whole document and external reference.
    r = env.client.post(
        f"/facts/{fid}/evidence", data={"kind": "document", "doc_id": env.email.doc_id}
    )
    assert r.status_code == 303
    r = env.client.post(
        f"/facts/{fid}/evidence", data={"kind": "external", "reference": "Smith Dep. 45:3"}
    )
    assert r.status_code == 303
    kinds = [e.kind for e in env.store.get_fact(fid).evidence]
    assert kinds.count("document") == 1 and kinds.count("external") == 1
    assert (
        env.client.post(
            f"/facts/{fid}/evidence", data={"kind": "external", "reference": " "}
        ).status_code
        == 422
    )


def test_add_fact_from_selection(env: Env) -> None:
    start = TEXT.index("Rent was due")
    end = start + len("Rent was due")
    r = env.client.post(
        "/cases/demo/facts",
        data={
            "title": "Rent due date",
            "doc_id": env.doc.doc_id,
            "char_start": start,
            "char_end": end,
            "quote": "Rent was due",
        },
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith(f"/documents/{env.doc.doc_id}/view?evidence=E-")
    fact = next(f for f in env.store.list_facts("demo") if f.title == "Rent due date")
    assert fact.review == "accepted" and fact.evidence[0].quote == "Rent was due"
    r = env.client.post(
        "/cases/demo/facts",
        data={
            "title": "Mismatch",
            "doc_id": env.doc.doc_id,
            "char_start": start,
            "char_end": end,
            "quote": "Rent was paid",
        },
    )
    assert r.status_code == 422
    assert not [f for f in env.store.list_facts("demo") if f.title == "Mismatch"]


def test_evidence_review(env: Env) -> None:
    span = SpanIn(doc_id=env.email.doc_id, char_start=0, char_end=11, quote=EMAIL_TEXT[:11])
    ev = env.store.add_evidence(env.f3.fact_id, kind="span", by=AGENT, span=span)
    assert ev.review == "draft"
    card = env.client.get(f"/facts/{env.f3.fact_id}").text
    assert f'hx-post="/evidence/{ev.evidence_id}/review"' in card
    r = env.client.post(f"/evidence/{ev.evidence_id}/review", data={"action": "accept"}, headers=HX)
    assert r.status_code == 200 and "accepted" in r.text
    assert env.store.get_evidence(ev.evidence_id)[1].review == "accepted"
    ev2 = env.store.add_evidence(env.f3.fact_id, kind="document", by=AGENT, doc_id=env.doc.doc_id)
    r = env.client.post(
        f"/evidence/{ev2.evidence_id}/review", data={"action": "reject", "reason": "irrelevant"}
    )
    assert r.status_code == 303
    last = env.store.list_review_events("demo")[-1]
    assert (last.evidence_id, last.action, last.reason) == (ev2.evidence_id, "reject", "irrelevant")
    assert env.client.post("/evidence/E-9999/review", data={"action": "accept"}).status_code == 404


def test_notes(env: Env) -> None:
    r = env.client.post(f"/facts/{env.f1.fact_id}/notes", data={"body": "Check the date"})
    assert r.status_code == 303 and r.headers["location"].endswith("#notes")
    notes = env.store.list_notes(env.f1.fact_id)
    assert [(n.author, n.body) for n in notes] == [("Alice", "Check the date")]
    assert "Check the date" in env.client.get(f"/facts/{env.f1.fact_id}").text
    assert env.client.post(f"/facts/{env.f1.fact_id}/notes", data={"body": ""}).status_code == 422


# --------------------------------------------------------------------------- witnesses, tags


def test_witnesses(env: Env) -> None:
    s = env.store
    sug = s.add_witness("demo", "Jon Smith", by=AGENT)
    s.update_fact(env.f1.fact_id, {"witness_ids": [sug.witness_id]}, by=AGENT)
    html = env.client.get("/cases/demo/witnesses").text
    assert "Suggested by AI" in html and "Jon Smith" in html
    # merge the suggestion into the confirmed witness
    r = env.client.post(
        f"/witnesses/{sug.witness_id}", data={"action": "merge", "into": env.witness.witness_id}
    )
    assert r.status_code == 303 and r.headers["location"] == f"/witnesses/{env.witness.witness_id}"
    w = s.get_witness(env.witness.witness_id)
    assert s.get_witness(sug.witness_id) is None and "Jon Smith" in w.aliases
    assert env.f1.fact_id in {f.fact_id for f in s.witness_facts(w.witness_id)}
    # confirm and dismiss
    sug2 = s.add_witness("demo", "Mary Major", by=AGENT)
    env.client.post(f"/witnesses/{sug2.witness_id}", data={"action": "confirm"})
    assert s.get_witness(sug2.witness_id).confirmed
    sug3 = s.add_witness("demo", "Nobody", by=AGENT)
    r = env.client.post(f"/witnesses/{sug3.witness_id}", data={"action": "dismiss"})
    assert r.headers["location"] == "/cases/demo/witnesses"
    assert s.get_witness(sug3.witness_id) is None
    # flag toggle and profile edit (form POST and real PATCH)
    env.client.post(f"/witnesses/{w.witness_id}", data={"has": "flagged", "flagged": "1"})
    assert s.get_witness(w.witness_id).flagged
    env.client.post(f"/witnesses/{w.witness_id}", data={"has": "flagged"})
    assert not s.get_witness(w.witness_id).flagged
    r = env.client.patch(
        f"/witnesses/{w.witness_id}", data={"role": "Tenant", "aliases": "J. Smith, Johnny"}
    )
    assert r.status_code == 200 and r.json()["role"] == "Tenant"
    assert r.json()["aliases"] == ["J. Smith", "Johnny"]
    # create, profile page lists facts in date order and documents mentioning them
    r = env.client.post("/cases/demo/witnesses", data={"name": "Jones", "flagged": "1"})
    assert r.status_code == 303
    profile = env.client.get(f"/witnesses/{w.witness_id}").text
    assert profile.index(env.f3.fact_id) < profile.index(env.f1.fact_id)
    assert "complaint.txt" in profile
    assert env.client.post("/cases/demo/witnesses", data={"name": ""}).status_code == 422


def test_tags(env: Env) -> None:
    r = env.client.post("/cases/demo/tags", data={"name": "Damages", "color": "#00ff00"})
    assert r.status_code == 303
    tag = next(t for t in env.store.list_tags("demo") if t.name == "Damages")
    assert tag.color == "#00ff00"
    env.client.post(
        "/cases/demo/tags",
        data={"action": "update", "tag_id": tag.tag_id, "name": "Harm", "color": "#112233"},
    )
    tag = env.store.get_tag(tag.tag_id)
    assert (tag.name, tag.color) == ("Harm", "#112233")
    bad = env.client.post("/cases/demo/tags", data={"name": "X", "color": "red;}body{x"})
    assert bad.status_code == 422
    page = env.client.get("/cases/demo/tags").text
    assert "Harm" in page and "Breach" in page
    env.client.post("/cases/demo/tags", data={"action": "delete", "tag_id": tag.tag_id})
    assert env.store.get_tag(tag.tag_id) is None
    r = env.client.post("/cases/demo/tags", data={"action": "delete", "tag_id": "T-9999"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- contradictions, alerts


def test_contradiction_confirm_marks_disputed(env: Env) -> None:
    c, _ = env.store.add_contradiction(
        "demo", [env.f1.fact_id, env.f3.fact_id], "date", "Sources disagree on the date"
    )
    html = env.client.get("/cases/demo/contradictions").text
    assert "Sources disagree on the date" in html and "disputed" in html
    assert QUOTE in html  # both sides with quotes
    assert "Conflict" in env.client.get("/cases/demo/facts").text
    r = env.client.post(f"/contradictions/{c.contradiction_id}", data={"action": "confirm"})
    assert r.status_code == 303
    assert env.store.get_contradiction(c.contradiction_id).status == "confirmed"
    for fid in (env.f1.fact_id, env.f3.fact_id):
        assert "disputed" in env.store.get_fact(fid).statuses
    env.client.post(f"/contradictions/{c.contradiction_id}", data={"action": "reopen"})
    assert env.store.get_contradiction(c.contradiction_id).status == "open"
    env.client.post(f"/contradictions/{c.contradiction_id}", data={"action": "dismiss"})
    assert env.store.get_contradiction(c.contradiction_id).status == "dismissed"
    assert env.client.post("/contradictions/C-9999", data={"action": "confirm"}).status_code == 404
    bad = env.client.post(f"/contradictions/{c.contradiction_id}", data={"action": "x"})
    assert bad.status_code == 422


def test_alerts_page(env: Env) -> None:
    s = env.store
    a1 = s.add_alert(
        "demo", "possible_deadline", fact_ids=[env.f1.fact_id], doc_ids=[env.doc.doc_id]
    )
    a2 = s.add_alert("demo", "flagged_witness", doc_ids=[env.email.doc_id])
    s.mark_alerts_delivered([a2.alert_id])
    html = env.client.get("/cases/demo/alerts").text
    assert html.index(a2.alert_id) < html.index(a1.alert_id)  # newest first
    block = html[html.index(f'id="{a1.alert_id}"') :]
    block = block[: block.index("</article>")]
    assert "Needs attorney review" in block and "pending" in block
    assert f"/facts/{env.f1.fact_id}" in block and f"/documents/{env.doc.doc_id}/view" in block
    block2 = html[html.index(f'id="{a2.alert_id}"') :]
    assert "delivered" in block2[: block2.index("</article>")]
    r = env.client.get(f"/cases/demo/alerts/{a1.alert_id}")
    assert r.status_code == 303 and r.headers["location"].endswith(f"#{a1.alert_id}")


# --------------------------------------------------------------------------- documents, jobs


def test_upload_goes_to_inbox_and_queues_job(env: Env) -> None:
    woke: list[bool] = []

    class FakeWorker:
        def wake(self) -> None:
            woke.append(True)

    env.app.state.deps.worker = FakeWorker()
    r = env.client.post(
        "/cases/demo/documents",
        files=[
            ("files", ("../new memo.txt", b"hello", "text/plain")),
            ("files", ("evil.exe", b"MZ", "application/octet-stream")),
        ],
    )
    assert r.status_code == 200
    inbox = env.settings.data_dir / "inbox"
    assert (inbox / "new memo.txt").read_bytes() == b"hello"
    assert list(env.evidence.iterdir()) == []
    assert not (inbox / "evil.exe").exists()
    assert "unsupported file type" in r.text
    job = env.store.latest_job("demo")
    assert job is not None and job.trigger == "upload" and job.status == "queued"
    assert woke == [True]
    anon = env.new_client()
    r = anon.post("/cases/demo/documents", files=[("files", ("x.txt", b"x", "text/plain"))])
    assert r.status_code == 303
    assert not (inbox / "x.txt").exists()


def test_documents_page_and_kind(env: Env) -> None:
    html = env.client.get("/cases/demo/documents").text
    assert "complaint.txt" in html and "email.eml" in html and "pleading" in html
    r = env.client.post(f"/documents/{env.doc.doc_id}/kind", data={"kind": "motion"})
    assert r.status_code == 303
    assert env.store.get_document(env.doc.doc_id).kind == "motion"
    assert env.store.alleged_fact_ids("demo") == set()
    r = env.client.post(f"/documents/{env.doc.doc_id}/kind", data={"kind": "poem"})
    assert r.status_code == 422


def test_process_and_job_partial(env: Env) -> None:
    r = env.client.post("/cases/demo/process", headers=HX)
    assert r.status_code == 200
    job = env.store.latest_job("demo")
    assert job.trigger == "ui" and job.status == "queued"
    assert 'hx-trigger="every 2s"' in r.text and f"/jobs/{job.job_id}" in r.text
    r = env.client.post("/cases/demo/process")
    assert r.status_code == 303 and r.headers["location"] == f"/jobs/{job.job_id}"  # reused
    claimed = env.store.claim_next_job()
    env.store.finish_job(claimed.job_id, "done", counts={"docs_parsed": 2, "facts_created": 5})
    r = env.client.get(f"/jobs/{job.job_id}", headers=HX)
    assert r.status_code == 200 and "every 2s" not in r.text
    assert "Facts created" in r.text and ">5<" in r.text
    page = env.client.get(f"/jobs/{job.job_id}")
    assert page.status_code == 200 and "<html" in page.text
    assert env.client.get("/jobs/J-9999").status_code == 404


def test_reprocess(env: Env) -> None:
    r = env.client.post(f"/documents/{env.email.doc_id}/reprocess")
    assert r.status_code == 303 and r.headers["location"].startswith("/jobs/J-")
    assert env.store.get_fact(env.f2.fact_id) is None  # its only evidence was this doc
    assert env.store.get_document(env.email.doc_id).parse_status == "pending"
    assert env.store.latest_job("demo").trigger == "ui"
    assert env.store.get_fact(env.f1.fact_id) is not None


# --------------------------------------------------------------------------- export


def test_export_endpoints(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str], Any]] = []

    def fake(kind: str, payload: bytes | str) -> Any:
        def fn(store: Store, case_id: str, facts: list, arg: Any = None) -> bytes | str:
            calls.append((kind, [f.fact_id for f in facts], arg))
            return payload

        return fn

    monkeypatch.setattr(export, "to_csv", fake("csv", b"Date,Title\n"))
    monkeypatch.setattr(export, "to_xlsx", fake("xlsx", b"PK\x03\x04"))
    monkeypatch.setattr(export, "to_svg", fake("svg", "<svg></svg>"))
    monkeypatch.setattr(export, "to_png", fake("png", b"\x89PNG"))
    expected = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "svg": "image/svg+xml",
        "png": "image/png",
    }
    for fmt, ctype in expected.items():
        r = env.client.get(f"/cases/demo/export?format={fmt}&review=accepted&zoom=year")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith(ctype)
        assert "attachment" in r.headers["content-disposition"]
        assert "sandbox" in r.headers["content-security-policy"]
    assert [c[1] for c in calls] == [[env.f3.fact_id]] * 4  # filters respected
    assert calls[0][2] == list(export.DEFAULT_COLUMNS) and calls[2][2] == "year"
    r = env.client.get("/cases/demo/export?format=csv&columns=title&columns=date&kind=pleading")
    assert calls[-1][1:] == ([env.f1.fact_id], ["title", "date"])
    assert env.client.get("/cases/demo/export?format=doc").status_code == 422
    assert env.client.get("/cases/demo/export?format=svg&zoom=century").status_code == 422
    html = env.client.get("/cases/demo/facts?status=key_fact").text
    assert 'action="/cases/demo/export"' in html
    assert 'name="status" value="key_fact"' in html


def test_export_with_real_module_when_available(env: Env) -> None:
    try:
        export.to_csv(env.store, "demo", [], export.DEFAULT_COLUMNS)
    except NotImplementedError:
        pytest.skip("pipeline.export not implemented yet")
    r = env.client.get("/cases/demo/export?format=csv&status=key_fact")
    assert r.status_code == 200 and "Lease term" in r.text and "Smith signed" not in r.text


# --------------------------------------------------------------------------- source viewer, XSS


def test_source_viewer_highlights_span(env: Env) -> None:
    ev = env.f1.evidence[0]
    r = env.client.get(f"/documents/{env.doc.doc_id}/view?evidence={ev.evidence_id}")
    assert r.status_code == 200
    html = r.text
    assert f'<mark class="hl" id="hl">{QUOTE}</mark>' in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and XSS not in html
    assert 'data-page="2"' in html and "Page 2" in html
    assert f'data-start="{len(PAGE1)}"' in html
    assert env.f1.fact_id in html and "/static/source.js" in html
    assert f'hx-post="/facts/{env.f1.fact_id}/review"' in html  # accept/reject for drafts
    # evidence from another document is not highlighted here
    other = env.f2.evidence[0].evidence_id
    html = env.client.get(f"/documents/{env.doc.doc_id}/view?evidence={other}").text
    assert 'id="hl"' not in html and "does not point into this document" in html


def test_source_segments_preserve_offsets(env: Env) -> None:
    import html as html_mod

    parsed = env.store.get_parsed(env.doc.doc_id)
    segs = source_segments(parsed, Q_START, len(PAGE1) + 5)  # highlight crosses a page
    text = ""
    for s in segs:
        inner = re.sub(r"<[^>]+>", "", str(s["html"]))
        assert int(s["start"]) == len(text)
        text += html_mod.unescape(inner)
    assert text == TEXT
    marks = "".join(str(s["html"]) for s in segs).count("<mark")
    assert marks == 2 and "".join(str(s["html"]) for s in segs).count('id="hl"') == 1


def test_case_content_is_escaped_everywhere(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    s = env.store
    bad = s.add_fact("demo", XSS, by="Mallory", details=XSS, date_text=XSS)
    s.add_note(bad.fact_id, author="Mallory", body=XSS)
    tag = s.add_tag("demo", XSS, by="Mallory", color="#000000")
    w = s.add_witness("demo", XSS, by="Mallory", role=XSS)
    s.update_fact(
        bad.fact_id, {"tag_ids": [tag.tag_id], "witness_ids": [w.witness_id]}, by="Mallory"
    )
    s.add_evidence(bad.fact_id, kind="external", by="Mallory", reference=f'x" onclick="{XSS}')
    c, _ = s.add_contradiction("demo", [bad.fact_id, env.f1.fact_id], "fact", XSS)
    s.add_alert("demo", "contradiction", fact_ids=[bad.fact_id])

    class FakeLLM:
        @staticmethod
        def chat(messages: list[dict], **kw: Any) -> str:
            return f'Answer {XSS} [{bad.fact_id}] <a href="x" onmouseover=alert(3)>'

    monkeypatch.setattr(search, "search_facts", lambda *a, **k: [bad.fact_id])
    monkeypatch.setattr(search, "keyword_fact_ids", lambda *a, **k: [bad.fact_id])
    deps = env.app.state.deps
    deps.llm, deps.chat_backend = FakeLLM(), "local"
    client = env.new_client(user="<b>Mallory</b>")
    pages = [
        client.get("/cases/demo/facts"),
        client.get("/cases/demo/facts?q=x"),
        client.get("/cases/demo/review"),
        client.get(f"/facts/{bad.fact_id}"),
        client.get(f"/facts/{bad.fact_id}?partial=1"),
        client.get(f"/documents/{env.doc.doc_id}/view?evidence={env.f1.evidence[0].evidence_id}"),
        client.get("/cases/demo/witnesses"),
        client.get(f"/witnesses/{w.witness_id}"),
        client.get("/cases/demo/tags"),
        client.get("/cases/demo/contradictions"),
        client.get("/cases/demo/alerts"),
        client.post("/cases/demo/chat", data={"question": XSS}),
        client.post("/cases/demo/chat", data={"question": XSS}, headers=HX),
    ]
    for r in pages:
        assert r.status_code == 200, r.url
        assert "<script>alert" not in r.text, r.url
        assert "<img src=x" not in r.text and '<a href="x"' not in r.text, r.url
        assert '" onclick="' not in r.text, r.url
        assert "<b>Mallory" not in r.text, r.url
    assert "&lt;script&gt;" in pages[3].text
    data = client.get("/cases/demo/facts?format=json").json()  # JSON; JS uses textContent
    assert any(i["title"] == XSS for i in data["items"])


# --------------------------------------------------------------------------- chat, today


def test_chat_local_fallback_links_fact_ids(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeLLM:
        @staticmethod
        def chat(messages: list[dict], **kw: Any) -> str:
            seen["messages"] = messages
            return f"Smith signed on March 3 [{env.f1.fact_id}]. See also F-9999."

    monkeypatch.setattr(
        search, "search_facts", lambda store, case_id, q, k=20, **kw: [env.f1.fact_id]
    )
    deps = env.app.state.deps
    deps.llm, deps.chat_backend = FakeLLM(), "auto"
    deps.openclaw_bin = None  # OpenClaw unavailable -> local fallback
    r = env.client.post("/cases/demo/chat", data={"question": "When was the lease signed?"})
    assert r.status_code == 200
    assert f'<a class="fact-ref" href="/facts/{env.f1.fact_id}">{env.f1.fact_id}</a>' in r.text
    assert 'href="/facts/F-9999"' not in r.text  # unknown ids are not linked
    assert "unreviewed draft" in r.text
    context = seen["messages"][1]["content"]
    assert f"[{env.f1.fact_id}]" in context and QUOTE in context and "complaint.txt" in context
    assert "p. 1" in context

    def offline(*a: Any, **k: Any) -> str:
        raise ConnectionError("model down")

    deps.llm = type("Down", (), {"chat": staticmethod(offline)})()
    r = env.client.post("/cases/demo/chat", data={"question": "lease?"}, headers=HX)
    assert r.status_code == 200 and "local model is unavailable" in r.text
    assert env.f1.fact_id in r.text
    assert env.client.post("/cases/demo/chat", data={"question": " "}).status_code == 422


def test_chat_openclaw_reply(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = env.app.state.deps
    deps.openclaw_bin = Path("/bin/true")
    sent: dict[str, Any] = {}

    class Proc:
        returncode = 0
        stdout = f'{{"reply": "Per [{env.f3.fact_id}] the lease ran all year."}}'
        stderr = ""

    def fake_run(cmd: list[str], **kw: Any) -> Proc:
        sent["cmd"] = cmd
        return Proc()

    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    r = env.client.post("/cases/demo/chat", data={"question": "How long was the lease?"})
    assert r.status_code == 200
    assert f'href="/facts/{env.f3.fact_id}"' in r.text and "OpenClaw" in r.text
    assert "F-0012" in sent["cmd"][2]  # the agent is told to cite fact ids


def test_today_panel(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        digest,
        "daily_digest",
        lambda store, case_id, **kw: {
            "text": f"New draft {env.f1.fact_id} and {XSS}",
            "counts": {"new_facts": 1},
            "fact_ids": [env.f1.fact_id],
        },
    )
    html = env.client.get("/cases/demo").text
    assert f'href="/facts/{env.f1.fact_id}"' in html and XSS not in html
    assert "Drafts to review" in html and "Process now" in html

    def broken(*a: Any, **k: Any) -> dict:
        raise NotImplementedError

    monkeypatch.setattr(digest, "daily_digest", broken)
    r = env.client.get("/cases/demo")
    assert r.status_code == 200 and "not available yet" in r.text


def test_nav_counts(env: Env) -> None:
    html = env.client.get("/cases/demo/tags").text
    nav = html[html.index("<nav") : html.index("</nav>")]
    assert 'Needs review <span class="count">2</span>' in nav
    for label in ("Timeline", "Documents", "Witnesses", "Tags", "Contradictions", "Alerts", "Ask"):
        assert label in nav


def test_real_search_and_digest(env: Env) -> None:
    """The real pipeline.search / pipeline.digest behind the list view and Today panel."""
    try:
        search.keyword_fact_ids(env.store, "demo", "lease")
        digest.daily_digest(env.store, "demo")
    except NotImplementedError:
        pytest.skip("pipeline.search / pipeline.digest not implemented yet")
    html = env.client.get("/cases/demo/facts?q=lease").text
    assert set(row_ids(html)) == {env.f1.fact_id, env.f3.fact_id}
    html = env.client.get("/cases/demo/facts?q=%22unbalanced").text
    assert "Search not understood" in html and row_ids(html) == []
    html = env.client.get("/cases/demo").text
    assert f'<a class="fact-ref" href="/facts/{env.f1.fact_id}">' in html
