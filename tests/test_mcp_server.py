"""MCP server tests (SPEC §8.2) against a real temp Store with fixture data.

Most tests go through the SDK's in-process client; two spawn the stdio server to prove stdout
carries JSON-RPC only.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from mcp_server import server as srv
from pipeline.schema import AGENT, Line, Page, SpanIn
from pipeline.store import Store

ROOT = Path(__file__).resolve().parents[1]
CASE = "demo"
PERSON = "Ana Reviewer"
NOISE = "NOISE FROM A LIBRARY THAT PRINTS"
EXPECTED_TOOLS = {
    "scan_case",
    "job_status",
    "get_timeline",
    "search_facts",
    "get_fact",
    "get_witness",
    "list_contradictions",
    "get_pending_alerts",
    "mark_alerts_delivered",
    "daily_digest",
}
WRITE_TOOLS = {"scan_case", "mark_alerts_delivered"}

MEMO_P1 = "Page one intro. On March 3, 2024 Alice Ng met Bob Stone at the warehouse.\n"
MEMO_P2 = (
    "Page two. Bob says the meeting was on March 5, 2024. Someone claims the meeting was "
    "cancelled on March 4, 2024. The filing deadline is June 1, 2024."
)
MEMO = MEMO_P1 + MEMO_P2
COMPLAINT = "Plaintiff alleges that Acme shipped defective forklifts on May 1, 2024."
DEPO_LINES = [
    "Q. Who else was present at that meeting?",
    "A. Priya Nand from HR sat in. Nobody else.",
    "Q. Was Tom Becker in the room?",
]
DEPO = "\n".join(DEPO_LINES)
REJECTED_TITLE = "Meeting was cancelled"


@dataclass
class Fixture:
    store: Store
    ids: dict[str, str] = field(default_factory=dict)


def _doc(
    store: Store, filename: str, text: str, kind: str, pages: list[Page], lines: list[Line] = ()
) -> str:  # type: ignore[assignment]
    doc, _ = store.register_document(
        CASE,
        filename=filename,
        rel_path=filename,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        location="evidence",
    )
    store.save_parsed_text(doc.doc_id, text, pages, lines)
    store.set_parse_result(doc.doc_id, "parsed", pages=len(pages), kind=kind)
    store.mark_drafted(doc.doc_id)
    return doc.doc_id


def _span(doc_id: str, text: str, quote: str) -> SpanIn:
    start = text.index(quote)
    return SpanIn(doc_id=doc_id, char_start=start, char_end=start + len(quote), quote=quote)


def build_case(db_path: Path) -> Fixture:
    """A small case: accepted, draft, rejected, alleged, transcript and person-made facts."""
    store = Store(db_path, embedder=lambda texts: [[0.0] * 1024 for _ in texts])
    fx = Fixture(store)
    ids = fx.ids
    ids["memo"] = memo = _doc(
        store,
        "memo.pdf",
        MEMO,
        "exhibit",
        [
            Page(number=1, char_start=0, char_end=len(MEMO_P1)),
            Page(number=2, char_start=len(MEMO_P1), char_end=len(MEMO)),
        ],
    )
    ids["complaint"] = complaint = _doc(
        store,
        "complaint.pdf",
        COMPLAINT,
        "pleading",
        [Page(number=1, char_start=0, char_end=len(COMPLAINT))],
    )
    depo_lines, pos = [], 0
    for n, line in enumerate(DEPO_LINES, start=7):
        depo_lines.append(Line(page=14, number=n, char_start=pos, char_end=pos + len(line)))
        pos += len(line) + 1
    ids["depo"] = depo = _doc(
        store,
        "depo_ellison.txt",
        DEPO,
        "transcript",
        [Page(number=14, char_start=0, char_end=len(DEPO))],
        depo_lines,
    )

    alice = store.add_witness(CASE, "Alice Ng", by=PERSON, aliases=["A. Ng"], flagged=True)
    bob = store.add_witness(CASE, "Bob Stone", by=AGENT)  # an AI suggestion
    ids["alice"], ids["bob"] = alice.witness_id, bob.witness_id
    tag = store.add_tag(CASE, "Liability", by=PERSON)
    ids["tag"] = tag.tag_id

    f = store.add_fact(
        CASE,
        "Alice met Bob at the warehouse",
        by=AGENT,
        date_text="March 3, 2024",
        date_start="2024-03-03",
        date_certainty="exact",
        confidence=0.9,
        witness_ids=[alice.witness_id, bob.witness_id],
        spans=[_span(memo, MEMO, "On March 3, 2024 Alice Ng met Bob Stone at the warehouse.")],
    )
    store.review_fact(f.fact_id, "accept", reviewer=PERSON)
    store.add_note(f.fact_id, author=PERSON, body="Check the gate log.")
    ids["accepted"] = f.fact_id

    f = store.add_fact(
        CASE,
        "Bob dates the meeting March 5",
        by=AGENT,
        date_text="March 5, 2024",
        date_start="2024-03-05",
        date_certainty="exact",
        confidence=0.7,
        witness_ids=[bob.witness_id],
        spans=[_span(memo, MEMO, "Bob says the meeting was on March 5, 2024.")],
    )
    ids["draft"] = f.fact_id

    f = store.add_fact(
        CASE,
        REJECTED_TITLE,
        by=AGENT,
        date_text="March 4, 2024",
        date_start="2024-03-04",
        date_certainty="exact",
        confidence=0.3,
        witness_ids=[alice.witness_id],
        spans=[_span(memo, MEMO, "the meeting was cancelled on March 4, 2024")],
    )
    store.review_fact(f.fact_id, "reject", reviewer=PERSON, reason="misread")
    ids["rejected"] = f.fact_id

    f = store.add_fact(
        CASE,
        "Acme shipped defective forklifts",
        by=AGENT,
        date_text="May 1, 2024",
        date_start="2024-05-01",
        date_certainty="exact",
        confidence=0.8,
        spans=[_span(complaint, COMPLAINT, "Acme shipped defective forklifts on May 1, 2024")],
    )
    ids["alleged"] = f.fact_id

    f = store.add_fact(
        CASE,
        "Only Nand attended with Ellison",
        by=AGENT,
        confidence=0.6,
        spans=[_span(depo, DEPO, "A. Priya Nand from HR sat in. Nobody else.")],
    )
    ids["transcript"] = f.fact_id

    f = store.add_fact(
        CASE,
        "Filing deadline",
        by=PERSON,
        date_text="June 1, 2024",
        date_start="2024-06-01",
        date_certainty="exact",
        statuses=["key_fact"],
        tag_ids=[tag.tag_id],
        spans=[_span(memo, MEMO, "The filing deadline is June 1, 2024.")],
    )
    ids["person"] = f.fact_id

    c, _ = store.add_contradiction(
        CASE, [ids["accepted"], ids["draft"]], "date", "March 3 vs March 5"
    )
    ids["conflict"] = c.contradiction_id
    c, _ = store.add_contradiction(
        CASE, [ids["accepted"], ids["rejected"]], "fact", "met vs cancelled"
    )
    ids["conflict_rejected"] = c.contradiction_id

    ids["alert_conflict"] = store.add_alert(
        CASE, "contradiction", fact_ids=[ids["accepted"], ids["draft"]], doc_ids=[memo]
    ).alert_id
    ids["alert_rejected"] = store.add_alert(
        CASE, "possible_deadline", fact_ids=[ids["rejected"]], doc_ids=[memo]
    ).alert_id
    ids["alert_deadline"] = store.add_alert(
        CASE, "possible_deadline", fact_ids=[ids["person"], ids["rejected"]]
    ).alert_id
    ids["alert_witness"] = store.add_alert(CASE, "flagged_witness", doc_ids=[depo]).alert_id
    return fx


# --------------------------------------------------------------------------- harness


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    fixture = build_case(tmp_path / "case.db")
    yield fixture
    fixture.store.close()


@pytest.fixture
def server(fx: Fixture, monkeypatch: pytest.MonkeyPatch) -> Any:
    from pipeline import digest, search

    ids = fx.ids

    def fake_search(
        store: Store, case_id: str, query: str, k: int = 20, *, semantic: bool = True
    ) -> list[str]:
        if query == "bad (":
            raise search.QueryError("unbalanced parenthesis")
        # The real search excludes rejected facts; the server must not rely on it.
        return [ids["rejected"], ids["accepted"], ids["draft"]][:k]

    def fake_digest(store: Store, case_id: str, *, now: Any = None) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "since": "2024-01-01T00:00:00+00:00",
            "text": f"2 new facts ({ids['draft']}, {ids['alleged']})",
            "counts": {"new_facts": 2, "alerts": 1, "reviews": 0},
            "fact_ids": [ids["draft"], ids["rejected"], ids["alleged"]],
        }

    monkeypatch.setattr(search, "search_facts", fake_search)
    monkeypatch.setattr(digest, "daily_digest", fake_digest)
    srv.set_backend(srv.Backend(store=fx.store, case_id=CASE))
    yield srv.build_server()
    srv.set_backend(None)


def _data(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content
    data = result.structured_content
    if data is None:
        data = json.loads(result.content[0].text)
    return data.get("result", data) if set(data) == {"result"} else data


async def _run(server: Any, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    async with Client(server) as client:
        return [await client.call_tool(name, args) for name, args in calls]


def call(server: Any, tool: str, /, **args: Any) -> dict[str, Any]:
    return _data(anyio.run(_run, server, [(tool, args)])[0])


def call_error(server: Any, tool: str, /, **args: Any) -> str:
    result = anyio.run(_run, server, [(tool, args)])[0]
    assert result.is_error
    return result.content[0].text


def _all_fact_ids(obj: Any) -> set[str]:
    """Every fact id anywhere in a tool result."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("fact_id", "anchor_fact_id") and isinstance(v, str):
                out.add(v)
            elif k in ("fact_ids", "other_fact_ids") and isinstance(v, list):
                out.update(v)
            else:
                out |= _all_fact_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _all_fact_ids(v)
    return out


# --------------------------------------------------------------------------- tools


def test_tool_list_and_annotations(server: Any) -> None:
    async def listing() -> tuple[Any, Any]:
        async with Client(server) as client:
            return await client.list_tools(), client.instructions

    tools, instructions = anyio.run(listing)
    by_name = {t.name: t for t in tools.tools}
    assert set(by_name) == EXPECTED_TOOLS
    for name, tool in by_name.items():
        assert tool.description and len(tool.description) > 40, name
        assert tool.annotations.read_only_hint is (name not in WRITE_TOOLS), name
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False
    assert "possible conflict" in instructions and "allegations" in instructions


def test_scan_case_only_queues(server: Any, fx: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline import jobs

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("the MCP server must never run the pipeline")

    monkeypatch.setattr(jobs, "run_job", boom)
    monkeypatch.setattr(jobs.Worker, "run_pending", boom)
    first = call(server, "scan_case", case_id="other")
    again = call(server, "scan_case", case_id="other")
    assert first["status"] == "queued" and first["case_id"] == "other"
    assert again["job_id"] == first["job_id"]  # one queued scan per case
    job = fx.store.get_job(first["job_id"])
    assert job.status == "queued" and job.trigger == "agent" and job.started_at is None
    assert job.counts == {}
    assert fx.store.list_documents("other") == []  # nothing discovered or parsed
    status = call(server, "job_status", job_id=first["job_id"])
    assert status["found"] and status["status"] == "queued" and "hint" not in status
    default = call(server, "scan_case")
    assert default["case_id"] == CASE


def test_job_status_unknown_and_finished(server: Any, fx: Fixture) -> None:
    assert call(server, "job_status", job_id="J-9999")["found"] is False
    job, _ = fx.store.request_job(CASE, "ui")
    fx.store.claim_next_job()
    fx.store.finish_job(job.job_id, "done", counts={"docs_parsed": 3, "quotes_rejected": 1})
    out = call(server, "job_status", job_id=job.job_id)
    assert out["status"] == "done" and out["counts"]["docs_parsed"] == 3


def test_invalid_arguments_reach_the_agent(server: Any) -> None:
    assert "invalid case id" in call_error(server, "get_timeline", case_id="../etc")
    assert "invalid case id" in call_error(server, "scan_case", case_id="a/b")
    assert "status must be one of" in call_error(server, "get_timeline", status="accepted")
    assert "ISO date" in call_error(server, "get_timeline", start="March 3")


def test_get_timeline_fields_and_order(server: Any, fx: Fixture) -> None:
    ids = fx.ids
    out = call(server, "get_timeline")
    order = [f["fact_id"] for f in out["facts"]]
    assert order == [
        ids["accepted"],
        ids["draft"],
        ids["alleged"],
        ids["person"],
        ids["transcript"],
    ]  # date order, undated last, rejected gone
    facts = {f["fact_id"]: f for f in out["facts"]}

    acc = facts[ids["accepted"]]
    assert acc["review"] == "accepted" and acc["date_start"] == "2024-03-03"
    assert acc["date_certainty"] == "exact" and acc["date_text"] == "March 3, 2024"
    assert acc["witnesses"] == ["Alice Ng", "Bob Stone"]
    assert acc["possible_conflict"] is True and acc["disputed"] is False
    assert acc["alleged"] is False
    ev = acc["evidence"][0]
    assert ev["filename"] == "memo.pdf" and ev["page"] == 1 and ev["cite"] == "memo.pdf p. 1"
    assert ev["quote"].startswith("On March 3, 2024")
    assert ev["source_link"].startswith("http://127.0.0.1:8080/documents/")

    assert facts[ids["draft"]]["review"] == "draft"
    assert facts[ids["draft"]]["evidence"][0]["page"] == 2
    assert facts[ids["alleged"]]["alleged"] is True
    person = facts[ids["person"]]
    assert person["statuses"] == ["key_fact"] and person["tags"] == ["Liability"]
    assert person["origin"] == "user"
    depo = facts[ids["transcript"]]["evidence"][0]
    assert depo["line_start"] == 8 and depo["cite"] == "depo_ellison.txt 14:8"
    assert facts[ids["transcript"]]["date_start"] is None


def test_get_timeline_filters(server: Any, fx: Fixture) -> None:
    ids = fx.ids

    def fact_ids(**kw: Any) -> list[str]:
        return [f["fact_id"] for f in call(server, "get_timeline", **kw)["facts"]]

    assert fact_ids(start="2024-03-04", end="2024-05-31") == [ids["draft"], ids["alleged"]]
    # Witness by alias, id and fuzzy name; Alice's rejected fact stays hidden.
    assert fact_ids(witness="A. Ng") == [ids["accepted"]]
    assert fact_ids(witness=ids["alice"]) == [ids["accepted"]]
    assert fact_ids(witness="Stone") == [ids["accepted"], ids["draft"]]
    assert fact_ids(tag="liability") == [ids["person"]]
    assert fact_ids(tag=ids["tag"]) == [ids["person"]]
    assert fact_ids(status="key_fact") == [ids["person"]]
    assert fact_ids(status="disputed") == []
    missing = call(server, "get_timeline", witness="Nobody Atall")
    assert missing["facts"] == [] and "no witness" in missing["error"]
    assert "no issue tag" in call(server, "get_timeline", tag="Damages")["error"]
    limited = call(server, "get_timeline", limit=2)
    assert limited["truncated"] is True and len(limited["facts"]) == 2


def test_search_facts(server: Any, fx: Fixture) -> None:
    out = call(server, "search_facts", query="warehouse")
    assert [f["fact_id"] for f in out["facts"]] == [fx.ids["accepted"], fx.ids["draft"]]
    bad = call(server, "search_facts", query="bad (")
    assert bad["facts"] == [] and "unbalanced" in bad["error"] and "syntax_help" in bad


def test_get_fact_quotes_in_context(server: Any, fx: Fixture) -> None:
    out = call(server, "get_fact", fact_id=fx.ids["draft"])
    assert out["found"] and out["case_id"] == CASE
    ev = out["evidence"][0]
    assert ev["filename"] == "memo.pdf" and ev["page"] == 2
    assert ev["quote"] == "Bob says the meeting was on March 5, 2024."
    assert "[[Bob says the meeting was on March 5, 2024.]]" in ev["highlighted"]
    assert ev["before"].endswith("Page two. ") and len(ev["before"]) <= srv.CONTEXT_CHARS
    assert out["possible_conflicts"][0]["other_fact_ids"] == [fx.ids["accepted"]]

    acc = call(server, "get_fact", fact_id=fx.ids["accepted"])
    assert acc["notes"][0]["body"] == "Check the gate log."
    # The open conflict with the rejected fact is not reported.
    assert [c["other_fact_ids"] for c in acc["possible_conflicts"]] == [[fx.ids["draft"]]]

    depo = call(server, "get_fact", fact_id=fx.ids["transcript"])["evidence"][0]
    assert depo["cite"] == "depo_ellison.txt 14:8"
    assert call(server, "get_fact", fact_id=fx.ids["rejected"])["found"] is False
    assert call(server, "get_fact", fact_id="F-9999")["found"] is False


def test_get_witness(server: Any, fx: Fixture) -> None:
    out = call(server, "get_witness", name="a. ng")
    assert out["found"] and out["witness"]["name"] == "Alice Ng"
    assert out["witness"]["flagged"] is True and out["witness"]["confirmed"] is True
    assert [f["fact_id"] for f in out["facts"]] == [fx.ids["accepted"]]
    assert "memo.pdf" in {d["filename"] for d in out["documents"]}
    fuzzy = call(server, "get_witness", name="Bob")
    assert fuzzy["witness"]["name"] == "Bob Stone" and fuzzy["witness"]["confirmed"] is False
    nobody = call(server, "get_witness", name="Zed Q")
    assert nobody["found"] is False and "Alice Ng" in nobody["witnesses"]


def test_list_contradictions(server: Any, fx: Fixture) -> None:
    out = call(server, "list_contradictions")
    assert out["count"] == 1  # the conflict with a rejected fact is dropped
    c = out["contradictions"][0]
    assert c["contradiction_id"] == fx.ids["conflict"] and c["label"] == "possible conflict"
    assert {f["fact_id"] for f in c["facts"]} == {fx.ids["accepted"], fx.ids["draft"]}
    assert all(f["evidence"][0]["quote"] for f in c["facts"])

    fx.store.review_contradiction(fx.ids["conflict"], "confirm", reviewer=PERSON)
    assert call(server, "list_contradictions")["count"] == 0
    everything = call(server, "list_contradictions", open_only=False)
    confirmed = everything["contradictions"][0]
    assert confirmed["status"] == "confirmed" and confirmed["label"] == "disputed"
    assert all(f["disputed"] for f in confirmed["facts"])


def test_pending_alerts_are_content_free(server: Any, fx: Fixture) -> None:
    ids = fx.ids
    out = call(server, "get_pending_alerts")
    got = {a["alert_id"]: a for a in out["alerts"]}
    assert set(got) == {ids["alert_conflict"], ids["alert_deadline"], ids["alert_witness"]}
    link = "http://127.0.0.1:8080/cases/demo/alerts"
    assert out["dashboard_url"] == link and "internal" in out["relay_policy"]
    for a in got.values():
        assert a["dashboard_url"] == link
        assert a["message"] == f"Case demo: {a['label']}. Review: {link}"
        assert a["internal"]["do_not_relay"] is True
        assert ids["rejected"] not in a["internal"]["fact_ids"]
    deadline = got[ids["alert_deadline"]]
    assert deadline["needs_attorney_review"] is True
    assert "needs attorney review" in deadline["message"]
    assert deadline["internal"]["fact_ids"] == [ids["person"]]
    assert got[ids["alert_witness"]]["internal"]["doc_ids"] == [ids["depo"]]
    # Messages carry no case content: no names, titles, quotes or filenames.
    blob = " ".join(a["message"] for a in got.values())
    for secret in ("Alice", "Bob", "memo.pdf", "warehouse", "June", "F-", "D-"):
        assert secret not in blob


def test_mark_alerts_delivered_is_idempotent(server: Any, fx: Fixture) -> None:
    ids = fx.ids
    pending = [a["alert_id"] for a in call(server, "get_pending_alerts")["alerts"]]
    first = call(server, "mark_alerts_delivered", alert_ids=pending)
    assert first == {"requested": 3, "marked": 3}
    again = call(server, "mark_alerts_delivered", alert_ids=pending + ["A-9999"])
    assert again == {"requested": 4, "marked": 0}
    assert call(server, "get_pending_alerts")["alerts"] == []
    assert fx.store.get_alert(ids["alert_conflict"]).delivered_at is not None
    assert fx.store.get_alert(ids["alert_rejected"]).delivered_at is None  # never shown
    assert call(server, "mark_alerts_delivered", alert_ids=[]) == {"requested": 0, "marked": 0}


def test_daily_digest(server: Any, fx: Fixture) -> None:
    out = call(server, "daily_digest")
    assert out["fact_ids"] == [fx.ids["draft"], fx.ids["alleged"]]
    assert out["dashboard_url"] == "http://127.0.0.1:8080/cases/demo"
    assert out["message"] == (
        "Case demo daily digest: 2 new facts, 1 alerts. Review: http://127.0.0.1:8080/cases/demo"
    )
    assert out["text"].startswith("2 new facts")


def test_tools_never_return_rejected_facts(server: Any, fx: Fixture) -> None:
    ids = fx.ids
    results = [
        call(server, "get_timeline"),
        call(server, "get_timeline", witness="Alice Ng"),
        call(server, "get_timeline", start="2024-03-04", end="2024-03-04"),
        call(server, "search_facts", query="cancelled"),
        call(server, "get_fact", fact_id=ids["rejected"]),
        call(server, "get_fact", fact_id=ids["accepted"]),
        call(server, "get_witness", name="Alice Ng"),
        call(server, "list_contradictions", open_only=False),
        call(server, "get_pending_alerts"),
        call(server, "daily_digest"),
    ]
    assert results[4] == {"fact_id": ids["rejected"], "found": False, "error": "unknown fact id"}
    for out in results[:4] + results[5:]:
        assert ids["rejected"] not in _all_fact_ids(out)
        assert REJECTED_TITLE not in json.dumps(out)
    assert results[2]["facts"] == []


# --------------------------------------------------------------------------- stdout safety


def test_main_keeps_stray_prints_off_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() swaps in a write-through stdout, so nothing is left buffered for exit."""
    captured: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            captured["stdout"] = srv.sys.stdout
            captured["transport"] = transport

    monkeypatch.setattr(srv, "configure_logging", lambda: None)
    monkeypatch.setattr(srv, "build_server", lambda: FakeServer())
    monkeypatch.setattr(srv.sys, "stdout", srv.sys.stdout)
    srv.main()
    out = captured["stdout"]
    assert captured["transport"] == "stdio"
    assert out.write_through and out.buffer.fileno() == 1  # type: ignore[attr-defined]
    assert srv.sys.stdout is srv.sys.stderr


def noisy_backend() -> srv.Backend:
    """Backend factory for the subprocess test: prints like a badly behaved library."""
    print(NOISE)  # noqa: T201 - the point of the test
    return srv.Backend(store=Store(os.environ["CT_DB_PATH"]), case_id=CASE)


def _env(data_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CT_")}
    env.update(
        CT_DATA_DIR=str(data_dir),
        CT_DB_PATH=str(data_dir / "case.db"),
        CT_EVIDENCE_DIR=str(data_dir.parent / "evidence"),
        CT_MCP_BACKEND="tests.test_mcp_server:noisy_backend",
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
    )
    return env


def _rpc(proc: subprocess.Popen, lines: queue.Queue, msg: dict, expect: bool = True) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if not expect:
        return {}
    return json.loads(lines.get(timeout=30))


def test_stdout_is_clean_json_rpc(tmp_path: Path) -> None:
    """Raw stdio: every stdout line must be JSON-RPC, even when code prints."""
    data_dir = tmp_path / "data"
    (tmp_path / "evidence").mkdir()
    fx = build_case(data_dir / "case.db")
    fx.store.close()
    with open(tmp_path / "server.stderr", "w") as errlog:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_server"],
            cwd=ROOT,
            env=_env(data_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errlog,
            text=True,
        )
    raw: list[str] = []
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            raw.append(line)
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()
    results: dict[str, Any] = {}
    try:
        init = _rpc(
            proc,
            lines,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert init["result"]["serverInfo"]["name"] == "case-timeline"
        _rpc(proc, lines, {"jsonrpc": "2.0", "method": "notifications/initialized"}, False)
        listed = _rpc(proc, lines, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert {t["name"] for t in listed["result"]["tools"]} == EXPECTED_TOOLS
        for i, name in enumerate(["get_timeline", "scan_case", "get_pending_alerts"], start=3):
            resp = _rpc(
                proc,
                lines,
                {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {}},
                },
            )
            assert resp["id"] == i and not resp["result"].get("isError"), resp
            results[name] = resp["result"]
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    for line in raw:
        assert json.loads(line)["jsonrpc"] == "2.0", line
    assert not any(NOISE in line for line in raw)
    assert NOISE in (tmp_path / "server.stderr").read_text()
    assert "starting case-timeline MCP server" in (data_dir / "mcp.log").read_text()
    timeline = json.dumps(results["get_timeline"])
    assert "Alice met Bob" in timeline and REJECTED_TITLE not in timeline
