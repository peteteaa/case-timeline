"""Regression tests for problems found in the adversarial review."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from pipeline import store as store_mod
from pipeline.contracts import Event, SourceRef
from pipeline.store import Store


def _event(eid: str) -> Event:
    return Event(
        id=eid,
        case_id="c",
        date_text="March 3, 2024",
        description="something happened",
        source=SourceRef(doc_id="d1", char_start=0, char_end=4, quote="text"),
        confidence=0.8,
    )


def test_get_changes_never_skips_rows_committed_late(tmp_path: Path, monkeypatch) -> None:
    """A writer that stamped created_at before a reader's cursor must still be reported."""
    db = tmp_path / "case.db"
    writer = Store(db, embedder=lambda texts: [[0.0] * 1024 for _ in texts])
    reader = Store(db)
    stamped = threading.Event()
    real_now = store_mod._now

    def slow_now() -> str:
        stamp = real_now()
        if threading.current_thread().name == "writer":
            stamped.set()
            time.sleep(0.3)  # created_at is chosen; commit happens later
        return stamp

    monkeypatch.setattr(store_mod, "_now", slow_now)
    seen: list[str] = []

    def read() -> None:
        stamped.wait(5)
        seen.extend(e.id for e in reader.get_changes("c", "bot").new_events)

    t_w = threading.Thread(target=writer.add_events, args=([_event("e1")],), name="writer")
    t_r = threading.Thread(target=read, name="reader")
    t_w.start()
    t_r.start()
    t_w.join()
    t_r.join()
    seen.extend(e.id for e in reader.get_changes("c", "bot").new_events)
    assert seen == ["e1"]
    writer.close()
    reader.close()


def test_mcp_main_keeps_stray_prints_off_the_wire(monkeypatch) -> None:
    """main() swaps in a write-through stdout, so nothing is left buffered for exit."""
    from mcp_server import server

    captured: dict[str, object] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            captured["stdout"] = server.sys.stdout
            captured["transport"] = transport

    monkeypatch.setattr(server, "configure_logging", lambda: None)
    monkeypatch.setattr(server, "build_server", lambda: FakeServer())
    monkeypatch.setattr(server.sys, "stdout", server.sys.stdout)
    server.main()
    out = captured["stdout"]
    assert captured["transport"] == "stdio"
    assert out.write_through and out.buffer.fileno() == 1  # type: ignore[attr-defined]
    assert server.sys.stdout is server.sys.stderr


def test_llm_refuses_remote_endpoint_and_ignores_proxies(monkeypatch) -> None:
    from pipeline import llm

    assert llm._require_loopback("http://127.0.0.1:8000/v1")
    assert llm._require_loopback("http://[::1]:8000/v1")
    assert llm._require_loopback("http://host.openshell.internal:8000/v1")
    monkeypatch.setattr(llm, "_is_host_local", lambda h: h == "172.17.0.1")
    assert llm._require_loopback("http://172.17.0.1:8000/v1")
    monkeypatch.undo()
    for url in (
        "https://api.example.com/v1",
        "http://203.0.113.5:8000/v1",
        "http://0.0.0.0:1",
        "http://evil.example:8000/v1",
    ):
        with pytest.raises(llm.LLMError):
            llm._require_loopback(url)

    monkeypatch.setenv("HTTP_PROXY", "http://10.9.9.9:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://10.9.9.9:3128")
    llm._client.cache_clear()
    try:
        assert not llm._client()._client._mounts  # no proxy transport mounted
    finally:
        llm._client.cache_clear()


XSS = "<script>alert(1)</script><img src=x onerror=alert(2)>\"'&"


def test_ui_escapes_case_content(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipeline.config import Settings
    from pipeline.contracts import (
        ConflictField,
        Contradiction,
        Document,
        Page,
    )
    from ui.app import create_app

    evidence = tmp_path / "ev"
    evidence.mkdir()
    settings = Settings(evidence_dir=evidence, data_dir=tmp_path / "data", case_id="c")
    store = Store(settings.db_path, embedder=lambda texts: [[0.0] * 1024 for _ in texts])
    text = f"Intro. {XSS} outro."
    store.upsert_document(
        Document(
            id="d1", case_id="c", path="/x", filename=f"{XSS}.txt", mime="text/plain", sha256="0"
        )
    )
    store.add_parsed_text("d1", text, [Page(number=1, char_start=0, char_end=len(text))])
    start = text.index("<script>")
    events = []
    for eid in ("e1", "e2"):
        ev = _event(eid).model_copy(
            update={
                "description": XSS,
                "actors": [XSS],
                "date_text": XSS,
                "source": SourceRef(
                    doc_id="d1", char_start=start, char_end=start + len(XSS), quote=XSS
                ),
            }
        )
        events.append(ev)
    store.add_events(events)
    store.add_contradictions(
        [
            Contradiction(
                id="k1",
                case_id="c",
                event_ids=["e1", "e2"],
                field=ConflictField.FACT,
                explanation=XSS,
            )
        ]
    )

    class FakeLLM:
        @staticmethod
        def chat(messages, **kw):
            return f'Answer {XSS} [E1] /source/d1?start=1&end=2&event=x" onmouseover=alert(3) [E9]'

    app = create_app(settings)
    deps = app.state.deps
    deps.store, deps.llm, deps.chat_backend = store, FakeLLM(), "local"
    client = TestClient(app, base_url="http://127.0.0.1:8080")
    pages = [
        client.get("/timeline"),
        client.get("/source/d1", params={"start": start, "end": start + len(XSS), "event": "e1"}),
        client.get("/contradictions"),
        client.get("/alerts"),
        client.post("/chat", data={"question": XSS}),
        client.post("/chat", data={"question": XSS}, headers={"HX-Request": "true"}),
    ]
    for r in pages:
        assert r.status_code == 200, r.url
        body = r.text
        assert "<script>alert" not in body and "<img src=x" not in body, r.url
        assert '" onmouseover' not in body, r.url  # no attribute break-out from links
    assert "&lt;script&gt;" in pages[1].text and 'id="hl"' in pages[1].text
    data = client.get("/api/events").json()  # JSON; timeline.js inserts text, not HTML
    assert data["items"][0]["label"] == XSS
    store.close()


def test_search_survives_zero_vectors(tmp_path: Path) -> None:
    from pipeline import llm

    store = Store(tmp_path / "x.db", embedder=lambda t: [llm.hashed_embedding(x) for x in t])
    store.add_events([_event("e1"), _event("e2").model_copy(update={"description": "!!!"})])
    assert store.search_events("c", "??!", 5) == []
    assert [e.id for e in store.search_events("c", "something happened", 5)][0] == "e1"
    store.close()


def _doc(path: Path) -> object:
    import hashlib

    from pipeline.contracts import Document
    from pipeline.parse import MIME_TYPES

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return Document(
        id=sha[:16],
        case_id="c",
        path=str(path),
        filename=path.name,
        mime=MIME_TYPES[path.suffix.lower()],
        sha256=sha,
    )


def test_parse_survives_awkward_real_world_files(tmp_path: Path) -> None:
    import pymupdf

    from pipeline.parse import parse

    files = {
        "empty.docx": b"",
        "empty.pdf": b"",
        "utf16.txt": "On March 4, 2024 Zoë left.".encode("utf-16"),
        "cp1252.txt": "José “quoted”".encode("cp1252"),
        "bogus.eml": b"Subject: hi\nContent-Type: text/plain; charset=bogus-charset\n\nWe met.\n",
    }
    blank = pymupdf.open()
    blank.new_page()
    files["scanned.pdf"] = blank.tobytes()
    texts = {}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        texts[name] = parse(_doc(tmp_path / name)).text
    assert texts["empty.docx"] == texts["empty.pdf"] == texts["scanned.pdf"] == ""
    assert texts["utf16.txt"] == "On March 4, 2024 Zoë left."
    assert texts["cp1252.txt"] == "José “quoted”"
    assert "We met." in texts["bogus.eml"]


@pytest.mark.parametrize("bad", ["../../etc", "a/b", ".hidden", "", "x" * 65, "..", "a\\b"])
def test_case_id_cannot_be_a_path(bad: str) -> None:
    from pipeline.config import valid_case_id

    with pytest.raises(ValueError):
        valid_case_id(bad)


def test_layout_refuses_db_or_inbox_in_evidence(tmp_path: Path) -> None:
    from pipeline.config import Settings

    ev = tmp_path / "ev"
    ev.mkdir()
    with pytest.raises(RuntimeError):
        Settings(evidence_dir=ev, data_dir=tmp_path / "d", db_path=ev / "x.db").ensure_dirs()
    with pytest.raises(RuntimeError):
        Settings(evidence_dir=tmp_path / "d" / "inbox", data_dir=tmp_path / "d").check_layout()
    Settings(evidence_dir=ev, data_dir=tmp_path / "d").check_layout()
