"""MCP server tests: spawn the stdio server against a temp data dir with a fake backend."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "scan_case",
    "job_status",
    "get_changes",
    "get_timeline",
    "search_events",
    "get_source",
    "list_contradictions",
    "daily_digest",
}


def _env(data_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CT_")}
    env.update(
        CT_DATA_DIR=str(data_dir),
        CT_MCP_BACKEND="tests.mcp_server_fakes:make_backend",
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
    )
    return env


def _structured(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content
    data = result.structured_content
    if data is None:
        data = json.loads(result.content[0].text)
    return data.get("result", data) if set(data) == {"result"} else data


async def _session_calls(data_dir: Path) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server"], env=_env(data_dir), cwd=str(ROOT)
    )
    out: dict[str, Any] = {}
    with open(data_dir / "server.stderr", "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                out["server_name"] = init.server_info.name
                tools = await session.list_tools()
                out["tools"] = {t.name: t for t in tools.tools}
                call = session.call_tool
                out["scan"] = _structured(await call("scan_case", {}))
                out["status"] = _structured(await call("job_status", {"job_id": "job-1"}))
                out["missing_job"] = _structured(await call("job_status", {"job_id": "nope"}))
                out["changes1"] = _structured(await call("get_changes", {}))
                out["changes2"] = _structured(await call("get_changes", {}))
                out["other_consumer"] = _structured(await call("get_changes", {"consumer": "ui"}))
                out["timeline"] = _structured(await call("get_timeline", {}))
                out["window"] = _structured(
                    await call("get_timeline", {"start": "2024-03-04", "end": "2024-12-31"})
                )
                out["accepted"] = _structured(await call("get_timeline", {"status": "accepted"}))
                out["search"] = _structured(await call("search_events", {"query": "warehouse"}))
                ev = out["search"]["events"][0]["source"]
                out["source"] = _structured(
                    await call(
                        "get_source",
                        {
                            "doc_id": ev["doc_id"],
                            "char_start": ev["char_start"],
                            "char_end": ev["char_end"],
                            "context": 10,
                        },
                    )
                )
                out["bad_source"] = _structured(
                    await call("get_source", {"doc_id": "doc1", "char_start": 5, "char_end": 10**6})
                )
                out["contradictions"] = _structured(await call("list_contradictions", {}))
                out["dismissed"] = _structured(
                    await call("list_contradictions", {"status": "dismissed"})
                )
                out["digest"] = _structured(await call("daily_digest", {}))
    return out


@pytest.fixture(scope="module")
def calls(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    data_dir = tmp_path_factory.mktemp("mcp_data")
    out = anyio.run(_session_calls, data_dir)
    out["data_dir"] = data_dir
    return out


def test_exactly_eight_tools(calls: dict[str, Any]) -> None:
    assert calls["server_name"] == "case-timeline"
    assert set(calls["tools"]) == EXPECTED_TOOLS
    for tool in calls["tools"].values():
        assert tool.description and len(tool.description) > 40
    assert calls["tools"]["get_timeline"].annotations.read_only_hint is True
    assert calls["tools"]["scan_case"].annotations.read_only_hint is False


def test_scan_and_job_status(calls: dict[str, Any]) -> None:
    assert calls["scan"] == {"job_id": "job-1", "case_id": "demo", "state": "queued"}
    assert calls["status"]["state"] == "done" and calls["status"]["found"] is True
    assert calls["missing_job"]["found"] is False


def test_get_changes_advances_cursor_per_consumer(calls: dict[str, Any]) -> None:
    c1, c2 = calls["changes1"], calls["changes2"]
    assert c1["counts"]["events"] == 4 and c1["counts"]["needs_attorney_review"] == 1
    assert c1["new_alerts"][0]["dashboard_url"] == "http://127.0.0.1:8080/alerts"
    assert c2["counts"] == {
        "events": 0,
        "contradictions": 0,
        "alerts": 0,
        "needs_attorney_review": 0,
    }
    assert c2["since"] is not None
    assert calls["other_consumer"]["counts"]["events"] == 4
    assert (calls["data_dir"] / "mcp_cursors.json").exists()


def test_timeline(calls: dict[str, Any]) -> None:
    ids = [e["id"] for e in calls["timeline"]["events"]]
    assert ids == ["e1", "e2", "e3", "e4"]  # sorted by date, undated last
    first = calls["timeline"]["events"][0]
    assert first["source"]["quote"].startswith("On March 3")
    assert first["source"]["link"].startswith("http://127.0.0.1:8080/source/doc1?")
    assert [e["id"] for e in calls["window"]["events"]] == ["e2", "e3"]
    assert [e["id"] for e in calls["accepted"]["events"]] == ["e1"]
    e3 = next(e for e in calls["timeline"]["events"] if e["id"] == "e3")
    assert e3["needs_attorney_review"] is True


def test_search_and_source(calls: dict[str, Any]) -> None:
    assert [e["id"] for e in calls["search"]["events"]] == ["e1"]
    src = calls["source"]
    assert src["found"] and src["page"] == 1 and src["filename"] == "memo.pdf"
    assert src["quote"] == "On March 3, 2024 Alice met Bob at the warehouse."
    assert "[[On March 3, 2024 Alice met Bob at the warehouse.]]" in src["highlighted"]
    assert len(src["before"]) <= 10 and len(src["after"]) <= 10
    assert calls["bad_source"]["found"] is False


def test_contradictions_and_digest(calls: dict[str, Any]) -> None:
    c = calls["contradictions"]
    assert c["count"] == 1
    assert {e["id"] for e in c["contradictions"][0]["events"]} == {"e1", "e2"}
    assert calls["dismissed"]["count"] == 0
    assert calls["digest"]["new_events"] == 4
    assert calls["digest"]["dashboard_url"] == "http://127.0.0.1:8080/alerts"


def test_logs_go_to_data_dir_not_stdout(calls: dict[str, Any]) -> None:
    log = (calls["data_dir"] / "mcp.log").read_text()
    assert "starting case-timeline MCP server" in log
    stderr = (calls["data_dir"] / "server.stderr").read_text()
    assert "NOISE FROM A LIBRARY THAT PRINTS" in stderr


def _rpc(proc: subprocess.Popen, lines: queue.Queue, msg: dict, expect: bool = True) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    if not expect:
        return {}
    return json.loads(lines.get(timeout=30))


def test_stdout_is_clean_json_rpc(tmp_path: Path) -> None:
    """Raw stdio: every stdout line must be JSON-RPC, even when tools print."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        cwd=ROOT,
        env=_env(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    raw: list[str] = []
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            raw.append(line)
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()
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
        for i, name in enumerate(["scan_case", "daily_digest"], start=3):
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
            assert resp["id"] == i and not resp["result"].get("isError")
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    for line in raw:
        assert json.loads(line)["jsonrpc"] == "2.0", line
    assert not any("NOISE" in line for line in raw)
