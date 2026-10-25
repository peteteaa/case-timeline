"""Regression tests for problems found in the adversarial review.

The MCP stdout-safety tests live in ``tests/test_mcp_server.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


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


@pytest.mark.parametrize("bad", ["../../etc", "a/b", ".hidden", "", "x" * 65, "..", "a\\b"])
def test_case_id_cannot_be_a_path(bad: str) -> None:
    from pipeline.config import Settings, valid_case_id

    with pytest.raises(ValueError):
        valid_case_id(bad)
    with pytest.raises(ValueError):
        Settings(case_id=bad)


def test_layout_refuses_db_or_inbox_in_evidence(tmp_path: Path) -> None:
    from pipeline.config import Settings

    ev = tmp_path / "ev"
    ev.mkdir()
    with pytest.raises(RuntimeError):
        Settings(evidence_dir=ev, data_dir=tmp_path / "d", db_path=ev / "x.db").ensure_dirs()
    with pytest.raises(RuntimeError):
        Settings(evidence_dir=ev, data_dir=ev / "data").check_layout()
    with pytest.raises(RuntimeError):
        Settings(evidence_dir=tmp_path / "d" / "inbox", data_dir=tmp_path / "d").check_layout()
    with pytest.raises(RuntimeError):
        _ = Settings(evidence_dir=tmp_path / "d" / "inbox", data_dir=tmp_path / "d").inbox
    ok = Settings(evidence_dir=ev, data_dir=tmp_path / "d")
    ok.check_layout()
    assert ok.inbox == (tmp_path / "d" / "inbox").resolve() and ok.inbox.is_dir()
    assert not any(ev.iterdir())  # nothing was written to the evidence folder
