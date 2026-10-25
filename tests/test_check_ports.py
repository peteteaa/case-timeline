"""scripts/check_ports.sh with a fake `ss` (no real sockets are inspected)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_ports.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(tmp_path: Path, listeners: list[str]) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / "bin"
    fake.mkdir()
    lines = "\n".join(f"LISTEN 0 4096 {addr} 0.0.0.0:*" for addr in listeners)
    ss = fake / "ss"
    ss.write_text(f"#!/usr/bin/env bash\ncat <<'OUT'\n{lines}\nOUT\n")
    ss.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ.get('PATH', '')}"}
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env, check=False
    )


@pytest.mark.parametrize(
    "listeners",
    [
        ["127.0.0.1:8090", "[::1]:8090", "0.0.0.0:22", "[::]:22"],
        ["127.0.0.1:8090", "172.17.0.1:8000"],
        ["127.0.0.1:8090", "172.17.0.1:8000", "172.18.0.1:8000"],  # vLLM + socat relay
    ],
)
def test_allowed_listeners_pass(tmp_path: Path, listeners: list[str]) -> None:
    res = _run(tmp_path, listeners)
    assert res.returncode == 0, res.stderr
    assert "OK:" in res.stdout and "172.18.0.1:8000" in res.stdout


@pytest.mark.parametrize(
    "addr",
    [
        "0.0.0.0:8000",
        "172.18.0.1:8090",  # relay address, wrong port
        "172.18.0.2:8000",  # another bridge host
        "172.19.0.1:8000",
        "[::]:8000",
        "192.168.1.5:8090",
    ],
)
def test_other_non_loopback_listeners_fail(tmp_path: Path, addr: str) -> None:
    res = _run(tmp_path, ["127.0.0.1:8090", addr])
    assert res.returncode == 1
    assert f"FAIL: {addr}" in res.stderr
