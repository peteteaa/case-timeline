"""The app's eval page: empty state, re-run, and rendering of a stored report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_app import Env


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    yield e
    e.store.close()


def _report_path(env: Env) -> Path:
    return Path(env.settings.data_dir) / "eval_demo.json"


def test_eval_page_without_report(env: Env) -> None:
    r = env.client.get("/cases/demo/eval")
    assert r.status_code == 200
    assert "No eval report yet" in r.text
    assert 'href="/cases/demo/eval"' in r.text  # nav link


def test_eval_rerun_writes_report_and_renders(env: Env) -> None:
    r = env.client.post("/cases/demo/eval")
    assert r.status_code == 303
    assert r.headers["location"] == "/cases/demo/eval"
    res = json.loads(_report_path(env).read_text(encoding="utf-8"))
    assert res["case_id"] == "demo"
    page = env.client.get("/cases/demo/eval")
    assert page.status_code == 200
    assert "Fact precision" in page.text and "Date accuracy" in page.text


def test_eval_rerun_requires_user(env: Env) -> None:
    anon = env.new_client(user=None)
    r = anon.post("/cases/demo/eval")
    assert r.status_code in (303, 401, 403)
    assert not _report_path(env).exists()


def test_eval_page_tolerates_bad_report(env: Env) -> None:
    path = _report_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"facts": {}}', encoding="utf-8")
    r = env.client.get("/cases/demo/eval")
    assert r.status_code == 200
    assert "No eval report yet" in r.text


def test_eval_page_escapes_gold_titles(env: Env) -> None:
    path = _report_path(env)
    env.client.post("/cases/demo/eval")
    res = json.loads(path.read_text(encoding="utf-8"))
    res["facts"]["matches"] = [
        {
            "gold_id": "<script>x</script>",
            "doc": "<b>d</b>",
            "fact_id": None,
            "score": None,
            "gold_date": None,
            "pred_date": None,
            "date_correct": None,
        }
    ]
    path.write_text(json.dumps(res), encoding="utf-8")
    r = env.client.get("/cases/demo/eval")
    assert "<script>x</script>" not in r.text and "&lt;script&gt;" in r.text
