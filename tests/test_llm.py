"""llm.py tests with a mocked OpenAI client (no vLLM needed)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from pipeline import llm
from pipeline.config import Settings, get_settings


class Item(BaseModel):
    name: str
    count: int


class FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_client(monkeypatch):
    def install(replies: list[str]) -> FakeCompletions:
        completions = FakeCompletions(replies)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_client", lambda: client)
        return completions

    return install


def test_chat_json_request_shape(fake_client):
    fc = fake_client(['{"name": "x", "count": 2}'])
    out = llm.chat_json("sys", "user text", Item)
    assert out == Item(name="x", count=2)
    [call] = fc.calls
    assert call["model"] == get_settings().llm_model
    assert call["temperature"] == 0.0
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Item", "schema": Item.model_json_schema()},
    }
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_chat_json_retries_once_with_error(fake_client):
    fc = fake_client(['{"name": "x"}', '```json\n{"name": "x", "count": 3}\n```'])
    assert llm.chat_json("s", "u", Item).count == 3
    assert len(fc.calls) == 2
    retry_msgs = fc.calls[1]["messages"]
    assert retry_msgs[-2] == {"role": "assistant", "content": '{"name": "x"}'}
    assert "count" in retry_msgs[-1]["content"]


def test_chat_json_gives_up_after_retry(fake_client):
    fc = fake_client(["not json", "still not json"])
    with pytest.raises(llm.LLMError):
        llm.chat_json("s", "u", Item)
    assert len(fc.calls) == 2


def test_chat_json_strips_think_block(fake_client):
    fake_client(['<think>hmm</think>{"name": "y", "count": 1}'])
    assert llm.chat_json("s", "u", Item).name == "y"


def test_chat_merges_extra_body(fake_client):
    fc = fake_client(["hi"])
    out = llm.chat(
        [{"role": "user", "content": "hello"}],
        extra_body={"chat_template_kwargs": {"foo": 1}, "top_k": 5},
    )
    assert out == "hi"
    body = fc.calls[0]["extra_body"]
    assert body == {"chat_template_kwargs": {"enable_thinking": False, "foo": 1}, "top_k": 5}


def test_hashed_embedding_deterministic_and_normalized():
    a = llm.hashed_embedding("Alice met Bob on Monday")
    b = llm.hashed_embedding("alice met bob on monday")
    assert a == b and len(a) == llm.EMBED_DIM
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert llm.hashed_embedding("") == [0.0] * llm.EMBED_DIM


def test_embed_falls_back_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(llm, "_st_model", None)
    monkeypatch.setattr(llm, "_st_failed", False)
    import builtins

    real_import = builtins.__import__

    def no_st(name, *args, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_st)
    with caplog.at_level("WARNING", logger="pipeline.llm"):
        v1 = llm.embed(["contract signed", "rent due"])
        v2 = llm.embed(["contract signed"])
    assert len(v1) == 2 and len(v1[0]) == 1024
    assert v1[0] == v2[0]
    assert sum("fallback" in r.message for r in caplog.records) == 1
    assert llm.embed([]) == []
    assert llm.embed_backend() == "hashed-bow"


def test_embed_uses_sentence_transformer_when_loaded(monkeypatch):
    class FakeST:
        def encode(self, texts, normalize_embeddings, batch_size):
            assert normalize_embeddings
            return [[0.5] * 1024 for _ in texts]

    monkeypatch.setattr(llm, "_st_model", FakeST())
    assert llm.embed(["a"]) == [[0.5] * 1024]


def test_health_ok(monkeypatch):
    def fake_get(url, timeout, headers, **kw):
        assert url == "http://172.17.0.1:8000/v1/models"
        req = httpx.Request("GET", url)
        return httpx.Response(200, json={"data": [{"id": "qwen3.6-35b-a3b"}]}, request=req)

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    h = llm.health()
    assert h["ok"] and h["model_available"] and h["error"] is None


def test_health_down(monkeypatch):
    def boom(url, timeout, headers, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm.httpx, "get", boom)
    h = llm.health()
    assert not h["ok"] and "ConnectError" in h["error"]


def test_settings_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CT_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("CT_CASE_ID", "c42")
    s = Settings()
    assert s.case_id == "c42"
    assert s.db_path == tmp_path / "d" / "case.db"
    assert s.evidence_dir.is_absolute()
    assert s.inbox == tmp_path / "d" / "inbox" and s.inbox.is_dir()
    assert s.ui_host == "127.0.0.1"


def test_llm_concurrency_setting(monkeypatch):
    monkeypatch.delenv("CT_LLM_CONCURRENCY", raising=False)
    assert Settings().llm_concurrency == 8
    monkeypatch.setenv("CT_LLM_CONCURRENCY", "3")
    assert Settings().llm_concurrency == 3
    monkeypatch.setenv("CT_LLM_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        Settings()


def test_map_ordered_keeps_order_and_returns_errors_in_place(monkeypatch):
    import threading
    import time

    def work(i: int) -> int:
        time.sleep(0.01 * (5 - i))  # later items finish first
        if i == 2:
            raise ValueError("bad item")
        return i * 10

    out = llm.map_ordered(work, list(range(5)), workers=5)
    assert out[:2] == [0, 10] and out[3:] == [30, 40]
    assert isinstance(out[2], ValueError)
    # One worker runs inline on the calling thread, in order.
    seen: list[str] = []
    llm.map_ordered(lambda i: seen.append(threading.current_thread().name), [1, 2], workers=1)
    assert seen == [threading.current_thread().name] * 2
    assert llm.map_ordered(work, []) == []


def test_request_slot_limits_requests_in_flight(monkeypatch):
    import threading
    import time

    monkeypatch.setenv("CT_LLM_CONCURRENCY", "2")
    get_settings.cache_clear()
    try:
        llm.reset_limiter_stats()
        lock = threading.Lock()
        flight = {"now": 0, "max": 0}

        def request(_: int) -> None:
            with llm.request_slot():
                with lock:
                    flight["now"] += 1
                    flight["max"] = max(flight["max"], flight["now"])
                time.sleep(0.03)
                with lock:
                    flight["now"] -= 1

        llm.map_ordered(request, list(range(8)), workers=8)  # more threads than slots
        assert flight["max"] == 2
        assert llm.limiter_stats() == {"in_flight": 0, "peak": 2, "total": 8}
    finally:
        get_settings.cache_clear()


def test_chat_holds_a_limiter_slot(fake_client):
    fake_client(["ok"])
    llm.reset_limiter_stats()
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert llm.limiter_stats()["total"] == 1 and llm.limiter_stats()["in_flight"] == 0


@pytest.mark.llm
def test_live_chat_json():
    assert llm.health()["ok"]
    out = llm.chat_json("Reply with JSON.", "name=apple, count=3", Item)
    assert out.count == 3
