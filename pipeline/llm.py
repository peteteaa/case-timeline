"""The only model client: local vLLM (OpenAI-compatible) for chat, local BGE-M3 for embeddings."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import math
import re
import socket
import threading
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

import httpx
from openai import DefaultHttpxClient, OpenAI
from pydantic import BaseModel, ValidationError

from pipeline.config import get_settings

log = logging.getLogger(__name__)

EMBED_DIM = 1024
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class LLMError(RuntimeError):
    """The model returned no usable output."""


def _is_host_local(host: str) -> bool:
    """True if ``host`` is loopback or an address assigned to this machine (e.g. docker0)."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if addr.is_loopback:
        return True
    if addr.is_unspecified or addr.is_multicast:
        return False
    family = socket.AF_INET6 if addr.version == 6 else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind((host, 0))  # only succeeds for addresses on a local interface
        except OSError:
            return False
    return True


def _require_loopback(url: str) -> str:
    """Return ``url`` if its host stays on this machine; case text must never leave it.

    Allowed: loopback, addresses assigned to a local interface (the docker0 bridge vLLM
    binds to), and hostnames listed in ``settings.llm_allowed_hosts`` (e.g.
    ``host.openshell.internal`` inside the NemoClaw sandbox, which resolves to the host).
    """
    host = (urlsplit(url).hostname or "").strip("[]")
    ok = host == "localhost" or host in get_settings().llm_allowed_hosts or _is_host_local(host)
    if not ok:
        raise LLMError(f"refusing non-local LLM endpoint {url!r}; models must run on this machine")
    return url


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        base_url=_require_loopback(s.llm_base_url),
        api_key="local",
        timeout=s.llm_timeout,
        max_retries=1,
        # Ignore HTTP(S)_PROXY etc.: a proxy would carry case text off the machine.
        http_client=DefaultHttpxClient(trust_env=False, timeout=s.llm_timeout),
    )


def _merge_extra_body(extra: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {k: dict(v) for k, v in _NO_THINK.items()}
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(body.get(key), dict):
            body[key].update(value)
        else:
            body[key] = value
    return body


def chat(messages: list[dict], **kw: Any) -> str:
    """Plain chat completion with thinking disabled; returns the assistant text."""
    kw.setdefault("model", get_settings().llm_model)
    kw.setdefault("temperature", 0.0)
    kw["extra_body"] = _merge_extra_body(kw.get("extra_body"))
    t0 = time.monotonic()
    resp = _client().chat.completions.create(messages=messages, **kw)
    content = resp.choices[0].message.content or ""
    log.debug(
        "chat: %d messages -> %d chars in %.1fs",
        len(messages),
        len(content),
        time.monotonic() - t0,
    )
    return content


def _clean_json(text: str) -> str:
    return _FENCE_RE.sub("", _THINK_RE.sub("", text)).strip()


def chat_json[T: BaseModel](
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> T:
    """Structured output validated against ``schema``; one retry with the validation error."""
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    extra: dict[str, Any] = {}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    last_error: ValidationError | None = None
    for attempt in range(2):
        raw = chat(messages, temperature=temperature, response_format=response_format, **extra)
        try:
            return schema.model_validate_json(_clean_json(raw))
        except ValidationError as exc:
            last_error = exc
            log.warning(
                "chat_json(%s): validation failed (attempt %d): %s",
                schema.__name__,
                attempt + 1,
                exc.errors(include_input=False)[:3],
            )
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "Your previous reply did not match the required JSON schema:\n"
                    f"{exc}\nReturn only corrected JSON that matches the schema.",
                },
            ]
    raise LLMError(f"{schema.__name__}: invalid structured output after retry") from last_error


# --- embeddings -------------------------------------------------------------------------------

_embed_lock = threading.Lock()
_st_model: Any = None
_st_failed = False


def _load_sentence_transformer() -> Any:
    """Load BGE-M3 once; returns None (and warns once) if unavailable."""
    global _st_model, _st_failed
    if _st_model is not None or _st_failed:
        return _st_model
    with _embed_lock:
        if _st_model is not None or _st_failed:
            return _st_model
        try:
            from sentence_transformers import SentenceTransformer  # lazy, optional dep

            _st_model = SentenceTransformer(get_settings().embed_model)
            log.info("embed: loaded %s", get_settings().embed_model)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            _st_failed = True
            log.warning(
                "embed: sentence-transformers/%s unavailable (%s); using hashed bag-of-words "
                "fallback vectors",
                get_settings().embed_model,
                exc,
            )
    return _st_model


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic L2-normalized signed hashed bag-of-words vector."""
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        vec[idx] += 1.0 if h[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts (normalized, dim 1024) locally; hashed fallback if BGE-M3 is unavailable."""
    if not texts:
        return []
    model = _load_sentence_transformer()
    if model is not None:
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
        return [[float(x) for x in v] for v in vectors]
    return [hashed_embedding(t) for t in texts]


def embed_backend() -> str:
    """Which embedding backend is active (without forcing a model load)."""
    if _st_model is not None:
        return get_settings().embed_model
    return "hashed-bow" if _st_failed else "not-loaded"


def health() -> dict:
    """Probe the local vLLM server's /models endpoint."""
    s = get_settings()
    url = s.llm_base_url.rstrip("/") + "/models"
    info: dict[str, Any] = {
        "ok": False,
        "base_url": s.llm_base_url,
        "model": s.llm_model,
        "models": [],
        "model_available": False,
        "embed_backend": embed_backend(),
        "error": None,
    }
    try:
        _require_loopback(url)
        resp = httpx.get(
            url, timeout=5.0, headers={"Authorization": "Bearer local"}, trust_env=False
        )
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
        info.update(ok=True, models=models, model_available=s.llm_model in models)
    except (httpx.HTTPError, ValueError, LLMError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        log.info("health: LLM server unreachable at %s (%s)", url, exc)
    return info
