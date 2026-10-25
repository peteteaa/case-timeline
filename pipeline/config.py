"""Runtime settings (env prefix ``CT_``). Relative paths resolve against the project root."""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def valid_case_id(case_id: str) -> str:
    """Return ``case_id`` if it is a safe identifier (also used as a folder name), else raise.

    Rejects path separators and leading dots, so a case id can never point the inbox scan
    (``inbox/<case_id>``) or any per-case file outside its parent folder.
    """
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(
            f"invalid case id {case_id!r}: use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return case_id


def _overlaps(a: Path, b: Path) -> bool:
    """True if one path equals or contains the other."""
    a, b = a.resolve(), b.resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _abs(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class Settings(BaseSettings):
    """Case Timeline Agent configuration. Everything stays on this machine."""

    model_config = SettingsConfigDict(
        env_prefix="CT_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    evidence_dir: Path = Path("./tests/fixtures/case_demo")  # read-only, never written
    data_dir: Path = Path("./data")
    db_path: Path | None = None  # defaults to data_dir / "case.db"
    case_id: str = "demo"

    # vLLM binds the docker0 bridge so the NemoClaw sandbox can reach it; never 0.0.0.0.
    llm_base_url: str = "http://172.17.0.1:8000/v1"
    # Hosts trusted to be this machine even from inside the sandbox (docker host-gateway).
    llm_allowed_hosts: list[str] = ["host.openshell.internal", "172.17.0.1"]
    llm_model: str = "qwen3.6-35b-a3b"
    llm_timeout: float = 300.0  # seconds per request
    # Most model requests in flight at once (shared by every caller in this process). vLLM
    # batches concurrent requests, so a few parallel calls raise throughput a lot; 1 restores
    # strictly sequential processing. The shared Qwen3.6 server has crashed at about 10-16
    # concurrent requests (docs/HANDOFF.md), so the default stays at 4.
    llm_concurrency: int = Field(default=4, ge=1, le=256)
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    ui_host: str = "127.0.0.1"
    ui_port: int = 8090

    # Case time zone (IANA name, e.g. "America/Chicago"). A time stated explicitly in UTC/GMT/
    # Zulu is converted to it before its calendar day is taken; None = no conversion.
    case_tz: str | None = None
    # The case's incident date, when the documents' consensus should not be used.
    case_incident_date: date | None = None

    max_drafts_per_doc: int = 40  # keep the most confident drafts per document (0 = no cap)
    worker_poll_seconds: float = 2.0
    run_worker: bool = True  # the app server starts the job worker

    @field_validator("case_tz", mode="before")
    @classmethod
    def _valid_tz(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str):
            raise ValueError("case_tz must be an IANA time zone name")
        try:
            ZoneInfo(value.strip())
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone {value!r}") from exc
        return value.strip()

    @field_validator("case_incident_date", mode="before")
    @classmethod
    def _empty_date(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        self.evidence_dir = _abs(self.evidence_dir)
        self.data_dir = _abs(self.data_dir)
        self.db_path = _abs(self.db_path) if self.db_path else self.data_dir / "case.db"
        valid_case_id(self.case_id)
        return self

    def check_layout(self) -> None:
        """Raise RuntimeError if anything we write could land in the read-only evidence dir."""
        evidence = Path(self.evidence_dir).resolve()
        assert self.db_path is not None
        for name, path in (("data_dir", self.data_dir), ("db_path", self.db_path)):
            if Path(path).resolve().is_relative_to(evidence):
                raise RuntimeError(f"{name} must not be inside the evidence folder")
        if _overlaps(self.data_dir / "inbox", evidence):
            raise RuntimeError("the upload inbox and the evidence folder must not overlap")

    @property
    def inbox(self) -> Path:
        """Upload folder (inside data_dir, never the evidence dir); created on access."""
        self.check_layout()
        path = self.data_dir / "inbox"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def ui_base_url(self) -> str:
        return f"http://{self.ui_host}:{self.ui_port}"

    def ensure_dirs(self) -> None:
        """Create data_dir (and the db's parent) if missing; refuses an unsafe layout."""
        self.check_layout()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        assert self.db_path is not None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings; creates data_dir. Call ``get_settings.cache_clear()`` in tests."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
