"""Runtime settings (env prefix ``CT_``). Relative paths resolve against the project root."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
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

    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = "qwen3.6-35b-a3b"
    llm_timeout: float = 300.0  # seconds per request
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    ui_host: str = "127.0.0.1"
    ui_port: int = 8080

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
