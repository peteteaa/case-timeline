"""Download the public evaluation documents listed in eval/docs/manifest.json (``make fetch-eval``).

Run this in setup mode (SPEC §11 step 6): it is the only part of the project that uses the
internet. Files land in ``eval/docs/<group>/<filename>`` (git-ignored), one folder per case, so
``CT_EVIDENCE_DIR=eval/docs/ftx CT_CASE_ID=ftx make scan`` works. Existing files are not
downloaded again. The first fetch records each file's sha256 and retrieval date in the manifest;
later runs verify the hash and fail loudly on a mismatch. HTML documents (SEC EDGAR exhibits)
also get a plain-text rendition (``text_filename``) because the parser reads PDF, DOCX, TXT,
MD and EML only.

    uv run python scripts/fetch_eval.py [--manifest PATH] [--only GROUP ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval" / "docs" / "manifest.json"
DEFAULT_UA = "case-timeline-eval lozhang@dons.usfca.edu"  # SEC asks for contact info
PAUSE_SECONDS = 1.0  # well under SEC's 10 requests/second limit
TIMEOUT = 120
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,200}$")

log = logging.getLogger("fetch_eval")


class _Text(HTMLParser):
    """Visible text with line breaks at block elements; drops script and style."""

    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK or tag == "td":
            self.parts.append("\n" if tag != "td" else " ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(re.sub(r"\s+", " ", data))  # source line wraps are not breaks


def html_to_text(raw: bytes) -> str:
    parser = _Text()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _target(docs_dir: Path, doc: dict[str, Any], key: str = "filename") -> Path:
    name, group = doc[key], doc.get("group") or "misc"
    for part in (name, group):
        if not _SAFE_NAME.fullmatch(part) or part in (".", ".."):
            raise ValueError(f"unsafe name in manifest: {part!r}")
    return docs_dir / group / name


def download(url: str, dest: Path, user_agent: str) -> None:
    """Fetch ``url`` to ``dest`` atomically (a partial download never looks complete)."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - manifest URLs
                shutil.copyfileobj(resp, tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(dest)


def fetch_all(
    manifest_path: Path = MANIFEST,
    *,
    only: list[str] | None = None,
    dry_run: bool = False,
    pause: float = PAUSE_SECONDS,
) -> dict[str, list[str]]:
    """Fetch every document; returns ids by outcome (fetched, present, failed, mismatch)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    docs_dir = manifest_path.parent
    ua = manifest.get("user_agent") or DEFAULT_UA
    out: dict[str, list[str]] = {"fetched": [], "present": [], "failed": [], "mismatch": []}
    first = True
    for doc in manifest.get("documents", []):
        if only and doc.get("group") not in only:
            continue
        dest = _target(docs_dir, doc)
        if dest.exists():
            outcome = "present"
        elif dry_run:
            log.info("would fetch %s -> %s", doc["url"], dest.relative_to(docs_dir))
            continue
        else:
            if not first:
                time.sleep(pause)
            first = False
            log.info("fetching %s", doc["url"])
            try:
                download(doc["url"], dest, ua)
            except Exception as exc:  # keep going; report at the end
                log.error("%s: %s", doc["id"], exc)
                out["failed"].append(doc["id"])
                continue
            outcome = "fetched"
        digest = sha256_file(dest)
        if doc.get("sha256") and doc["sha256"] != digest:
            log.error(
                "%s: sha256 mismatch (manifest %s, file %s)", doc["id"], doc["sha256"], digest
            )
            out["mismatch"].append(doc["id"])
            if outcome == "fetched":
                dest.unlink()  # never keep unverified content
            continue
        if not doc.get("sha256"):
            doc["sha256"] = digest
        if outcome == "fetched" or not doc.get("retrieved_at"):
            doc["retrieved_at"] = doc.get("retrieved_at") or datetime.now(UTC).date().isoformat()
        if doc.get("text_filename"):
            text_dest = _target(docs_dir, doc, "text_filename")
            if not text_dest.exists():
                text_dest.write_text(html_to_text(dest.read_bytes()), encoding="utf-8")
        out[outcome].append(doc["id"])
    if not dry_run:
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--only", nargs="*", help="groups to fetch (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    res = fetch_all(args.manifest, only=args.only, dry_run=args.dry_run)
    sys.stdout.write(
        f"fetched {len(res['fetched'])}, already present {len(res['present'])}, "
        f"failed {len(res['failed'])}, sha256 mismatch {len(res['mismatch'])}\n"
    )
    for key in ("failed", "mismatch"):
        if res[key]:
            sys.stdout.write(f"{key}: {', '.join(res[key])}\n")
    return 1 if res["failed"] or res["mismatch"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
