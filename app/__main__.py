"""Run the web UI and job worker: ``uv run python -m app``. Binds to loopback only."""

from __future__ import annotations

import logging
import sys

import uvicorn

from app.app import create_app, host_is_loopback
from pipeline.config import get_settings

log = logging.getLogger("app")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    host = str(settings.ui_host)
    if not host_is_loopback(host):
        log.error("refusing to start: ui_host %r is not a loopback address", host)
        return 2
    uvicorn.run(
        create_app(settings),
        host=host,
        port=settings.ui_port,
        proxy_headers=False,
        server_header=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
