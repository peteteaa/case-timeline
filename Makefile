# Case Timeline Agent - developer targets. Everything runs locally; services bind to 127.0.0.1.
SHELL := /bin/bash
UV    ?= uv
PY    := $(UV) run python
CASE  ?= demo

.PHONY: help setup lint fmt test test-llm eval llm ui mcp scan fixtures ports preflight

help:
	@echo "targets: setup lint fmt test test-llm eval CASE=<id> llm ui mcp scan CASE=<id> fixtures ports preflight"

setup:
	$(UV) sync

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest -q

test-llm:
	$(UV) run pytest -q -m llm

eval:
	$(PY) -m eval --case $(CASE)

llm:
	scripts/serve_llm.sh

ui:
	$(PY) -m ui

mcp:
	$(PY) -m mcp_server

# Run one scan of the evidence folder and wait for the job to finish.
define SCAN_PY
import sys, time
from pipeline.config import get_settings
from pipeline.runner import scan_case
from pipeline.store import Store

job = scan_case(sys.argv[1])
store = Store(get_settings().db_path)
while str(job.state) not in ("done", "failed"):
    time.sleep(2)
    job = store.get_job(job.id) or job
print(job.model_dump_json(indent=2))
sys.exit(1 if str(job.state) == "failed" else 0)
endef
export SCAN_PY

scan:
	$(PY) -c "$$SCAN_PY" $(CASE)

fixtures:
	$(PY) scripts/make_fixtures.py

ports:
	scripts/check_ports.sh

preflight:
	scripts/preflight.sh
