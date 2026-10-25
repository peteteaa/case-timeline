# Case Timeline Agent - developer targets (SPEC §10). Everything runs locally; services bind to
# 127.0.0.1 (vLLM: the docker0 bridge 172.17.0.1).
SHELL := /bin/bash
UV    ?= uv
PY    := $(UV) run python
CASE  ?= demo

.PHONY: help setup lint fmt test test-llm eval fetch-eval llm app ui serve-mcp mcp scan \
	fixtures ports preflight

help:
	@echo "targets:"
	@echo "  setup                uv sync (needs internet once)"
	@echo "  test / test-llm      unit tests (no model) / live-model tests"
	@echo "  lint / fmt           ruff check + format check / auto-fix"
	@echo "  llm                  start vLLM on 172.17.0.1:8000"
	@echo "  app (alias ui)       web UI + job worker on http://127.0.0.1:8090"
	@echo "  serve-mcp (alias mcp) MCP server over stdio (normally spawned by OpenClaw)"
	@echo "  scan CASE=<id>       queue a scan and run it inline (no app server needed)"
	@echo "  eval CASE=<id>       metrics vs gold; report in data/eval_<id>.json"
	@echo "  fetch-eval           download the public eval documents into eval/docs/"
	@echo "  fixtures             rebuild the fictional demo case"
	@echo "  ports / preflight    exposure check / host checks"

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

fetch-eval:
	$(PY) scripts/fetch_eval.py

llm:
	scripts/serve_llm.sh

# The app server: web UI plus the job worker that runs queued scans.
app:
	$(PY) -m app

ui: app

serve-mcp:
	$(PY) -m mcp_server

mcp: serve-mcp

# Queue a scan and run the queue inline (same code path as the app server's worker).
scan:
	$(PY) -m pipeline.jobs scan --case $(CASE)

fixtures:
	$(PY) scripts/make_fixtures.py

ports:
	scripts/check_ports.sh

preflight:
	scripts/preflight.sh
