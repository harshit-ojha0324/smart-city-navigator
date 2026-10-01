.PHONY: help install lock test lint eval graph run demo clean

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := $(PWD)/src

help:
	@echo "Smart City Navigator — make targets"
	@echo "  install    create venv + install from requirements.lock"
	@echo "  lock       re-resolve requirements*.txt into requirements.lock"
	@echo "  test       run the pytest suite"
	@echo "  lint       ruff check"
	@echo "  eval       run the core 20-prompt eval suite (in-process); pass flags in ARGS:"
	@echo "             ARGS='--set heldout'  (core | heldout | fresh | wild | all)"
	@echo "             ARGS='--transport mcp'  through the live MCP servers"
	@echo "  graph      rebuild the station graph from the MTA GTFS feed"
	@echo "  run        start 3 MCP servers + gateway (http://localhost:8000)"
	@echo "  demo       CLI demo: make demo Q='from Times Square to Coney Island'"
	@echo "  clean      remove venv + caches"

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q --require-hashes -r requirements.lock

# Resolve requirements*.txt into the hash-pinned lockfile CI and Docker install.
# Run after changing a version bound; commit the result. 3.10 is CI's oldest
# Python — without it uv resolves for the running interpreter and drops 3.10 pins.
lock:
	uv pip compile requirements-dev.txt --universal --python-version 3.10 --generate-hashes -o requirements.lock

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check .

eval:
	$(BIN)/python eval/run_eval.py $(ARGS)

graph:
	$(BIN)/python scripts/build_graph.py

run:
	PYTHON=$(BIN)/python bash scripts/run_all.sh

demo:
	$(BIN)/python scripts/demo.py $(Q)

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ *.sqlite checkpoints.sqlite
