.PHONY: help venv install test eval eval-mcp servers gateway run demo lint clean

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := $(PWD)/src

help:
	@echo "Smart City Navigator — make targets"
	@echo "  install    create venv + install deps"
	@echo "  test       run the pytest suite"
	@echo "  eval       run the 20-prompt eval suite (in-process)"
	@echo "  eval-mcp   run the eval suite through the live MCP servers"
	@echo "  servers    start the 3 FastMCP servers (foreground)"
	@echo "  run        start 3 MCP servers + gateway (http://localhost:8000)"
	@echo "  demo       CLI demo: make demo Q='from Times Square to Coney Island'"
	@echo "  clean      remove venv + caches"

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements-dev.txt

test:
	$(BIN)/pytest -q

eval:
	$(BIN)/python eval/run_eval.py

eval-mcp:
	$(BIN)/python eval/run_eval.py --transport mcp

servers:
	NAVIGATOR_SIMULATE_FEED=$${NAVIGATOR_SIMULATE_FEED:-0} bash scripts/run_all.sh

run:
	PYTHON=$(BIN)/python bash scripts/run_all.sh

demo:
	$(BIN)/python scripts/demo.py $(Q)

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ *.sqlite checkpoints.sqlite
