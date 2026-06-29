MODEL ?= llama3.2:3b
EMBED_MODEL ?= nomic-embed-text
PORT ?= 8000
PROMPT ?= Hello!
DATA ?= data
LIMIT ?= 20
ID ?=
GOLDEN ?= evals/golden.json
BASELINE ?=
OUTPUT ?=
MAX_STEPS ?=

.DEFAULT_GOAL := help

.PHONY: help setup install-ollama check lint format typecheck test run chat chat-once rag rag-once ingest traces trace eval agent clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup: ## Install deps, git hooks, .env, and pull the Ollama models (MODEL=llama3.2:3b EMBED_MODEL=nomic-embed-text)
	uv sync
	uv run pre-commit install
	test -f .env || cp .env.example .env
	@if ! command -v ollama > /dev/null 2>&1; then \
		echo ""; \
		echo "  ollama is not installed. Run 'make install-ollama' (or install it"; \
		echo "  manually), then re-run 'make setup'."; \
		echo ""; \
		exit 1; \
	fi
	ollama pull $(MODEL)
	ollama pull $(EMBED_MODEL)

install-ollama: ## Install Ollama (macOS via Homebrew, Linux via official script)
	@if command -v ollama > /dev/null 2>&1; then \
		echo "ollama already installed: $$(ollama --version 2>/dev/null || echo present)"; \
	elif [ "$$(uname -s)" = "Darwin" ]; then \
		if command -v brew > /dev/null 2>&1; then \
			brew install ollama; \
		else \
			echo "Homebrew not found. Install it from https://brew.sh, then re-run."; \
			exit 1; \
		fi; \
	elif [ "$$(uname -s)" = "Linux" ]; then \
		curl -fsSL https://ollama.com/install.sh | sh; \
	else \
		echo "Unsupported OS: $$(uname -s). See https://ollama.com/download"; \
		exit 1; \
	fi

check: lint typecheck ## Run ruff (lint + format check) and mypy

lint: ## Lint and verify formatting (read-only, fails if unformatted)
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint issues and reformat
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Type-check with mypy
	uv run mypy src

test: ## Run the test suite
	uv run pytest

run: ## Start the API (uvicorn via app.main)
	uv run python -m app.main

chat: ## Send a sample /chat request (PROMPT="..." PORT=8000)
	@curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { \
		echo "API not running on port $(PORT). Start it in another terminal with 'make run',"; \
		echo "or use 'make chat-once' to start it, send the prompt, and stop it automatically."; \
		exit 1; \
	}
	@curl -s http://localhost:$(PORT)/chat \
		-H 'Content-Type: application/json' \
		-d '{"messages": [{"role": "user", "content": "$(PROMPT)"}]}'
	@echo

chat-once: ## Start API, send one /chat, then stop it (PROMPT="..." PORT=8000)
	@uv run uvicorn app.main:app --host 127.0.0.1 --port $(PORT) > /dev/null 2>&1 & \
	SERVER_PID=$$!; \
	trap 'kill $$SERVER_PID 2>/dev/null' EXIT; \
	echo "Waiting for API on port $(PORT)..."; \
	for i in $$(seq 1 30); do \
		curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 && break; \
		sleep 0.5; \
	done; \
	curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { echo "API failed to start"; exit 1; }; \
	echo "--- response ---"; \
	curl -s http://localhost:$(PORT)/chat \
		-H 'Content-Type: application/json' \
		-d '{"messages": [{"role": "user", "content": "$(PROMPT)"}]}' | jq .

rag: ## Send a sample /rag request (PROMPT="..." PORT=8000); run `make ingest` first
	@curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { \
		echo "API not running on port $(PORT). Start it in another terminal with 'make run',"; \
		echo "or use 'make rag-once' to start it, send the question, and stop it automatically."; \
		exit 1; \
	}
	@echo 'Asking: "$(PROMPT)"'
	@curl -s http://localhost:$(PORT)/rag \
		-H 'Content-Type: application/json' \
		-d '{"question": "$(PROMPT)"}'
	@echo

rag-once: ## Start API, send one /rag, then stop it (PROMPT="..."); run `make ingest` first
	@uv run uvicorn app.main:app --host 127.0.0.1 --port $(PORT) > /dev/null 2>&1 & \
	SERVER_PID=$$!; \
	trap 'kill $$SERVER_PID 2>/dev/null' EXIT; \
	echo "Waiting for API on port $(PORT)..."; \
	for i in $$(seq 1 30); do \
		curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 && break; \
		sleep 0.5; \
	done; \
	curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { echo "API failed to start"; exit 1; }; \
	echo "--- response ---"; \
	echo "Asking: \"$(PROMPT)\""; \
	curl -s http://localhost:$(PORT)/rag \
		-H 'Content-Type: application/json' \
		-d '{"question": "$(PROMPT)"}' | jq .

ingest: ## Ingest documents into the RAG store (DATA=data)
	uv run python -m app.rag.ingest $(DATA)

traces: ## List recent traces (LIMIT=20 PORT=8000); run the API first with 'make run'
	@curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { \
		echo "API not running on port $(PORT). Start it in another terminal with 'make run'."; \
		exit 1; \
	}
	@curl -s "http://localhost:$(PORT)/traces?limit=$(LIMIT)" | jq .

trace: ## Show one trace and its spans (ID=<trace_id> PORT=8000); get an id from 'make traces'
	@test -n "$(ID)" || { echo "Usage: make trace ID=<trace_id>  (list ids with 'make traces')"; exit 1; }
	@curl -sf http://localhost:$(PORT)/health > /dev/null 2>&1 || { \
		echo "API not running on port $(PORT). Start it in another terminal with 'make run'."; \
		exit 1; \
	}
	@curl -s "http://localhost:$(PORT)/traces/$(ID)" | jq .

eval: ## Run the eval harness over the golden set (GOLDEN=… OUTPUT=… BASELINE=…); needs Ollama + `make ingest`
	uv run python -m app.evals $(GOLDEN) $(if $(OUTPUT),--output $(OUTPUT)) $(if $(BASELINE),--baseline $(BASELINE))

agent: ## Ask the ReAct agent a question (PROMPT="…" MAX_STEPS=…); needs Ollama + `make ingest`
	uv run python -m app.agent "$(PROMPT)" $(if $(MAX_STEPS),--max-steps $(MAX_STEPS))

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
