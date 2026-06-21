MODEL ?= llama3.2:3b
EMBED_MODEL ?= nomic-embed-text
PORT ?= 8000
PROMPT ?= Hello!
DATA ?= data

.DEFAULT_GOAL := help

.PHONY: help setup install-ollama check lint format typecheck test run chat ingest clean

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
	curl -s http://localhost:$(PORT)/chat \
		-H 'Content-Type: application/json' \
		-d '{"messages": [{"role": "user", "content": "$(PROMPT)"}]}'
	@echo

ingest: ## Ingest documents into the RAG store (DATA=data)
	uv run python -m app.rag.ingest $(DATA)

clean: ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
