# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Smart Router — an OpenAI-compatible proxy that sits in front of a LiteLLM backend and automatically routes requests to the best model based on query complexity. Uses a 3-tier system (SMALL/MEDIUM/LARGE) with hybrid complexity analysis (heuristics + LLM classifier fallback).

## Commands

```bash
# Install (dev)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run locally
uvicorn smart_router.main:app --reload

# Run tests
pytest tests/ -v

# Run single test
pytest tests/test_heuristics.py::TestHeuristicScoring::test_simple_question_scores_low -v

# Container (podman, rootless)
podman build -t llm-smart-router .
podman run -d --name smart-router -v ./router_config.yaml:/app/router_config.yaml:Z,ro -p 8000:8000 llm-smart-router
```

## Architecture

```
Client -> [Smart Router :8000] -> [LiteLLM backend] -> Model providers
```

All source code is in `src/smart_router/`:

- **config.py** — Central configuration loaded entirely from `router_config.yaml`. No .env or environment variables (except `CONFIG_PATH` to override the config file location). `RouterConfig` holds connection, server, routing, and model filtering settings.
- **models.py** — Discovers models from LiteLLM `/v1/models`, extracts parameter counts from model names (incl. MoE total params), assigns tiers, caches results. `ModelRegistry` handles tier lookup with fallback logic.
- **heuristics.py** — Rule-based complexity scoring (0.0-1.0) based on: token count, conversation depth, tool usage, system prompt complexity, code blocks, images, keyword analysis. Keywords are matched in both English and German. Returns a confidence flag.
- **classifier.py** — When heuristics are uncertain (score between 0.3-0.7), uses the smallest available model to classify complexity into a tier via structured JSON prompt.
- **router.py** — Orchestrates heuristics -> classifier -> model selection. Honors explicit model requests. Detects coding requests to prefer coder-specialized models.
- **proxy.py** — Forwards requests to LiteLLM with the selected model. Supports both streaming (SSE) and non-streaming responses.
- **main.py** — FastAPI app with endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, `POST /admin/reload`. Injects `X-Smart-Router-Model` and `X-Smart-Router-Tier` response headers.

## Key Design Decisions

- All configuration is centralized in `router_config.yaml` (connection, server, routing, model selection)
- Model tier assignment is based on **total parameters** for all models including MoE — e.g. `Qwen3-30B-A3B` (30B total) is Tier LARGE
- Tier boundaries: SMALL <=8B, MEDIUM <=27B, LARGE >27B (configurable in YAML)
- Models can be assigned to tiers manually via tier-grouped dict format in the YAML config, or automatically by parameter count
- Embedding/OCR/TTS models are automatically excluded from routing
- If no model exists in the target tier, fallback goes upward first, then downward
- When `prefer_coder=False`, non-coder models are preferred within the tier
- The classifier model is auto-selected as the smallest available model
- Config is re-read on every model refresh cycle (default 5 min) or immediately via `POST /admin/reload`
- Container volume-mounts the config: edit `router_config.yaml` on host, then `curl -X POST localhost:8000/admin/reload`
- The `router_config.yaml` is in `.gitignore` (contains credentials); `router_config.example.yaml` is committed
