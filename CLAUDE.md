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
podman run -d --name smart-router --env-file .env -p 8000:8000 llm-smart-router

# Or with podman-compose
podman-compose up --build
```

## Architecture

```
Client → [Smart Router :8000] → [LiteLLM backend] → Model providers
```

All source code is in `src/smart_router/`:

- **config.py** — Pydantic settings from `.env` + `RouterConfig` loaded from `router_config.yaml` (model allowlist/blocklist, tier overrides)
- **models.py** — Discovers models from LiteLLM `/v1/models`, extracts parameter counts from model names (incl. MoE active params), assigns tiers, caches results. `ModelRegistry` handles tier lookup with fallback logic.
- **heuristics.py** — Rule-based complexity scoring (0.0–1.0) based on: token count, conversation depth, tool usage, system prompt complexity, code blocks, images, keyword analysis. Returns a confidence flag.
- **classifier.py** — When heuristics are uncertain (score between 0.3–0.7), uses the smallest available model to classify complexity into a tier via structured JSON prompt.
- **router.py** — Orchestrates heuristics → classifier → model selection. Honors explicit model requests. Detects coding requests to prefer coder-specialized models.
- **proxy.py** — Forwards requests to LiteLLM with the selected model. Supports both streaming (SSE) and non-streaming responses.
- **main.py** — FastAPI app with endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, `POST /admin/reload`. Injects `X-Smart-Router-Model` and `X-Smart-Router-Tier` response headers.

## Key Design Decisions

- Model tier assignment is based on **active parameters** (not total) for MoE models — e.g. `Qwen3-30B-A3B` (30B total, 3B active) is Tier SMALL
- Tier boundaries: SMALL ≤8B, MEDIUM ≤27B, LARGE >27B (configurable via env vars)
- Embedding/OCR/TTS models are automatically excluded from routing
- If no model exists in the target tier, fallback goes upward first, then downward
- The classifier model is auto-selected as the smallest available model
- `router_config.yaml` controls which models are active (allowlist/blocklist) and allows tier overrides
- Config is re-read on every model refresh cycle (5 min) or immediately via `POST /admin/reload`
- Container volume-mounts the config: edit `router_config.yaml` on host, then `curl -X POST localhost:8000/admin/reload`
