# LLM Smart Router

An intelligent, OpenAI-compatible proxy that sits in front of a [LiteLLM](https://github.com/BerriAI/litellm) backend and automatically routes requests to the optimal model based on query complexity.

## How It Works

```
Client / Open WebUI
        │
        ▼
┌─────────────────────────────────┐
│       Smart Router (:8000)      │
│                                 │
│  1. Receive request             │
│  2. Score complexity            │
│     ├─ Heuristics (fast)        │
│     └─ LLM Classifier (fallback)│
│  3. Select tier: S / M / L      │
│  4. Pick best model in tier     │
│  5. Forward to LiteLLM          │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│       LiteLLM Backend           │
│  (manages model providers)      │
└─────────────────────────────────┘
```

The router analyzes each incoming request and assigns it to one of three complexity tiers:

| Tier | Parameter Range | Use Case | Example Models |
|------|----------------|----------|----------------|
| **SMALL** | ≤ 8B (active) | Simple questions, translations, formatting | qwen-3-4b, gemma-3-4b |
| **MEDIUM** | ≤ 27B | Standard coding, analysis, summarization | gemma-3-27b, Mistral-Small-3.2-24B |
| **LARGE** | > 27B | Complex reasoning, architecture design, multi-step tasks | qwen2.5-coder-32b, Qwen3-Coder-30B |

## Features

- **OpenAI-compatible API** — Drop-in replacement, clients just change the URL
- **Hybrid complexity analysis** — Fast rule-based heuristics with LLM classifier fallback for uncertain cases
- **Automatic model discovery** — Queries LiteLLM `/v1/models` and categorizes models by parameter count
- **MoE-aware** — Uses active parameters (not total) for Mixture-of-Experts models (e.g. `Qwen3-30B-A3B` → 3B active → Tier SMALL)
- **Coder model preference** — Detects code-related requests and prefers specialized coder models
- **Streaming support** — Passes through SSE streaming responses
- **Configurable model selection** — YAML config for allowlist/blocklist and tier overrides
- **Hot reload** — Change config without restarting via `POST /admin/reload`
- **Open WebUI integration** — Appears as a single virtual model in the UI

## Quick Start

### With Podman/Docker

```bash
# Clone and configure
git clone <repo-url> && cd llm-smart-router
cp .env.example .env
# Edit .env with your LiteLLM URL and API key

# Start
podman build -t llm-smart-router .
podman run -d --name smart-router \
  --env-file .env \
  -v ./router_config.yaml:/app/router_config.yaml:Z,ro \
  -p 8000:8000 \
  llm-smart-router
```

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn smart_router.main:app --reload
```

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000/v1` | LiteLLM API base URL |
| `LITELLM_API_KEY` | — | API key for LiteLLM |
| `ROUTER_PORT` | `8000` | Port the router listens on |
| `LOG_LEVEL` | `info` | Logging level (debug, info, warning, error) |
| `TIER1_MAX_PARAMS` | `8.0` | Max active params (B) for SMALL tier |
| `TIER2_MAX_PARAMS` | `27.0` | Max active params (B) for MEDIUM tier |
| `HEURISTIC_LOW_THRESHOLD` | `0.3` | Score below this → SMALL tier |
| `HEURISTIC_HIGH_THRESHOLD` | `0.7` | Score above this → LARGE tier |
| `CLASSIFIER_MODEL` | (auto) | Model for classification; auto-selects smallest if empty |
| `MODEL_CACHE_TTL` | `300` | Seconds between model list refreshes |

### Model Configuration (`router_config.yaml`)

Controls which models are used and how they're categorized:

```yaml
# Name shown in Open WebUI and other clients
model_name: "smart-router"

# "allowlist" = only listed models; "blocklist" = all except listed
filter_mode: allowlist

# Models to include (allowlist mode)
models:
  - qwen-3-4b
  - gemma-3-4b
  - qwen2.5-coder-7b
  - gemma-3-27b
  - Mistral-Small-3.2-24B
  - qwen2.5-coder-32b
  - Qwen3-Coder-30B

# Models to exclude (blocklist mode)
# excluded:
#   - granite-vision-3.3-2b

# Override automatic tier assignments
# tier_overrides:
#   gpt-oss20b: large
```

Changes are picked up automatically every 5 minutes, or immediately via:

```bash
curl -X POST http://localhost:8000/admin/reload
```

## API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint. The router selects the model automatically.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

Response headers include routing information:
- `X-Smart-Router-Model` — The actual model used
- `X-Smart-Router-Tier` — The tier (SMALL, MEDIUM, LARGE)

The response body also includes a `_routing` field with detailed metadata:

```json
{
  "_routing": {
    "routing": "heuristic",
    "heuristic_score": 0.0,
    "heuristic_reasons": ["very short (3 est. tokens)", "simple keywords: What is"],
    "tier": "SMALL",
    "selected_model": "qwen-3-4b",
    "prefer_coder": false
  }
}
```

### `GET /v1/models`

Returns the configured virtual model name (for Open WebUI integration).

### `GET /health`

Health check endpoint.

### `POST /admin/reload`

Hot-reloads `router_config.yaml` and refreshes the model list from LiteLLM. Returns the current model-to-tier mapping.

## Complexity Scoring

The heuristic scorer evaluates requests on multiple dimensions:

| Signal | Impact | Example |
|--------|--------|---------|
| **Token count** | 0.0 – 0.5 | Short questions score low, long conversations score high |
| **Conversation depth** | 0.0 – 0.15 | 10+ turns adds significant complexity |
| **Tool/function calls** | 0.1 – 0.2 | More tools = more complex orchestration |
| **System prompt length** | 0.0 – 0.15 | Long system prompts suggest complex tasks |
| **Code blocks** | 0.05 – 0.15 | Multiple code blocks indicate involved tasks |
| **Image content** | 0.1 | Multimodal requests need capable models |
| **Complex keywords** | 0.3 – 0.6 | "analyze", "step-by-step", "trade-offs", "implement" |
| **Simple keywords** | -0.15 | "translate", "what is", "yes or no" |

When the heuristic score falls in the uncertain range (0.3–0.7), the router automatically queries the smallest available model to classify the request's complexity.

## Open WebUI Integration

The router is designed to work seamlessly with [Open WebUI](https://github.com/open-webui/open-webui):

```bash
# Create a shared network
podman network create smartrouter-net

# Start Smart Router
podman run -d --name smart-router \
  --network smartrouter-net \
  --env-file .env \
  -v ./router_config.yaml:/app/router_config.yaml:Z,ro \
  -p 8000:8000 \
  llm-smart-router

# Start Open WebUI
podman run -d --name open-webui \
  --network smartrouter-net \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://smart-router:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -e WEBUI_AUTH=false \
  -v open-webui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Open WebUI will show a single model called "smart-router" (configurable via `model_name` in `router_config.yaml`). Select it and every request is automatically routed to the best backend model.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_heuristics.py -v

# Run a single test
pytest tests/test_router.py::TestRouteRequest::test_simple_request_routes_to_small -v
```

## Project Structure

```
src/smart_router/
├── config.py        # Settings (.env) + RouterConfig (router_config.yaml)
├── models.py        # Model discovery, parameter extraction, tier assignment
├── heuristics.py    # Rule-based complexity scoring
├── classifier.py    # LLM-based classification fallback
├── router.py        # Orchestrates heuristics → classifier → model selection
├── proxy.py         # Forwards requests to LiteLLM (sync + streaming)
└── main.py          # FastAPI application and endpoints
```

## License

MIT
