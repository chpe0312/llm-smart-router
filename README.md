# LLM Smart Router

An intelligent, OpenAI-compatible proxy that sits in front of a [LiteLLM](https://github.com/BerriAI/litellm) backend and automatically routes requests to the optimal model based on query complexity.

## How It Works

```
Client / Open WebUI
        |
        v
+----------------------------------+
|       Smart Router (:8000)       |
|                                  |
|  1. Receive request              |
|  2. Score complexity             |
|     +-- Heuristics (fast)        |
|     +-- LLM Classifier (fallback)|
|  3. Select tier: S / M / L       |
|  4. Pick best model in tier      |
|  5. Forward to LiteLLM           |
+----------------+-----------------+
                 |
                 v
+----------------------------------+
|       LiteLLM Backend            |
|  (manages model providers)       |
+----------------------------------+
```

The router analyzes each incoming request and assigns it to one of three complexity tiers:

| Tier | Parameter Range | Use Case | Example Models |
|------|----------------|----------|----------------|
| **SMALL** | <= 8B | Simple questions, translations, formatting | qwen-3-4b, gemma-3-4b |
| **MEDIUM** | <= 27B | Standard coding, analysis, summarization | gemma-3-27b, Mistral-Small-3.2-24B |
| **LARGE** | > 27B | Complex reasoning, architecture design, multi-step tasks | qwen2.5-coder-32b, Qwen3-Coder-30B |

## Features

- **OpenAI-compatible API** — Drop-in replacement, clients just change the URL
- **Hybrid complexity analysis** — Fast rule-based heuristics with LLM classifier fallback for uncertain cases
- **Multilingual keyword detection** — Heuristic keywords work in English and German
- **Automatic model discovery** — Queries LiteLLM `/v1/models` and categorizes models by parameter count
- **MoE-aware** — Extracts both total and active parameters from MoE model names, uses total parameters for tier assignment (e.g. `Qwen3-30B-A3B` -> 30B total -> Tier LARGE)
- **Coder model preference** — Detects code-related requests and prefers specialized coder models
- **Streaming support** — Passes through SSE streaming responses
- **Centralized YAML config** — All settings (connection, routing, model selection) in a single `router_config.yaml`
- **Hot reload** — Change config without restarting via `POST /admin/reload`
- **Open WebUI integration** — Appears as a single virtual model in the UI

## Quick Start

### With Podman/Docker

```bash
# Clone and configure
git clone <repo-url> && cd llm-smart-router
cp router_config.example.yaml router_config.yaml
# Edit router_config.yaml with your LiteLLM URL and API key

# Build and start
podman build -t llm-smart-router .
podman run -d --name smart-router \
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

All settings are centralized in a single file: `router_config.yaml`. Copy the example to get started:

```bash
cp router_config.example.yaml router_config.yaml
```

The config file has four sections:

### Connection

```yaml
connection:
  litellm_base_url: "http://localhost:4000/v1"
  litellm_api_key: "sk-your-key-here"
```

### Server

```yaml
server:
  port: 8000
  log_level: "info"               # debug, info, warning, error
  model_name: "smart-router"      # Name shown in Open WebUI and /v1/models
```

### Routing

```yaml
routing:
  tier_boundaries:
    small_max: 8.0                # <= 8B  -> SMALL
    medium_max: 27.0              # <= 27B -> MEDIUM
                                  # > 27B  -> LARGE
  heuristic_low_threshold: 0.3    # Score <= this -> SMALL (confident)
  heuristic_high_threshold: 0.7   # Score >= this -> LARGE (confident)
  classifier_model: ""            # Empty = auto-select smallest model
  model_cache_ttl: 300            # Seconds between model list refreshes
```

### Model Selection

```yaml
# "allowlist" = only listed models; "blocklist" = all except listed
filter_mode: allowlist

# Option 1: Manually grouped by tier (recommended)
models:
  small:
    - qwen-3-4b
    - gemma-3-4b
  medium:
    - gemma-3-27b
    - Mistral-Small-3.2-24B
  large:
    - qwen2.5-coder-32b
    - Qwen3-Coder-30B

# Option 2: Flat list (tier determined automatically by parameter count)
# models:
#   - qwen-3-4b
#   - gemma-3-27b
#   - qwen2.5-coder-32b

# Models to exclude (only in blocklist mode)
# excluded:
#   - granite-vision-3.3-2b
```

Changes are picked up automatically every 5 minutes, or immediately via:

```bash
curl -X POST http://localhost:8000/admin/reload
```

The config file path can be overridden with the `CONFIG_PATH` environment variable.

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
| **Token count** | 0.0 - 0.5 | Short questions score low, long conversations score high |
| **Conversation depth** | 0.0 - 0.15 | 10+ turns adds significant complexity |
| **Tool/function calls** | 0.1 - 0.2 | More tools = more complex orchestration |
| **System prompt length** | 0.0 - 0.15 | Long system prompts suggest complex tasks |
| **Code blocks** | 0.05 - 0.15 | Multiple code blocks indicate involved tasks |
| **Image content** | 0.1 | Multimodal requests need capable models |
| **Complex keywords** | 0.3 - 0.6 | "analyze", "step-by-step", "trade-offs", "analysiere", "Schritt fur Schritt" |
| **Simple keywords** | -0.15 | "translate", "what is", "ubersetze", "was ist" |

Keywords are matched in both English and German.

When the heuristic score falls in the uncertain range (0.3-0.7), the router automatically queries the smallest available model to classify the request's complexity.

## Open WebUI Integration

The router is designed to work seamlessly with [Open WebUI](https://github.com/open-webui/open-webui):

```bash
# Create a shared network
podman network create smartrouter-net

# Start Smart Router
podman run -d --name smart-router \
  --network smartrouter-net \
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
+-- config.py        # Central YAML config loader (router_config.yaml)
+-- models.py        # Model discovery, parameter extraction, tier assignment
+-- heuristics.py    # Rule-based complexity scoring (EN + DE keywords)
+-- classifier.py    # LLM-based classification fallback
+-- router.py        # Orchestrates heuristics -> classifier -> model selection
+-- proxy.py         # Forwards requests to LiteLLM (sync + streaming)
+-- main.py          # FastAPI application and endpoints
```

## License

MIT
