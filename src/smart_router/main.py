"""FastAPI application — OpenAI-compatible smart router."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_config, get_config
from .models import refresh_models, get_registry, Tier
from .router import route_request
from .proxy import proxy_chat_completion, proxy_chat_completion_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Smart Router on port %d", cfg.router_port)
    logger.info("LiteLLM backend: %s", cfg.litellm_base_url)
    await refresh_models(force=True)
    registry = get_registry()
    logger.info("Loaded %d models", len(registry.models))
    yield


app = FastAPI(title="LLM Smart Router", lifespan=lifespan)
logger = logging.getLogger(__name__)


@app.get("/health")
async def health():
    registry = get_registry()
    return {
        "status": "ok",
        "models_loaded": len(registry.models),
    }


@app.get("/v1/models")
async def list_models():
    await refresh_models()
    cfg = get_config()
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "smart-router",
            }
        ],
    }


@app.post("/admin/reload")
async def reload_config():
    """Reload router_config.yaml and refresh model list immediately."""
    load_config()
    registry = await refresh_models(force=True)
    models_by_tier = {
        tier.name: [m.id for m in registry.by_tier(tier)]
        for tier in Tier
    }
    cfg = get_config()
    return {
        "status": "reloaded",
        "model_name": cfg.model_name,
        "active_models": len(registry.models),
        "models_by_tier": models_by_tier,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    requested_model = body.get("model")
    stream = body.get("stream", False)

    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages is required", "type": "invalid_request_error"}},
        )

    try:
        model, routing_meta = await route_request(messages, tools, requested_model)
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(e), "type": "server_error"}},
        )

    logger.info(
        "Request routed: model=%s tier=%s method=%s score=%.3f",
        model.id,
        routing_meta.get("tier", "?"),
        routing_meta.get("routing", "?"),
        routing_meta.get("heuristic_score", 0),
    )

    if stream:
        return StreamingResponse(
            proxy_chat_completion_stream(body, model.id),
            media_type="text/event-stream",
            headers={
                "X-Smart-Router-Model": model.id,
                "X-Smart-Router-Tier": routing_meta.get("tier", ""),
            },
        )

    result = await proxy_chat_completion(body, model.id)
    result["_routing"] = routing_meta
    return JSONResponse(
        content=result,
        headers={
            "X-Smart-Router-Model": model.id,
            "X-Smart-Router-Tier": routing_meta.get("tier", ""),
        },
    )
