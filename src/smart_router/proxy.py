"""Proxies requests to LiteLLM backend, with streaming support."""

import logging
from collections.abc import AsyncIterator

import httpx

from .config import get_config

logger = logging.getLogger(__name__)


async def proxy_chat_completion(
    body: dict,
    model_id: str,
) -> dict:
    """Forward a non-streaming chat completion request to LiteLLM."""
    body = {**body, "model": model_id, "stream": False}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{get_config().litellm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {get_config().litellm_api_key}"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def proxy_chat_completion_stream(
    body: dict,
    model_id: str,
) -> AsyncIterator[bytes]:
    """Forward a streaming chat completion request to LiteLLM."""
    body = {**body, "model": model_id, "stream": True}

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{get_config().litellm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {get_config().litellm_api_key}"},
            json=body,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield (line + "\n").encode()
