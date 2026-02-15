import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from smart_router.main import app
from smart_router.config import RouterConfig
from smart_router.models import ModelInfo, ModelRegistry, Tier


@pytest.fixture
def mock_registry():
    return ModelRegistry(models={
        "test-small": ModelInfo(id="test-small", total_params=4, tier=Tier.SMALL),
        "test-large": ModelInfo(id="test-large", total_params=32, tier=Tier.LARGE),
    })


@pytest.fixture
def mock_completion_response():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, mock_registry):
        with patch("smart_router.main.get_registry", return_value=mock_registry):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert data["models_loaded"] == 2


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_list_models_returns_virtual_name(self, mock_registry):
        cfg = RouterConfig()
        cfg.model_name = "my-smart-router"
        with patch("smart_router.main.refresh_models", new_callable=AsyncMock, return_value=mock_registry), \
             patch("smart_router.main.get_config", return_value=cfg):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                data = resp.json()
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["id"] == "my-smart-router"


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, mock_registry):
        with patch("smart_router.main.get_registry", return_value=mock_registry):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={"messages": []})
                assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_registry, mock_completion_response):
        with patch("smart_router.main.route_request", new_callable=AsyncMock) as mock_route, \
             patch("smart_router.main.proxy_chat_completion", new_callable=AsyncMock, return_value=mock_completion_response):
            mock_route.return_value = (
                ModelInfo(id="test-small", total_params=4, tier=Tier.SMALL),
                {"routing": "heuristic", "tier": "SMALL", "heuristic_score": 0.1},
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hi"}]},
                )
                assert resp.status_code == 200
                assert resp.headers["x-smart-router-model"] == "test-small"
                data = resp.json()
                assert data["choices"][0]["message"]["content"] == "Hello!"
