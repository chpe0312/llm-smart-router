import pytest
from unittest.mock import patch, AsyncMock

from smart_router.models import Tier, ModelInfo, ModelRegistry
from smart_router.router import route_request, _score_to_tier, _is_coding_request


class TestScoreToTier:
    def test_low_score(self):
        assert _score_to_tier(0.1) == Tier.SMALL
        assert _score_to_tier(0.0) == Tier.SMALL

    def test_mid_score(self):
        assert _score_to_tier(0.5) == Tier.MEDIUM

    def test_high_score(self):
        assert _score_to_tier(0.8) == Tier.LARGE
        assert _score_to_tier(1.0) == Tier.LARGE


class TestIsCodingRequest:
    def test_coding_keywords(self):
        messages = [{"role": "user", "content": "Write a Python function to sort a list"}]
        assert _is_coding_request(messages)

    def test_non_coding(self):
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        assert not _is_coding_request(messages)

    def test_bug_keyword(self):
        messages = [{"role": "user", "content": "I have a bug in my code"}]
        assert _is_coding_request(messages)

    def test_german_coding_keywords(self):
        messages = [{"role": "user", "content": "Implementiere einen Algorithmus für Sortierung"}]
        assert _is_coding_request(messages)

    def test_german_quellcode(self):
        messages = [{"role": "user", "content": "Schau dir den Quellcode an"}]
        assert _is_coding_request(messages)

    def test_code_fence_detection(self):
        messages = [{"role": "user", "content": "Fix this:\n```\nfoo()\n```"}]
        assert _is_coding_request(messages)

    def test_german_non_coding(self):
        messages = [{"role": "user", "content": "analysiere e autos vs verbrenner"}]
        assert not _is_coding_request(messages)


class TestRouteRequest:
    @pytest.fixture
    def mock_registry(self):
        return ModelRegistry(models={
            "small-model": ModelInfo(id="small-model", total_params=4, tier=Tier.SMALL),
            "medium-model": ModelInfo(id="medium-model", total_params=24, tier=Tier.MEDIUM),
            "large-model": ModelInfo(id="large-model", total_params=32, tier=Tier.LARGE),
            "coder-model": ModelInfo(id="coder-model", total_params=32, tier=Tier.LARGE, is_coder=True),
        })

    @pytest.mark.asyncio
    async def test_simple_request_routes_to_small(self, simple_messages, mock_registry):
        with patch("smart_router.router.refresh_models", new_callable=AsyncMock, return_value=mock_registry), \
             patch("smart_router.router.get_registry", return_value=mock_registry):
            model, meta = await route_request(simple_messages)
            assert model.tier == Tier.SMALL

    @pytest.mark.asyncio
    async def test_explicit_model_honored(self, simple_messages, mock_registry):
        with patch("smart_router.router.refresh_models", new_callable=AsyncMock, return_value=mock_registry), \
             patch("smart_router.router.get_registry", return_value=mock_registry):
            model, meta = await route_request(
                simple_messages, requested_model="large-model"
            )
            assert model.id == "large-model"
            assert meta["routing"] == "explicit"

    @pytest.mark.asyncio
    async def test_complex_request_routes_to_large(self, complex_messages, mock_registry):
        with patch("smart_router.router.refresh_models", new_callable=AsyncMock, return_value=mock_registry), \
             patch("smart_router.router.get_registry", return_value=mock_registry):
            model, meta = await route_request(complex_messages)
            assert model.tier in (Tier.MEDIUM, Tier.LARGE)

    @pytest.mark.asyncio
    async def test_coding_request_prefers_coder(self, coding_messages, mock_registry):
        with patch("smart_router.router.refresh_models", new_callable=AsyncMock, return_value=mock_registry), \
             patch("smart_router.router.get_registry", return_value=mock_registry):
            model, meta = await route_request(coding_messages)
            assert meta["prefer_coder"] is True
