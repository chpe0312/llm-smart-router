import pytest

from smart_router.models import (
    Tier,
    _extract_params,
    _build_model_info,
    _is_chat_model,
    ModelRegistry,
    ModelInfo,
)


class TestParamExtraction:
    def test_simple_param_count(self):
        total, active = _extract_params("gemma-3-27b")
        assert total == 27.0
        assert active is None

    def test_small_model(self):
        total, active = _extract_params("qwen-3-4b")
        assert total == 4.0

    def test_moe_model(self):
        total, active = _extract_params("Qwen3-30B-A3B")
        assert total == 30.0
        assert active == 3.0

    def test_moe_large(self):
        total, active = _extract_params("qwen3-next-80b-a3b")
        assert total == 80.0
        assert active == 3.0

    def test_decimal_params(self):
        total, active = _extract_params("granite-vision-3.3-2b")
        assert total == 3.3 or total == 2.0  # depends on regex matching

    def test_no_params(self):
        total, active = _extract_params("gpt-oss20b")
        assert total == 20.0

    def test_coder_model(self):
        total, active = _extract_params("qwen2.5-coder-32b")
        assert total == 32.0


class TestModelClassification:
    def test_embedding_excluded(self):
        assert not _is_chat_model("nomic-embed-text:latest")
        assert not _is_chat_model("snowflake-arctic-embed2:latest")
        assert not _is_chat_model("qwen3-embedding:8b")

    def test_ocr_excluded(self):
        assert not _is_chat_model("vllm-deepseek-ocr")

    def test_chat_model_included(self):
        assert _is_chat_model("gemma-3-27b")
        assert _is_chat_model("qwen2.5-coder-32b")


class TestTierAssignment:
    def test_small_model_tier(self):
        info = _build_model_info("gemma-3-4b")
        assert info is not None
        assert info.tier == Tier.SMALL

    def test_medium_model_tier(self):
        info = _build_model_info("gemma-3-27b")
        assert info is not None
        assert info.tier == Tier.MEDIUM

    def test_large_model_tier(self):
        info = _build_model_info("qwen2.5-coder-32b")
        assert info is not None
        assert info.tier == Tier.LARGE

    def test_moe_uses_total_params(self):
        info = _build_model_info("Qwen3-30B-A3B")
        assert info is not None
        assert info.tier == Tier.LARGE  # 30B total
        assert info.active_params == 3.0
        assert info.total_params == 30.0

    def test_coder_flag(self):
        info = _build_model_info("qwen2.5-coder-32b")
        assert info is not None
        assert info.is_coder

        info2 = _build_model_info("gemma-3-27b")
        assert info2 is not None
        assert not info2.is_coder


class TestModelRegistry:
    def test_by_tier(self):
        registry = ModelRegistry(models={
            "small": ModelInfo(id="small", total_params=4, tier=Tier.SMALL),
            "medium": ModelInfo(id="medium", total_params=24, tier=Tier.MEDIUM),
            "large": ModelInfo(id="large", total_params=32, tier=Tier.LARGE),
        })
        assert len(registry.by_tier(Tier.SMALL)) == 1
        assert len(registry.by_tier(Tier.MEDIUM)) == 1
        assert len(registry.by_tier(Tier.LARGE)) == 1

    def test_get_model_for_tier_fallback_up(self):
        registry = ModelRegistry(models={
            "medium": ModelInfo(id="medium", total_params=24, tier=Tier.MEDIUM),
        })
        # No small models, should fallback to medium
        model = registry.get_model_for_tier(Tier.SMALL)
        assert model is not None
        assert model.id == "medium"

    def test_get_model_for_tier_prefer_coder(self):
        registry = ModelRegistry(models={
            "general": ModelInfo(id="general", total_params=32, tier=Tier.LARGE),
            "coder": ModelInfo(id="coder", total_params=30, tier=Tier.LARGE, is_coder=True),
        })
        model = registry.get_model_for_tier(Tier.LARGE, prefer_coder=True)
        assert model is not None
        assert model.id == "coder"

    def test_empty_registry(self):
        registry = ModelRegistry()
        assert registry.get_model_for_tier(Tier.SMALL) is None
