import re
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum

import httpx

from .config import get_config, load_config

logger = logging.getLogger(__name__)

EMBEDDING_PATTERNS = re.compile(r"embed|embedding", re.IGNORECASE)
EXCLUDE_PATTERNS = re.compile(r"ocr|whisper|tts|rerank", re.IGNORECASE)

# Matches parameter counts like "27b", "3.2-24B", "7b", "80b"
PARAM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|$)")
# Matches MoE active parameter patterns like "A3B" (3B active)
MOE_ACTIVE_PATTERN = re.compile(r"[Aa](\d+(?:\.\d+)?)[bB]")


class Tier(IntEnum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


@dataclass
class ModelInfo:
    id: str
    total_params: float | None = None  # billions
    active_params: float | None = None  # billions (for MoE)
    tier: Tier = Tier.MEDIUM
    is_coder: bool = False

    @property
    def effective_params(self) -> float:
        return self.total_params or self.active_params or 0.0


@dataclass
class ModelRegistry:
    models: dict[str, ModelInfo] = field(default_factory=dict)
    _last_refresh: float = 0.0

    def by_tier(self, tier: Tier) -> list[ModelInfo]:
        return [m for m in self.models.values() if m.tier == tier]

    def get_model_for_tier(self, tier: Tier, prefer_coder: bool = False) -> ModelInfo | None:
        candidates = self.by_tier(tier)
        if not candidates:
            # Fallback: try next higher tier
            for fallback_tier in Tier:
                if fallback_tier > tier:
                    candidates = self.by_tier(fallback_tier)
                    if candidates:
                        break
        if not candidates:
            # Last resort: try lower tiers
            for fallback_tier in reversed(list(Tier)):
                if fallback_tier < tier:
                    candidates = self.by_tier(fallback_tier)
                    if candidates:
                        break
        if not candidates:
            return None

        if prefer_coder:
            coders = [m for m in candidates if m.is_coder]
            if coders:
                return max(coders, key=lambda m: m.effective_params)
        else:
            # Prefer non-coder models for general requests
            general = [m for m in candidates if not m.is_coder]
            if general:
                return max(general, key=lambda m: m.effective_params)

            # No non-coder in this tier — try adjacent tiers for non-coder
            for fallback_tier in list(Tier):
                if fallback_tier != tier:
                    adj_general = [m for m in self.by_tier(fallback_tier) if not m.is_coder]
                    if adj_general:
                        return max(adj_general, key=lambda m: m.effective_params)

        # Last resort: pick the largest model in the tier regardless
        return max(candidates, key=lambda m: m.effective_params)


_registry = ModelRegistry()


def _extract_params(model_id: str) -> tuple[float | None, float | None]:
    """Extract total and active parameter counts from model name."""
    moe_match = MOE_ACTIVE_PATTERN.search(model_id)
    active_params = float(moe_match.group(1)) if moe_match else None

    all_params = PARAM_PATTERN.findall(model_id)
    if all_params:
        # Take the largest number as total params
        total_params = max(float(p) for p in all_params)
        # If MoE active params were found, don't confuse them with total
        if active_params and total_params == active_params and len(all_params) > 1:
            total_params = max(
                float(p) for p in all_params if float(p) != active_params
            )
    else:
        total_params = None

    return total_params, active_params


def _classify_tier(effective_params: float) -> Tier:
    if effective_params <= get_config().tier1_max_params:
        return Tier.SMALL
    elif effective_params <= get_config().tier2_max_params:
        return Tier.MEDIUM
    else:
        return Tier.LARGE


def _is_chat_model(model_id: str) -> bool:
    if EMBEDDING_PATTERNS.search(model_id):
        return False
    if EXCLUDE_PATTERNS.search(model_id):
        return False
    return True


def _build_model_info(model_id: str) -> ModelInfo | None:
    if not _is_chat_model(model_id):
        return None

    total_params, active_params = _extract_params(model_id)
    effective = total_params or active_params

    if effective is not None:
        tier = _classify_tier(effective)
    else:
        tier = Tier.MEDIUM  # Default for unknown models

    is_coder = bool(re.search(r"coder|code", model_id, re.IGNORECASE))

    return ModelInfo(
        id=model_id,
        total_params=total_params,
        active_params=active_params,
        tier=tier,
        is_coder=is_coder,
    )


async def refresh_models(force: bool = False) -> ModelRegistry:
    global _registry

    now = time.time()
    if not force and (now - _registry._last_refresh) < get_config().model_cache_ttl:
        return _registry

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{get_config().litellm_base_url}/models",
                headers={"Authorization": f"Bearer {get_config().litellm_api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Reload router_config.yaml on every model refresh
        router_cfg = load_config()

        new_models: dict[str, ModelInfo] = {}
        skipped: list[str] = []
        for entry in data.get("data", []):
            model_id = entry["id"]
            info = _build_model_info(model_id)
            if not info:
                continue

            # Apply router_config.yaml filter
            if not router_cfg.is_model_enabled(model_id):
                skipped.append(model_id)
                continue

            # Apply tier overrides from config
            tier_override = router_cfg.get_tier_override(model_id)
            if tier_override:
                try:
                    info.tier = Tier[tier_override]
                    logger.info("Model %s: tier overridden to %s", model_id, tier_override)
                except KeyError:
                    logger.warning("Invalid tier override '%s' for model %s", tier_override, model_id)

            new_models[model_id] = info
            logger.info(
                "Model %s: %.0f%sB params -> tier %s%s",
                model_id,
                info.effective_params,
                " active" if info.active_params else "",
                info.tier.name,
                " (coder)" if info.is_coder else "",
            )

        if skipped:
            logger.info("Skipped %d models (filtered by config): %s", len(skipped), ", ".join(skipped))

        _registry = ModelRegistry(models=new_models, _last_refresh=now)
        logger.info(
            "Active models: %d total (%d small, %d medium, %d large)",
            len(new_models),
            len(_registry.by_tier(Tier.SMALL)),
            len(_registry.by_tier(Tier.MEDIUM)),
            len(_registry.by_tier(Tier.LARGE)),
        )
    except Exception:
        logger.exception("Failed to refresh models from LiteLLM")
        if not _registry.models:
            raise

    return _registry


def get_registry() -> ModelRegistry:
    return _registry
