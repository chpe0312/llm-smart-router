import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

VALID_TIERS = {"small", "medium", "large"}

# Config file path: override via env var CONFIG_PATH, default router_config.yaml
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "router_config.yaml"))


class RouterConfig:
    """Central configuration — loaded entirely from router_config.yaml."""

    def __init__(self):
        # Connection
        self.litellm_base_url: str = "http://localhost:4000/v1"
        self.litellm_api_key: str = ""

        # Server
        self.router_port: int = 8000
        self.log_level: str = "info"
        self.model_name: str = "smart-router"

        # Tier boundaries (billions of parameters)
        self.tier1_max_params: float = 8.0
        self.tier2_max_params: float = 27.0

        # Heuristic scoring thresholds
        self.heuristic_low_threshold: float = 0.3
        self.heuristic_high_threshold: float = 0.7

        # Classifier model (auto-selects smallest if empty)
        self.classifier_model: str = ""

        # How often to re-fetch models from LiteLLM (seconds)
        self.model_cache_ttl: int = 300

        # Model filtering
        self.filter_mode: str = "blocklist"
        self.allowed_models: set[str] = set()
        self.excluded_models: set[str] = set()
        self.tier_overrides: dict[str, str] = {}

    def is_model_enabled(self, model_id: str) -> bool:
        if self.filter_mode == "allowlist":
            return model_id in self.allowed_models
        else:
            return model_id not in self.excluded_models

    def get_tier_override(self, model_id: str) -> str | None:
        return self.tier_overrides.get(model_id)


_config = RouterConfig()


def _parse_models_config(cfg: RouterConfig, models_raw) -> None:
    """Parse the models section which can be a flat list or a tier-grouped dict."""
    if isinstance(models_raw, list):
        cfg.allowed_models = set(models_raw)
    elif isinstance(models_raw, dict):
        for key, value in models_raw.items():
            if key.lower() in VALID_TIERS and isinstance(value, list):
                tier_name = key.upper()
                for model_id in value:
                    cfg.allowed_models.add(model_id)
                    cfg.tier_overrides[model_id] = tier_name
            elif isinstance(key, str):
                cfg.allowed_models.add(key)
    else:
        cfg.allowed_models = set()


def load_config() -> RouterConfig:
    global _config

    if not CONFIG_PATH.exists():
        logger.warning("Config file %s not found, using defaults", CONFIG_PATH)
        _config = RouterConfig()
        return _config

    try:
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        cfg = RouterConfig()

        # Connection
        conn = data.get("connection", {})
        cfg.litellm_base_url = conn.get("litellm_base_url", cfg.litellm_base_url)
        cfg.litellm_api_key = conn.get("litellm_api_key", cfg.litellm_api_key)

        # Server
        server = data.get("server", {})
        cfg.router_port = server.get("port", cfg.router_port)
        cfg.log_level = server.get("log_level", cfg.log_level)
        cfg.model_name = server.get("model_name", cfg.model_name)

        # Routing
        routing = data.get("routing", {})
        cfg.heuristic_low_threshold = routing.get("heuristic_low_threshold", cfg.heuristic_low_threshold)
        cfg.heuristic_high_threshold = routing.get("heuristic_high_threshold", cfg.heuristic_high_threshold)
        cfg.classifier_model = routing.get("classifier_model", cfg.classifier_model)
        cfg.model_cache_ttl = routing.get("model_cache_ttl", cfg.model_cache_ttl)

        # Tier boundaries
        tiers = routing.get("tier_boundaries", {})
        cfg.tier1_max_params = tiers.get("small_max", cfg.tier1_max_params)
        cfg.tier2_max_params = tiers.get("medium_max", cfg.tier2_max_params)

        # Models
        cfg.filter_mode = data.get("filter_mode", cfg.filter_mode)
        cfg.excluded_models = set(data.get("excluded", []))
        _parse_models_config(cfg, data.get("models", []))

        _config = cfg

        logger.info("Config loaded from %s", CONFIG_PATH)
        logger.info("LiteLLM backend: %s", cfg.litellm_base_url)
        logger.info("Model name: %s", cfg.model_name)
        if cfg.filter_mode == "allowlist":
            logger.info("Allowlist mode: %d models", len(cfg.allowed_models))
        else:
            logger.info("Blocklist mode: %d excluded", len(cfg.excluded_models))

    except Exception:
        logger.exception("Failed to load %s, using defaults", CONFIG_PATH)
        _config = RouterConfig()

    return _config


def get_config() -> RouterConfig:
    return _config
