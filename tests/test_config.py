from smart_router.config import RouterConfig, _parse_models_config


class TestRouterConfig:
    def test_allowlist_mode(self):
        cfg = RouterConfig()
        cfg.filter_mode = "allowlist"
        cfg.allowed_models = {"model-a", "model-b"}

        assert cfg.is_model_enabled("model-a")
        assert cfg.is_model_enabled("model-b")
        assert not cfg.is_model_enabled("model-c")

    def test_blocklist_mode(self):
        cfg = RouterConfig()
        cfg.filter_mode = "blocklist"
        cfg.excluded_models = {"model-bad"}

        assert cfg.is_model_enabled("model-a")
        assert not cfg.is_model_enabled("model-bad")

    def test_default_allows_all(self):
        cfg = RouterConfig()
        assert cfg.is_model_enabled("anything")

    def test_tier_override(self):
        cfg = RouterConfig()
        cfg.tier_overrides = {"model-x": "LARGE"}

        assert cfg.get_tier_override("model-x") == "LARGE"
        assert cfg.get_tier_override("model-y") is None


class TestParseModelsConfig:
    def test_flat_list(self):
        cfg = RouterConfig()
        _parse_models_config(cfg, ["model-a", "model-b"])
        assert cfg.allowed_models == {"model-a", "model-b"}
        assert cfg.tier_overrides == {}

    def test_tier_grouped_dict(self):
        cfg = RouterConfig()
        _parse_models_config(cfg, {
            "small": ["model-s1", "model-s2"],
            "medium": ["model-m1"],
            "large": ["model-l1", "model-l2"],
        })
        assert cfg.allowed_models == {"model-s1", "model-s2", "model-m1", "model-l1", "model-l2"}
        assert cfg.tier_overrides["model-s1"] == "SMALL"
        assert cfg.tier_overrides["model-s2"] == "SMALL"
        assert cfg.tier_overrides["model-m1"] == "MEDIUM"
        assert cfg.tier_overrides["model-l1"] == "LARGE"

    def test_tier_keys_case_insensitive(self):
        cfg = RouterConfig()
        _parse_models_config(cfg, {"Small": ["model-a"], "LARGE": ["model-b"]})
        assert cfg.tier_overrides["model-a"] == "SMALL"
        assert cfg.tier_overrides["model-b"] == "LARGE"

    def test_empty_models(self):
        cfg = RouterConfig()
        _parse_models_config(cfg, [])
        assert cfg.allowed_models == set()

    def test_none_models(self):
        cfg = RouterConfig()
        _parse_models_config(cfg, None)
        assert cfg.allowed_models == set()
