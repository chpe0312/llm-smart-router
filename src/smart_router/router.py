"""Core routing logic: determines which model to use for a request."""

import logging

from .config import get_config
from .heuristics import score_request
from .classifier import classify_complexity
from .models import Tier, ModelInfo, get_registry, refresh_models

logger = logging.getLogger(__name__)


def _score_to_tier(score: float) -> Tier:
    if score <= get_config().heuristic_low_threshold:
        return Tier.SMALL
    elif score >= get_config().heuristic_high_threshold:
        return Tier.LARGE
    else:
        return Tier.MEDIUM


async def route_request(
    messages: list[dict],
    tools: list[dict] | None = None,
    requested_model: str | None = None,
) -> tuple[ModelInfo, dict]:
    """Determine the best model for a request.

    Returns the selected ModelInfo and a metadata dict with routing details.
    """
    # Ensure models are loaded
    registry = await refresh_models()

    # If client explicitly requests a specific model that exists, honor it
    if requested_model and requested_model in registry.models:
        model = registry.models[requested_model]
        return model, {
            "routing": "explicit",
            "requested_model": requested_model,
        }

    # Step 1: Heuristic scoring
    heuristic = score_request(messages, tools)
    logger.info(
        "Heuristic score=%.3f confident=%s reasons=%s",
        heuristic.score, heuristic.confident, heuristic.reasons,
    )

    # Step 2: Determine tier
    if heuristic.confident:
        tier = _score_to_tier(heuristic.score)
        routing_method = "heuristic"
        classifier_reason = ""
    else:
        # Heuristics uncertain -> ask classifier
        tier, classifier_reason = await classify_complexity(messages)
        routing_method = "classifier"

    # Step 3: Check if this looks like a coding task
    prefer_coder = _is_coding_request(messages)

    # Step 4: Select model from tier
    model = registry.get_model_for_tier(tier, prefer_coder=prefer_coder)
    if model is None:
        raise RuntimeError("No models available for routing")

    metadata = {
        "routing": routing_method,
        "heuristic_score": heuristic.score,
        "heuristic_reasons": heuristic.reasons,
        "tier": tier.name,
        "selected_model": model.id,
        "prefer_coder": prefer_coder,
    }
    if classifier_reason:
        metadata["classifier_reason"] = classifier_reason

    logger.info("Routed to %s (tier=%s, method=%s)", model.id, tier.name, routing_method)
    return model, metadata


def _is_coding_request(messages: list[dict]) -> bool:
    """Check if the request is likely code-related."""
    import re

    code_indicators = re.compile(
        r"\b(code|function|class|implement|bug|error|exception|stacktrace|"
        r"api|endpoint|database|query|sql|html|css|javascript|python|"
        r"typescript|rust|golang|java|refactor|test|unittest|"
        # German
        r"implementiere|debugge|Quellcode|Quelltext|Programmier|kompilier|"
        r"Algorithmus|Algorithmen|Skript)\b",
        re.IGNORECASE,
    )
    code_fence = re.compile(r"```")

    # Check last 3 user messages
    user_msgs = [m for m in messages if m.get("role") == "user"][-3:]
    for msg in user_msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            if code_indicators.search(content):
                return True
            if code_fence.search(content):
                return True
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if code_indicators.search(text):
                        return True
                    if code_fence.search(text):
                        return True
    return False
