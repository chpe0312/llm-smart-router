"""LLM-based complexity classifier for ambiguous requests.

Used when heuristics alone are not confident about the complexity tier.
"""

import json
import logging

import httpx

from .config import get_config
from .models import Tier, get_registry

logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """\
You are a request complexity classifier. Given a user's conversation with an AI assistant, \
classify the complexity of the LATEST user request into one of three tiers:

1 = SIMPLE: Short factual questions, translations, simple formatting, yes/no questions, \
basic lookups, trivial code fixes.
2 = MEDIUM: Standard coding tasks, summarization of longer texts, moderate analysis, \
explanations of concepts, typical chat interactions.
3 = COMPLEX: Multi-step reasoning, architecture design, complex debugging, in-depth analysis, \
creative writing with specific constraints, tasks requiring deep domain expertise.

Respond with ONLY a JSON object: {"tier": 1|2|3, "reason": "brief explanation"}"""


async def classify_complexity(messages: list[dict]) -> tuple[Tier, str]:
    """Use a small LLM to classify the complexity of a request.

    Returns the tier and a brief reason.
    """
    registry = get_registry()

    # Pick classifier model: configured, or smallest available
    classifier_model = get_config().classifier_model
    if not classifier_model:
        small_models = registry.by_tier(Tier.SMALL)
        if small_models:
            # Pick smallest by effective params
            classifier_model = min(
                small_models, key=lambda m: m.effective_params
            ).id
        else:
            # Fallback to any available model
            if registry.models:
                classifier_model = min(
                    registry.models.values(), key=lambda m: m.effective_params
                ).id
            else:
                logger.warning("No models available for classification, defaulting to MEDIUM")
                return Tier.MEDIUM, "no classifier model available"

    # Build a condensed version of the conversation for classification
    condensed = _condense_messages(messages)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{get_config().litellm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {get_config().litellm_api_key}"},
                json={
                    "model": classifier_model,
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                        {"role": "user", "content": condensed},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
            )
            resp.raise_for_status()
            result_text = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        # Handle cases where model wraps in markdown code block
        if result_text.startswith("```"):
            result_text = result_text.strip("`").removeprefix("json").strip()

        parsed = json.loads(result_text)
        tier_num = int(parsed["tier"])
        reason = parsed.get("reason", "")
        tier = Tier(max(1, min(3, tier_num)))

        logger.info(
            "Classifier (%s) result: tier=%s reason=%s",
            classifier_model, tier.name, reason,
        )
        return tier, reason

    except Exception:
        logger.exception("Classifier failed, defaulting to MEDIUM")
        return Tier.MEDIUM, "classifier error, defaulting to medium"


def _condense_messages(messages: list[dict], max_chars: int = 2000) -> str:
    """Create a condensed representation of the conversation for classification."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "image_url":
                    text_parts.append("[image]")
            content = " ".join(text_parts)
        parts.append(f"{role}: {content}")

    full_text = "\n".join(parts)
    if len(full_text) > max_chars:
        # Keep system prompt + last messages
        return full_text[:500] + "\n...\n" + full_text[-1500:]
    return full_text
