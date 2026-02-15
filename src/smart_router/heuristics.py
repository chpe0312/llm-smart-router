"""Rule-based complexity scoring for incoming requests.

Returns a score between 0.0 (trivial) and 1.0 (very complex).
"""

import re
from dataclasses import dataclass


@dataclass
class HeuristicResult:
    score: float  # 0.0 - 1.0
    reasons: list[str]
    confident: bool  # True if heuristics alone can decide the tier


# Patterns that suggest higher complexity
COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"analy[sz]e|compare|contrast|explain\s+in\s+detail|step[- ]by[- ]step|"
    r"implement|architect|design|refactor|optimize|debug|"
    r"write\s+(a\s+)?(complete|full|entire)|"
    r"multi[- ]step|comprehensive|thorough|in[- ]depth|"
    r"trade[- ]?offs?|pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?"
    r")\b",
    re.IGNORECASE,
)

# Patterns that suggest simple tasks
SIMPLE_KEYWORDS = re.compile(
    r"\b("
    r"translate|summarize|summarise|tldr|tl;dr|"
    r"yes\s+or\s+no|true\s+or\s+false|"
    r"what\s+is|who\s+is|when\s+did|where\s+is|"
    r"define|list|name|count|"
    r"fix\s+(this|the)\s+(typo|spelling|grammar)|"
    r"convert|format|reformat"
    r")\b",
    re.IGNORECASE,
)

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def _extract_text(messages: list[dict]) -> str:
    """Extract all text content from messages."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        parts.append("[IMAGE]")
    return "\n".join(parts)


def score_request(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> HeuristicResult:
    reasons: list[str] = []
    score = 0.0
    full_text = _extract_text(messages)
    total_tokens = _estimate_tokens(full_text)
    num_turns = len(messages)

    # --- Token count scoring ---
    if total_tokens < 50:
        score += 0.0
        reasons.append(f"very short ({total_tokens} est. tokens)")
    elif total_tokens < 200:
        score += 0.1
    elif total_tokens < 800:
        score += 0.25
        reasons.append(f"medium length ({total_tokens} est. tokens)")
    elif total_tokens < 2000:
        score += 0.4
        reasons.append(f"long ({total_tokens} est. tokens)")
    else:
        score += 0.5
        reasons.append(f"very long ({total_tokens} est. tokens)")

    # --- Conversation depth ---
    if num_turns > 10:
        score += 0.15
        reasons.append(f"deep conversation ({num_turns} turns)")
    elif num_turns > 4:
        score += 0.08
        reasons.append(f"multi-turn ({num_turns} turns)")

    # --- Tool/function calling ---
    if tools:
        tool_count = len(tools)
        if tool_count > 3:
            score += 0.2
            reasons.append(f"many tools ({tool_count})")
        else:
            score += 0.1
            reasons.append(f"tool use ({tool_count} tools)")

    # --- System prompt complexity ---
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if system_msgs:
        system_text = _extract_text(system_msgs)
        system_tokens = _estimate_tokens(system_text)
        if system_tokens > 500:
            score += 0.15
            reasons.append(f"complex system prompt ({system_tokens} est. tokens)")
        elif system_tokens > 100:
            score += 0.05

    # --- Code blocks ---
    code_blocks = CODE_BLOCK_PATTERN.findall(full_text)
    if len(code_blocks) > 2:
        score += 0.15
        reasons.append(f"multiple code blocks ({len(code_blocks)})")
    elif code_blocks:
        score += 0.05

    # --- Images ---
    if "[IMAGE]" in full_text:
        score += 0.1
        reasons.append("contains images")

    # --- Keyword analysis (on last user message) ---
    user_msgs = [m for m in messages if m.get("role") == "user"]
    last_user_text = _extract_text(user_msgs[-1:]) if user_msgs else ""

    complex_matches = [m.group() for m in COMPLEX_KEYWORDS.finditer(last_user_text)]
    if complex_matches:
        # Scale with number of complex keywords: 1=0.3, 2=0.45, 3+=0.6
        keyword_score = min(0.6, 0.15 + 0.15 * len(complex_matches))
        score += keyword_score
        reasons.append(f"complex keywords ({len(complex_matches)}): {', '.join(set(complex_matches[:3]))}")

    simple_matches = [m.group() for m in SIMPLE_KEYWORDS.finditer(last_user_text)]
    if simple_matches and not complex_matches:
        score -= 0.15
        reasons.append(f"simple keywords: {', '.join(set(simple_matches[:3]))}")

    # Clamp
    score = max(0.0, min(1.0, score))

    # Confidence: high/low scores are confident, middle range is uncertain
    confident = score <= 0.2 or score >= 0.8

    return HeuristicResult(score=round(score, 3), reasons=reasons, confident=confident)
