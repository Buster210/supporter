"""Deterministic request router.

WHY: routing (direct/research/task) is normally decided by the model
re-reading a triage block baked into the system prompt every turn -- which
rots as the bloated orchestrator's context grows. This module makes the same
decision with one isolated, fresh-context call (no conversation history),
mirroring the one-shot pattern in history_summarizer.summarize_turns. Only
used when config.router_enabled is True.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .llm.types import GenOptions
from .logger import logger
from .prompts import ROUTER_PROMPT, ROUTER_SYSTEM_INSTRUCTION
from .types import LLMProvider

Route = Literal["direct", "research", "task"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class RouteDecision:
    route: Route
    needs_research: bool = False


def _fallback(reason: str) -> RouteDecision:
    # WHY: mirrors _TASK_TRIAGE's own tiebreak -- unsure means "task", never crash.
    logger.warning(f"Router: falling back to route=task ({reason})")
    return RouteDecision(route="task", needs_research=False)


async def route_prompt(provider: LLMProvider, prompt: str) -> RouteDecision:
    """Classify prompt into a route via one isolated, history-free LLM call.

    Never raises: any parse/validation/provider failure falls back to
    RouteDecision(route="task", needs_research=False).
    """
    try:
        result = await provider.generate(
            # WHY: not str.format -- ROUTER_PROMPT's JSON example contains
            # literal braces that .format() would try to parse as fields.
            ROUTER_PROMPT.replace("{prompt}", prompt),
            GenOptions(
                system_instruction=ROUTER_SYSTEM_INSTRUCTION,
                temperature=0.0,
                # ponytail: prompt-instructed strict JSON only. gemini_codec
                # already threads extras["response_mime_type"] through as a
                # no-op default, so flip this on once response_schema-based
                # validation is worth the extra plumbing.
                extras={"response_mime_type": "application/json"},
            ),
        )
    except Exception as e:  # fail-safe boundary, must never crash
        return _fallback(f"provider error: {e}")

    text = (result.text or "").strip()
    if not text:
        return _fallback("empty response")

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return _fallback(f"invalid JSON: {e}")

    if not isinstance(data, dict):
        return _fallback("JSON was not an object")

    route = data.get("route")
    if route not in ("direct", "research", "task"):
        return _fallback(f"invalid route: {route!r}")

    needs_research = bool(data.get("needs_research", False))
    return RouteDecision(route=route, needs_research=needs_research)
