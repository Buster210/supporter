"""History summarization module.

WHY: Long sessions lose early context when history exceeds model limits. Instead of
hard-dropping oldest turns, fold them into a dense summary for context preservation.
"""

from __future__ import annotations

import hashlib

from supporter.llm.types import Message, TextPart, ToolCallPart, ToolResultPart

from .config import config

__all__ = [
    "clear_summarizer_cache",
    "render_turns",
    "summarize_turns",
    "summarizer_cache_info",
]

_SUMMARIZATION_INSTRUCTION = (
    "You are a helpful assistant that summarizes conversation history. "
    "Create a concise but complete summary of the preceding dialogue, "
    "preserving key facts, decisions, context, and technical details. "
    "Write in present tense, be factual and neutral, avoid redundancy."
)

# In-process cache for summarization results. Caching is safe because
# summarize_turns is a pure function of the transcript (no side effects,
# no streaming state). Kept as a plain dict because the function is async
# and functools.lru_cache does not support async callables.
_SUMMARIZER_CACHE: dict[str, str] = {}


def render_turns(turns: list[Message]) -> str:
    """Flatten conversation turns to a 'User:/Assistant:' transcript.

    WHY: Summarizer needs plain text input; render function_call/response parts
    compactly to preserve tool interactions without raw JSON noise.
    """
    lines: list[str] = []

    for turn in turns:
        role = getattr(turn, "role", None)
        if role == "user":
            speaker = "User:"
        elif role == "model":
            speaker = "Assistant:"
        else:
            continue

        parts = getattr(turn, "parts", None) or []
        part_texts: list[str] = []

        for part in parts:
            if isinstance(part, TextPart):
                part_texts.append(part.text)
                continue

            text = getattr(part, "text", None)
            if text:
                part_texts.append(text)
                continue

            if isinstance(part, ToolCallPart):
                args_str = ", ".join(f"{k}={v!r}" for k, v in part.args.items())
                part_texts.append(f"[tool_call: {part.name}({args_str})]")
                continue

            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                name = fc.name
                args = getattr(fc, "args", None) or {}
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                part_texts.append(f"[tool_call: {name}({args_str})]")
                continue

            if isinstance(part, ToolResultPart):
                resp_str = ", ".join(f"{k}={v!r}" for k, v in part.response.items())
                part_texts.append(f"[tool_response: {part.name}({resp_str})]")
                continue

            fr = getattr(part, "function_response", None)
            if fr is not None and getattr(fr, "name", None):
                name = fr.name
                response = getattr(fr, "response", None) or {}
                resp_str = ", ".join(f"{k}={v!r}" for k, v in response.items())
                part_texts.append(f"[tool_response: {name}({resp_str})]")
                continue

        if part_texts:
            lines.append(f"{speaker} {' '.join(part_texts)}")

    return "\n".join(lines)


async def summarize_turns(turns: list[Message]) -> str:
    """Summarize turns into a dense summary string.

    WHY: Provide compressed context for long sessions; decoupled from live provider
    to avoid cross-modality contamination (e.g., GeminiLiveProvider audio state).

    Returns empty string if transcript empty or no API keys available (RuntimeError).
    """
    transcript = render_turns(turns)
    if not transcript:
        return ""

    # WHY: A long-lived session can ask for the same summary multiple
    # times (the agent recomputes coverage after each turn). Caching by
    # transcript hash skips the network round-trip and the LLM cost on
    # the common case. The cache is bounded so it cannot grow without
    # limit across many distinct transcripts.
    cache_key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if cache_key in _SUMMARIZER_CACHE:
        return _SUMMARIZER_CACHE[cache_key]

    if not config.gemini_api_keys:
        raise RuntimeError("No Gemini API keys configured for summarization")

    from .pool import get_provider

    summarizer = get_provider(
        shared=True,
        model_name=config.gemini_model,
    )

    result = await summarizer.generate(
        transcript,
        {
            "system_instruction": _SUMMARIZATION_INSTRUCTION,
            "temperature": 0.2,
        },
    )

    summary = result.text.strip()
    _SUMMARIZER_CACHE[cache_key] = summary
    return summary


def summarizer_cache_info() -> dict[str, int]:
    """Return the size of the in-process summarizer cache.

    Useful for observability / tests.
    """
    return {"size": len(_SUMMARIZER_CACHE)}


def clear_summarizer_cache() -> None:
    """Drop the in-process summarizer cache. Tests / config changes."""
    _SUMMARIZER_CACHE.clear()
