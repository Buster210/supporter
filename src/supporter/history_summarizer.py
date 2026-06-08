"""History summarization module.

WHY: Long sessions lose early context when history exceeds model limits. Instead of
hard-dropping oldest turns, fold them into a dense summary for context preservation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import config

if TYPE_CHECKING:
    from google.genai.types import Content


__all__ = ["render_turns", "summarize_turns"]

_SUMMARIZATION_INSTRUCTION = (
    "You are a helpful assistant that summarizes conversation history. "
    "Create a concise but complete summary of the preceding dialogue, "
    "preserving key facts, decisions, context, and technical details. "
    "Write in present tense, be factual and neutral, avoid redundancy."
)


def render_turns(turns: list[Content]) -> str:
    """Flatten Content turns to a 'User:/Assistant:' transcript.

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
            text = getattr(part, "text", None)
            if text:
                part_texts.append(text)
                continue

            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                name = fc.name
                args = getattr(fc, "args", None) or {}
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                part_texts.append(f"[tool_call: {name}({args_str})]")
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


async def summarize_turns(turns: list[Content]) -> str:
    """Summarize turns into a dense summary string.

    WHY: Provide compressed context for long sessions; decoupled from live provider
    to avoid cross-modality contamination (e.g., GeminiLiveProvider audio state).

    Returns empty string if transcript empty or no API keys available (RuntimeError).
    """
    transcript = render_turns(turns)
    if not transcript:
        return ""

    if not config.gemini_api_keys:
        raise RuntimeError("No Gemini API keys configured for summarization")

    from .providers.gemini_provider import GeminiProvider

    summarizer = GeminiProvider(
        api_key=config.gemini_api_keys[0],
        model_name=config.gemini_model,
    )

    result = await summarizer.generate(
        transcript,
        {
            "system_instruction": _SUMMARIZATION_INSTRUCTION,
            "temperature": 0.2,
        },
    )

    return result.text.strip()
