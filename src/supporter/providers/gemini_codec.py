"""Gemini-specific codec — the ONLY module that bridges neutral types ↔ google.genai.

Converts Message ↔ Content, decodes AFC history, builds GenerateContentConfig
from neutral GenOptions, and constructs Gemini Tool objects from ToolDefs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..llm.types import (
    GenOptions,
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

if TYPE_CHECKING:
    from google.genai.types import Content


def message_to_content(msg: Message) -> Content:
    """Convert a neutral Message to a Gemini Content object."""
    from google.genai.types import (
        Blob,
        FunctionCall,
        FunctionResponse,
        Part,
    )

    gemini_parts: list[Any] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            gemini_parts.append(Part(text=part.text))
        elif isinstance(part, ToolCallPart):
            gemini_parts.append(
                Part(
                    function_call=FunctionCall(
                        name=part.name,
                        args=part.args,
                    )
                )
            )
        elif isinstance(part, ToolResultPart):
            gemini_parts.append(
                Part(
                    function_response=FunctionResponse(
                        name=part.name,
                        response=part.response,
                    )
                )
            )
        elif isinstance(part, ImagePart):
            if part.data:
                gemini_parts.append(
                    Part(
                        inline_data=Blob(
                            data=part.data,
                            mime_type=part.mime_type,
                        )
                    )
                )
            else:
                gemini_parts.append(Part(text=f"[image:{part.ref or 'unavailable'}]"))

    from google.genai.types import Content

    return Content(role=msg.role, parts=gemini_parts)


def content_to_message(c: Content) -> Message:
    """Convert a Gemini Content object to a neutral Message."""
    role = getattr(c, "role", "user")
    parts_raw = getattr(c, "parts", None) or []
    parts: list[Any] = []

    for p in parts_raw:
        text = getattr(p, "text", None)
        if text:
            parts.append(TextPart(text=text))
            continue

        fc = getattr(p, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            parts.append(
                ToolCallPart(
                    name=fc.name,
                    args=getattr(fc, "args", None) or {},
                )
            )
            continue

        fr = getattr(p, "function_response", None)
        if fr is not None and getattr(fr, "name", None):
            parts.append(
                ToolResultPart(
                    name=fr.name,
                    response=getattr(fr, "response", None) or {},
                )
            )
            continue

        idata = getattr(p, "inline_data", None)
        if idata is not None and getattr(idata, "data", None):
            parts.append(
                ImagePart(
                    mime_type=getattr(idata, "mime_type", "application/octet-stream"),
                    data=getattr(idata, "data", None),
                )
            )
            continue

    return Message(role=role, parts=parts)


def afc_history_to_messages(history: list[Content]) -> list[Message]:
    """Decode a Gemini automatic_function_calling_history to neutral Messages."""
    return [content_to_message(c) for c in history]


def gen_options_to_config(
    opts: GenOptions,
    transformed_tools: list[Any] | None,
    *,
    is_gemma: bool = False,
    default_system_instruction: str = "",
) -> Any:
    """Build a Gemini GenerateContentConfig from neutral GenOptions.

    Reads neutral fields + opts.extras for Gemini-only knobs
    (top_k, thinking_level, response_schema, response_mime_type, etc.).
    """
    from google.genai import types
    from google.genai.types import GenerateContentConfig

    extras = opts.extras

    thinking: Any
    if is_gemma:
        thinking = types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    else:
        thinking_level_name = extras.get("thinking_level")
        if thinking_level_name:
            thinking = types.ThinkingConfig(
                thinking_level=getattr(
                    types.ThinkingLevel,
                    thinking_level_name.upper(),
                    types.ThinkingLevel.MEDIUM,
                ),
            )
        else:
            thinking = types.ThinkingConfig(include_thoughts=True)

    return GenerateContentConfig(
        system_instruction=opts.system_instruction or default_system_instruction,
        temperature=opts.temperature,
        top_p=opts.top_p,
        top_k=extras.get("top_k"),
        max_output_tokens=opts.max_output_tokens,
        tools=transformed_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        if transformed_tools
        else None,
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True)
        if transformed_tools
        else None,
        response_mime_type=extras.get("response_mime_type"),
        response_schema=extras.get("response_schema"),
        thinking_config=thinking,
    )


def gemini_error_classes() -> tuple[type[BaseException], ...]:
    """Return Gemini error classes for error detection in pool.py."""
    try:
        from google.genai import errors as genai_errors

        return (genai_errors.ClientError, genai_errors.ServerError)
    except ImportError:
        return ()
