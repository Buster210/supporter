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
    ToolDef,
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


def tooldefs_to_gemini(
    tooldefs: list[ToolDef],
    *,
    use_search: bool = False,
    use_code_execution: bool = False,
    model_name: str = "",
    registry: dict[str, Any] | None = None,
) -> list[Any]:
    """Build Gemini Tool objects from neutral ToolDefs + server-side tools.

    Passes user callables through (AFC introspects them). Handles
    google_search and code_execution server-side tools. Strips
    code_execution from Gemma models.
    """
    from google.genai import types

    final_tools: list[Any] = []
    declared_names: set[str] = set()

    # Add user callables from tooldefs — AFC introspects them directly.
    for td in tooldefs:
        final_tools.append(td.callable)
        declared_names.add(td.name)

    # Add registry functions not already declared.
    if registry:
        for name, func in registry.items():
            if name not in declared_names:
                final_tools.append(func)

    is_gemma = model_name.lower().startswith("gemma")

    # Server-side search tool.
    if use_search:
        has_server_search = any(
            getattr(t, "google_search", None) is not None for t in final_tools
        )
        if not has_server_search:
            final_tools.append(types.Tool(google_search=types.GoogleSearch()))

    # Server-side code execution tool (skip for Gemma).
    if use_code_execution and not is_gemma:
        has_code_exec = any(
            getattr(t, "code_execution", None) is not None for t in final_tools
        )
        if not has_code_exec:
            final_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

    # Gemma: strip code_execution if somehow present.
    if is_gemma:
        final_tools = [
            t for t in final_tools if getattr(t, "code_execution", None) is None
        ]

    return final_tools
