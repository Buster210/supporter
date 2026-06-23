"""Provider-neutral contract types for the LLM seam.

Adding a new text provider = one new file in providers/ + one registry
entry. Everything above the provider boundary speaks these types.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Part variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ToolCallPart:
    name: str
    args: dict[str, typing.Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ToolResultPart:
    name: str
    response: dict[str, typing.Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ImagePart:
    mime_type: str
    ref: str | None = None
    data: bytes | None = None


Part = TextPart | ToolCallPart | ToolResultPart | ImagePart


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "model" | "tool"
    parts: list[Part]


# ---------------------------------------------------------------------------
# Tool definition (provider-neutral)
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, typing.Any]  # JSON-Schema object
    callable: Callable[..., typing.Any]


_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_hint_to_json_schema(hint: typing.Any) -> dict[str, typing.Any]:
    origin = getattr(hint, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in hint.__args__ if a is not type(None)]
        if len(args) == 1:
            return _type_hint_to_json_schema(args[0])
        return {"anyOf": [_type_hint_to_json_schema(a) for a in args]}
    if origin is list:
        args = list(getattr(hint, "__args__", [str]))
        return {"type": "array", "items": _type_hint_to_json_schema(args[0])}
    if origin is dict:
        return {"type": "object"}
    return {"type": _PYTHON_TO_JSON_TYPE.get(hint, "string")}


def tool_def_from_callable(fn: Callable[..., typing.Any]) -> ToolDef:
    """Derive a neutral ToolDef from a Python callable via introspection.

    Matches the JSON-Schema shape that Gemini AFC introspection produces:
    name, description from docstring, required params from type hints.
    """
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, typing.Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        hint = hints.get(pname, str)
        properties[pname] = _type_hint_to_json_schema(hint)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].strip() if doc else fn.__name__
    # Strip leading Args:/Returns: lines that sometimes follow the first line.
    lines = description.split("\n")
    desc_lines = [
        line
        for line in lines
        if not line.strip().lower().startswith(("args:", "returns:", "raises:"))
    ]
    description = " ".join(desc_lines).strip()

    parameters_schema: dict[str, typing.Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return ToolDef(
        name=fn.__name__,
        description=description,
        parameters=parameters_schema,
        callable=fn,
    )


# ---------------------------------------------------------------------------
# Generation options (provider-neutral)
# ---------------------------------------------------------------------------


@dataclass
class GenOptions:
    model: str | None = None
    system_instruction: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    use_search: bool = False
    extras: dict[str, typing.Any] = field(default_factory=dict)
