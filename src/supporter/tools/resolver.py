from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _get_google_search() -> Any:
    from .search import google_search

    return google_search


def needs_function_search(model_name: str) -> bool:
    return "gemini" in model_name.lower() and "-3." in model_name.lower()


def _needs_live_server_search(model_name: str) -> bool:
    model = model_name.lower()
    return "2.5" in model or "gemma" in model or "fallback" in model


def extract_declared_tool_names(tools: Sequence[Any]) -> set[str]:
    names = set()
    for tool in tools:
        declarations = getattr(tool, "function_declarations", None) or []
        if isinstance(tool, dict):
            declarations = tool.get("function_declarations") or []

        for declaration in declarations:
            name = (
                declaration.get("name")
                if isinstance(declaration, dict)
                else getattr(declaration, "name", None)
            )
            if name:
                names.add(name)
    return names


def _has_tool_attr(tools: Sequence[Any], attr_name: str) -> bool:
    """ponytail: unified helper to check for any tool attribute."""
    return any(getattr(tool, attr_name, None) is not None for tool in tools)


def _has_server_search_tool(tools: Sequence[Any]) -> bool:
    return _has_tool_attr(tools, "google_search")


def _has_code_execution_tool(tools: Sequence[Any]) -> bool:
    return _has_tool_attr(tools, "code_execution")


def ensure_function_search_tool(
    model_name: str,
    registry: dict[str, Callable[..., Any]],
) -> None:
    if needs_function_search(model_name) and "google_search" not in registry:
        registry["google_search"] = _get_google_search()


def resolve_provider_tools(
    *,
    model_name: str,
    tools: Sequence[Any],
    registry: Mapping[str, Callable[..., Any]],
    use_search: bool,
    use_code_execution: bool,
    google_types: Any,
) -> list[Any]:
    final_tools: list[Any] = list(tools)
    declared_names = extract_declared_tool_names(final_tools)

    for name, func in registry.items():
        if name not in declared_names:
            final_tools.append(func)

    if use_search:
        if needs_function_search(model_name):
            if (
                "google_search" not in registry
                and "google_search" not in declared_names
            ):
                final_tools.append(_get_google_search())
        elif not _has_server_search_tool(final_tools):
            final_tools.append(
                google_types.Tool(google_search=google_types.GoogleSearch())
            )

    if use_code_execution and not _has_code_execution_tool(final_tools):
        final_tools.append(
            google_types.Tool(code_execution=google_types.ToolCodeExecution())
        )

    return final_tools


def resolve_live_provider_tools(
    *,
    model_name: str,
    tools: Sequence[Any],
    registry: Mapping[str, Callable[..., Any]],
    google_types: Any,
) -> list[Any]:
    final_tools = list(tools) + list(registry.values())
    if _needs_live_server_search(model_name) and not _has_server_search_tool(
        final_tools
    ):
        final_tools.append(google_types.Tool(google_search=google_types.GoogleSearch()))
    return final_tools
