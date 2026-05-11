from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

_ToolSelection = Iterable[str] | Literal["all"]

ORCHESTRATOR_TOOL_NAMES = (
    "read_file",
    "write_file",
    "delegate_tasks",
    "check_delegation",
    "cancel_delegation",
    "query_delegation",
    "execute_bash",
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    callable: Callable[..., Any]
    default_enabled: bool = True
    delegate_allowed: bool = False


def build_tool_catalog(
    *,
    include_bash: bool = True,
    extra_tools: Mapping[str, ToolSpec | Callable[..., Any]] | None = None,
) -> dict[str, ToolSpec]:
    from .bash.executor import execute_bash
    from .delegate.api import cancel_delegation, check_delegation, delegate_tasks
    from .delegate.capsule_query import query_delegation
    from .file_ops import read_file, write_file
    from .search import google_search

    catalog: dict[str, ToolSpec] = {
        "read_file": ToolSpec("read_file", read_file, delegate_allowed=True),
        "write_file": ToolSpec("write_file", write_file, delegate_allowed=True),
        "delegate_tasks": ToolSpec("delegate_tasks", delegate_tasks),
        "check_delegation": ToolSpec("check_delegation", check_delegation),
        "cancel_delegation": ToolSpec("cancel_delegation", cancel_delegation),
        "query_delegation": ToolSpec("query_delegation", query_delegation),
        "google_search": ToolSpec(
            "google_search", google_search, delegate_allowed=True
        ),
        "execute_bash": ToolSpec(
            "execute_bash",
            execute_bash,
            default_enabled=include_bash,
            delegate_allowed=True,
        ),
    }
    if extra_tools:
        for name, tool in extra_tools.items():
            catalog[name] = (
                tool
                if isinstance(tool, ToolSpec)
                else ToolSpec(name=name, callable=tool)
            )
    return catalog


def select_tools(
    catalog: Mapping[str, ToolSpec],
    names: _ToolSelection,
    *,
    include_disabled: bool = False,
) -> dict[str, Callable[..., Any]]:
    selected_names = set(catalog) if names == "all" else set(names)
    return {
        name: spec.callable
        for name, spec in catalog.items()
        if name in selected_names and (include_disabled or spec.default_enabled)
    }


def select_delegate_tools(
    catalog: Mapping[str, ToolSpec],
    names: _ToolSelection,
    *,
    include_disabled: bool = False,
) -> dict[str, Callable[..., Any]]:
    selected_names = set(catalog) if names == "all" else set(names)
    return {
        name: spec.callable
        for name, spec in catalog.items()
        if name in selected_names
        and spec.delegate_allowed
        and (include_disabled or spec.default_enabled)
    }
