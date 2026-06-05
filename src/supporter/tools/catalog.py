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
    "browser_supervise",
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    callable: Callable[..., Any]
    default_enabled: bool = True
    delegate_allowed: bool = False
    allowed_roles: frozenset[str] | None = None


_BUILTIN_CATALOG_CACHE: dict[bool, dict[str, ToolSpec]] = {}


def _builtin_catalog(include_bash: bool) -> dict[str, ToolSpec]:
    if include_bash in _BUILTIN_CATALOG_CACHE:
        return _BUILTIN_CATALOG_CACHE[include_bash]

    from .bash.executor import execute_bash
    from .browser.supervisor import browser_supervise
    from .browser.task import (
        delete_playbook,
        finish_task,
        list_playbooks,
        query_playbook,
        replay_playbook,
        start_task,
    )
    from .browser.tool import browse
    from .delegate.api import cancel_delegation, check_delegation, delegate_tasks
    from .delegate.capsule_query import query_delegation
    from .file_ops import read_file, write_file
    from .search import google_search

    catalog: dict[str, ToolSpec] = {
        "read_file": ToolSpec("read_file", read_file, delegate_allowed=True),
        "write_file": ToolSpec("write_file", write_file, delegate_allowed=True),
        "browse": ToolSpec(
            "browse",
            browse,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "browser_supervise": ToolSpec(
            "browser_supervise",
            browser_supervise,
        ),
        "start_task": ToolSpec(
            "start_task",
            start_task,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "finish_task": ToolSpec(
            "finish_task",
            finish_task,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "query_playbook": ToolSpec(
            "query_playbook",
            query_playbook,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "replay_playbook": ToolSpec(
            "replay_playbook",
            replay_playbook,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "list_playbooks": ToolSpec(
            "list_playbooks",
            list_playbooks,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
        "delete_playbook": ToolSpec(
            "delete_playbook",
            delete_playbook,
            delegate_allowed=True,
            allowed_roles=frozenset({"page-pilot"}),
        ),
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
    # ToolSpec is frozen=True dataclass — safe to share cached refs without copying
    _BUILTIN_CATALOG_CACHE[include_bash] = catalog
    return catalog


def build_tool_catalog(
    *,
    include_bash: bool = True,
    extra_tools: Mapping[str, ToolSpec | Callable[..., Any]] | None = None,
) -> dict[str, ToolSpec]:
    catalog = dict(_builtin_catalog(include_bash))
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
    role: str | None = None,
) -> dict[str, Callable[..., Any]]:
    selected_names = set(catalog) if names == "all" else set(names)
    return {
        name: spec.callable
        for name, spec in catalog.items()
        if name in selected_names
        and spec.delegate_allowed
        and (include_disabled or spec.default_enabled)
        and (spec.allowed_roles is None or role in spec.allowed_roles)
    }
