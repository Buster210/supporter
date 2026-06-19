from __future__ import annotations

from typing import Any

from supporter.tools.catalog import (
    ORCHESTRATOR_TOOL_NAMES,
    ToolSpec,
    build_tool_catalog,
    select_delegate_tools,
    select_tools,
)


def test_catalog_exposes_current_tool_surface() -> None:
    catalog = build_tool_catalog()

    assert set(catalog) == {
        "read_file",
        "write_file",
        "browse",
        "browser_supervise",
        "start_task",
        "finish_task",
        "query_playbook",
        "replay_playbook",
        "list_playbooks",
        "delete_playbook",
        "delegate_tasks",
        "check_delegation",
        "cancel_delegation",
        "query_delegation",
        "google_search",
        "web_search",
        "deep_research",
        "research_assess",
        "verify_claims",
        "research_report",
        "execute_bash",
        "memory_write",
        "memory_read",
        "memory_search",
        "memory_list_kinds",
        "memory_compact",
        "memory_clear",
        "memory_status",
        "recipe_save",
        "recipe_find",
        "recipe_search",
        "recipe_run",
        "recipe_delete",
        "recipe_list",
        "recipe_status",
        "plan",
    }
    assert all(spec.name == name for name, spec in catalog.items())
    assert all(callable(spec.callable) for spec in catalog.values())


def test_orchestrator_selects_root_tools_without_search_function() -> None:
    registry = select_tools(build_tool_catalog(), ORCHESTRATOR_TOOL_NAMES)

    assert set(registry) == {
        "read_file",
        "write_file",
        "delegate_tasks",
        "check_delegation",
        "cancel_delegation",
        "query_delegation",
        "execute_bash",
        "browser_supervise",
        "web_search",
        "research_assess",
        "verify_claims",
        "research_report",
        "plan",
    }
    assert "google_search" not in registry
    assert "browse" not in registry
    assert "start_task" not in registry
    assert "finish_task" not in registry
    assert "query_playbook" not in registry
    assert "replay_playbook" not in registry
    assert "list_playbooks" not in registry
    assert "delete_playbook" not in registry


def test_bash_can_be_disabled_without_changing_selection_sites() -> None:
    registry = select_tools(build_tool_catalog(include_bash=False), "all")

    assert "execute_bash" not in registry


def test_delegate_selection_rejects_recursive_control_tools() -> None:
    registry = select_delegate_tools(
        build_tool_catalog(),
        {
            "read_file",
            "write_file",
            "execute_bash",
            "google_search",
            "delegate_tasks",
            "check_delegation",
            "cancel_delegation",
            "query_delegation",
        },
    )

    assert set(registry) == {
        "read_file",
        "write_file",
        "execute_bash",
        "google_search",
    }


def test_catalog_only_tool_can_be_selected_without_wiring_changes() -> None:
    def dummypreview(_: Any) -> str:
        return "ready"

    catalog = build_tool_catalog(
        extra_tools={
            "dummypreview": ToolSpec(
                name="dummypreview",
                callable=dummypreview,
                delegate_allowed=True,
            )
        }
    )

    root_registry = select_tools(catalog, {"dummypreview"})
    delegate_registry = select_delegate_tools(catalog, {"dummypreview"})

    assert root_registry == {"dummypreview": dummypreview}
    assert delegate_registry == {"dummypreview": dummypreview}
