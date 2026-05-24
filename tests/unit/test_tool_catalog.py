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
        "start_task",
        "finish_task",
        "query_playbook",
        "replay_playbook",
        "delegate_tasks",
        "check_delegation",
        "cancel_delegation",
        "query_delegation",
        "google_search",
        "execute_bash",
    }
    assert all(spec.name == name for name, spec in catalog.items())
    assert all(callable(spec.callable) for spec in catalog.values())


def test_orchestrator_selects_root_tools_without_search_function() -> None:
    registry = select_tools(build_tool_catalog(), ORCHESTRATOR_TOOL_NAMES)

    assert set(registry) == {
        "read_file",
        "write_file",
        "browse",
        "start_task",
        "finish_task",
        "query_playbook",
        "replay_playbook",
        "delegate_tasks",
        "check_delegation",
        "cancel_delegation",
        "query_delegation",
        "execute_bash",
    }
    assert "google_search" not in registry


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
