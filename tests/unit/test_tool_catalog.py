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
    def dummy_preview(_: Any) -> str:
        return "ready"

    catalog = build_tool_catalog(
        extra_tools={
            "dummy_preview": ToolSpec(
                name="dummy_preview",
                callable=dummy_preview,
                delegate_allowed=True,
            )
        }
    )

    root_registry = select_tools(catalog, {"dummy_preview"})
    delegate_registry = select_delegate_tools(catalog, {"dummy_preview"})

    assert root_registry == {"dummy_preview": dummy_preview}
    assert delegate_registry == {"dummy_preview": dummy_preview}
