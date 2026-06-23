"""Tests for orchestrator planning — now a delegate sub-agent role, not a tool."""

from __future__ import annotations

from supporter.prompts import DELEGATE_AGENT_ROSTER
from supporter.tools.catalog import (
    ORCHESTRATOR_TOOL_NAMES,
    build_tool_catalog,
    select_delegate_tools,
    select_tools,
)

# ---------------------------------------------------------------------------
# Catalog / scoping — plan tool removed
# ---------------------------------------------------------------------------


class TestPlanToolRemoved:
    def test_plan_not_in_orchestrator_tool_names(self) -> None:
        assert "plan" not in ORCHESTRATOR_TOOL_NAMES

    def test_plan_not_in_catalog(self) -> None:
        catalog = build_tool_catalog()
        assert "plan" not in catalog

    def test_plan_not_selectable_for_orchestrator(self) -> None:
        catalog = build_tool_catalog()
        tools = select_tools(catalog, ORCHESTRATOR_TOOL_NAMES)
        assert "plan" not in tools

    def test_plan_not_selectable_for_delegates(self) -> None:
        catalog = build_tool_catalog()
        delegate_tools = select_delegate_tools(
            catalog, ORCHESTRATOR_TOOL_NAMES, role="explorer"
        )
        assert "plan" not in delegate_tools

        delegate_tools2 = select_delegate_tools(catalog, "all", role="page-pilot")
        assert "plan" not in delegate_tools2


# ---------------------------------------------------------------------------
# New planner delegate role
# ---------------------------------------------------------------------------


class TestPlannerDelegateRole:
    def test_planner_in_roster(self) -> None:
        planner = DELEGATE_AGENT_ROSTER["planner"]
        assert planner["model"] == "gemma-4-31b-it"
        assert planner["live"] is False
        assert "read_file" in planner["tools"]
        assert "execute_bash" in planner["tools"]
        assert "google_search" in planner["tools"]
        assert "web_search" in planner["tools"]

    def test_planner_not_worker_planner(self) -> None:
        """planner (orchestration) and worker_planner (browser) are distinct."""
        planner = DELEGATE_AGENT_ROSTER["planner"]
        worker_planner = DELEGATE_AGENT_ROSTER["worker_planner"]
        assert planner is not worker_planner
        # worker_planner has no tools (browser worker plans, doesn't recon)
        assert worker_planner["tools"] == set()

    def test_planner_tools_are_delegate_allowed(self) -> None:
        """All planner recon tools must be delegate_allowed in the catalog."""
        catalog = build_tool_catalog()
        planner_tools = DELEGATE_AGENT_ROSTER["planner"]["tools"]
        for tool_name in planner_tools:
            spec = catalog[tool_name]
            assert spec.delegate_allowed, f"{tool_name} not delegate_allowed"


# ---------------------------------------------------------------------------
# Tool ownership: orchestrator has recipe_*, not memory_*; page-pilot has memory_*
# ---------------------------------------------------------------------------


class TestToolOwnership:
    def test_orchestrator_no_recipes(self) -> None:
        """Recipe tools removed from orchestrator — orchestrator doesn't use them."""
        catalog = build_tool_catalog()
        tools = select_tools(catalog, ORCHESTRATOR_TOOL_NAMES)
        for name in (
            "recipe_save",
            "recipe_find",
            "recipe_search",
            "recipe_run",
            "recipe_delete",
            "recipe_list",
            "recipe_status",
        ):
            assert name not in tools, f"{name} should NOT be in orchestrator tools"

    def test_orchestrator_no_memory(self) -> None:
        catalog = build_tool_catalog()
        tools = select_tools(catalog, ORCHESTRATOR_TOOL_NAMES)
        for name in (
            "memory_write",
            "memory_read",
            "memory_search",
            "memory_list_kinds",
            "memory_compact",
            "memory_clear",
            "memory_status",
        ):
            assert name not in tools, f"{name} should NOT be in orchestrator tools"

    def test_orchestrator_no_plan(self) -> None:
        catalog = build_tool_catalog()
        tools = select_tools(catalog, ORCHESTRATOR_TOOL_NAMES)
        assert "plan" not in tools

    def test_page_pilot_has_memory(self) -> None:
        catalog = build_tool_catalog()
        tools = select_delegate_tools(catalog, "all", role="page-pilot")
        for name in (
            "memory_write",
            "memory_read",
            "memory_search",
            "memory_list_kinds",
            "memory_compact",
            "memory_clear",
            "memory_status",
        ):
            assert name in tools, f"{name} missing from page-pilot delegate tools"

    def test_recipe_tools_not_delegate_allowed(self) -> None:
        """recipe_* are orchestrator-only, not available to delegates."""
        catalog = build_tool_catalog()
        for name in (
            "recipe_save",
            "recipe_find",
            "recipe_search",
            "recipe_run",
            "recipe_delete",
            "recipe_list",
            "recipe_status",
        ):
            spec = catalog[name]
            assert not spec.delegate_allowed, f"{name} should not be delegate_allowed"

    def test_memory_tools_page_pilot_only(self) -> None:
        """memory_* are page-pilot-only, not available to other delegates."""
        catalog = build_tool_catalog()
        for name in (
            "memory_write",
            "memory_read",
            "memory_search",
            "memory_list_kinds",
            "memory_compact",
            "memory_clear",
            "memory_status",
        ):
            spec = catalog[name]
            assert spec.allowed_roles == frozenset({"page-pilot"}), (
                f"{name} allowed_roles should be page-pilot only"
            )
