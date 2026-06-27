"""Tests for orchestrator-driven planning via the plan tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from supporter.tools.catalog import (
    ORCHESTRATOR_TOOL_NAMES,
    ToolSpec,
    build_tool_catalog,
    select_delegate_tools,
    select_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str, options: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self._response)


def _patch_provider(provider: Any) -> Any:
    return patch(
        "supporter.worker.get_provider",
        return_value=provider,
    )


# ---------------------------------------------------------------------------
# Catalog / scoping
# ---------------------------------------------------------------------------


class TestPlanToolCatalog:
    def test_plan_in_orchestrator_tool_names(self) -> None:
        assert "plan" in ORCHESTRATOR_TOOL_NAMES

    def test_plan_tool_in_catalog(self) -> None:
        catalog = build_tool_catalog()
        assert "plan" in catalog
        spec = catalog["plan"]
        assert isinstance(spec, ToolSpec)
        assert spec.name == "plan"
        assert callable(spec.callable)

    def test_plan_tool_orchestrator_only(self) -> None:
        """plan must NOT be selectable for sub-agents (delegate_allowed=False)."""
        catalog = build_tool_catalog()
        # Selecting tools for a delegate role should never include "plan"
        delegate_tools = select_delegate_tools(
            catalog, ORCHESTRATOR_TOOL_NAMES, role="explorer"
        )
        assert "plan" not in delegate_tools

        delegate_tools2 = select_delegate_tools(catalog, "all", role="page-pilot")
        assert "plan" not in delegate_tools2

    def test_plan_tool_selected_for_orchestrator(self) -> None:
        catalog = build_tool_catalog()
        tools = select_tools(catalog, ORCHESTRATOR_TOOL_NAMES)
        assert "plan" in tools
        assert callable(tools["plan"])


# ---------------------------------------------------------------------------
# plan_tool handler
# ---------------------------------------------------------------------------


class TestPlanToolHandler:
    @pytest.mark.asyncio
    async def test_returns_plan_text(self) -> None:
        from supporter.tools.planning import plan_tool

        with _patch_provider(_FakeProvider("GOAL: test\nSTEPS:\n1. do thing")):
            result = await plan_tool("do a multi-step thing")
        assert "GOAL: test" in result
        assert "STEPS:" in result

    @pytest.mark.asyncio
    async def test_sets_agent_last_plan(self) -> None:
        from supporter.tools.planning import bind_agent, clear_agent, plan_tool

        class _FakeAgent:
            last_plan: str = ""
            last_plan_objective: str = ""

        agent = _FakeAgent()
        bind_agent(agent)
        try:
            with _patch_provider(_FakeProvider("PLAN: step1\nstep2")):
                result = await plan_tool("do something complex")
            assert result == "PLAN: step1\nstep2"
            assert agent.last_plan == "PLAN: step1\nstep2"
            assert agent.last_plan_objective == "do something complex"
        finally:
            clear_agent()

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self) -> None:
        from supporter.tools.planning import plan_tool

        failing = AsyncMock()
        failing.generate = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("supporter.worker.get_provider", return_value=failing):
            result = await plan_tool("will fail")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_plan(self) -> None:
        from supporter.tools.planning import plan_tool

        with _patch_provider(_FakeProvider("")):
            result = await plan_tool("trivial thing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_agent_binding_still_works(self) -> None:
        from supporter.tools.planning import clear_agent, plan_tool

        clear_agent()  # ensure no agent bound
        with _patch_provider(_FakeProvider("PLAN: ok")):
            result = await plan_tool("task")
        assert result == "PLAN: ok"


# ---------------------------------------------------------------------------
# Pre-classifier funnel removed
# ---------------------------------------------------------------------------


class TestFunnelRemoved:
    def test_is_substantive_task_removed(self) -> None:
        """_is_substantive_task no longer exists on supporter.tui."""
        import supporter.tui as tui_mod

        assert not hasattr(tui_mod, "_is_substantive_task")

    def test_trivial_responses_removed(self) -> None:
        import supporter.tui as tui_mod

        assert not hasattr(tui_mod, "_TRIVIAL_RESPONSES")

    def test_plan_before_act_removed_from_config(self) -> None:
        from supporter.types import AppConfig

        # plan_before_act field should not exist
        assert not hasattr(AppConfig, "plan_before_act") or not any(
            f.name == "plan_before_act"
            for f in AppConfig.__dataclass_fields__.values()
            if hasattr(f, "name")
        )

    def test_run_planner_and_inject_removed(self) -> None:
        from supporter.tui import SupporterApp

        assert not hasattr(SupporterApp, "_run_planner_and_inject")
