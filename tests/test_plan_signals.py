"""plan_tool surfaces planner activity live and in order."""

from typing import ClassVar
from unittest.mock import patch

import pytest

from supporter.tools import planning


@pytest.mark.asyncio
async def test_plan_tool_emits_consulting_then_plan_in_order():
    events: list[tuple[str, str]] = []
    planning.set_plan_signal_callback(lambda kind, text: events.append((kind, text)))

    async def fake_make_plan(objective, persona, model, tools_roster=""):
        # The consult signal must already have fired before the plan returns.
        assert events == [("consulting", "Consulting planner sub-agent…")]
        return "1. do it"

    try:
        with patch("supporter.worker.make_plan", fake_make_plan):
            plan = await planning.plan_tool("obj")
    finally:
        planning.clear_plan_signal_callback()

    assert plan == "1. do it"
    assert events == [
        ("consulting", "Consulting planner sub-agent…"),
        ("plan", "1. do it"),
    ]


@pytest.mark.asyncio
async def test_plan_signal_callback_never_required():
    # Headless (no sink bound) must not raise.
    planning.clear_plan_signal_callback()

    async def fake_make_plan(objective, persona, model, tools_roster=""):
        return "x"

    with patch("supporter.worker.make_plan", fake_make_plan):
        assert await planning.plan_tool("obj") == "x"


@pytest.mark.asyncio
async def test_plan_tool_passes_orchestrator_tool_roster():
    """The planner is told which tools the orchestrator actually has."""

    def bash_run():
        """Run a sandboxed shell command."""

    def google_search():
        """Search the web."""

    class FakeAgent:
        registry: ClassVar[dict] = {
            "bash_run": bash_run,
            "google_search": google_search,
        }
        last_plan = ""
        last_plan_objective = ""

    planning.bind_agent(FakeAgent())
    captured = {}

    async def fake_make_plan(objective, persona, model, tools_roster=""):
        captured["roster"] = tools_roster
        return "1. do it"

    try:
        with patch("supporter.worker.make_plan", fake_make_plan):
            await planning.plan_tool("obj")
    finally:
        planning.clear_agent()

    roster = captured["roster"]
    assert "bash_run: Run a sandboxed shell command." in roster
    assert "google_search: Search the web." in roster


def test_format_tool_roster_handles_missing_registry():
    assert planning._format_tool_roster(object()) == ""
