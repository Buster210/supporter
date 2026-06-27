"""Orchestrator-only planning tool."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..logger import logger

# Module-level agent reference, bound before streaming so the handler can
# store the plan on the agent for TUI display.  Set via bind_agent().
_agent_ref: object = None  # ChatAgent | None, avoided to skip import.

# Optional live UI signal callback (kind, text). Bound by the TUI so plan_tool
# can surface "consulting planner" and the produced plan IN ORDER as they
# happen.  Headless/tests leave it None — _emit never raises.
_signal_cb: Callable[[str, str], None] | None = None


def bind_agent(agent: object) -> None:
    """Bind the current agent so plan_tool can store results on it."""
    global _agent_ref
    _agent_ref = agent


def clear_agent() -> None:
    """Unbind after streaming completes."""
    global _agent_ref
    _agent_ref = None


def set_plan_signal_callback(cb: Callable[[str, str], None]) -> None:
    """Bind a live UI signal sink: cb(kind, text), kind in {consulting, plan}."""
    global _signal_cb
    _signal_cb = cb


def clear_plan_signal_callback() -> None:
    """Unbind the signal sink after streaming completes."""
    global _signal_cb
    _signal_cb = None


def _emit_signal(kind: str, text: str) -> None:
    """Fire the UI signal if bound; never raise (headless has no sink)."""
    cb = _signal_cb
    if cb is None:
        return
    try:
        cb(kind, text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"plan signal callback failed: {exc}")


def _format_tool_roster(agent: object) -> str:
    """Render the orchestrator's live tool registry as a name + purpose list."""
    registry = getattr(agent, "registry", None) or {}
    lines = []
    for name, fn in sorted(registry.items()):
        doc = (getattr(fn, "__doc__", "") or "").strip().split("\n", 1)[0].strip()
        lines.append(f"- {name}: {doc}" if doc else f"- {name}")
    return "\n".join(lines)


async def plan_tool(objective: str) -> str:
    """Run the orchestrator planner on *objective* and return the plan text.

    Sets ``agent.last_plan`` / ``agent.last_plan_objective`` on the bound
    agent so the TUI can display the plan.  Never raises — returns ``""`` on
    failure.
    """
    from ..prompts import DELEGATE_AGENT_ROSTER, ORCHESTRATION_PLANNER_PERSONA
    from ..worker import make_plan

    profile = DELEGATE_AGENT_ROSTER.get("planner", {})
    model: str = profile.get("model", "gemma-4-31b-it")
    # Tell the planner what tools the orchestrator actually has, so the plan is
    # grounded in real capabilities (read the live registry off the bound agent).
    tools_roster = _format_tool_roster(_agent_ref)
    # Live, in-order: surface the consult BEFORE make_plan blocks (up to 15s).
    _emit_signal("consulting", "Consulting planner sub-agent…")
    try:
        plan = await asyncio.wait_for(
            make_plan(objective, ORCHESTRATION_PLANNER_PERSONA, model, tools_roster),
            timeout=15.0,
        )
    except Exception as exc:
        logger.warning(f"plan_tool failed: {exc}")
        return ""

    # Store on the agent so the TUI can display it.
    agent = _agent_ref
    if agent is not None:
        agent.last_plan = plan  # type: ignore[attr-defined]
        agent.last_plan_objective = objective  # type: ignore[attr-defined]
    # Render the plan IN ORDER now — not retroactively after the turn ends.
    if plan:
        _emit_signal("plan", plan)
    return plan
