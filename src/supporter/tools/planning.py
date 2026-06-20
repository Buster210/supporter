"""Orchestrator-only planning tool."""

from __future__ import annotations

import asyncio

from ..logger import logger

# Module-level agent reference, bound before streaming so the handler can
# store the plan on the agent for TUI display.  Set via bind_agent().
_agent_ref: object = None  # ChatAgent | None, avoided to skip import.


def bind_agent(agent: object) -> None:
    """Bind the current agent so plan_tool can store results on it."""
    global _agent_ref
    _agent_ref = agent


def clear_agent() -> None:
    """Unbind after streaming completes."""
    global _agent_ref
    _agent_ref = None


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
    try:
        plan = await asyncio.wait_for(
            make_plan(objective, ORCHESTRATION_PLANNER_PERSONA, model),
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
    return plan
