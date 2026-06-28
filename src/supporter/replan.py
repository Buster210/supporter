"""Plan → Implement → Verify → Replan loop (G2).

When a non-trivial task runs:
1. PLAN   -- ask the planner to produce a plan.
2. IMPLEMENT -- execute per the plan.
3. VERIFY -- run checks; if all pass, done.
4. REPLAN (if failed) -- feed failure context back to planner, try again.

Loop bounded by max_cycles; each cycle increments the attempt counter
and includes prior failure context so the planner can refine.
"""

from __future__ import annotations

from .decision_log import log_decision

__all__ = ["ReplanContext", "format_replan_prompt"]


class ReplanContext:
    """Tracks state across replan cycles."""

    def __init__(self, objective: str, max_cycles: int = 3) -> None:
        self.objective = objective
        self.max_cycles = max_cycles
        self.cycle = 0
        self.plan = ""
        self.last_result = ""
        self.failures: list[str] = []

    def next_cycle(self) -> bool:
        """Advance to the next cycle. Return True if within budget."""
        if self.cycle >= self.max_cycles:
            return False
        self.cycle += 1
        return True

    def record_failure(self, reason: str) -> None:
        """Record why verification failed."""
        self.failures.append(reason)
        log_decision(
            site="replan.failure",
            chosen="record",
            options=("record", "discard"),
            reason=reason,
            correlation_id=f"replan_cycle_{self.cycle}",
        )

    def format_replan_prompt_context(self) -> str:
        """Format context for the replan prompt (used by caller)."""
        return format_replan_prompt(
            self.objective, self.plan, self.last_result, self.failures
        )


def format_replan_prompt(
    objective: str, plan: str, result: str, failures: list[str]
) -> str:
    """Format a replan prompt with objective, plan, result, and failure context.

    ponytail: Failure context is concatenated directly; richer failure parsing
    could be added when the planner's feedback loop proves insufficient.
    """
    parts = [f"OBJECTIVE:\n{objective}\n"]

    if plan:
        parts.append(f"PREVIOUS PLAN:\n{plan}\n")

    if result:
        parts.append(f"IMPLEMENTATION RESULT:\n{result}\n")

    if failures:
        failure_text = "\n".join(f"- {f}" for f in failures)
        parts.append(f"VERIFICATION FAILURES:\n{failure_text}\n")

    parts.append(
        "Please revise the plan to address the failures and produce a NEW plan."
    )

    return "".join(parts)
