"""Stub module — plan tool removed; planning is now a delegate sub-agent role.

The TUI still imports bind_agent / clear_agent / set_plan_signal_callback /
clear_plan_signal_callback for backwards compatibility.  They are harmless
no-ops: the plan now arrives inside the delegation capsule.
"""

from __future__ import annotations

from collections.abc import Callable


def bind_agent(agent: object) -> None:  # pragma: no cover
    """No-op — kept for TUI import compat."""


def clear_agent() -> None:  # pragma: no cover
    """No-op — kept for TUI import compat."""


def set_plan_signal_callback(  # pragma: no cover
    cb: Callable[[str, str], None],
) -> None:
    """No-op — kept for TUI import compat."""


def clear_plan_signal_callback() -> None:  # pragma: no cover
    """No-op — kept for TUI import compat."""
