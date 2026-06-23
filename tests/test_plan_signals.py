"""plan_tool removed — planning is now a delegate sub-agent role.

Stub tests for the backwards-compat signal functions in planning.py.
"""

from supporter.tools import planning


def test_bind_agent_is_noop() -> None:
    """bind_agent is a no-op stub — should not raise."""
    planning.bind_agent(object())


def test_clear_agent_is_noop() -> None:
    """clear_agent is a no-op stub — should not raise."""
    planning.clear_agent()


def test_set_plan_signal_callback_is_noop() -> None:
    """set_plan_signal_callback is a no-op stub — should not raise."""
    planning.set_plan_signal_callback(lambda kind, text: None)


def test_clear_plan_signal_callback_is_noop() -> None:
    """clear_plan_signal_callback is a no-op stub — should not raise."""
    planning.clear_plan_signal_callback()
