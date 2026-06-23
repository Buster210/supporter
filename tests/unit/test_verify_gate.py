"""Post-execution plan-verification gate (``_verify_planned_turn``).

A planned turn is verified once it finishes; an incomplete verdict mounts a
warning bubble, DONE is silent, and an unplanned turn skips verify entirely.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from supporter.tui import SupporterApp


class _FakeTarget:
    """Records mounted widgets."""

    def __init__(self) -> None:
        self.mounted: list[Any] = []

    async def mount(self, widget: Any) -> None:
        self.mounted.append(widget)


class _FakeChatView:
    def jump_to_bottom(self) -> None:
        pass


async def _run(objective: str, verdict: tuple[bool, str]) -> _FakeTarget:
    target = _FakeTarget()
    with patch(
        "supporter.worker.verify_plan", new=AsyncMock(return_value=verdict)
    ) as mock_verify:
        await SupporterApp._verify_planned_turn(
            object(),  # type: ignore[arg-type]  # self is unused beyond imports
            objective,
            "PLAN: do X",
            "the result text",
            target,  # type: ignore[arg-type]
            _FakeChatView(),  # type: ignore[arg-type]
        )
    _run.last_mock = mock_verify  # type: ignore[attr-defined]
    return target


@pytest.mark.asyncio
async def test_incomplete_verdict_mounts_warning() -> None:
    target = await _run("ship the feature", (False, "missing tests"))
    assert len(target.mounted) == 1
    assert "missing tests" in target.mounted[0].content
    assert "Verification" in target.mounted[0].content


@pytest.mark.asyncio
async def test_done_verdict_mounts_nothing() -> None:
    target = await _run("ship the feature", (True, "looks complete"))
    assert target.mounted == []


@pytest.mark.asyncio
async def test_unplanned_turn_skips_verify() -> None:
    target = await _run("", (False, "should not be reached"))
    assert target.mounted == []
    _run.last_mock.assert_not_called()  # type: ignore[attr-defined]
