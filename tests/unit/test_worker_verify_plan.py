"""Tests for supporter.worker.verify_plan (G10 self-verify loop)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from supporter import worker


async def test_verify_plan_done() -> None:
    """VERDICT: DONE -> returns (True, reason)."""
    fake_gen = SimpleNamespace(text="VERDICT: DONE\nall 50 items gathered")
    fake_provider = SimpleNamespace(generate=AsyncMock(return_value=fake_gen))

    with patch("supporter.worker.get_provider", return_value=fake_provider):
        ok, reason = await worker.verify_plan(
            "collect 50 posts", "PLAN BODY", "RESULT BODY", "gemma-4-31b-it"
        )

    assert ok is True
    assert reason == "all 50 items gathered"


async def test_verify_plan_not_done() -> None:
    """VERDICT: NOT_DONE -> returns (False, reason)."""
    fake_gen = SimpleNamespace(text="VERDICT: NOT_DONE\nonly 12 of 50 posts collected")
    fake_provider = SimpleNamespace(generate=AsyncMock(return_value=fake_gen))

    with patch("supporter.worker.get_provider", return_value=fake_provider):
        ok, reason = await worker.verify_plan(
            "collect 50 posts", "PLAN BODY", "RESULT BODY", "gemma-4-31b-it"
        )

    assert ok is False
    assert reason == "only 12 of 50 posts collected"


async def test_verify_plan_fail_open_on_exception() -> None:
    """provider.generate raises -> fail-open: returns (True, ...), never raises."""
    fake_provider = SimpleNamespace(
        generate=AsyncMock(side_effect=RuntimeError("network error"))
    )

    with patch("supporter.worker.get_provider", return_value=fake_provider):
        ok, reason = await worker.verify_plan(
            "collect 50 posts", "PLAN BODY", "RESULT BODY", "gemma-4-31b-it"
        )

    assert ok is True
    assert "network error" in reason
