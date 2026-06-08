from __future__ import annotations

from unittest.mock import patch

import pytest

from supporter.tools.browser import support, tool
from supporter.tools.browser.core import BrowseRequest


async def test_browse_action_emits_elapsed_ms_at_debug() -> None:
    """Every browse action logs its total wall-clock at DEBUG."""

    async def handler(req: BrowseRequest) -> str:
        return "ok"

    async def record(_req: BrowseRequest, _result: str) -> None:
        return None

    with (
        patch.object(tool, "HANDLERS", {"click": handler}),
        patch.object(tool, "_record_step", record),
        patch("supporter.tools.browser.tool.logger.debug") as mock_debug,
    ):
        result = await tool.browse("click", ref="e1")

    assert result == "ok"
    timing_calls = [
        call
        for call in mock_debug.call_args_list
        if "action=click" in str(call) and "elapsed_ms" in str(call)
    ]
    assert timing_calls, (
        f"No browse action timing log emitted; got {mock_debug.call_args_list}"
    )


async def test_capture_emits_snapshot_elapsed_ms_at_debug() -> None:
    """The aria-snapshot path logs its own elapsed_ms separately."""
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "has_baseline", lambda k: False)
    monkeypatch.setattr(snap_mod, "remember_snapshot", lambda k, v: None)
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)
    monkeypatch.setattr(snap_mod, "clean_snapshot", lambda s, u: s)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return '- document [ref=e1]:\n  - button "OK" [ref=e2]'

    req = BrowseRequest(action="snapshot", compact=False)
    try:
        with patch("supporter.tools.browser.support.logger.debug") as mock_debug:
            result = await support._capture(FakePage(), req, force_full=True, label="")
        assert "ref=e1" in result
        snap_calls = [
            call
            for call in mock_debug.call_args_list
            if "snapshot" in str(call) and "elapsed_ms" in str(call)
        ]
        assert snap_calls, (
            f"No snapshot timing log emitted; got {mock_debug.call_args_list}"
        )
    finally:
        monkeypatch.undo()
