from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from supporter.config import config
from supporter.tools import _resolve_path
from supporter.tools.browser import snapshot
from supporter.tools.browser.tool import browse

from .conftest import FakeSession

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    snapshot._LAST_SNAPSHOT.clear()


async def test_snapshot_first_sight_returns_full_tree(
    fake_session: FakeSession,
) -> None:
    result = await browse("snapshot")

    # First capture of a page has no baseline, so the full cleaned tree comes
    # back (document ref stripped, interactive button ref kept).
    assert 'button "OK" [ref=e2]' in result
    assert "[ref=e1]" not in result


async def test_snapshot_second_sight_reports_no_changes(
    fake_session: FakeSession,
) -> None:
    await browse("snapshot")

    result = await browse("snapshot")

    assert result == "(no changes since last snapshot)"


async def test_diff_without_baseline_announces_baseline_stored(
    fake_session: FakeSession,
) -> None:
    result = await browse("diff")

    assert result == "(no previous snapshot to diff against; baseline stored)"


async def test_diff_after_change_shows_the_delta(
    fake_session: FakeSession,
) -> None:
    await browse("diff")  # establishes the baseline
    fake_session.page.aria_text = (
        '- document [ref=e1]:\n  - button "OK" [ref=e2]\n  - link "Next" [ref=e3]'
    )

    result = await browse("diff")

    # The added line carries a unified-diff "+" prefix; context lines (space
    # prefix) may precede it, so assert the added line is present, not position.
    assert '+  - link "Next" [ref=e3]' in result


async def test_frame_clear_resets_to_top_page(fake_session: FakeSession) -> None:
    result = await browse("frame")

    assert result == "Frame cleared (acting on the top page now)."


async def test_frame_drill_returns_frame_aria(fake_session: FakeSession) -> None:
    result = await browse("frame", selector="iframe#inner")

    assert "Drilled into frame 'iframe#inner':" in result
    assert 'button "OK"' in result


async def test_screenshot_saves_file_and_reports_dims(
    fake_session: FakeSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_path.cache_clear()
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])

    result = await browse("screenshot", stamp="shot")

    saved = tmp_path / ".supporter" / "screenshots" / "shot.png"
    assert saved.exists()
    # The fake yields a 12-byte PNG (8-byte signature + b"FAKE").
    assert result == f"Screenshot saved to {saved} (1280x800, 12 bytes)."
    # The image sink (a Live session's view) receives the same bytes.
    assert fake_session.image_sink.images == [(b"\x89PNG\r\n\x1a\nFAKE", "image/png")]
