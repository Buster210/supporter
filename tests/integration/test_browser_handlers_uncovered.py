from __future__ import annotations

import pytest

from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    from supporter.tools.browser import snapshot

    snapshot._LAST_SNAPSHOT.clear()


async def test_extract_handler_ref_in_frame_error(fake_session: FakeSession) -> None:
    import supporter.tools.browser.session as session_module

    original_frame_selector = session_module.active_frame_selector

    def mock_frame_selector() -> str | None:
        return "iframe#test"

    session_module.active_frame_selector = mock_frame_selector

    try:
        result = await browse("extract", ref="e2")
        assert (
            "Error: inside a frame, extract needs a CSS 'selector', not a ref."
            in result
        )
    finally:
        session_module.active_frame_selector = original_frame_selector


async def test_extract_handler_no_ref_or_selector(fake_session: FakeSession) -> None:
    result = await browse("extract")
    assert "Error: extract needs a 'ref' or a CSS 'selector'." in result


async def test_eval_handler_no_script(fake_session: FakeSession) -> None:
    result = await browse("eval")
    assert "Error: 'script' is required for eval." in result


async def test_eval_handler_confirmation_denied(fake_session: FakeSession) -> None:
    fake_session.confirm.allow = False

    result = await browse("eval", script="console.log('test')")
    assert "Error: action cancelled." in result


async def test_frame_handler_clear_frame(fake_session: FakeSession) -> None:
    result = await browse("frame")
    assert "Frame cleared (acting on the top page now)." in result


async def test_tab_handler_negative_index(fake_session: FakeSession) -> None:
    result = await browse("closetab", index=-1)
    assert "Closed the last tab" in result or "no tabs remain open" in result


async def test_close_handler_rejected_as_orchestrator_only(
    fake_session: FakeSession,
) -> None:
    result = await browse("close")
    assert "orchestrator-only" in result


async def test_closenow_handler_rejected_as_orchestrator_only(
    fake_session: FakeSession,
) -> None:
    result = await browse("closenow")
    assert "orchestrator-only" in result


async def test_upload_handler_no_ref(fake_session: FakeSession) -> None:
    result = await browse("upload")
    assert "Error: 'ref' is required for upload. Get a snapshot first." in result


async def test_upload_handler_no_path(fake_session: FakeSession) -> None:
    result = await browse("upload", ref="e2")
    assert "Error: 'path' is required for upload (the file to attach)." in result


async def test_download_handler_no_ref(fake_session: FakeSession) -> None:
    result = await browse("download")
    assert "Error: 'ref' is required for download. Get a snapshot first." in result


async def test_download_handler_no_path(fake_session: FakeSession) -> None:
    result = await browse("download", ref="e2")
    assert "Error: 'path' is required for download (where to save the file)." in result
