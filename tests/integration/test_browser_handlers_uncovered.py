from __future__ import annotations

import pytest

from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    from supporter.tools.browser import snapshot

    snapshot._LAST_SNAPSHOT.clear()


async def test_extract_handler_ref_in_frame_error(fake_session: FakeSession) -> None:
    # Set up frame selector state
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
    # Set confirm recorder to deny
    fake_session.confirm.allow = False

    result = await browse("eval", script="console.log('test')")
    assert "Error: action cancelled." in result


async def test_frame_handler_clear_frame(fake_session: FakeSession) -> None:
    result = await browse("frame")
    assert "Frame cleared (acting on the top page now)." in result


async def test_tab_handler_negative_index(fake_session: FakeSession) -> None:
    # This tests the closetab path with negative index
    result = await browse("closetab", index=-1)
    # Should not error and should close the last tab
    assert "Closed the last tab" in result or "no tabs remain open" in result


async def test_close_handler_browser_already_closed(fake_session: FakeSession) -> None:
    # Mock is_active to return False
    import supporter.tools.browser.session as session_module

    original_is_active_fn = session_module.is_active

    def mock_is_active() -> bool:
        return False

    session_module.is_active = mock_is_active

    try:
        result = await browse("close")
        assert "Browser already closed." in result
    finally:
        session_module.is_active = original_is_active_fn


async def test_close_handler_pinned_open(fake_session: FakeSession) -> None:
    # Mock pinned_open to return True
    import supporter.tools.browser.session as session_module

    original_pinned_open_fn = session_module.pinned_open

    def mock_pinned_open() -> bool:
        return True

    session_module.pinned_open = mock_pinned_open

    try:
        result = await browse("close")
        assert "Browser left open (you chose to keep it open)." in result
    finally:
        session_module.pinned_open = original_pinned_open_fn


async def test_close_handler_confirmation_callback_none(
    fake_session: FakeSession,
) -> None:
    # Temporarily remove the callback
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    guardrails.browse_confirmation_callback = None

    try:
        result = await browse("close")
        assert "Error: browser confirmation not wired." in result
    finally:
        guardrails.browse_confirmation_callback = original_callback


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
