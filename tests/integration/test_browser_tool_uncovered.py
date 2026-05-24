from __future__ import annotations

import pytest

from supporter.tools.browser.tool import BrowseRequest, browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    from supporter.tools.browser import snapshot

    snapshot._LAST_SNAPSHOT.clear()


async def test_resolve_target_frame_selector_error(fake_session: FakeSession) -> None:
    # Set up frame selector state using monkeypatch from conftest
    import supporter.tools.browser.session as session_module

    original_frame_selector = session_module.active_frame_selector

    def mock_frame_selector() -> str | None:
        return "iframe#test"

    session_module.active_frame_selector = mock_frame_selector

    try:
        result = await browse("click", ref="e2")
        assert (
            "Error: inside a frame, click/type/hover needs a CSS 'selector', not a ref."
            in result
        )
    finally:
        session_module.active_frame_selector = original_frame_selector


async def test_resolve_target_no_ref_error(fake_session: FakeSession) -> None:
    result = await browse("click")
    assert (
        "Error: 'ref' is required for click/type. "
        "Get a snapshot first to find [ref=eN]." in result
    )


async def test_record_locator_frame_path(fake_session: FakeSession) -> None:
    # Set up frame selector state
    import supporter.tools.browser.session as session_module

    original_frame_selector2 = session_module.active_frame_selector

    def mock_frame_selector2() -> str | None:
        return "iframe#test"

    session_module.active_frame_selector = mock_frame_selector2

    try:
        # This should return None when no selector provided
        from supporter.tools.browser.tool import _record_locator

        req = BrowseRequest(action="click", ref="e2", selector="")
        result = _record_locator(fake_session.page, req)
        assert result is None
    finally:
        session_module.active_frame_selector = original_frame_selector2


async def test_record_locator_no_ref_path(fake_session: FakeSession) -> None:
    from supporter.tools.browser.tool import _record_locator

    req = BrowseRequest(action="click", ref="", selector="")
    result = _record_locator(fake_session.page, req)
    assert result is None


async def test_confirm_or_block_callback_none(fake_session: FakeSession) -> None:
    # _confirm_or_block (not _confirm_script) is called by click/type/press/
    # select/upload/download.  Use click on a non-sensitive host and force
    # needs_confirmation True so we land on the cb=None guard.
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    original_needs = guardrails.needs_confirmation

    guardrails.browse_confirmation_callback = None
    guardrails.needs_confirmation = lambda action, role, name, host: True

    try:
        result = await browse("click", ref="e2")
        assert "Error: browser confirmation not wired. Action cancelled." in result
    finally:
        guardrails.needs_confirmation = original_needs
        guardrails.browse_confirmation_callback = original_callback


async def test_confirm_script_callback_none(fake_session: FakeSession) -> None:
    # Temporarily remove the callback
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    guardrails.browse_confirmation_callback = None

    try:
        result = await browse("eval", script="console.log('test')")
        assert "Error: browser confirmation not wired. Action cancelled." in result
    finally:
        # Restore callback
        guardrails.browse_confirmation_callback = original_callback


async def test_confirm_script_full_confirmation(fake_session: FakeSession) -> None:
    # Set confirm recorder to deny
    fake_session.confirm.allow = False

    result = await browse("eval", script="console.log('test')")
    assert "Error: action cancelled." in result


async def test_confirm_always_callback_none(fake_session: FakeSession) -> None:
    # Temporarily remove the callback
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    guardrails.browse_confirmation_callback = None

    try:
        result = await browse("storage", key="test")
        assert "Error: browser confirmation not wired. Action cancelled." in result
    finally:
        # Restore callback
        guardrails.browse_confirmation_callback = original_callback


async def test_render_script_result_json_failure(fake_session: FakeSession) -> None:
    from supporter.tools.browser.tool import _render_script_result

    # Create a result that can't be JSON serialized
    class Unserializable:
        def __str__(self) -> str:
            return "test"

    result = _render_script_result(Unserializable())
    assert "test" in result
    assert len(result) <= 2000
