from __future__ import annotations

import pytest

from supporter.tools.browser.core import BrowseRequest
from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    from supporter.tools.browser import snapshot

    snapshot._LAST_SNAPSHOT.clear()


async def test_resolve_target_frame_selector_error(fake_session: FakeSession) -> None:
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
    import supporter.tools.browser.session as session_module

    original_frame_selector2 = session_module.active_frame_selector

    def mock_frame_selector2() -> str | None:
        return "iframe#test"

    session_module.active_frame_selector = mock_frame_selector2

    try:
        from supporter.tools.browser.support import _record_locator

        req = BrowseRequest(action="click", ref="e2", selector="")
        result = _record_locator(fake_session.page, req)
        assert result is None
    finally:
        session_module.active_frame_selector = original_frame_selector2


async def test_record_locator_no_ref_path(fake_session: FakeSession) -> None:
    from supporter.tools.browser.support import _record_locator

    req = BrowseRequest(action="click", ref="", selector="")
    result = _record_locator(fake_session.page, req)
    assert result is None


async def test_confirm_or_block_callback_none(fake_session: FakeSession) -> None:
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
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    guardrails.browse_confirmation_callback = None

    try:
        result = await browse("eval", script="console.log('test')")
        assert "Error: browser confirmation not wired. Action cancelled." in result
    finally:
        guardrails.browse_confirmation_callback = original_callback


async def test_confirm_script_full_confirmation(fake_session: FakeSession) -> None:
    fake_session.confirm.allow = False

    result = await browse("eval", script="console.log('test')")
    assert "Error: action cancelled." in result


async def test_confirm_always_callback_none(fake_session: FakeSession) -> None:
    import supporter.tools.browser.guardrails as guardrails

    original_callback = guardrails.browse_confirmation_callback
    guardrails.browse_confirmation_callback = None

    try:
        result = await browse("storage", key="test")
        assert "Error: browser confirmation not wired. Action cancelled." in result
    finally:
        guardrails.browse_confirmation_callback = original_callback


async def test_render_script_result_json_failure(fake_session: FakeSession) -> None:
    from supporter.tools.browser.support import _render_script_result

    class Unserializable:
        def __str__(self) -> str:
            return "test"

    result = _render_script_result(Unserializable())
    assert "test" in result
    assert len(result) <= 2000
