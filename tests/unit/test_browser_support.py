from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.tools.browser import guardrails, session, support
from supporter.tools.browser.core import BrowseRequest


async def test_require_ref_returns_locator_on_success() -> None:
    class FakePage:
        def locator(self, sel: str) -> object:
            return FakeLocator()

    class FakeLocator:
        async def wait_for(self, **kw: object) -> None:
            return None

    result = await support._require_ref(FakePage(), "e1")
    assert result is not None


async def test_require_ref_returns_none_on_playwright_timeout() -> None:
    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    class FailingLocator:
        async def wait_for(self, **kw: object) -> None:
            raise PlaywrightTimeoutError("timeout")

    class FakePage:
        def locator(self, sel: str) -> object:
            return FailingLocator()

    result = await support._require_ref(FakePage(), "e99")
    assert result is None


async def test_resolve_target_frame_with_selector_returns_locator() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(session, "active_frame_selector", lambda: "iframe#main")
    try:
        fake_locator = object()
        frame_loc = type(
            "FL",
            (),
            {"locator": lambda self, s: type("L", (), {"first": fake_locator})()},
        )()

        class FakePage:
            def frame_locator(self, sel: str) -> object:
                return frame_loc

        req = BrowseRequest(action="click", ref="e1", selector="#btn")
        locator, err = await support._resolve_target(FakePage(), req)
        assert err is None
        assert locator is fake_locator
    finally:
        monkeypatch.undo()


async def test_resolve_target_ref_not_found_returns_error() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(session, "active_frame_selector", lambda: None)

    from patchright.async_api import TimeoutError as PlaywrightTimeoutError

    class TimeoutLocator:
        async def wait_for(self, **kw: object) -> None:
            raise PlaywrightTimeoutError("not visible")

    class FakePage:
        def locator(self, sel: str) -> object:
            return TimeoutLocator()

    req = BrowseRequest(action="click", ref="e99")
    locator, err = await support._resolve_target(FakePage(), req)
    assert locator is None
    assert err is not None and "ref e99 not found" in err
    monkeypatch.undo()


async def test_record_locator_frame_with_selector_returns_locator() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(session, "active_frame_selector", lambda: "iframe#t")
    try:
        fake_loc = object()

        class FakePage:
            def frame_locator(self, sel: str) -> object:
                return type(
                    "FL",
                    (),
                    {"locator": lambda self, s: type("L", (), {"first": fake_loc})()},
                )()

        req = BrowseRequest(action="click", ref="", selector="#btn")
        result = support._record_locator(FakePage(), req)
        assert result is fake_loc
    finally:
        monkeypatch.undo()


async def test_confirm_script_bypasses_when_no_confirmation_needed() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails,
        "needs_confirmation",
        lambda action, role, name, host: False,
    )

    class FakePage:
        url = "https://trusted.test/"

    result = await support._confirm_script(FakePage(), "alert(1)")
    assert result is None
    monkeypatch.undo()


async def test_confirm_script_truncates_long_script() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails,
        "needs_confirmation",
        lambda action, role, name, host: True,
    )
    captured_detail: list[str] = []

    async def fake_cb(title: str, detail: str) -> bool:
        captured_detail.append(detail)
        return True

    monkeypatch.setattr(guardrails, "browse_confirmation_callback", fake_cb)

    class FakePage:
        url = "https://eval.test/"

    long_script = "x" * 600
    result = await support._confirm_script(FakePage(), long_script)
    assert result is None
    assert len(captured_detail) == 1
    assert "…(+100 more chars)" in captured_detail[0]
    monkeypatch.undo()


async def test_confirm_script_accepted_returns_none() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails,
        "needs_confirmation",
        lambda action, role, name, host: True,
    )
    monkeypatch.setattr(
        guardrails,
        "browse_confirmation_callback",
        AsyncMock(return_value=True),
    )

    class FakePage:
        url = "https://eval.test/"

    result = await support._confirm_script(FakePage(), "console.log(1)")
    assert result is None
    monkeypatch.undo()


async def test_confirm_always_returns_none_when_accepted() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails,
        "browse_confirmation_callback",
        AsyncMock(return_value=True),
    )
    result = await support._confirm_always("Title", "Detail")
    assert result is None
    monkeypatch.undo()


async def test_confirm_always_rejected_returns_error() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails,
        "browse_confirmation_callback",
        AsyncMock(return_value=False),
    )
    result = await support._confirm_always("Title", "Detail")
    assert result is not None
    assert "action cancelled" in result
    monkeypatch.undo()


async def test_live_refs_snapshot_returns_cleaned_snapshot() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(support, "_page_key", lambda page: "https://test.test/")

    class FakePage:
        async def aria_snapshot(self, **kw: object) -> str:
            return '- document [ref=e1]:\n  - button "OK" [ref=e2]'

    from supporter.tools.browser import snapshot as snap_mod

    original = snap_mod.clean_snapshot

    def fake_clean(snap: str, url: str) -> str:
        return snap

    monkeypatch.setattr(snap_mod, "clean_snapshot", fake_clean)
    try:
        result = await support._live_refs_snapshot(FakePage())
        assert "button" in result
    finally:
        monkeypatch.setattr(snap_mod, "clean_snapshot", original)
        monkeypatch.undo()


async def test_resolve_role_and_name_success() -> None:
    class FakeLocator:
        async def evaluate(self, script: str) -> list[str]:
            return ["button", "Submit"]

    role, name = await support._resolve_role_and_name(FakeLocator(), "e1")
    assert role == "button"
    assert name == "Submit"


async def test_resolve_role_and_name_empty_role() -> None:
    class FakeLocator:
        async def evaluate(self, script: str) -> list[str]:
            return ["", "some name"]

    role, name = await support._resolve_role_and_name(FakeLocator(), "e1")
    assert role == ""
    assert name == "some name"


async def test_render_snapshot_compact_non_empty() -> None:
    tree = (
        '- document [ref=e1]:\n  - button "OK" [ref=e2]\n  - link "Learn more" [ref=e3]'
    )
    req = BrowseRequest(action="snapshot", compact=True)
    result = support._render_snapshot(tree, req, " after click", "https://x.test/")
    assert "interactive elements after click" in result
    assert "OK" in result


async def test_render_snapshot_compact_empty_returns_empty_page() -> None:
    req = BrowseRequest(action="snapshot", compact=True)
    result = support._render_snapshot(
        "- document [ref=e1]:", req, "", "https://x.test/"
    )
    assert result == "(empty page)"


async def test_render_snapshot_non_compact_returns_cleaned() -> None:
    req = BrowseRequest(action="snapshot", compact=False)
    result = support._render_snapshot(
        '- document [ref=e1]:\n  - button "OK" [ref=e2]',
        req,
        "",
        "https://x.test/",
    )
    assert "OK" in result
    assert "interactive elements" not in result


async def test_page_key_returns_url() -> None:
    class P:
        url = "https://x.test/p"

    assert support._page_key(P()) == "https://x.test/p"


async def test_page_key_exception_returns_empty() -> None:
    class P:
        @property
        def url(self) -> str:
            raise RuntimeError("detached")

    assert support._page_key(P()) == ""


async def test_page_baseline_key_exception_returns_empty() -> None:
    assert support._page_baseline_key(123) == ""


async def test_diff_header_long_key_truncates() -> None:
    key = "https://x.test/" + "a" * 100
    header = support._diff_header(key)
    assert header.endswith("...):")
    inner = header[len("diff vs last snapshot (") : -len("):")]
    assert len(inner) == 60


async def test_diff_header_empty() -> None:
    assert support._diff_header("") == "diff vs last snapshot:"


async def test_render_script_result_truncates_long_output() -> None:
    out = support._render_script_result("z" * 5000)
    assert out.endswith("...(truncated)")
    assert len(out) == 2000 + len("...(truncated)")


async def test_validate_path_or_error_rejects_traversal() -> None:
    resolved, err = support._validate_path_or_error("../../etc/passwd")
    assert resolved is None
    assert err is not None
    assert err.startswith("Error: ")


async def test_validate_path_or_error_accepts_valid() -> None:
    resolved, err = support._validate_path_or_error("notes.txt")
    assert err is None
    assert resolved is not None


async def test_confirm_or_block_no_confirmation_needed() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails, "needs_confirmation", lambda action, role, name, host: False
    )
    result = await support._confirm_or_block(
        MagicMock(), BrowseRequest(action="click", ref="e1"), None
    )
    assert result is None
    monkeypatch.undo()


async def test_confirm_or_block_callback_none() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails, "needs_confirmation", lambda action, role, name, host: True
    )
    monkeypatch.setattr(guardrails, "browse_confirmation_callback", None)
    result = await support._confirm_or_block(
        MagicMock(), BrowseRequest(action="click", ref="e1"), None
    )
    assert result is not None and "confirmation not wired" in result
    monkeypatch.undo()


async def test_confirm_or_block_accepted() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails, "needs_confirmation", lambda action, role, name, host: True
    )
    monkeypatch.setattr(
        guardrails, "browse_confirmation_callback", AsyncMock(return_value=True)
    )
    result = await support._confirm_or_block(
        MagicMock(), BrowseRequest(action="click", ref="e1"), None
    )
    assert result is None
    monkeypatch.undo()


async def test_confirm_or_block_rejected() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails, "needs_confirmation", lambda action, role, name, host: True
    )
    monkeypatch.setattr(
        guardrails, "browse_confirmation_callback", AsyncMock(return_value=False)
    )
    result = await support._confirm_or_block(
        MagicMock(), BrowseRequest(action="click", ref="e1"), None
    )
    assert result is not None and "action cancelled" in result
    monkeypatch.undo()


async def test_confirm_or_block_type_includes_text_length() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        guardrails, "needs_confirmation", lambda action, role, name, host: True
    )
    captured: list[str] = []

    async def fake_cb(title: str, detail: str) -> bool:
        captured.append(detail)
        return True

    monkeypatch.setattr(guardrails, "browse_confirmation_callback", fake_cb)
    await support._confirm_or_block(
        MagicMock(), BrowseRequest(action="type", ref="e1", text="hello"), None
    )
    assert len(captured) == 1
    assert "Text length: 5 chars" in captured[0]
    monkeypatch.undo()


async def test_capture_force_full_returns_cleaned_snapshot() -> None:
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "has_baseline", lambda k: False)
    monkeypatch.setattr(snap_mod, "remember_snapshot", lambda k, v: None)
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return '- document [ref=e1]:\n  - button "OK" [ref=e2]'

    req = BrowseRequest(action="snapshot", compact=False)
    result = await support._capture(FakePage(), req, force_full=True, label=" test")
    assert "OK" in result
    monkeypatch.undo()


async def test_snapshot_text_uses_diff_when_baseline_exists() -> None:
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "has_baseline", lambda k: True)
    monkeypatch.setattr(snap_mod, "clean_snapshot", lambda snap, url: snap)
    monkeypatch.setattr(snap_mod, "diff_snapshot", lambda k, c: "(no changes)")
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return "- document [ref=e1]:"

    req = BrowseRequest(action="snapshot", compact=False)
    result = await support._snapshot_text(FakePage(), req)
    assert result == "(no changes)"
    monkeypatch.undo()


async def test_snapshot_full_returns_full_snapshot() -> None:
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "clean_snapshot", lambda snap, url: snap)
    monkeypatch.setattr(snap_mod, "remember_snapshot", lambda k, v: None)
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return '- document [ref=e1]:\n  - link "Go" [ref=e2]'

    req = BrowseRequest(action="snapshot", compact=False)
    result = await support._snapshot_full(FakePage(), req, label=" full")
    assert "Go" in result
    monkeypatch.undo()


async def test_post_action_snapshot_waits_and_captures() -> None:
    from supporter.tools.browser import humanize
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "has_baseline", lambda k: False)
    monkeypatch.setattr(snap_mod, "remember_snapshot", lambda k, v: None)
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)
    monkeypatch.setattr(humanize, "jitter_ms", lambda *a: 100)

    slept: list[float] = []

    async def fake_sleep(ms: float) -> None:
        slept.append(ms)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return "- document [ref=e1]:"

        async def wait_for_timeout(self, ms: float) -> None:
            slept.append(ms)

        async def wait_for_load_state(self, state: str, **kw: object) -> None:
            slept.append(0.0)

    req = BrowseRequest(action="click", compact=False)
    await support._post_action_snapshot(FakePage(), req)
    assert len(slept) >= 1
    monkeypatch.undo()


async def test_diff_text_returns_diff() -> None:
    from supporter.tools.browser import snapshot as snap_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(snap_mod, "clean_snapshot", lambda snap, url: snap)
    monkeypatch.setattr(snap_mod, "diff_snapshot", lambda k, c: "(no changes)")
    monkeypatch.setattr(snap_mod, "log_snapshot", lambda a, r: None)

    class FakePage:
        url = "https://x.test/"

        async def aria_snapshot(self, **kw: object) -> str:
            return "- document [ref=e1]:"

    req = BrowseRequest(action="diff", compact=False)
    result = await support._diff_text(FakePage(), req)
    assert result == "(no changes)"
    monkeypatch.undo()


async def test_effective_fast_allowlisted() -> None:
    from unittest.mock import patch

    with (
        patch.object(support, "_page_host", AsyncMock(return_value="google.com")),
        patch.object(support.config, "browser_debug_overlay", False),
    ):
        assert await support._effective_fast(object()) is True


async def test_effective_fast_non_allowlisted() -> None:
    from unittest.mock import patch

    with patch.object(support, "_page_host", AsyncMock(return_value="example.com")):
        assert await support._effective_fast(object()) is False


async def test_effective_fast_forced_off_by_debug_overlay() -> None:
    from unittest.mock import patch

    with (
        patch.object(support, "_page_host", AsyncMock(return_value="google.com")),
        patch.object(support.config, "browser_debug_overlay", True),
    ):
        assert await support._effective_fast(object()) is False


async def test_record_locator_no_frame_no_ref() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(session, "active_frame_selector", lambda: None)
    req = BrowseRequest(action="click", ref="")
    result = support._record_locator(MagicMock(), req)
    assert result is None
    monkeypatch.undo()
