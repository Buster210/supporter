from __future__ import annotations

from typing import Any
from unittest.mock import patch

from supporter.tools.browser import handlers
from supporter.tools.browser.core import BrowseRequest


class _FakePage:
    pass


class _FakeLocator:
    def __init__(
        self, box: dict[str, float] | None = None, *, raises: bool = False
    ) -> None:
        self._box = box
        self._raises = raises

    async def bounding_box(self) -> dict[str, float] | None:
        if self._raises:
            raise RuntimeError("no box")
        return self._box


async def test_overlay_mark_flag_off_no_draw() -> None:
    calls: list[Any] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        calls.append(("click", x, y))

    async def fake_move(_page: Any, x: float, y: float) -> None:
        calls.append(("move", x, y))

    with (
        patch.object(handlers.config, "browser_debug_overlay", False),
        patch.object(handlers.debug_overlay, "overlay_click", fake_click),
        patch.object(handlers.debug_overlay, "overlay_move", fake_move),
    ):
        locator = _FakeLocator(box={"x": 10, "y": 20, "width": 40, "height": 60})
        await handlers._overlay_mark(_FakePage(), locator, "click")
        await handlers._overlay_mark(_FakePage(), locator, "move")

    assert calls == []


async def test_overlay_mark_flag_on_click_fires_at_center() -> None:
    clicks: list[tuple[float, float]] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        clicks.append((x, y))

    async def fake_move(_page: Any, x: float, y: float) -> None:
        pass

    locator = _FakeLocator(box={"x": 10, "y": 20, "width": 40, "height": 60})
    with (
        patch.object(handlers.config, "browser_debug_overlay", True),
        patch.object(handlers.debug_overlay, "overlay_click", fake_click),
        patch.object(handlers.debug_overlay, "overlay_move", fake_move),
    ):
        await handlers._overlay_mark(_FakePage(), locator, "click")

    assert clicks == [(30.0, 50.0)]


async def test_overlay_mark_flag_on_move_fires_at_center() -> None:
    moves: list[tuple[float, float]] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        pass

    async def fake_move(_page: Any, x: float, y: float) -> None:
        moves.append((x, y))

    locator = _FakeLocator(box={"x": 10, "y": 20, "width": 40, "height": 60})
    with (
        patch.object(handlers.config, "browser_debug_overlay", True),
        patch.object(handlers.debug_overlay, "overlay_click", fake_click),
        patch.object(handlers.debug_overlay, "overlay_move", fake_move),
    ):
        await handlers._overlay_mark(_FakePage(), locator, "move")

    assert moves == [(30.0, 50.0)]


async def test_overlay_mark_flag_on_box_none_no_draw() -> None:
    calls: list[Any] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        calls.append(x)

    async def fake_move(_page: Any, x: float, y: float) -> None:
        calls.append(x)

    locator = _FakeLocator(box=None)
    with (
        patch.object(handlers.config, "browser_debug_overlay", True),
        patch.object(handlers.debug_overlay, "overlay_click", fake_click),
        patch.object(handlers.debug_overlay, "overlay_move", fake_move),
    ):
        await handlers._overlay_mark(_FakePage(), locator, "click")

    assert calls == []


async def test_overlay_mark_flag_on_bounding_box_raises_no_draw() -> None:
    calls: list[Any] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        calls.append(x)

    async def fake_move(_page: Any, x: float, y: float) -> None:
        calls.append(x)

    locator = _FakeLocator(raises=True)
    with (
        patch.object(handlers.config, "browser_debug_overlay", True),
        patch.object(handlers.debug_overlay, "overlay_click", fake_click),
        patch.object(handlers.debug_overlay, "overlay_move", fake_move),
    ):
        await handlers._overlay_mark(_FakePage(), locator, "click")

    assert calls == []


async def test_newtab_injects_overlay_when_flag_on() -> None:
    injected: list[Any] = []

    class _FakeTab:
        url = "about:blank"

        async def bring_to_front(self) -> None: ...

    tab = _FakeTab()

    async def fake_session_parts() -> tuple[Any, Any, Any]:
        return None, None, None

    async def fake_inject(page: Any) -> None:
        injected.append(page)

    async def fake_snapshot_full(page: Any, req: Any) -> str:
        return "ok"

    with (
        patch.object(handlers, "_session_parts", fake_session_parts),
        patch.object(handlers.session, "list_pages", return_value=[tab]),
        patch.object(handlers.session, "set_active"),
        patch.object(handlers.config, "browser_debug_overlay", True),
        patch.object(handlers.debug_overlay, "inject_overlay", fake_inject),
        patch.object(handlers, "_snapshot_full", fake_snapshot_full),
    ):
        await handlers._handle_newtab(BrowseRequest(action="newtab"))

    assert injected == [tab]


async def test_newtab_skips_overlay_when_flag_off() -> None:
    injected: list[Any] = []

    class _FakeTab:
        url = "about:blank"

        async def bring_to_front(self) -> None: ...

    tab = _FakeTab()

    async def fake_session_parts() -> tuple[Any, Any, Any]:
        return None, None, None

    async def fake_inject(page: Any) -> None:
        injected.append(page)

    async def fake_snapshot_full(page: Any, req: Any) -> str:
        return "ok"

    with (
        patch.object(handlers, "_session_parts", fake_session_parts),
        patch.object(handlers.session, "list_pages", return_value=[tab]),
        patch.object(handlers.session, "set_active"),
        patch.object(handlers.config, "browser_debug_overlay", False),
        patch.object(handlers.debug_overlay, "inject_overlay", fake_inject),
        patch.object(handlers, "_snapshot_full", fake_snapshot_full),
    ):
        await handlers._handle_newtab(BrowseRequest(action="newtab"))

    assert injected == []
