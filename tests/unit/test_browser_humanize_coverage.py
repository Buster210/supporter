from __future__ import annotations

import random
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from supporter.tools.browser import humanize

if TYPE_CHECKING:
    from patchright.async_api import Locator, Mouse, Page


class _FakeMouse:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.wheels: list[tuple[int, int]] = []

    async def move(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    async def wheel(self, dx: int, dy: int) -> None:
        self.wheels.append((dx, dy))

    async def click(self, x: float, y: float, button: str = "left") -> None:
        pass


class _FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def press(self, key: str) -> None:
        self.events.append(("press", key))

    async def down(self, key: str) -> None:
        self.events.append(("down", key))

    async def up(self, key: str) -> None:
        self.events.append(("up", key))


class _FakeLocator:
    def __init__(self, box: dict[str, float] | None = None) -> None:
        self._box = box

    async def click(self) -> None:
        pass

    async def bounding_box(self) -> dict[str, float] | None:
        return self._box

    @property
    def first(self) -> _FakeLocator:
        return self

    def locator(self, sel: str) -> _FakeLocator:
        return self


class _FakePage:
    def __init__(
        self,
        *,
        geometry: dict[str, int] | None = None,
        box: dict[str, float] | None = None,
    ) -> None:
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.viewport_size: dict[str, int] | None = {"width": 1280, "height": 800}
        self._geometry = geometry or {"y": 0, "ih": 800, "sh": 800}
        self._box = box or {
            "x": 100,
            "y": 200,
            "width": 50,
            "height": 20,
        }

    def locator(self, _selector: str) -> _FakeLocator:
        return _FakeLocator(self._box)

    async def evaluate(self, _script: str) -> dict[str, int]:
        return self._geometry


def _no_sleep() -> AbstractContextManager[object]:
    async def fake_sleep(_seconds: float) -> None:
        return None

    return patch("supporter.tools.browser.humanize.asyncio.sleep", fake_sleep)


def test_scroll_step_zero() -> None:
    assert humanize._scroll_step(0) == 0


def test_scroll_step_positive() -> None:
    random.seed(1)
    step = humanize._scroll_step(500)
    assert 80 <= step <= 180
    assert step > 0


def test_scroll_step_negative() -> None:
    random.seed(1)
    step = humanize._scroll_step(-500)
    assert -180 <= step <= -80
    assert step < 0


def test_scroll_step_small_remaining() -> None:
    random.seed(1)
    step = humanize._scroll_step(50)
    assert step == 50


def test_scroll_step_negative_small() -> None:
    random.seed(1)
    step = humanize._scroll_step(-30)
    assert step == -30


def test_wrong_key_returns_neighbor() -> None:
    random.seed(1)
    result = humanize._wrong_key("a")
    assert result is not None
    assert result in humanize._QWERTY_NEIGHBORS["a"]


def test_wrong_key_returns_none_for_non_letter() -> None:
    assert humanize._wrong_key("1") is None
    assert humanize._wrong_key(" ") is None


def test_wrong_key_preserves_case() -> None:
    random.seed(42)
    for _ in range(100):
        result = humanize._wrong_key("A")
        if result is not None:
            assert result.isupper()


async def test_overshoot_correction_short_distance() -> None:
    mouse = _FakeMouse()
    await humanize._overshoot_correction(cast("Mouse", mouse), (0.0, 0.0), (0.5, 0.5))
    assert mouse.moves == []


async def test_overshoot_correction_normal() -> None:
    random.seed(1)
    mouse = _FakeMouse()
    with _no_sleep():
        await humanize._overshoot_correction(
            cast("Mouse", mouse), (0.0, 0.0), (100.0, 100.0)
        )
    assert len(mouse.moves) >= 1


async def test_element_target_with_locator() -> None:
    page = _FakePage(box={"x": 100, "y": 200, "width": 50, "height": 20})
    x, y = await humanize._element_target(cast("Page", page), "e1")
    assert 100 <= x <= 150
    assert 200 <= y <= 220


async def test_element_target_without_locator() -> None:
    page = _FakePage(box={"x": 100, "y": 200, "width": 50, "height": 20})
    x, y = await humanize._element_target(cast("Page", page), "e1", locator=None)
    assert 100 <= x <= 150
    assert 200 <= y <= 220


async def test_element_target_raises_on_no_bounding_box() -> None:
    class NoBoxLocator:
        async def bounding_box(self) -> None:
            return None

    class PageWithNoBox:
        def locator(self, sel: str) -> NoBoxLocator:
            return NoBoxLocator()

    with pytest.raises(ValueError, match="no bounding box"):
        await humanize._element_target(cast("Page", PageWithNoBox()), "e1")


async def test_idle_flourish_hover_reconsider() -> None:
    humanize._LAST_POS = (300.0, 300.0)
    page = _FakePage()
    with _no_sleep(), patch("random.random", return_value=0.0):
        await humanize.idle_flourish(cast("Page", page), rate=1.0)
    assert page.mouse.moves
    humanize.reset_cursor()


async def test_idle_flourish_scroll_back_reread() -> None:
    humanize.reset_cursor()
    page = _FakePage(geometry={"y": 200, "ih": 800, "sh": 4000})
    with (
        _no_sleep(),
        patch("random.random", return_value=0.99),
        patch(
            "supporter.tools.browser.humanize.random.choice",
            return_value=humanize._idle_scroll_back_reread,
        ),
    ):
        await humanize.idle_flourish(cast("Page", page), rate=1.0)
    assert len(page.mouse.wheels) > 0


async def test_idle_flourish_cursor_drift_noop_when_no_last_pos() -> None:
    humanize._LAST_POS = None
    page = _FakePage()
    with _no_sleep(), patch("random.random", return_value=0.0):
        await humanize.idle_flourish(cast("Page", page), rate=1.0)
    assert page.mouse.moves == []


async def test_idle_hover_reconsider_noop_when_no_last_pos() -> None:
    humanize._LAST_POS = None
    page = _FakePage()
    with _no_sleep():
        await humanize._idle_hover_reconsider(cast("Page", page))
    assert page.mouse.moves == []


async def test_idle_cursor_drift_noop_when_no_last_pos() -> None:
    humanize._LAST_POS = None
    page = _FakePage()
    with _no_sleep():
        await humanize._idle_cursor_drift(cast("Page", page))
    assert page.mouse.moves == []


async def test_human_type_typo_rate_injects_errors() -> None:
    random.seed(42)
    page = _FakePage()
    with (
        _no_sleep(),
        patch(
            "supporter.tools.browser.humanize.random.random",
            return_value=0.0,
        ),
    ):
        await humanize.human_type(cast("Page", page), "e1", "test", sensitive=False)
    assert any(k == "Backspace" for op, k in page.keyboard.events if op == "press")


async def test_human_type_inter_key_delay_pause_chars() -> None:
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    random.seed(1)
    page = _FakePage()
    with (
        patch("supporter.tools.browser.humanize.asyncio.sleep", record_sleep),
        patch("supporter.tools.browser.humanize.random.random", return_value=1.0),
    ):
        await humanize.human_type(cast("Page", page), "e1", "a b", sensitive=True)
    assert len(delays) > 0


async def test_human_type_with_locator() -> None:
    random.seed(8)
    page = _FakePage()
    locator = _FakeLocator()
    with (
        _no_sleep(),
        patch("supporter.tools.browser.humanize.random.random", return_value=0.0),
    ):
        await humanize.human_type(
            cast("Page", page),
            "e1",
            "hi",
            sensitive=True,
            locator=cast("Locator", locator),
        )
    downs = [k for op, k in page.keyboard.events if op == "down"]
    assert downs == ["h", "i"]


async def test_human_click_with_locator() -> None:
    random.seed(1)
    page = _FakePage()
    locator = _FakeLocator(box={"x": 100, "y": 200, "width": 50, "height": 20})
    with _no_sleep():
        await humanize.human_click(
            cast("Page", page), "e1", locator=cast("Locator", locator)
        )
    assert len(page.mouse.moves) > 0


async def test_human_hover_with_locator() -> None:
    random.seed(1)
    page = _FakePage()
    locator = _FakeLocator(box={"x": 100, "y": 200, "width": 50, "height": 20})
    with _no_sleep():
        await humanize.human_hover(
            cast("Page", page), "e1", locator=cast("Locator", locator)
        )
    assert len(page.mouse.moves) > 0


async def test_human_press() -> None:
    page = _FakePage()
    with _no_sleep():
        await humanize.human_press(cast("Page", page), "Enter")
    assert page.keyboard.events == [("press", "Enter")]


async def test_human_scroll_zero_delta() -> None:
    page = _FakePage()
    with _no_sleep():
        await humanize.human_scroll(cast("Page", page), 0, 0)
    assert page.mouse.wheels == []


async def test_scroll_overshoot_settle_small() -> None:
    mouse = _FakeMouse()
    with _no_sleep():
        await humanize._scroll_overshoot_settle(cast("Mouse", mouse), 0, 0)
    assert mouse.wheels == []


async def test_realize_and_fix() -> None:
    keyboard = _FakeKeyboard()
    with _no_sleep():
        await humanize._realize_and_fix(cast("Any", keyboard), 2)
    presses = [k for op, k in keyboard.events if op == "press"]
    assert presses == ["Backspace", "Backspace"]


async def test_human_click_drives_overlay_when_flag_enabled() -> None:
    random.seed(1)
    page = _FakePage()
    clicks: list[tuple[float, float]] = []

    async def fake_click(_page: Any, x: float, y: float) -> None:
        clicks.append((x, y))

    async def fake_move(_page: Any, x: float, y: float) -> None:
        pass

    with (
        _no_sleep(),
        patch.object(humanize.config, "browser_debug_overlay", True),
        patch.object(humanize.debug_overlay, "overlay_click", fake_click),
        patch.object(humanize.debug_overlay, "overlay_move", fake_move),
    ):
        await humanize.human_click(cast("Page", page), "e1")
    assert len(clicks) == 1
