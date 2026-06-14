from __future__ import annotations

import asyncio
import math
import random
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from supporter.tools.browser import humanize

if TYPE_CHECKING:
    from collections.abc import Iterator

    from patchright.async_api import Mouse, Page, ViewportSize


@pytest.fixture(autouse=True)
def _reset_cursor() -> Iterator[None]:
    humanize.reset_cursor()
    try:
        yield
    finally:
        humanize.reset_cursor()


def test_bezier_endpoints_are_exact() -> None:
    pts = ((0.0, 0.0), (5.0, 10.0), (10.0, 0.0))
    assert humanize.bezier_2d(0.0, pts) == (0.0, 0.0)
    end_x, end_y = humanize.bezier_2d(1.0, pts)
    assert math.isclose(end_x, 10.0)
    assert math.isclose(end_y, 0.0)


def test_bezier_midpoint_is_inside_hull() -> None:
    pts = ((0.0, 0.0), (10.0, 0.0))
    x, y = humanize.bezier_2d(0.5, pts)
    assert math.isclose(x, 5.0)
    assert math.isclose(y, 0.0)


def test_minimum_jerk_boundaries() -> None:
    assert math.isclose(humanize.minimum_jerk(0.0, 3.0, 9.0), 3.0)
    assert math.isclose(humanize.minimum_jerk(1.0, 3.0, 9.0), 9.0)
    mid = humanize.minimum_jerk(0.5, 0.0, 1.0)
    assert math.isclose(mid, 0.5)


def test_minimum_jerk_is_monotonic() -> None:
    prev = -1.0
    for i in range(101):
        v = humanize.minimum_jerk(i / 100, 0.0, 1.0)
        assert v >= prev - 1e-9
        prev = v


def test_fitts_duration_grows_with_distance() -> None:
    near = humanize.fitts_duration((0.0, 0.0), (10.0, 0.0))
    far = humanize.fitts_duration((0.0, 0.0), (1000.0, 0.0))
    assert far > near
    assert near >= humanize._FITTS_BASE_MS


def test_random_control_points_starts_and_ends_fixed() -> None:
    random.seed(1)
    start, end = (0.0, 0.0), (100.0, 50.0)
    pts = humanize.random_control_points(start, end, count=3)
    assert pts[0] == start
    assert pts[-1] == end
    assert len(pts) == 4


def test_lognormal_delay_clamped_to_bounds() -> None:
    random.seed(7)
    for _ in range(1000):
        d = humanize._lognormal_delay(0.08, 0.45, 0.03, 0.4)
        assert 0.03 <= d <= 0.4


def test_lognormal_delay_centers_near_median() -> None:
    random.seed(7)
    samples = [humanize._lognormal_delay(0.1, 0.4, 0.0, 10.0) for _ in range(5000)]
    samples.sort()
    median = samples[len(samples) // 2]
    assert math.isclose(median, 0.1, abs_tol=0.02)


def test_origin_uses_tracked_position_when_set() -> None:
    random.seed(0)
    humanize._LAST_POS = (123.0, 456.0)
    tol = 6 * humanize._ORIGIN_SETTLE_JITTER
    viewports: tuple[ViewportSize | None, ...] = (
        None,
        cast("ViewportSize", {"width": 800, "height": 600}),
    )
    for vp in viewports:
        x, y = humanize._origin_for(vp)
        assert abs(x - 123.0) <= tol
        assert abs(y - 456.0) <= tol


def test_origin_falls_back_inside_viewport_when_unset() -> None:
    random.seed(3)
    x, y = humanize._origin_for({"width": 800, "height": 600})
    assert 0.0 <= x <= 800.0
    assert 0.0 <= y <= 600.0


def test_origin_fallback_box_when_no_viewport() -> None:
    random.seed(3)
    x, y = humanize._origin_for(None)
    assert humanize._ORIGIN_FALLBACK_X[0] <= x <= humanize._ORIGIN_FALLBACK_X[1]
    assert humanize._ORIGIN_FALLBACK_Y[0] <= y <= humanize._ORIGIN_FALLBACK_Y[1]
    assert (x, y) != (0.0, 0.0)


def test_reset_cursor_clears_position() -> None:
    humanize._LAST_POS = (10.0, 10.0)
    humanize.reset_cursor()
    assert humanize._LAST_POS is None


async def test_reading_pause_sleeps_within_documented_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random.seed(11)
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record)
    for _ in range(50):
        await humanize.reading_pause()

    assert len(slept) == 50
    assert all(0.25 <= s <= humanize._READ_HI_BASE for s in slept)


class _FakeTextPage:
    def __init__(self, chars: int) -> None:
        self._chars = chars

    async def evaluate(self, _script: str) -> int:
        return self._chars


def _median_sleep_for(page: _FakeTextPage, n: int = 2001) -> float:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    random.seed(11)
    with patch("supporter.tools.browser.humanize.asyncio.sleep", fake_sleep):
        for _ in range(n):
            asyncio.run(humanize.reading_pause(cast("Page", page)))
    slept.sort()
    return slept[len(slept) // 2]


def test_reading_pause_scales_median_with_content() -> None:
    short = _median_sleep_for(_FakeTextPage(100))
    dense = _median_sleep_for(_FakeTextPage(40_000))
    assert dense > short
    assert short <= humanize._READ_MEDIAN_BASE + 0.2


def test_reading_pause_median_is_capped() -> None:
    huge = _median_sleep_for(_FakeTextPage(10_000_000))
    assert huge <= humanize._READ_MEDIAN_CAP + 1e-9


async def test_reading_pause_subthreshold_page_keeps_pass1_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record)
    random.seed(11)
    page = _FakeTextPage(50)
    for _ in range(500):
        await humanize.reading_pause(cast("Page", page))
    assert all(0.25 <= s <= humanize._READ_HI_BASE for s in slept)


def test_jitter_ms_within_bounds_and_int() -> None:
    random.seed(5)
    for _ in range(500):
        v = humanize.jitter_ms(0.3, 0.3, 0.15, 0.6)
        assert isinstance(v, int)
        assert 150 <= v <= 600


class _FakeMouse:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.wheels: list[tuple[int, int]] = []
        self.clicks: list[tuple[float, float]] = []

    async def move(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    async def wheel(self, dx: int, dy: int) -> None:
        self.wheels.append((dx, dy))

    async def click(self, x: float, y: float, button: str = "left") -> None:
        self.clicks.append((x, y))


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
    async def click(self) -> None:
        pass


class _FakePage:
    def __init__(self, *, geometry: dict[str, int] | None = None) -> None:
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.viewport_size = {"width": 1280, "height": 800}
        self._geometry = geometry or {"y": 0, "ih": 800, "sh": 800}

    def locator(self, _selector: str) -> _FakeLocator:
        return _FakeLocator()

    async def evaluate(self, _script: str) -> dict[str, int]:
        return self._geometry


def _no_sleep() -> AbstractContextManager[object]:
    async def fake_sleep(_seconds: float) -> None:
        return None

    return patch("supporter.tools.browser.humanize.asyncio.sleep", fake_sleep)


def _reconstruct(events: list[tuple[str, str]]) -> str:
    out: list[str] = []
    for op, key in events:
        if op == "down":
            continue
        char = key
        if op == "up":
            out.append(char)
        elif op == "press":
            if char == "Backspace":
                if out:
                    out.pop()
            elif char == "Enter":
                out.append("\n")
            else:
                out.append(char)
    return "".join(out)


def test_move_cursor_lands_exactly_on_target() -> None:
    humanize.reset_cursor()
    random.seed(2)
    mouse = _FakeMouse()
    with _no_sleep():
        asyncio.run(humanize._move_cursor(cast("Mouse", mouse), (640.0, 400.0), None))
    assert mouse.moves[-1] == (640.0, 400.0)
    assert humanize._LAST_POS == (640.0, 400.0)
    humanize.reset_cursor()


def test_tremor_sigma_shrinks_with_speed() -> None:
    slow = humanize._tremor_sigma(1.0, 0.0, 0.1)
    fast = humanize._tremor_sigma(100.0, 0.0, 0.1)
    assert slow > fast
    assert fast >= humanize._TREMOR_AMPLITUDE * humanize._TREMOR_MIN_SCALE - 1e-9


def test_idle_flourish_never_acts_at_rate_zero() -> None:
    page = _FakePage()
    with _no_sleep():
        asyncio.run(humanize.idle_flourish(cast("Page", page), rate=0.0))
    assert page.mouse.moves == []
    assert page.mouse.wheels == []


def _force_branch(behavior: object) -> AbstractContextManager[object]:
    return patch(
        "supporter.tools.browser.humanize.random.choice", return_value=behavior
    )


def test_idle_flourish_cursor_drift_branch() -> None:
    humanize._LAST_POS = (300.0, 300.0)
    page = _FakePage()
    with (
        _no_sleep(),
        patch("random.random", return_value=0.0),
        _force_branch(humanize._idle_cursor_drift),
    ):
        asyncio.run(humanize.idle_flourish(cast("Page", page), rate=1.0))
    assert page.mouse.moves
    humanize.reset_cursor()


def test_idle_flourish_scrolls_down_when_room_below() -> None:
    humanize.reset_cursor()
    page = _FakePage(geometry={"y": 0, "ih": 800, "sh": 4000})
    with (
        _no_sleep(),
        patch("random.random", return_value=0.99),
        _force_branch(humanize._idle_position_scroll),
    ):
        asyncio.run(humanize.idle_flourish(cast("Page", page), rate=1.0))
    total_dy = sum(dy for _dx, dy in page.mouse.wheels)
    assert total_dy > 0
    assert humanize._SCROLL_DRIFT_MIN <= total_dy <= humanize._SCROLL_DRIFT_MAX


def test_idle_flourish_scrolls_up_near_bottom() -> None:
    humanize.reset_cursor()
    page = _FakePage(geometry={"y": 3200, "ih": 800, "sh": 4000})
    with (
        _no_sleep(),
        patch("random.random", return_value=0.99),
        _force_branch(humanize._idle_position_scroll),
    ):
        asyncio.run(humanize.idle_flourish(cast("Page", page), rate=1.0))
    total_dy = sum(dy for _dx, dy in page.mouse.wheels)
    assert total_dy < 0


def test_idle_flourish_noop_when_unscrollable() -> None:
    humanize.reset_cursor()
    page = _FakePage(geometry={"y": 0, "ih": 800, "sh": 800})
    with (
        _no_sleep(),
        patch("random.random", return_value=0.99),
        _force_branch(humanize._idle_position_scroll),
    ):
        asyncio.run(humanize.idle_flourish(cast("Page", page), rate=1.0))
    assert page.mouse.wheels == []


def test_human_scroll_nets_to_target_without_overshoot() -> None:
    page = _FakePage()
    with _no_sleep(), patch("random.random", return_value=1.0):
        asyncio.run(humanize.human_scroll(cast("Page", page), 0, 500))
    assert sum(dy for _dx, dy in page.mouse.wheels) == 500


def test_human_scroll_overshoot_settles_back_to_target() -> None:
    page = _FakePage()
    random.seed(4)
    with _no_sleep(), patch("random.random", return_value=0.0):
        asyncio.run(humanize.human_scroll(cast("Page", page), 0, 500))
    assert sum(dy for _dx, dy in page.mouse.wheels) == 500
    running = 0
    peak = 0
    for _dx, dy in page.mouse.wheels:
        running += dy
        peak = max(peak, running)
    assert peak > 500


def test_human_type_normal_field_reconstructs_input() -> None:
    random.seed(8)
    page = _FakePage()
    with (
        _no_sleep(),
        patch("supporter.tools.browser.humanize.random.random", return_value=0.0),
    ):
        asyncio.run(
            humanize.human_type(
                cast("Page", page), "e1", "hello world", sensitive=False
            )
        )
    assert _reconstruct(page.keyboard.events) == "hello world"
    assert any(k == "Backspace" for op, k in page.keyboard.events if op == "press")


def test_human_type_sensitive_field_types_clean() -> None:
    random.seed(8)
    page = _FakePage()
    with (
        _no_sleep(),
        patch("supporter.tools.browser.humanize.random.random", return_value=0.0),
    ):
        asyncio.run(
            humanize.human_type(cast("Page", page), "e1", "secret", sensitive=True)
        )
    assert _reconstruct(page.keyboard.events) == "secret"
    assert not any(k == "Backspace" for op, k in page.keyboard.events if op == "press")


def test_human_type_uses_key_hold_dwell() -> None:
    random.seed(8)
    page = _FakePage()
    with (
        _no_sleep(),
        patch("supporter.tools.browser.humanize.random.random", return_value=1.0),
    ):
        asyncio.run(
            humanize.human_type(cast("Page", page), "e1", "abc", sensitive=False)
        )
    downs = [k for op, k in page.keyboard.events if op == "down"]
    ups = [k for op, k in page.keyboard.events if op == "up"]
    assert downs == ["a", "b", "c"]
    assert ups == ["a", "b", "c"]
