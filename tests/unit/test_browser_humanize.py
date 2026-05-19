from __future__ import annotations

import asyncio
import math
import random
from unittest.mock import patch

from supporter.tools.browser import humanize


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
    assert math.isclose(mid, 0.5)  # symmetric profile crosses 0.5 at t=0.5


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
    assert near >= 200.0  # base_ms floor


def test_random_control_points_starts_and_ends_fixed() -> None:
    random.seed(1)
    start, end = (0.0, 0.0), (100.0, 50.0)
    pts = humanize.random_control_points(start, end, count=3)
    assert pts[0] == start
    assert pts[-1] == end
    assert len(pts) == 4  # start + (count-1=2) interior points + end


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
    humanize._LAST_POS = (123.0, 456.0)
    try:
        assert humanize._origin_for(None) == (123.0, 456.0)
        assert humanize._origin_for({"width": 800, "height": 600}) == (123.0, 456.0)
    finally:
        humanize.reset_cursor()


def test_origin_falls_back_inside_viewport_when_unset() -> None:
    humanize.reset_cursor()
    random.seed(3)
    x, y = humanize._origin_for({"width": 800, "height": 600})
    assert 0.0 <= x <= 800.0
    assert 0.0 <= y <= 600.0


def test_origin_fallback_box_when_no_viewport() -> None:
    humanize.reset_cursor()
    random.seed(3)
    x, y = humanize._origin_for(None)
    assert humanize._ORIGIN_FALLBACK_X[0] <= x <= humanize._ORIGIN_FALLBACK_X[1]
    assert humanize._ORIGIN_FALLBACK_Y[0] <= y <= humanize._ORIGIN_FALLBACK_Y[1]
    assert (x, y) != (0.0, 0.0)


def test_reset_cursor_clears_position() -> None:
    humanize._LAST_POS = (10.0, 10.0)
    humanize.reset_cursor()
    assert humanize._LAST_POS is None


def test_reading_pause_sleeps_within_bounds() -> None:
    random.seed(11)
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    with patch("asyncio.sleep", fake_sleep):
        for _ in range(500):
            asyncio.run(humanize.reading_pause())

    assert slept
    assert all(0.4 <= s <= 5.0 for s in slept)
