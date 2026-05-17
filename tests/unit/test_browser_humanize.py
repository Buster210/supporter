from __future__ import annotations

import math
import random

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
