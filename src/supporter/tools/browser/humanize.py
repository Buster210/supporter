from __future__ import annotations

import asyncio
import math
import random
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from patchright.async_api import Keyboard, Locator, Mouse, Page, ViewportSize


def bezier_2d(t: float, points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    n = len(points) - 1
    x = 0.0
    y = 0.0
    for i, (px, py) in enumerate(points):
        binomial = math.comb(n, i)
        coeff = binomial * (t**i) * ((1 - t) ** (n - i))
        x += coeff * px
        y += coeff * py
    return (x, y)


def random_control_points(
    start: tuple[float, float],
    end: tuple[float, float],
    count: int = 3,
) -> tuple[tuple[float, float], ...]:
    points = [start]
    for i in range(1, count):
        progress = i / (count + 1)
        base_x = start[0] + (end[0] - start[0]) * progress
        base_y = start[1] + (end[1] - start[1]) * progress
        jitter_x = random.uniform(-30, 30)
        jitter_y = random.uniform(-30, 30)
        points.append((base_x + jitter_x, base_y + jitter_y))
    points.append(end)
    return tuple(points)


def fitts_duration(
    start: tuple[float, float],
    end: tuple[float, float],
    base_ms: float = 200.0,
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    return base_ms + math.log2(distance + 1) * 80.0


def minimum_jerk(t: float, start: float, end: float) -> float:
    factor = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    return start + (end - start) * factor


_LAST_POS: tuple[float, float] | None = None

_ORIGIN_FALLBACK_X: tuple[float, float] = (100.0, 800.0)
_ORIGIN_FALLBACK_Y: tuple[float, float] = (100.0, 600.0)

_OVERSHOOT_PROBABILITY = 0.15


def reset_cursor() -> None:
    global _LAST_POS
    _LAST_POS = None


def _lognormal_delay(median: float, sigma: float, lo: float, hi: float) -> float:
    value = median * math.exp(random.gauss(0.0, sigma))
    return max(lo, min(hi, value))


async def reading_pause() -> None:
    await asyncio.sleep(_lognormal_delay(1.1, 0.5, 0.4, 5.0))


def _origin_for(viewport: ViewportSize | None) -> tuple[float, float]:
    if _LAST_POS is not None:
        return _LAST_POS
    if viewport:
        return (
            random.uniform(0.1, 0.9) * viewport["width"],
            random.uniform(0.1, 0.9) * viewport["height"],
        )
    return (
        random.uniform(*_ORIGIN_FALLBACK_X),
        random.uniform(*_ORIGIN_FALLBACK_Y),
    )


async def _move_cursor(
    mouse: Mouse, target: tuple[float, float], viewport: ViewportSize | None
) -> None:
    global _LAST_POS
    start = _origin_for(viewport)

    control_points = random_control_points(start, target, count=3)
    duration_ms = fitts_duration(start, target) + random.uniform(100, 300)
    steps = max(10, int(duration_ms / 16.0))
    step_delay = duration_ms / 1000.0 / steps

    for step in range(steps + 1):
        jt = minimum_jerk(step / steps, 0.0, 1.0)
        mx, my = bezier_2d(jt, control_points)
        await mouse.move(mx, my)
        await asyncio.sleep(_lognormal_delay(step_delay, 0.3, 0.0, step_delay * 4))

    if random.random() < _OVERSHOOT_PROBABILITY:
        await _overshoot_correction(mouse, start, target)

    await mouse.move(*target)
    _LAST_POS = target


async def _overshoot_correction(
    mouse: Mouse, start: tuple[float, float], target: tuple[float, float]
) -> None:
    dx, dy = target[0] - start[0], target[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        return
    overshoot = random.uniform(2.0, 8.0)
    past = (
        target[0] + dx / distance * overshoot,
        target[1] + dy / distance * overshoot,
    )
    for _ in range(random.randint(1, 3)):
        await mouse.move(*past)
        await asyncio.sleep(_lognormal_delay(0.012, 0.3, 0.004, 0.04))
    await asyncio.sleep(_lognormal_delay(0.05, 0.3, 0.02, 0.15))


async def _element_target(
    page: Page, ref: str, locator: Locator | None = None
) -> tuple[float, float]:
    if locator is None:
        locator = page.locator(f"aria-ref={ref}")
    box = await locator.bounding_box()
    if box is None:
        raise ValueError(f"Element {ref} has no bounding box")
    return (
        box["x"] + box["width"] * random.uniform(0.3, 0.7),
        box["y"] + box["height"] * random.uniform(0.3, 0.7),
    )


async def human_click(
    page: Page,
    ref: str,
    *,
    button: Literal["left", "middle", "right"] = "left",
    locator: Locator | None = None,
) -> None:
    target = await _element_target(page, ref, locator)
    mouse: Mouse = page.mouse

    await _move_cursor(mouse, target, page.viewport_size)
    await asyncio.sleep(_lognormal_delay(0.09, 0.3, 0.05, 0.15))
    await mouse.click(target[0], target[1], button=button)
    await asyncio.sleep(_lognormal_delay(0.18, 0.3, 0.1, 0.3))


async def human_hover(page: Page, ref: str, *, locator: Locator | None = None) -> None:
    target = await _element_target(page, ref, locator)
    await _move_cursor(page.mouse, target, page.viewport_size)
    await asyncio.sleep(_lognormal_delay(0.09, 0.3, 0.05, 0.15))


async def human_scroll(page: Page, dx: int, dy: int) -> None:
    mouse: Mouse = page.mouse
    remaining_x = dx
    remaining_y = dy
    while remaining_x != 0 or remaining_y != 0:
        step_x = _scroll_step(remaining_x)
        step_y = _scroll_step(remaining_y)
        await mouse.wheel(step_x, step_y)
        remaining_x -= step_x
        remaining_y -= step_y
        await asyncio.sleep(_lognormal_delay(0.06, 0.3, 0.03, 0.2))


def _scroll_step(remaining: int) -> int:
    if remaining == 0:
        return 0
    sign = 1 if remaining > 0 else -1
    chunk = random.randint(80, 180)
    return sign * min(chunk, abs(remaining))


async def human_press(page: Page, keys: str) -> None:
    await asyncio.sleep(_lognormal_delay(0.1, 0.3, 0.05, 0.2))
    await page.keyboard.press(keys)
    await asyncio.sleep(_lognormal_delay(0.16, 0.3, 0.1, 0.3))


async def human_type(
    page: Page,
    ref: str,
    text: str,
    *,
    typo_rate: float = 0.02,
    locator: Locator | None = None,
) -> None:
    if locator is None:
        locator = page.locator(f"aria-ref={ref}")
    await locator.click()
    await asyncio.sleep(random.uniform(0.1, 0.3))

    keyboard: Keyboard = page.keyboard
    for i, char in enumerate(text):
        await keyboard.press(char)
        delay = _lognormal_delay(0.08, 0.45, 0.03, 0.4)
        if delay > 0.12 and random.random() < typo_rate:
            await keyboard.press("Backspace")
            await asyncio.sleep(_lognormal_delay(0.07, 0.3, 0.05, 0.15))
            await keyboard.press(char)
        await asyncio.sleep(delay)

        if i > 0 and i % random.randint(10, 30) == 0:
            await asyncio.sleep(_lognormal_delay(0.35, 0.3, 0.2, 0.9))
