from __future__ import annotations

import asyncio
import math
import random
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from patchright.async_api import Keyboard, Mouse, Page


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


async def human_click(
    page: Page,
    ref: str,
    *,
    button: Literal["left", "middle", "right"] = "left",
) -> None:
    locator = page.locator(f"aria-ref={ref}")
    box = await locator.bounding_box()
    if box is None:
        raise ValueError(f"Element {ref} has no bounding box")

    target_x = box["x"] + box["width"] * random.uniform(0.3, 0.7)
    target_y = box["y"] + box["height"] * random.uniform(0.3, 0.7)

    mouse: Mouse = page.mouse
    current = (0.0, 0.0)
    target = (target_x, target_y)

    control_points = random_control_points(current, target, count=3)
    duration_ms = fitts_duration(current, target) + random.uniform(100, 300)
    steps = max(10, int(duration_ms / 16.0))

    for step in range(steps + 1):
        t = step / steps
        jt = minimum_jerk(t, 0.0, 1.0)
        mx, my = bezier_2d(jt, control_points)
        await mouse.move(mx, my)
        await asyncio.sleep(duration_ms / 1000.0 / steps)

    await asyncio.sleep(random.uniform(0.05, 0.15))
    await mouse.click(target_x, target_y, button=button)
    await asyncio.sleep(random.uniform(0.1, 0.3))


async def human_type(
    page: Page,
    ref: str,
    text: str,
    *,
    typo_rate: float = 0.02,
) -> None:
    locator = page.locator(f"aria-ref={ref}")
    await locator.click()
    await asyncio.sleep(random.uniform(0.1, 0.3))

    keyboard: Keyboard = page.keyboard
    for i, char in enumerate(text):
        await keyboard.press(char)
        delay = random.uniform(0.03, 0.15)
        if delay > 0.12 and random.random() < typo_rate:
            await keyboard.press("Backspace")
            await asyncio.sleep(random.uniform(0.05, 0.1))
            await keyboard.press(char)
        await asyncio.sleep(delay)

        if i > 0 and i % random.randint(10, 30) == 0:
            await asyncio.sleep(random.uniform(0.2, 0.6))
