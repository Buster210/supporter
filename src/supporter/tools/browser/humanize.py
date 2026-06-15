from __future__ import annotations

import asyncio
import math
import random
from typing import TYPE_CHECKING, Final, Literal

from ...config import config
from . import debug_overlay

if TYPE_CHECKING:
    from patchright.async_api import Keyboard, Locator, Mouse, Page, ViewportSize

__all__ = [
    "config",
    "debug_overlay",
]


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


_FITTS_BASE_MS: Final = 75.0
_FITTS_DISTANCE_COEF: Final = 15.0
_MOVE_OVERHEAD_MS: Final = (10.0, 50.0)


def fitts_duration(
    start: tuple[float, float],
    end: tuple[float, float],
    base_ms: float = _FITTS_BASE_MS,
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    return base_ms + math.log2(distance + 1) * _FITTS_DISTANCE_COEF


def minimum_jerk(t: float, start: float, end: float) -> float:
    factor = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    return start + (end - start) * factor


_LAST_POS: tuple[float, float] | None = None

_ORIGIN_FALLBACK_X: tuple[float, float] = (100.0, 800.0)
_ORIGIN_FALLBACK_Y: tuple[float, float] = (100.0, 600.0)

_OVERSHOOT_PROBABILITY = 0.10

_TREMOR_AMPLITUDE: Final = 0.6
_TREMOR_VELOCITY_FLOOR: Final = 500.0
_TREMOR_MIN_SCALE: Final = 0.2
_MID_MOVE_PAUSE_PROBABILITY: Final = 0.015
_ORIGIN_SETTLE_JITTER: Final = 3.0

MICRO_BEHAVIOR_RATE: Final = config.browser_micro_behavior_rate
_CURSOR_DRIFT_MAX: Final = 40.0
_SCROLL_DRIFT_MIN: Final = 20
_SCROLL_DRIFT_MAX: Final = 80
_RECONSIDER_REACH: Final = 80.0

_SCROLL_OVERSHOOT_PROBABILITY: Final = 0.15
_SCROLL_DECEL: Final = 0.85

_PAUSE_CHARS: Final = frozenset(" .,!?;:\n")
_TYPO_RATE: Final = 0.04
_KEY_HOLD_MIN: Final = 0.04
_KEY_HOLD_MAX: Final = 0.09
_QWERTY_NEIGHBORS: Final[dict[str, str]] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrds",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huiknm",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _clamp_to_viewport(
    x: float, y: float, viewport: ViewportSize | None, margin: float = 2.0
) -> tuple[float, float]:
    if viewport is None:
        return (x, y)
    w = viewport.get("width", 800)
    h = viewport.get("height", 600)
    return (
        max(margin, min(w - margin, x)),
        max(margin, min(h - margin, y)),
    )


def reset_cursor() -> None:
    global _LAST_POS
    _LAST_POS = None


def _lognormal_delay(median: float, sigma: float, lo: float, hi: float) -> float:
    value = median * math.exp(random.gauss(0.0, sigma))
    return max(lo, min(hi, value))


def jitter_ms(median: float, sigma: float, lo: float, hi: float) -> int:
    return int(_lognormal_delay(median, sigma, lo, hi) * 1000)


_READ_MEDIAN_BASE: Final = 0.7
_READ_HI_BASE: Final = 3.0
_READ_MEDIAN_CAP: Final = 3.5
_READ_SCALE_CHARS: Final = 400.0


async def reading_pause(page: Page | None = None) -> None:
    median = _READ_MEDIAN_BASE
    hi = _READ_HI_BASE
    if page is not None:
        raw = await page.evaluate(
            "() => (document.body && document.body.innerText"
            " ? document.body.innerText.length : 0)"
        )
        chars = raw if isinstance(raw, (int, float)) else 0
        if chars > _READ_SCALE_CHARS:
            median = min(
                _READ_MEDIAN_CAP,
                _READ_MEDIAN_BASE * (1 + math.log10(chars / _READ_SCALE_CHARS)),
            )
            hi = _READ_MEDIAN_CAP + 2.0
    await asyncio.sleep(_lognormal_delay(median, 0.5, hi=hi, lo=0.25))


def _origin_for(viewport: ViewportSize | None) -> tuple[float, float]:
    if _LAST_POS is not None:
        return (
            _LAST_POS[0] + random.gauss(0.0, _ORIGIN_SETTLE_JITTER),
            _LAST_POS[1] + random.gauss(0.0, _ORIGIN_SETTLE_JITTER),
        )
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
    mouse: Mouse,
    target: tuple[float, float],
    viewport: ViewportSize | None,
    page: Page | None = None,
) -> None:
    global _LAST_POS
    start = _clamp_to_viewport(*_origin_for(viewport), viewport)
    dbg = page is not None and config.browser_debug_overlay

    control_points = random_control_points(
        start, _clamp_to_viewport(*target, viewport), count=3
    )
    duration_ms = fitts_duration(start, target) + random.uniform(*_MOVE_OVERHEAD_MS)
    steps = max(10, int(duration_ms / 16.0))
    step_delay = duration_ms / 1000.0 / steps

    prev_x, prev_y = start
    for step in range(steps + 1):
        jt = minimum_jerk(step / steps, 0.0, 1.0)
        mx, my = bezier_2d(jt, control_points)
        sigma = _tremor_sigma(mx - prev_x, my - prev_y, step_delay)
        nx, ny = _clamp_to_viewport(
            mx + random.gauss(0.0, sigma),
            my + random.gauss(0.0, sigma),
            viewport,
        )
        await mouse.move(nx, ny)
        if dbg:
            await debug_overlay.overlay_move(page, nx, ny)
        prev_x, prev_y = mx, my
        await asyncio.sleep(_lognormal_delay(step_delay, 0.3, 0.0, step_delay * 4))
        if random.random() < _MID_MOVE_PAUSE_PROBABILITY:
            await asyncio.sleep(random.uniform(0.015, 0.040))

    target = _clamp_to_viewport(*target, viewport)
    if random.random() < _OVERSHOOT_PROBABILITY:
        await _overshoot_correction(mouse, start, target, viewport)

    await mouse.move(*target)
    if dbg:
        await debug_overlay.overlay_move(page, target[0], target[1])
    _LAST_POS = target


def _tremor_sigma(dx: float, dy: float, step_delay: float) -> float:
    velocity = math.hypot(dx, dy) / max(step_delay, 1e-3)
    scale = max(_TREMOR_MIN_SCALE, 1.0 - velocity / _TREMOR_VELOCITY_FLOOR)
    return _TREMOR_AMPLITUDE * scale


async def _overshoot_correction(
    mouse: Mouse,
    start: tuple[float, float],
    target: tuple[float, float],
    viewport: ViewportSize | None,
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
    # Reflect overshoot inward if it would exit the viewport
    if viewport is not None:
        w = viewport.get("width", 800)
        h = viewport.get("height", 600)
        margin = 2.0
        if past[0] > w - margin:
            overshoot_amount = past[0] - (w - margin)
            past = (target[0] - overshoot_amount, past[1])
        elif past[0] < margin:
            overshoot_amount = margin - past[0]
            past = (target[0] + overshoot_amount, past[1])
        if past[1] > h - margin:
            overshoot_amount = past[1] - (h - margin)
            past = (past[0], target[1] - overshoot_amount)
        elif past[1] < margin:
            overshoot_amount = margin - past[1]
            past = (past[0], target[1] + overshoot_amount)
        past = _clamp_to_viewport(*past, viewport, margin=2.0)
    for _ in range(random.randint(1, 3)):
        await mouse.move(*past)
        await asyncio.sleep(_lognormal_delay(0.012, 0.3, 0.004, 0.04))
    await asyncio.sleep(_lognormal_delay(0.05, 0.3, 0.02, 0.15))


async def _idle_cursor_drift(page: Page) -> None:
    if _LAST_POS is None:
        return
    ox, oy = _LAST_POS
    drift = _clamp_to_viewport(
        ox + random.uniform(-_CURSOR_DRIFT_MAX, _CURSOR_DRIFT_MAX),
        oy + random.uniform(-_CURSOR_DRIFT_MAX, _CURSOR_DRIFT_MAX),
        page.viewport_size,
    )
    await _move_cursor(page.mouse, drift, page.viewport_size, page=page)


async def _idle_position_scroll(page: Page) -> None:
    geo = await page.evaluate(
        "() => ({y: window.scrollY, ih: window.innerHeight,"
        " sh: document.body.scrollHeight})"
    )
    mag = random.randint(_SCROLL_DRIFT_MIN, _SCROLL_DRIFT_MAX)
    room_below = geo["sh"] - (geo["y"] + geo["ih"])
    if room_below > mag:
        await human_scroll(page, 0, mag)
    elif geo["y"] > mag:
        await human_scroll(page, 0, -mag)


async def _idle_hover_reconsider(page: Page) -> None:
    if _LAST_POS is None:
        return
    ox, oy = _LAST_POS
    away = _clamp_to_viewport(
        ox + random.uniform(-_RECONSIDER_REACH, _RECONSIDER_REACH),
        oy + random.uniform(-_RECONSIDER_REACH, _RECONSIDER_REACH),
        page.viewport_size,
    )
    await _move_cursor(page.mouse, away, page.viewport_size, page=page)
    await asyncio.sleep(_lognormal_delay(0.2, 0.4, 0.08, 0.8))
    await _move_cursor(page.mouse, (ox, oy), page.viewport_size, page=page)


async def _idle_scroll_back_reread(page: Page) -> None:
    geo = await page.evaluate("() => ({y: window.scrollY})")
    mag = random.randint(_SCROLL_DRIFT_MIN, _SCROLL_DRIFT_MAX)
    if geo["y"] <= mag:
        return
    await human_scroll(page, 0, -mag)
    await asyncio.sleep(_lognormal_delay(0.4, 0.4, 0.15, 1.2))
    await human_scroll(page, 0, mag)


async def idle_flourish(page: Page, *, rate: float = MICRO_BEHAVIOR_RATE) -> None:
    if random.random() >= rate:
        return
    behavior = random.choice(
        (
            _idle_cursor_drift,
            _idle_position_scroll,
            _idle_hover_reconsider,
            _idle_scroll_back_reread,
        )
    )
    await behavior(page)


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
    await idle_flourish(page)
    if locator is None:
        locator = page.locator(f"aria-ref={ref}")
    await locator.scroll_into_view_if_needed()
    target = await _element_target(page, ref, locator)
    mouse: Mouse = page.mouse

    await _move_cursor(mouse, target, page.viewport_size, page=page)
    clamped = _clamp_to_viewport(*target, page.viewport_size)
    await asyncio.sleep(_lognormal_delay(0.04, 0.3, 0.02, 0.08))
    await mouse.click(clamped[0], clamped[1], button=button)
    if config.browser_debug_overlay:
        await debug_overlay.overlay_click(page, clamped[0], clamped[1])
    await asyncio.sleep(_lognormal_delay(0.07, 0.3, 0.03, 0.13))


async def human_hover(page: Page, ref: str, *, locator: Locator | None = None) -> None:
    await idle_flourish(page)
    if locator is None:
        locator = page.locator(f"aria-ref={ref}")
    await locator.scroll_into_view_if_needed()
    target = await _element_target(page, ref, locator)
    await _move_cursor(page.mouse, target, page.viewport_size, page=page)
    await asyncio.sleep(_lognormal_delay(0.04, 0.3, 0.02, 0.08))


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

    if random.random() < _SCROLL_OVERSHOOT_PROBABILITY:
        await _scroll_overshoot_settle(mouse, dx, dy)


async def _scroll_overshoot_settle(mouse: Mouse, dx: int, dy: int) -> None:
    extra_x = int(dx * random.uniform(0.02, 0.08))
    extra_y = int(dy * random.uniform(0.02, 0.08))
    if extra_x == 0 and extra_y == 0:
        return
    await mouse.wheel(extra_x, extra_y)
    await asyncio.sleep(_lognormal_delay(0.05, 0.3, 0.02, 0.15))

    back_x, back_y = -extra_x, -extra_y
    vx, vy = float(back_x), float(back_y)
    while round(back_x) != 0 or round(back_y) != 0:
        step_x = int(vx) if abs(vx) >= 1 else back_x
        step_y = int(vy) if abs(vy) >= 1 else back_y
        await mouse.wheel(step_x, step_y)
        back_x -= step_x
        back_y -= step_y
        vx *= _SCROLL_DECEL
        vy *= _SCROLL_DECEL
        await asyncio.sleep(_lognormal_delay(0.04, 0.3, 0.02, 0.12))


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


async def _key_tap(keyboard: Keyboard, char: str) -> None:
    if char == "\n":
        await keyboard.press("Enter")
        return
    await keyboard.down(char)
    await asyncio.sleep(random.uniform(_KEY_HOLD_MIN, _KEY_HOLD_MAX))
    await keyboard.up(char)


def _wrong_key(char: str) -> str | None:
    neighbours = _QWERTY_NEIGHBORS.get(char.lower())
    if not neighbours:
        return None
    slip = random.choice(neighbours)
    return slip.upper() if char.isupper() else slip


async def _inter_key_delay(char: str) -> None:
    delay = random.uniform(0.02, 0.07)
    if char in _PAUSE_CHARS:
        delay += random.uniform(0.05, 0.12)
    if random.random() < 0.02:
        delay += random.uniform(0.2, 0.45)
    elif random.random() < 0.005:
        delay += random.uniform(0.35, 0.7)
    await asyncio.sleep(delay)


async def _realize_and_fix(keyboard: Keyboard, backspaces: int) -> None:
    await asyncio.sleep(random.uniform(0.10, 0.25))
    for _ in range(backspaces):
        await keyboard.press("Backspace")
        await asyncio.sleep(random.uniform(0.03, 0.08))


async def _type_with_typo(keyboard: Keyboard, text: str, i: int) -> int:
    char = text[i]
    kind = random.choices(
        ("adjacent", "transpose", "double", "skip", "missed_space"),
        weights=(0.55, 0.20, 0.12, 0.08, 0.05),
    )[0]
    nxt = text[i + 1] if i + 1 < len(text) else ""

    if kind == "transpose" and nxt:
        await _key_tap(keyboard, nxt)
        await _key_tap(keyboard, char)
        await _realize_and_fix(keyboard, 2)
        await _key_tap(keyboard, char)
        await _key_tap(keyboard, nxt)
        return 2

    if kind == "missed_space" and char == " " and nxt:
        await _key_tap(keyboard, nxt)
        await _realize_and_fix(keyboard, 1)
        await _key_tap(keyboard, char)
        await _key_tap(keyboard, nxt)
        return 2

    if kind == "double":
        await _key_tap(keyboard, char)
        await _key_tap(keyboard, char)
        await _realize_and_fix(keyboard, 1)
        return 1

    if kind == "skip" and nxt:
        await _key_tap(keyboard, nxt)
        await _realize_and_fix(keyboard, 1)
        await _key_tap(keyboard, char)
        await _key_tap(keyboard, nxt)
        return 2

    slip = _wrong_key(char)
    if slip is None:
        await _key_tap(keyboard, char)
        return 1
    await _key_tap(keyboard, slip)
    await _realize_and_fix(keyboard, 1)
    await _key_tap(keyboard, char)
    return 1


async def human_type(
    page: Page,
    ref: str,
    text: str,
    *,
    sensitive: bool = False,
    locator: Locator | None = None,
) -> None:
    if locator is None:
        locator = page.locator(f"aria-ref={ref}")
    await locator.scroll_into_view_if_needed()
    await locator.click()
    await asyncio.sleep(random.uniform(0.1, 0.3))

    keyboard: Keyboard = page.keyboard
    typo_rate = 0.0 if sensitive else _TYPO_RATE
    i = 0
    typed = 0
    while i < len(text):
        if random.random() < typo_rate:
            i += await _type_with_typo(keyboard, text, i)
        else:
            await _key_tap(keyboard, text[i])
            i += 1
        await _inter_key_delay(text[i - 1])
        typed += 1
        if typed > 0 and typed % random.randint(10, 30) == 0:
            await asyncio.sleep(_lognormal_delay(0.2, 0.3, 0.12, 0.5))
