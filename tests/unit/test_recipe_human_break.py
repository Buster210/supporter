from unittest.mock import AsyncMock, patch

import pytest

from supporter.recipes import RecipeStep, _execute_step, _parse_break_range


def test_parse_break_range_defaults_and_bounds() -> None:
    assert _parse_break_range("") == (300, 1500)
    assert _parse_break_range("nope") == (300, 1500)
    assert _parse_break_range("50||10") == (10, 50)  # swapped
    assert _parse_break_range("0||999999") == (0, 60000)  # clamped hi
    assert _parse_break_range("5||5") == (5, 5)


@pytest.mark.asyncio
async def test_human_break_draws_fresh_random_each_run() -> None:
    """Each run uses whatever random.randint returns over (lo, hi) — a fresh
    draw per replay, never a baked constant. Deterministic via mocked RNG/clock."""
    step = RecipeStep(kind="human_break", value="1||5")
    with (
        patch("supporter.recipes.random.randint", side_effect=[2, 4, 1]) as rnd,
        patch("supporter.recipes.asyncio.sleep", new_callable=AsyncMock) as slept,
    ):
        got = [
            (await _execute_step(0, step, {"emitted": [], "name": "t"}))["break_ms"]
            for _ in range(3)
        ]
    assert got == [2, 4, 1]  # fresh draw per run
    assert all(c.args == (1, 5) for c in rnd.call_args_list)  # over the stored range
    assert [c.args[0] for c in slept.call_args_list] == [2 / 1000, 4 / 1000, 1 / 1000]


@pytest.mark.asyncio
async def test_human_break_blank_uses_defaults() -> None:
    step = RecipeStep(kind="human_break", value="")
    with (
        patch("supporter.recipes.random.randint", return_value=900) as rnd,
        patch("supporter.recipes.asyncio.sleep", new_callable=AsyncMock),
    ):
        r = await _execute_step(0, step, {"emitted": [], "name": "t"})
    assert r["ok"] is True
    assert r["break_ms"] == 900
    rnd.assert_called_once_with(300, 1500)  # default range
