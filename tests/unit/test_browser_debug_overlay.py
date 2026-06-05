from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

from supporter.config import load_config
from supporter.tools.browser import debug_overlay
from supporter.tools.browser.debug_overlay import (
    _BOOTSTRAP_JS,
    _CURSOR_JS,
    inject_overlay,
    overlay_click,
    overlay_move,
)
from supporter.types import AppConfig


class _FakePage:
    def __init__(self, raise_on_eval: bool = False) -> None:
        self.evaluated: list[str] = []
        self._raise_on_eval = raise_on_eval

    async def evaluate(self, expression: str, *args: Any) -> None:
        if self._raise_on_eval:
            raise RuntimeError("boom")
        self.evaluated.append(expression)


class _FakeInjPage:
    def __init__(self, raise_on_add_init_script: bool = False) -> None:
        self.add_init_calls: list[str] = []
        self.evaluate_calls: list[str] = []
        self._raise = raise_on_add_init_script

    async def add_init_script(self, script: str) -> None:
        if self._raise:
            raise RuntimeError("add_init_script boom")
        self.add_init_calls.append(script)

    async def evaluate(self, expression: str) -> None:
        self.evaluate_calls.append(expression)


def test_cursor_js_invariants() -> None:
    assert "canvas" in _CURSOR_JS
    assert "__dbg_cursor" in _CURSOR_JS
    assert "getContext" in _CURSOR_JS
    assert "arc(" in _CURSOR_JS
    assert "__X__" in _CURSOR_JS
    assert "__Y__" in _CURSOR_JS
    assert "__KIND__" in _CURSOR_JS
    assert "__dbg_pointer" in _CURSOR_JS
    assert "sessionStorage" in _CURSOR_JS
    assert "translate(" in _CURSOR_JS


def test_bootstrap_js_invariants() -> None:
    assert "__dbg_cursor" in _BOOTSTRAP_JS
    assert "__dbg_pointer" in _BOOTSTRAP_JS
    assert "sessionStorage" in _BOOTSTRAP_JS
    assert "DOMContentLoaded" in _BOOTSTRAP_JS
    assert "__X__" not in _BOOTSTRAP_JS
    assert "__KIND__" not in _BOOTSTRAP_JS
    assert "window.innerWidth / 2" in _BOOTSTRAP_JS
    assert "window.innerHeight / 2" in _BOOTSTRAP_JS


def test_module_exports() -> None:
    assert debug_overlay.__all__ == [
        "_BOOTSTRAP_JS",
        "_CURSOR_JS",
        "inject_overlay",
        "overlay_click",
        "overlay_move",
    ]


async def test_overlay_move_substitutes_coords_and_kind() -> None:
    page = _FakePage()
    await overlay_move(page, 12.4, 34.6)
    script = page.evaluated[0]
    assert "kind = 'move'" in script
    assert "= 12," in script
    assert "= 35," in script
    assert "__X__" not in script


async def test_overlay_move_swallows_exception() -> None:
    page = _FakePage(raise_on_eval=True)
    await overlay_move(page, 1.0, 2.0)
    assert page.evaluated == []


async def test_overlay_click_substitutes_kind() -> None:
    page = _FakePage()
    await overlay_click(page, 5, 6)
    script = page.evaluated[0]
    assert "kind = 'click'" in script
    assert "__KIND__" not in script


async def test_overlay_click_swallows_exception() -> None:
    page = _FakePage(raise_on_eval=True)
    await overlay_click(page, 1.0, 2.0)
    assert page.evaluated == []


async def test_inject_overlay_arms_and_draws() -> None:
    page = _FakeInjPage()
    await inject_overlay(page)
    assert page.add_init_calls == [_BOOTSTRAP_JS]
    assert page.evaluate_calls == [_BOOTSTRAP_JS]


async def test_inject_overlay_swallows_exception() -> None:
    page = _FakeInjPage(raise_on_add_init_script=True)
    await inject_overlay(page)


def test_config_debug_overlay_defaults_false() -> None:
    assert AppConfig.browser_debug_overlay is False


def test_load_config_debug_overlay_default_false() -> None:
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
    try:
        with patch("supporter.config.load_dotenv"):
            assert load_config().browser_debug_overlay is False
    finally:
        os.environ.clear()
        os.environ.update(old_env)


async def test_first_draw_logs_at_info() -> None:
    old_flag = debug_overlay._first_draw_logged
    old_entry = debug_overlay._entry_logged
    debug_overlay._first_draw_logged = False
    debug_overlay._entry_logged = True
    try:
        with patch("supporter.tools.browser.debug_overlay.logger") as mock_logger:
            await overlay_move(_FakePage(), 1, 2)
            mock_logger.info.assert_called_once()
    finally:
        debug_overlay._first_draw_logged = old_flag
        debug_overlay._entry_logged = old_entry
