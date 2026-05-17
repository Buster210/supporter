from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...config import config
from ...logger import logger
from . import guardrails

if TYPE_CHECKING:
    from patchright.async_api import BrowserContext, Page, Playwright

_PWS: Playwright | None = None
_CONTEXT: BrowserContext | None = None
_PAGE: Page | None = None
_LAUNCHING: bool = False
_LAUNCH_LOOP: object | None = None
_ACTION_COUNT: int = 0
_LAST_ACTION_TS: float = 0.0
_KEEP_OPEN: bool | None = None

_STEALTH_ARGS: list[str] = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-features=TranslateUI,BlinkGenPropertyTrees",
    "--disable-extensions",
    "--disable-component-extensions-with-background-pages",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-fre",
    "--disable-ipc-flooding-protection",
    "--disable-hang-monitor",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--disable-default-apps",
    "--disable-demo-mode",
    "--enable-features=NetworkService,NetworkServiceInProcess",
    "--force-color-profile=srgb",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain",
    "--export-tagged-pdf",
]


async def _prompt_lifecycle() -> None:
    global _KEEP_OPEN

    if _KEEP_OPEN is not None:
        return

    cb = guardrails.browse_confirmation_callback
    if cb is None:
        _KEEP_OPEN = True
        return

    keep = await cb(
        "Keep browser open after task?",
        "Yes = leave it open until app exits; "
        "No = you'll be asked to close it when the task is done",
    )
    _KEEP_OPEN = keep


def keep_open() -> bool:
    return _KEEP_OPEN is not False


def is_active() -> bool:
    return _PAGE is not None


def _managed_profile_dir() -> Path:
    if config.browser_profile_path:
        return Path(config.browser_profile_path).expanduser().resolve()
    from .. import resolved_project_root

    return resolved_project_root() / ".supporter" / "chrome-profile"


def _source_profile_dir() -> Path:
    if config.browser_profile_path:
        return Path(config.browser_profile_path).expanduser().resolve()
    return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"


def _ensure_profile() -> None:
    managed = _managed_profile_dir()
    sentinel = managed / ".supporter_profile_ready"
    if sentinel.exists():
        return

    source = _source_profile_dir()
    if not source.exists():
        raise RuntimeError(
            f"Chrome profile not found at {source}. "
            "Set BROWSER_PROFILE_PATH to a valid Chrome profile directory."
        )

    managed.parent.mkdir(parents=True, exist_ok=True)
    if managed.exists():
        shutil.rmtree(managed)

    managed.mkdir(parents=True, exist_ok=True)

    defaults = source / "Default"
    if defaults.exists():
        shutil.copytree(defaults, managed / "Default", symlinks=False)
    local_state = source / "Local State"
    if local_state.exists():
        shutil.copy2(local_state, managed / "Local State")

    sentinel.touch()
    logger.info(f"Chrome profile copied to {managed}")


async def get_session() -> tuple[Any, Any, Any]:
    global _PWS, _CONTEXT, _PAGE, _LAUNCHING, _LAUNCH_LOOP, _LAST_ACTION_TS

    if _PAGE is not None and _CONTEXT is not None and _PWS is not None:
        return _PWS, _CONTEXT, _PAGE

    if _LAUNCHING:
        if _LAUNCH_LOOP is not None:
            loop = _LAUNCH_LOOP
            if loop is not asyncio.get_running_loop():
                raise RuntimeError(
                    "Browser session launched on a different event loop. "
                    "Expected singleton per loop."
                )
        while _LAUNCHING:
            await asyncio.sleep(0.1)
        if _PAGE is not None and _CONTEXT is not None and _PWS is not None:
            return _PWS, _CONTEXT, _PAGE
        raise RuntimeError("Browser session launch failed")

    loop = asyncio.get_running_loop()
    _LAUNCH_LOOP = loop
    _LAUNCHING = True

    try:
        _ensure_profile()

        from patchright.async_api import async_playwright

        _PWS = await async_playwright().start()

        managed = _managed_profile_dir()
        _CONTEXT = await _PWS.chromium.launch_persistent_context(
            user_data_dir=str(managed),
            channel="chrome",
            headless=config.browser_headless,
            no_viewport=True,
            args=_STEALTH_ARGS,
        )

        _PAGE = await _CONTEXT.new_page()

        await _prompt_lifecycle()

        _LAST_ACTION_TS = time.monotonic()
        logger.info("Browser session launched")

        return _PWS, _CONTEXT, _PAGE
    except Exception:
        _PWS = None
        _CONTEXT = None
        _PAGE = None
        raise
    finally:
        _LAUNCHING = False


async def close_session() -> None:
    global _PWS, _CONTEXT, _PAGE, _LAUNCH_LOOP, _ACTION_COUNT, _LAST_ACTION_TS
    global _KEEP_OPEN

    try:
        if _CONTEXT is not None:
            await _CONTEXT.close()
    except Exception as e:
        logger.warning(f"Error closing browser context: {e}")

    try:
        if _PWS is not None:
            await _PWS.stop()
    except Exception as e:
        logger.warning(f"Error stopping playwright: {e}")

    _PWS = None
    _CONTEXT = None
    _PAGE = None
    _LAUNCH_LOOP = None
    _ACTION_COUNT = 0
    _LAST_ACTION_TS = 0.0
    _KEEP_OPEN = None


async def pace() -> None:
    global _LAST_ACTION_TS, _ACTION_COUNT

    gap = guardrails.random_gap()
    elapsed = time.monotonic() - _LAST_ACTION_TS
    if elapsed < gap:
        await asyncio.sleep(gap - elapsed)

    _LAST_ACTION_TS = time.monotonic()
    _ACTION_COUNT += 1

    if _ACTION_COUNT >= guardrails.ACTION_CAP:
        cb = guardrails.browse_confirmation_callback
        if cb is not None:
            go = await cb(
                "Session Cap Reached",
                f"{_ACTION_COUNT} browser actions performed. Continue?",
            )
            if not go:
                raise RuntimeError("Action cap reached and user denied continuation")
        _ACTION_COUNT = 0
