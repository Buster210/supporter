from __future__ import annotations

import asyncio
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...config import config
from ...logger import logger
from . import guardrails, humanize, task_memory

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
_FRAME_SELECTOR: str | None = None

_STEALTH_ARGS: list[str] = []

_IGNORE_DEFAULT_ARGS: list[str] = ["--use-mock-keychain"]

_CLONE_ROOT = Path.home() / ".patchright-chrome"

_CLONE_SKIP: frozenset[str] = frozenset(
    {
        "Cache",
        "Code Cache",
        "GPUCache",
        "ShaderCache",
        "GraphiteDawnCache",
        "DawnCache",
        "GrShaderCache",
        "Service Worker",
        "component_crx_cache",
        "extensions_crx_cache",
        "Sessions",
        "blob_storage",
        "File System",
        "Crashpad",
        "Crash Reports",
        "lockfile",
    }
)


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


def active_page() -> Any:
    return _PAGE


def list_pages() -> list[Any]:
    if _CONTEXT is None:
        return []
    return list(_CONTEXT.pages)


def set_active(page: Any) -> None:
    global _PAGE, _FRAME_SELECTOR
    _PAGE = page
    _FRAME_SELECTOR = None


def active_frame_selector() -> str | None:
    return _FRAME_SELECTOR


def set_frame(selector: str | None) -> None:
    global _FRAME_SELECTOR
    _FRAME_SELECTOR = selector


def _profile_dir() -> Path:
    if config.browser_profile_path:
        return Path(config.browser_profile_path).expanduser().resolve()
    home = Path.home()
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Google" / "Chrome"
    return home / ".config" / "google-chrome"


def _profile_name() -> str:
    return config.browser_profile_name


def _clone_ignore(_dir: str, names: list[str]) -> set[str]:
    skip: set[str] = set()
    for n in names:
        if n in _CLONE_SKIP or n.startswith("Singleton"):
            skip.add(n)
    return skip


def _backup_cookie_db(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)
    d = sqlite3.connect(str(dst))
    try:
        with d:
            s.backup(d)
    finally:
        s.close()
        d.close()


def _build_clone(source_user_data: Path, profile: str) -> Path:
    src_profile = source_user_data / profile
    dst_profile = _CLONE_ROOT / profile

    if not src_profile.exists():
        logger.warning(
            f"Chrome profile not found at {src_profile}; launching without a "
            "cloned login. Check chrome://version -> 'Profile Path'."
        )
        return source_user_data

    if not dst_profile.exists():
        logger.info(f"Cloning profile to {_CLONE_ROOT} (one-time copy)...")
        shutil.copytree(
            src_profile, dst_profile, ignore=_clone_ignore, dirs_exist_ok=True
        )
        for name in ("Local State", "First Run"):
            f = source_user_data / name
            if f.exists():
                shutil.copy2(f, _CLONE_ROOT / name)

    for rel in (Path("Network") / "Cookies", Path("Cookies")):
        _backup_cookie_db(src_profile / rel, dst_profile / rel)

    return _CLONE_ROOT


_LOCK_ERROR_MARKERS: tuple[str, ...] = (
    "ProcessSingleton",
    "SingletonLock",
    "profile appears to be in use",
    "user data directory is already in use",
)


def _is_profile_lock_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _LOCK_ERROR_MARKERS)


async def _launch_context(
    pws: Any, user_data_dir: Path, profile_name: str | None
) -> Any:
    args = list(_STEALTH_ARGS)
    if profile_name:
        args.append("--profile-directory=" + profile_name)
    if sys.platform.startswith("linux"):
        args.append("--password-store=gnome-libsecret")

    return await pws.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        channel="chrome",
        headless=config.browser_headless,
        no_viewport=True,
        chromium_sandbox=True,
        args=args,
        ignore_default_args=_IGNORE_DEFAULT_ARGS,
    )


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
        from patchright.async_api import async_playwright

        _PWS = await async_playwright().start()

        user_data_dir = _profile_dir()
        profile = _profile_name()

        if config.browser_profile_path:
            launch_dir = user_data_dir
        else:
            launch_dir = await asyncio.to_thread(_build_clone, user_data_dir, profile)

        try:
            _CONTEXT = await _launch_context(_PWS, launch_dir, profile)
        except Exception as e:
            if _is_profile_lock_error(e):
                raise RuntimeError(
                    "Chrome is already using this profile. Fully quit Chrome and "
                    "try again, or set BROWSER_PROFILE_PATH to a separate profile "
                    "directory to run alongside it."
                ) from e
            raise

        _PAGE = await _CONTEXT.new_page()
        await _PAGE.bring_to_front()

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
    global _KEEP_OPEN, _FRAME_SELECTOR

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
    _FRAME_SELECTOR = None
    humanize.reset_cursor()
    task_memory.discard()


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
