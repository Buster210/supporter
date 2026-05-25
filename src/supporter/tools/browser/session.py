from __future__ import annotations

import asyncio
import shutil
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...config import config
from ...logger import logger
from . import guardrails, humanize, task

if TYPE_CHECKING:
    from patchright.async_api import BrowserContext, Page, Playwright

_PWS: Playwright | None = None
_CONTEXT: BrowserContext | None = None
_PAGE: Page | None = None
_LAUNCHING: bool = False
_LAUNCH_LOOP: object | None = None
_CLONE_LOCK: asyncio.Lock | None = None
_ACTION_COUNT: int = 0
_ACTION_CAP_CEILING: int = 0
_LAST_ACTION_TS: float = 0.0
_ACTION_TIMES: deque[float] = deque()
_SESSION_START_TS: float = 0.0
_TEMPO: float = 1.0

_RATE_WINDOW_SECONDS: float = 60.0
_KEEP_OPEN: bool | None = None
_LIFECYCLE_TASK: asyncio.Task[None] | None = None
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
        "DawnWebGPUCache",
        "DawnGraphiteCache",
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

_SESSION_SQLITE: tuple[Path, ...] = (
    Path("Network") / "Cookies",
    Path("Cookies"),
    Path("Login Data"),
    Path("Login Data For Account"),
    Path("Web Data"),
)

_SESSION_DIRS: tuple[str, ...] = (
    "Local Storage",
    "Session Storage",
    "IndexedDB",
    "WebStorage",
)

_ROOT_SESSION_FILES: tuple[str, ...] = ("Local State",)


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


def _start_lifecycle_prompt() -> None:
    global _LIFECYCLE_TASK

    if _KEEP_OPEN is not None or _LIFECYCLE_TASK is not None:
        return

    async def run() -> None:
        try:
            await _prompt_lifecycle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Browser lifecycle prompt failed: {e}")

    task = asyncio.ensure_future(run())
    _LIFECYCLE_TASK = task
    task.add_done_callback(_clear_lifecycle_task)


def _clear_lifecycle_task(_task: object) -> None:
    global _LIFECYCLE_TASK
    _LIFECYCLE_TASK = None


async def _await_lifecycle_answer() -> None:
    global _KEEP_OPEN
    if _KEEP_OPEN is not None:
        return
    if _LIFECYCLE_TASK is not None:
        try:
            await _LIFECYCLE_TASK
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Browser lifecycle prompt failed: {e}")
    if _KEEP_OPEN is None:
        _KEEP_OPEN = True


async def resolve_close_at_task_end() -> str:
    if not is_active():
        return ""
    await _await_lifecycle_answer()
    if pinned_open():
        return "Browser left open (persistent session)."
    cb = guardrails.browse_confirmation_callback
    if cb is None:
        return ""
    if not await cb("Close browser now?", "Task done — close browser now?"):
        return "Browser left open; will ask again when the next task finishes."
    await close_session()
    return "Browser closed."


def keep_open() -> bool:
    return _KEEP_OPEN is not False


def pinned_open() -> bool:
    return _KEEP_OPEN is True


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


def _clone_ignore_once(dir_: str, names: list[str]) -> set[str]:
    skip = _clone_ignore(dir_, names)
    skip.update(n for n in names if n in _SESSION_DIRS)
    return skip


def _newer(src: Path, dst: Path) -> bool:
    return not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime


def _mirror_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    skip = _clone_ignore(str(src), [p.name for p in src.iterdir()])
    wanted: set[Path] = set()
    for s in src.rglob("*"):
        rel = s.relative_to(src)
        if rel.parts[0] in skip:
            continue
        d = dst / rel
        wanted.add(rel)
        if s.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        elif _newer(s, d):
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

    if dst.exists():
        for d in sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.relative_to(dst) not in wanted:
                d.unlink() if d.is_file() else d.rmdir()


def _backup_sqlite(src: Path, dst: Path) -> None:
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
            src_profile, dst_profile, ignore=_clone_ignore_once, dirs_exist_ok=True
        )
        for name in ("Local State", "First Run"):
            f = source_user_data / name
            if f.exists():
                shutil.copy2(f, _CLONE_ROOT / name)

    for rel in _SESSION_SQLITE:
        _backup_sqlite(src_profile / rel, dst_profile / rel)
    for name in _SESSION_DIRS:
        _mirror_dir(src_profile / name, dst_profile / name)
    for name in _ROOT_SESSION_FILES:
        f = source_user_data / name
        if f.exists() and _newer(f, _CLONE_ROOT / name):
            shutil.copy2(f, _CLONE_ROOT / name)

    return _CLONE_ROOT


def _clone_lock() -> asyncio.Lock:
    global _CLONE_LOCK
    if _CLONE_LOCK is None:
        _CLONE_LOCK = asyncio.Lock()
    return _CLONE_LOCK


async def _clone_profile() -> Path:
    user_data_dir = _profile_dir()
    profile = _profile_name()
    async with _clone_lock():
        return await asyncio.to_thread(_build_clone, user_data_dir, profile)


async def prewarm_clone() -> None:
    if config.browser_profile_path or _PAGE is not None:
        return
    try:
        await _clone_profile()
        logger.info("Browser profile clone prewarmed")
    except Exception as e:
        logger.warning(f"Browser clone prewarm failed (will retry on launch): {e}")


_LOCK_ERROR_MARKERS: tuple[str, ...] = (
    "ProcessSingleton",
    "SingletonLock",
    "profile appears to be in use",
    "user data directory is already in use",
)


def _is_profile_lock_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _LOCK_ERROR_MARKERS)


async def _launch_or_lock_error(pws: Any, launch_dir: Path, profile: str) -> Any:
    try:
        return await _launch_context(pws, launch_dir, profile)
    except Exception as e:
        if _is_profile_lock_error(e):
            raise RuntimeError(
                "Chrome is already using this profile. Fully quit Chrome and "
                "try again, or set BROWSER_PROFILE_PATH to a separate profile "
                "directory to run alongside it."
            ) from e
        raise


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
        headless=False,
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
        _start_lifecycle_prompt()

        from patchright.async_api import async_playwright

        _PWS = await async_playwright().start()

        profile = _profile_name()

        if config.browser_profile_path:
            _CONTEXT = await _launch_or_lock_error(_PWS, _profile_dir(), profile)
        else:
            clone_dir = await _clone_profile()
            _CONTEXT = await _launch_or_lock_error(_PWS, clone_dir, profile)

        _PAGE = await _CONTEXT.new_page()
        await _PAGE.bring_to_front()

        _LAST_ACTION_TS = time.monotonic()
        logger.info("Browser session launched")

        return _PWS, _CONTEXT, _PAGE
    except Exception:
        if _CONTEXT is not None:
            try:
                await _CONTEXT.close()
            except Exception as e:
                logger.warning(f"Error closing context after launch failure: {e}")
        if _PWS is not None:
            try:
                await _PWS.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright after launch failure: {e}")
        _PWS = None
        _CONTEXT = None
        _PAGE = None
        raise
    finally:
        _LAUNCHING = False


async def close_session() -> None:
    global _PWS, _CONTEXT, _PAGE, _LAUNCH_LOOP, _ACTION_COUNT, _LAST_ACTION_TS
    global _KEEP_OPEN, _FRAME_SELECTOR, _CLONE_LOCK, _LIFECYCLE_TASK
    global _ACTION_CAP_CEILING, _SESSION_START_TS, _TEMPO

    if _LIFECYCLE_TASK is not None:
        _LIFECYCLE_TASK.cancel()
        _LIFECYCLE_TASK = None

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
    _ACTION_CAP_CEILING = 0
    _LAST_ACTION_TS = 0.0
    _ACTION_TIMES.clear()
    _SESSION_START_TS = 0.0
    _TEMPO = 1.0
    _KEEP_OPEN = None
    _FRAME_SELECTOR = None
    _CLONE_LOCK = None
    humanize.reset_cursor()
    task.discard()


async def pace() -> None:
    global _LAST_ACTION_TS, _ACTION_COUNT, _ACTION_CAP_CEILING
    global _SESSION_START_TS, _TEMPO

    now = time.monotonic()
    if _SESSION_START_TS == 0.0:
        _SESSION_START_TS = now

    session_minutes = (now - _SESSION_START_TS) / 60.0
    _TEMPO = guardrails.next_tempo(_TEMPO)
    gap = (
        guardrails.random_gap()
        * guardrails.fatigue_multiplier(session_minutes)
        * _TEMPO
    )
    elapsed = now - _LAST_ACTION_TS
    if elapsed < gap:
        await asyncio.sleep(gap - elapsed)

    now = time.monotonic()
    _ACTION_TIMES.append(now)
    cutoff = now - _RATE_WINDOW_SECONDS
    while _ACTION_TIMES and _ACTION_TIMES[0] < cutoff:
        _ACTION_TIMES.popleft()
    window_span = now - _ACTION_TIMES[0] if len(_ACTION_TIMES) > 1 else 0.0
    throttle = guardrails.rate_throttle_delay(len(_ACTION_TIMES), window_span)
    if throttle > 0.0:
        await asyncio.sleep(throttle)

    idle_gap = guardrails.maybe_idle_gap()
    if idle_gap > 0.0:
        await asyncio.sleep(idle_gap)

    _LAST_ACTION_TS = time.monotonic()
    _ACTION_COUNT += 1
    if _ACTION_CAP_CEILING == 0:
        _ACTION_CAP_CEILING = guardrails.action_cap()

    if _ACTION_COUNT >= _ACTION_CAP_CEILING:
        cb = guardrails.browse_confirmation_callback
        if cb is not None:
            go = await cb(
                "Session Cap Reached",
                f"{_ACTION_COUNT} browser actions performed. Continue?",
            )
            if not go:
                raise RuntimeError("Action cap reached and user denied continuation")
        else:
            logger.warning(
                f"Browser action cap ({_ACTION_CAP_CEILING}) reached with no "
                "confirmation callback; continuing autonomously."
            )
        _ACTION_COUNT = 0
        _ACTION_CAP_CEILING = 0
