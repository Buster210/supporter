from __future__ import annotations

import asyncio
import contextlib
import contextvars
import shutil
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...config import config
from ...logger import logger
from . import blocklist, debug_overlay, guardrails, humanize, recorder
from . import profiles as profiles_mod

__all__: list[str] = ["guardrails", "profiles_mod"]

if TYPE_CHECKING:
    from patchright.async_api import BrowserContext, Page, Playwright

_PWS: Playwright | None = None
_CONTEXT: BrowserContext | None = None
_LAUNCHING: bool = False
_LAUNCH_LOOP: object | None = None
_CLONE_LOCK: asyncio.Lock | None = None
_RATE_WINDOW_SECONDS: float = 60.0
_CLEANUP_TASK: asyncio.Task[None] | None = None

# Idle auto-close: a single context-level monotonic timestamp of the last
# browser interaction (any agent), plus the background monitor task that closes
# the whole session after config.browser_idle_close_seconds of inactivity.
_LAST_ACTIVITY_TS: float = 0.0
_IDLE_TASK: asyncio.Task[None] | None = None

# Per-agent identity via contextvar (default "main" for backward-compat)
_AGENT_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "browser_agent_id", default="main"
)

# Per-agent active pages and frame selectors
_PAGES: dict[str, Page] = {}
_FRAME_SELECTORS: dict[str, str | None] = {}

# Per-agent page ownership for strict own-tab enforcement
_OWNED_PAGES: dict[str, set[Page]] = {}

# Per-agent pacing state (global rate window remains shared)
_ACTION_COUNT: dict[str, int] = {}
_ACTION_CAP_CEILING: dict[str, int] = {}
_LAST_ACTION_TS: dict[str, float] = {}
_SESSION_START_TS: dict[str, float] = {}
_TEMPO: dict[str, float] = {}

_ACTION_TIMES: deque[float] = deque()  # Shared across all agents for rate limiting

_SELECTED_PROFILE: str | None = None

_BLANK_URLS: frozenset[str] = frozenset({"about:blank", "chrome://newtab/", ""})

# Lock for per-agent page allocation (distinct from _CLONE_LOCK for browser launch)
_ALLOC_LOCK: asyncio.Lock | None = None


def current_agent_id() -> str:
    """Get the current agent ID, respecting the parallel-pilots config flag.

    When browser_parallel_pilots is False, always returns "main" for backward-compat.
    """
    if not config.browser_parallel_pilots:
        return "main"
    return _AGENT_ID.get()


def set_agent_id(aid: str) -> contextvars.Token[str] | None:
    """Set the current agent ID. Returns a token for reset_contextvar."""
    if not config.browser_parallel_pilots:
        return None
    return _AGENT_ID.set(aid)


def reset_contextvar(token: contextvars.Token[str] | None) -> None:
    """Reset a contextvar to its previous state."""
    if not isinstance(token, contextvars.Token):
        return
    _AGENT_ID.reset(token)


def _get_alloc_lock() -> asyncio.Lock:
    """Get or create the per-agent page allocation lock."""
    global _ALLOC_LOCK
    if _ALLOC_LOCK is None:
        _ALLOC_LOCK = asyncio.Lock()
    return _ALLOC_LOCK


def _get_agent_pace_state(aid: str) -> dict[str, Any]:
    """Get per-agent pacing state, initializing if needed."""
    if aid not in _ACTION_COUNT:
        _ACTION_COUNT[aid] = 0
        _ACTION_CAP_CEILING[aid] = 0
        _LAST_ACTION_TS[aid] = 0.0
        _SESSION_START_TS[aid] = 0.0
        _TEMPO[aid] = 1.0
    return {
        "action_count": _ACTION_COUNT[aid],
        "action_cap_ceiling": _ACTION_CAP_CEILING[aid],
        "last_action_ts": _LAST_ACTION_TS[aid],
        "session_start_ts": _SESSION_START_TS[aid],
        "tempo": _TEMPO[aid],
    }


def _get_agent_page(aid: str) -> Page | None:
    """Get the active page for a given agent."""
    return _PAGES.get(aid)


def _set_agent_page(aid: str, page: Any) -> None:
    """Set the active page for a given agent."""
    _PAGES[aid] = page


def _get_agent_frame_selector(aid: str) -> str | None:
    """Get the frame selector for a given agent."""
    return _FRAME_SELECTORS.get(aid)


def _set_agent_frame_selector(aid: str, selector: str | None) -> None:
    """Set the frame selector for a given agent."""
    _FRAME_SELECTORS[aid] = selector


def _get_agent_owned_pages(aid: str) -> set[Page]:
    """Get the set of pages owned by a given agent."""
    return _OWNED_PAGES.get(aid, set())


def _register_owned_page(aid: str, page: Any) -> None:
    """Register a page as owned by a given agent."""
    if aid not in _OWNED_PAGES:
        _OWNED_PAGES[aid] = set()
    _OWNED_PAGES[aid].add(page)


def drop_page(page: Any) -> None:
    """Drop a (now-closed) page from the current agent's ownership.

    Clears the active slot if it pointed at this page so a stale, closed
    handle is never returned by active_page().
    """
    aid = current_agent_id()
    owned = _OWNED_PAGES.get(aid)
    if owned is not None:
        owned.discard(page)
    if _PAGES.get(aid) is page:
        _PAGES.pop(aid, None)


async def _install_block_route(context: Any) -> None:
    async def _handler(route: Any) -> None:
        request = route.request
        try:
            if (
                blocklist.host_blocked(request.url)
                or request.resource_type in blocklist.BLOCKED_TYPES
            ):
                await route.abort()
            else:
                await route.continue_()
        except Exception as exc:
            logger.debug(f"Block-route handler failed for {request.url}: {exc}")
            # Leaving a route unhandled hangs the request until timeout; fall back
            # to letting it through. If the route was already handled this no-ops.
            try:
                await route.continue_()
            except Exception:
                logger.debug(f"Block-route fallback failed for {request.url}")

    await context.route("**/*", _handler)


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


def note_activity() -> None:
    """Record a browser interaction for the idle auto-close monitor."""
    global _LAST_ACTIVITY_TS
    _LAST_ACTIVITY_TS = time.monotonic()


async def _idle_monitor() -> None:
    """Close the whole session after a stretch of no browser interaction.

    Context-level (not per-agent): any agent's action refreshes the shared
    _LAST_ACTIVITY_TS. After config.browser_idle_close_seconds with no activity
    the browser is torn down. A value <= 0 disables the monitor entirely, so the
    browser persists until an explicit model/user close.
    """
    idle_limit = config.browser_idle_close_seconds
    if idle_limit <= 0:
        return
    # Poll at a fraction of the limit so we close within ~one tick of the
    # deadline without busy-waiting; floor keeps tests/short limits responsive.
    interval = max(1.0, min(30.0, idle_limit / 10.0))
    while True:
        try:
            await asyncio.sleep(interval)
            if _CONTEXT is None:
                return
            idle_for = time.monotonic() - _LAST_ACTIVITY_TS
            should_close = idle_for >= idle_limit
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Browser idle monitor failed: {e}")
            return
        if should_close:
            logger.info(
                f"Browser idle for {idle_for:.0f}s "
                f"(limit {idle_limit}s) — auto-closing session."
            )
            await close_session()
            return


def _start_idle_monitor() -> None:
    global _IDLE_TASK
    if _IDLE_TASK is not None or config.browser_idle_close_seconds <= 0:
        return
    note_activity()
    task = asyncio.ensure_future(_idle_monitor())
    _IDLE_TASK = task
    task.add_done_callback(_clear_idle_task)


def _clear_idle_task(_task: object) -> None:
    global _IDLE_TASK
    _IDLE_TASK = None


def _clear_cleanup_task(_task: object) -> None:
    global _CLEANUP_TASK
    _CLEANUP_TASK = None


async def cleanup_blank_tabs() -> None:
    """Close unowned blank tabs to avoid clutter.

    Under parallel pilots, only blanks NOT owned by any agent are removed.
    The FIRST agent adopting the existing blank page is recorded as its owner.
    """
    if _CONTEXT is None:
        return

    # Collect all owned pages across all agents
    all_owned: set[Page] = set()
    for owned_set in _OWNED_PAGES.values():
        all_owned.update(owned_set)

    for tab in list(_CONTEXT.pages):
        # Skip tabs owned by any agent (they're not "orphan blank")
        if tab in all_owned:
            continue
        try:
            if tab.url in _BLANK_URLS:
                await tab.close()
        except Exception as e:
            logger.debug(f"cleanup_blank_tabs: failed to close tab: {e}")


async def resolve_close_at_task_end() -> str:
    """Called at task end; releases the current agent's pages, leaving the
    browser running.

    The browser is never closed at task end any more — it persists until the
    idle monitor fires (config.browser_idle_close_seconds) or an explicit
    model/user close. This only frees the finished agent's tabs (keeping at
    least one tab open in the shared context).
    """
    aid = current_agent_id()
    if not is_active():
        return ""
    await release_agent(aid)
    return "Browser left open (closes after idle timeout or on explicit close)."


def is_active() -> bool:
    """True when the current agent has a recorded active page."""
    return _PAGES.get(current_agent_id()) is not None


def active_page() -> Any:
    """Get the active page for the current agent, dropping a dead reference.

    If the recorded page has been closed, its entry is purged so the next
    get_session() lazily reallocates instead of returning a stale handle.
    """
    aid = current_agent_id()
    page = _PAGES.get(aid)
    if page is not None and _is_page_closed(page):
        _PAGES.pop(aid, None)
        return None
    return page


async def session_status() -> dict[str, Any]:
    """Read-only snapshot of the browser session state for supervisor decisions."""
    now = time.monotonic()
    aid = current_agent_id()
    page = _PAGES.get(aid)
    if page is None and aid == "main" and _PWS is not None:
        # For main agent, check if we have any page
        page = next(iter(_PAGES.values())) if _PAGES else None
    last_ts = _LAST_ACTION_TS.get(aid, 0.0)
    idle_s = (now - last_ts) if last_ts > 0.0 else -1.0
    tab_count = len(_CONTEXT.pages) if _CONTEXT is not None else 0
    url = ""
    title = ""
    if page is not None:
        with contextlib.suppress(Exception):
            url = page.url or ""
        with contextlib.suppress(Exception):
            title = await page.title()
    return {
        "active": page is not None,
        "launching": _LAUNCHING,
        "url": url,
        "title": title,
        "tabs": tab_count,
        "idle_seconds": round(idle_s, 2) if idle_s >= 0 else None,
        "idle_close_seconds": config.browser_idle_close_seconds,
    }


def list_pages() -> list[Any]:
    """List pages owned by the current agent (filtered against live context)."""
    if _CONTEXT is None:
        return []
    aid = current_agent_id()
    owned = _OWNED_PAGES.get(aid, set())
    # Return only pages that are both owned and still live
    return [p for p in owned if p in _CONTEXT.pages]


def is_blank(page: Any) -> bool:
    try:
        return page.url in _BLANK_URLS
    except Exception:
        return False


def set_active(page: Any) -> None:
    """Set the active page for the current agent and register ownership."""
    aid = current_agent_id()
    _PAGES[aid] = page
    _FRAME_SELECTORS[aid] = None
    _register_owned_page(aid, page)


def active_frame_selector() -> str | None:
    aid = current_agent_id()
    return _FRAME_SELECTORS.get(aid)


def set_frame(selector: str | None) -> None:
    aid = current_agent_id()
    _FRAME_SELECTORS[aid] = selector


def _profile_dir() -> Path:
    if config.browser_profile_path:
        return Path(config.browser_profile_path).expanduser().resolve()
    home = Path.home()
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Google" / "Chrome"
    return home / ".config" / "google-chrome"


async def _resolve_profile_name() -> str:
    global _SELECTED_PROFILE

    if config.browser_profile_name:
        return config.browser_profile_name
    if _SELECTED_PROFILE is not None:
        return _SELECTED_PROFILE

    profiles = profiles_mod.list_profiles(_profile_dir())
    if not profiles:
        _SELECTED_PROFILE = "Default"
        return _SELECTED_PROFILE
    if len(profiles) == 1:
        _SELECTED_PROFILE = profiles[0].dir_name
        return _SELECTED_PROFILE

    cb = guardrails.browse_profile_select_callback
    if cb is None:
        raise RuntimeError(
            "Browser profile not selected and no interactive picker available; "
            "set BROWSER_PROFILE_NAME environment variable."
        )
    chosen = await cb(profiles)
    if not chosen:
        raise RuntimeError("Browser profile selection cancelled.")
    _SELECTED_PROFILE = chosen
    return _SELECTED_PROFILE


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


async def _clone_profile(profile: str) -> Path:
    user_data_dir = _profile_dir()
    async with _clone_lock():
        return await asyncio.to_thread(_build_clone, user_data_dir, profile)


async def prewarm_clone() -> None:
    if config.browser_profile_path:
        return
    if not config.browser_profile_name:
        return
    # Check if any page already exists (per-agent dict check)
    if _PAGES:
        return
    try:
        await _clone_profile(config.browser_profile_name)
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


async def _acquire_agent_page(aid: str) -> Any:
    """Allocate (or adopt) the calling agent's page under the allocation lock.

    Adopts an existing blank tab only when it is not already owned by another
    agent, so concurrent pilots never land on the same tab; otherwise opens a
    fresh page. Idempotent: a live page already recorded for the agent is reused.
    """
    async with _get_alloc_lock():
        existing = _PAGES.get(aid)
        if existing is not None and not _is_page_closed(existing):
            return existing
        all_owned: set[Page] = set()
        for owned_set in _OWNED_PAGES.values():
            all_owned.update(owned_set)
        if _CONTEXT is None:
            raise RuntimeError("Browser context closed before page could be allocated")
        blank = next(
            (p for p in _CONTEXT.pages if is_blank(p) and p not in all_owned),
            None,
        )
        page = blank if blank is not None else await _CONTEXT.new_page()
        _PAGES[aid] = page
        _register_owned_page(aid, page)
        return page


async def get_session() -> tuple[Any, Any, Any]:
    global _PWS, _CONTEXT, _LAUNCHING, _LAUNCH_LOOP

    # Determine agent id early for per-agent page handling
    aid = current_agent_id()

    # Check if this agent already has an active page
    agent_page = _PAGES.get(aid)
    if (
        agent_page is not None
        and not _is_page_closed(agent_page)
        and _CONTEXT is not None
        and _PWS is not None
    ):
        return _PWS, _CONTEXT, agent_page

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
        # After launch completes, check if this agent has a page now
        agent_page = _PAGES.get(aid)
        if agent_page is not None and _CONTEXT is not None and _PWS is not None:
            return _PWS, _CONTEXT, agent_page
        # Context is live but this agent has no page yet — allocate one.
        if _CONTEXT is not None and _PWS is not None:
            agent_page = await _acquire_agent_page(aid)
            return _PWS, _CONTEXT, agent_page
        raise RuntimeError("Browser session launch failed")

    loop = asyncio.get_running_loop()
    _LAUNCH_LOOP = loop
    _LAUNCHING = True

    try:
        from patchright.async_api import async_playwright

        _PWS = await async_playwright().start()

        profile = await _resolve_profile_name()

        if config.browser_profile_path:
            _CONTEXT = await _launch_or_lock_error(_PWS, _profile_dir(), profile)
        else:
            clone_dir = await _clone_profile(profile)
            _CONTEXT = await _launch_or_lock_error(_PWS, clone_dir, profile)

        if blocklist.should_block_resources():
            try:
                await _install_block_route(_CONTEXT)
            except Exception as exc:
                logger.warning(f"Could not install resource block route: {exc}")

        # Per-agent page allocation under lock to prevent races on blank adoption.
        agent_page = await _acquire_agent_page(aid)

        await agent_page.bring_to_front()
        if config.browser_debug_overlay:
            await debug_overlay.inject_overlay(agent_page)

        _LAST_ACTION_TS[aid] = time.monotonic()
        logger.info("Browser session launched")

        _start_idle_monitor()

        global _CLEANUP_TASK
        _CLEANUP_TASK = asyncio.ensure_future(cleanup_blank_tabs())
        _CLEANUP_TASK.add_done_callback(_clear_cleanup_task)

        return _PWS, _CONTEXT, agent_page
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
        _PAGES.clear()
        raise
    finally:
        _LAUNCHING = False


def _is_page_closed(page: Any) -> bool:
    """Check if a page is closed (safe guard against stale references)."""
    try:
        return bool(page.is_closed())
    except Exception:
        return True


# Force-close bounds each teardown await so a wedged/hung session can't block
# recovery; on timeout we swallow and still reset all globals.
_FORCE_CLOSE_TIMEOUT_S = 5.0


async def close_session(*, force: bool = False) -> None:
    """Full teardown of browser context. Clears ALL per-agent state."""
    global _PWS, _CONTEXT, _LAUNCH_LOOP, _CLONE_LOCK, _ALLOC_LOCK
    global _IDLE_TASK, _CLEANUP_TASK, _LAST_ACTIVITY_TS

    # Cancel the idle monitor — but never cancel ourselves, since the monitor
    # itself calls close_session() on timeout (self-cancel would raise inside
    # the teardown awaits below).
    current = asyncio.current_task()
    if _IDLE_TASK is not None and _IDLE_TASK is not current:
        _IDLE_TASK.cancel()
    _IDLE_TASK = None

    if _CLEANUP_TASK is not None:
        _CLEANUP_TASK.cancel()
        _CLEANUP_TASK = None

    try:
        if _CONTEXT is not None:
            if force:
                await asyncio.wait_for(_CONTEXT.close(), timeout=_FORCE_CLOSE_TIMEOUT_S)
            else:
                await _CONTEXT.close()
    except Exception as e:
        logger.warning(f"Error closing browser context: {e}")

    try:
        if _PWS is not None:
            if force:
                await asyncio.wait_for(_PWS.stop(), timeout=_FORCE_CLOSE_TIMEOUT_S)
            else:
                await _PWS.stop()
    except Exception as e:
        logger.warning(f"Error stopping playwright: {e}")

    _PWS = None
    _CONTEXT = None
    _PAGES.clear()
    _FRAME_SELECTORS.clear()
    _OWNED_PAGES.clear()
    _ACTION_COUNT.clear()
    _ACTION_CAP_CEILING.clear()
    _LAST_ACTION_TS.clear()
    _SESSION_START_TS.clear()
    _TEMPO.clear()
    _ACTION_TIMES.clear()
    _LAUNCH_LOOP = None
    _LAST_ACTIVITY_TS = 0.0
    _CLONE_LOCK = None
    _ALLOC_LOCK = None
    humanize.reset_cursor()
    recorder.discard_all()  # Per-agent recorder cleanup


async def pace() -> None:
    """Apply pacing delays for the current agent.

    WHY: Per-agent pacing state (action count, timing) keyed by agent ID,
    but the 60s rate-window throttle (_ACTION_TIMES) remains GLOBAL so
    the shared Chrome profile's aggregate request rate stays bounded.
    """
    aid = current_agent_id()
    now = time.monotonic()

    # Initialize per-agent pacing state if needed (per-key so a partially
    # populated agent slot can't raise KeyError mid-pace).
    _ACTION_COUNT.setdefault(aid, 0)
    _ACTION_CAP_CEILING.setdefault(aid, 0)
    _LAST_ACTION_TS.setdefault(aid, 0.0)
    _SESSION_START_TS.setdefault(aid, 0.0)
    _TEMPO.setdefault(aid, 1.0)

    if _SESSION_START_TS[aid] == 0.0:
        _SESSION_START_TS[aid] = now

    session_minutes = (now - _SESSION_START_TS[aid]) / 60.0
    _TEMPO[aid] = guardrails.next_tempo(_TEMPO[aid])
    gap = (
        guardrails.random_gap()
        * guardrails.fatigue_multiplier(session_minutes)
        * _TEMPO[aid]
    )
    elapsed = now - _LAST_ACTION_TS[aid]
    if elapsed < gap:
        await asyncio.sleep(gap - elapsed)

    now = time.monotonic()
    _ACTION_TIMES.append(now)  # Global rate window
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

    _LAST_ACTION_TS[aid] = time.monotonic()
    note_activity()
    _ACTION_COUNT[aid] += 1
    if _ACTION_CAP_CEILING[aid] == 0:
        _ACTION_CAP_CEILING[aid] = guardrails.action_cap()

    if _ACTION_COUNT[aid] >= _ACTION_CAP_CEILING[aid]:
        cb = guardrails.browse_confirmation_callback
        if cb is not None:
            go = await cb(
                "Session Cap Reached",
                f"{_ACTION_COUNT[aid]} browser actions performed. Continue?",
            )
            if not go:
                raise RuntimeError("Action cap reached and user denied continuation")
        else:
            logger.warning(
                f"Browser action cap ({_ACTION_CAP_CEILING[aid]}) reached with no "
                "confirmation callback; continuing autonomously."
            )
        _ACTION_COUNT[aid] = 0
        _ACTION_CAP_CEILING[aid] = 0


async def release_agent(aid: str) -> None:
    """Release an agent's pages and clean up its state.

    Closes the agent's owned tabs (suppressing per-tab errors) and drops its
    entries from the per-agent dicts. The browser process is NEVER torn down
    here — it persists until the idle monitor fires or an explicit close. To
    avoid leaving a context with zero pages (which some Chromium builds treat
    as a quit), at least one tab is always kept open.
    """
    owned = _OWNED_PAGES.get(aid, set())

    # Compute how many live tabs would remain across the whole context after we
    # close this agent's owned tabs, so we can keep one alive if it's the last.
    live_owned = [p for p in owned if not _is_page_closed(p)]
    other_live_tabs = 0
    if _CONTEXT is not None:
        owned_set = set(live_owned)
        other_live_tabs = sum(
            1 for p in _CONTEXT.pages if p not in owned_set and not _is_page_closed(p)
        )

    keep_one = other_live_tabs == 0
    for page in live_owned:
        if keep_one:
            # Leave one tab open; blank it so it isn't tied to this task.
            keep_one = False
            try:
                await page.goto("about:blank")
            except Exception as e:
                logger.debug(f"release_agent: failed to blank kept tab: {e}")
            continue
        try:
            await page.close()
        except Exception as e:
            logger.debug(f"release_agent: failed to close page: {e}")

    # If we intended to keep one tab but had no live owned page to keep,
    # open a fresh blank tab so the context never reaches zero pages.
    if keep_one and _CONTEXT is not None:
        try:
            await _CONTEXT.new_page()
        except Exception as e:
            logger.debug(f"release_agent: failed to open replacement tab: {e}")

    # Drop per-agent state
    _PAGES.pop(aid, None)
    _FRAME_SELECTORS.pop(aid, None)
    _OWNED_PAGES.pop(aid, None)
    _ACTION_COUNT.pop(aid, None)
    _ACTION_CAP_CEILING.pop(aid, None)
    _LAST_ACTION_TS.pop(aid, None)
    _SESSION_START_TS.pop(aid, None)
    _TEMPO.pop(aid, None)

    # Discard this agent's in-progress recording, if any.
    try:
        recorder.discard(aid)
    except Exception as e:
        logger.debug(f"release_agent: recorder discard failed: {e}")
