from typing import Any

from ...logger import logger

__all__ = [
    "_BOOTSTRAP_JS",
    "_CURSOR_JS",
    "inject_overlay",
    "overlay_click",
    "overlay_move",
]

# Recreated on every document load via add_init_script so the overlay survives
# navigation. Idempotent: guards on existing elements.
_BOOTSTRAP_JS = r"""
(() => {
  const install = () => {
    const root = document.body || document.documentElement;
    if (!root) return;
    if (!document.getElementById('__dbg_cursor')) {
      const c = document.createElement('canvas');
      c.id = '__dbg_cursor';
      c.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;' +
        'pointer-events:none;z-index:2147483646;';
      c.width = window.innerWidth; c.height = window.innerHeight;
      root.appendChild(c);
    }
    if (!document.getElementById('__dbg_pointer')) {
      const p = document.createElement('div');
      p.id = '__dbg_pointer';
      p.style.cssText = 'position:fixed;top:0;left:0;width:14px;height:14px;' +
        'margin:-7px 0 0 -7px;border:2px solid rgba(0,0,0,0.65);border-radius:50%;' +
        'background:rgba(255,255,255,0.95);box-shadow:0 0 5px 1px rgba(0,0,0,0.5);' +
        'pointer-events:none;z-index:2147483647;will-change:transform;' +
        'transition:transform 60ms linear;transform:translate(-100px,-100px);';
      root.appendChild(p);
      // Default to viewport center (visible) so the cursor shows the instant a
      // page loads — not only after an action. sessionStorage is per-origin, so
      // a freshly-navigated site has no saved pos and would otherwise park the
      // ring off-screen and invisible.
      let px = window.innerWidth / 2, py = window.innerHeight / 2;
      try {
        const s = sessionStorage.getItem('__dbg_pos');
        if (s) { const a = s.split(','); px = parseFloat(a[0]); py = parseFloat(a[1]); }
      } catch (e) {}
      if (isNaN(px) || isNaN(py)) {
        px = window.innerWidth / 2; py = window.innerHeight / 2;
      }
      p.style.transform = 'translate(' + px + 'px,' + py + 'px)';
    }
  };
  if (document.body) install();
  else document.addEventListener('DOMContentLoaded', install, { once: true });
})();
"""

# pydoll-style accumulating dots (blue move, red click) PLUS a persistent cursor
# ring that follows the agent. Self-heals canvas + cursor if a navigation wiped them.
_CURSOR_JS = r"""
(() => {
  const root = document.body || document.documentElement;
  if (!root) return;
  let c = document.getElementById('__dbg_cursor');
  if (!c) {
    c = document.createElement('canvas');
    c.id = '__dbg_cursor';
    c.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;' +
      'pointer-events:none;z-index:2147483646;';
    c.width = window.innerWidth; c.height = window.innerHeight;
    root.appendChild(c);
  } else if (c.width !== window.innerWidth || c.height !== window.innerHeight) {
    // Re-stamp on viewport resize so dots stay in the right coordinate space
    // (this clears the accumulated trail — acceptable on a resize).
    c.width = window.innerWidth; c.height = window.innerHeight;
  }
  let p = document.getElementById('__dbg_pointer');
  if (!p) {
    p = document.createElement('div');
    p.id = '__dbg_pointer';
    p.style.cssText = 'position:fixed;top:0;left:0;width:14px;height:14px;' +
      'margin:-7px 0 0 -7px;border:2px solid rgba(0,0,0,0.65);border-radius:50%;' +
      'background:rgba(255,255,255,0.95);box-shadow:0 0 5px 1px rgba(0,0,0,0.5);' +
      'pointer-events:none;z-index:2147483647;will-change:transform;' +
      'transition:transform 60ms linear;transform:translate(-100px,-100px);';
    root.appendChild(p);
  }
  const ctx = c.getContext('2d');
  const x = __X__, y = __Y__, kind = '__KIND__';
  if (kind === 'click') {
    ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.283185);
    ctx.fillStyle = 'rgba(255,50,50,0.9)'; ctx.fill();
  } else {
    ctx.beginPath(); ctx.arc(x, y, 3, 0, 6.283185);
    ctx.fillStyle = 'rgba(0,150,255,0.6)'; ctx.fill();
  }
  p.style.transform = 'translate(' + x + 'px,' + y + 'px)';
  try { sessionStorage.setItem('__dbg_pos', x + ',' + y); } catch (e) {}
})();
"""

_first_draw_logged = False
_entry_logged = False


async def _draw(page: Any, x: float, y: float, kind: str) -> None:
    global _first_draw_logged, _entry_logged
    if not _entry_logged:
        _entry_logged = True
        logger.info(f"debug overlay: _draw entered (kind={kind})")
    try:
        script = (
            _CURSOR_JS.replace("__X__", str(round(x)))
            .replace("__Y__", str(round(y)))
            .replace("__KIND__", kind)
        )
        await page.evaluate(script)
        if not _first_draw_logged:
            _first_draw_logged = True
            logger.info(f"debug overlay: first draw OK (kind={kind})")
        else:
            logger.debug(f"overlay draw kind={kind} @ ({round(x)},{round(y)})")
    except Exception as exc:
        # Loud on purpose: a swallowed draw failure is exactly the invisible
        # case we are trying to diagnose in the live session.
        logger.warning(f"debug overlay draw FAILED (kind={kind}): {exc!r}")


async def inject_overlay(page: Any) -> None:
    """Install the persistent overlay on a page and re-arm it for future navigations."""
    try:
        await page.add_init_script(_BOOTSTRAP_JS)
        await page.evaluate(_BOOTSTRAP_JS)
        logger.info("debug overlay: injected (persists across navigation)")
    except Exception as exc:
        logger.warning(f"debug overlay inject FAILED: {exc!r}")


async def overlay_move(page: Any, x: float, y: float) -> None:
    await _draw(page, x, y, "move")


async def overlay_click(page: Any, x: float, y: float) -> None:
    await _draw(page, x, y, "click")
