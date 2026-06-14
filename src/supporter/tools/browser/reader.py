from __future__ import annotations

import asyncio
import re
from typing import Any

from ...logger import logger
from .core import BrowseRequest
from .support import _navigate_with_retry, _page_or_error, _wrap_action_errors

__all__ = ["_handle_links", "_handle_read"]

# Per-page and whole-batch caps on returned markdown, plus a link cap. Reading
# is for getting the substance into context, not mirroring the page -- bound it
# so a single huge page can't blow the window.
_PAGE_CHARS_CAP = 12_000
_BATCH_CHARS_CAP = 40_000
_MAX_LINKS = 40
_AUTOSCROLL_STEPS = 12
_AUTOSCROLL_PAUSE = 0.4
_URL_RE = re.compile(r"https?://[^\s]+")

# In-page reader: find the main content node, strip chrome, and emit clean
# markdown + metadata + a deduped in-content link list. Runs as a trusted fixed
# script (like _ROLE_NAME_JS / storage JS) -- not user `eval`, so no confirm.
_READER_JS = r"""
(opts) => {
  const BLOCK = 'h1,h2,h3,h4,h5,h6,p,li,blockquote,pre';
  const NOISE = 'script,style,noscript,nav,header,footer,aside,form,svg,' +
    'iframe,[aria-hidden="true"],[role="navigation"],[role="banner"],' +
    '[role="contentinfo"],.ad,.ads,.advert,.cookie,.newsletter';
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const meta = (sel, attr) => {
    const el = document.querySelector(sel);
    if (!el) return '';
    return norm(attr ? (el.getAttribute(attr) || '') : (el.textContent || ''));
  };

  const pickRoot = () => {
    if (opts.selector) {
      const el = document.querySelector(opts.selector);
      if (el) return el;
    }
    const named = document.querySelector('article, main, [role="main"]');
    if (named && norm(named.textContent).length > 200) return named;
    let best = document.body, bestScore = 0;
    const blocks = document.body
      ? document.body.querySelectorAll('div,section,article,main') : [];
    blocks.forEach((node) => {
      let score = 0;
      node.querySelectorAll(':scope > p').forEach((p) => {
        score += norm(p.textContent).length;
      });
      if (score > bestScore) { bestScore = score; best = node; }
    });
    return best;
  };

  const root = pickRoot();
  const clone = root.cloneNode(true);
  clone.querySelectorAll(NOISE).forEach((n) => n.remove());

  const lines = [];
  const nodes = clone.querySelectorAll(BLOCK);
  nodes.forEach((el) => {
    if (el.querySelector(BLOCK)) return;  // container, not a leaf block
    const tag = el.tagName.toLowerCase();
    if (tag === 'pre') {
      const code = (el.textContent || '').replace(/\s+$/, '');
      if (code.trim()) lines.push('```\n' + code + '\n```');
      return;
    }
    const text = norm(el.textContent);
    if (!text) return;
    if (/^h[1-6]$/.test(tag)) {
      lines.push('#'.repeat(Number(tag[1])) + ' ' + text);
    } else if (tag === 'li') {
      lines.push('- ' + text);
    } else if (tag === 'blockquote') {
      lines.push('> ' + text);
    } else {
      lines.push(text);
    }
  });

  const here = location.href;
  const seen = new Set();
  const links = [];
  const scope = (root === document.body ? document : root);
  const anchors = scope.querySelectorAll('a[href]');
  for (const a of anchors) {
    const href = a.href;
    if (!href || !/^https?:/.test(href) || href === here) continue;
    if (seen.has(href)) continue;
    const text = norm(a.textContent);
    if (!text) continue;
    seen.add(href);
    links.push({ text: text.slice(0, 160), href });
    if (links.length >= opts.maxLinks) break;
  }

  return {
    title: meta('meta[property="og:title"]', 'content') || norm(document.title),
    byline: meta('meta[name="author"]', 'content') ||
      meta('meta[property="article:author"]', 'content') ||
      meta('[rel="author"]'),
    published: meta('meta[property="article:published_time"]', 'content') ||
      meta('time[datetime]', 'datetime'),
    siteName: meta('meta[property="og:site_name"]', 'content'),
    url: here,
    markdown: lines.join('\n\n'),
    links,
    block_count: lines.length,
  };
};
"""


def _parse_urls(raw: str) -> list[str]:
    """Pull every http(s) URL out of the `url` field. Supports a single URL or
    several (whitespace/newline separated) for a batch read."""
    return _URL_RE.findall(raw or "")


def _format_read(data: dict[str, Any], *, char_cap: int = _PAGE_CHARS_CAP) -> str:
    """Render one extracted page as a clean, citable text block."""
    if not isinstance(data, dict):
        return "Error: reader returned no content."
    header: list[str] = []
    title = str(data.get("title", "")).strip()
    if title:
        header.append(f"# {title}")
    meta_bits = []
    for label, key in (("Source", "url"), ("Site", "siteName"), ("By", "byline"),
                       ("Published", "published")):
        val = str(data.get(key, "")).strip()
        if val:
            meta_bits.append(f"{label}: {val}")
    if meta_bits:
        header.append("\n".join(meta_bits))

    markdown = str(data.get("markdown", "")).strip()
    truncated = ""
    if len(markdown) > char_cap:
        omitted = len(markdown) - char_cap
        markdown = markdown[:char_cap].rstrip()
        truncated = (
            f"\n\n…(truncated: {omitted} more chars; scroll/read a selector "
            "for the rest)"
        )
    if not markdown:
        markdown = "(no readable article text found on this page)"

    links = data.get("links") or []
    link_lines = []
    for item in links:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href", "")).strip()
        if not href:
            continue
        text = str(item.get("text", "")).strip() or href
        link_lines.append(f"- {text} -> {href}")

    parts = ["\n\n".join(header)] if header else []
    parts.append(markdown + truncated)
    if link_lines:
        parts.append("## Links on this page\n" + "\n".join(link_lines))
    return "\n\n".join(p for p in parts if p).strip()


def _format_links(data: dict[str, Any]) -> str:
    """Render just the in-content link list for the `links` action."""
    links = data.get("links") or [] if isinstance(data, dict) else []
    lines = []
    for item in links:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href", "")).strip()
        if not href:
            continue
        text = str(item.get("text", "")).strip() or href
        lines.append(f"- {text} -> {href}")
    if not lines:
        return "(no outbound links found on this page)"
    return f"{len(lines)} links:\n" + "\n".join(lines)


async def _autoscroll(page: Any) -> None:
    """Scroll to the bottom in steps to trigger lazy-loaded content, stopping
    once the page height stabilizes or the step cap is hit."""
    prev = -1
    for _ in range(_AUTOSCROLL_STEPS):
        height = await page.evaluate(
            "() => { window.scrollTo(0, document.body.scrollHeight); "
            "return document.body.scrollHeight; }"
        )
        await asyncio.sleep(_AUTOSCROLL_PAUSE)
        if not isinstance(height, (int, float)) or height <= prev:
            break
        prev = height
    try:
        await page.evaluate("() => window.scrollTo(0, 0)")
    except Exception:
        logger.debug("autoscroll reset failed", exc_info=True)


async def _extract(page: Any, req: BrowseRequest) -> dict[str, Any]:
    if req.full_page:
        await _autoscroll(page)
    data = await page.evaluate(
        _READER_JS, {"maxLinks": _MAX_LINKS, "selector": req.selector or ""}
    )
    return data if isinstance(data, dict) else {}


@_wrap_action_errors("read")
async def _handle_read(req: BrowseRequest) -> str:
    """Read page(s) as clean markdown + metadata + links. Navigates first when
    a URL (or several) is given; otherwise reads the current page."""
    page = await _page_or_error()
    urls = _parse_urls(req.url)

    if not urls:
        data = await _extract(page, req)
        return _format_read(data)

    blocks: list[str] = []
    budget = _BATCH_CHARS_CAP
    for i, url in enumerate(urls, 1):
        try:
            await _navigate_with_retry(
                page,
                lambda u=url: page.goto(
                    u, wait_until="domcontentloaded", timeout=30_000
                ),
            )
            await asyncio.sleep(req.delay_ms / 1000.0)
            data = await _extract(page, req)
            per_cap = min(_PAGE_CHARS_CAP, max(0, budget))
            rendered = _format_read(data, char_cap=per_cap)
        except Exception as exc:  # one bad URL must not sink the batch
            rendered = f"Error reading {url}: {exc}"
        block = f"===== [{i}/{len(urls)}] {url} =====\n{rendered}"
        blocks.append(block)
        budget -= len(rendered)
        if budget <= 0 and i < len(urls):
            blocks.append(
                f"(batch char budget reached; stopped after {i}/{len(urls)} URLs)"
            )
            break
    return "\n\n".join(blocks)


@_wrap_action_errors("links")
async def _handle_links(req: BrowseRequest) -> str:
    """List the in-content outbound links (text -> absolute URL) on the current
    page, so you can decide what to open and follow next."""
    page = await _page_or_error()
    if req.url:
        urls = _parse_urls(req.url)
        if urls:
            await _navigate_with_retry(
                page,
                lambda: page.goto(
                    urls[0], wait_until="domcontentloaded", timeout=30_000
                ),
            )
            await asyncio.sleep(req.delay_ms / 1000.0)
    data = await page.evaluate(
        _READER_JS, {"maxLinks": _MAX_LINKS, "selector": req.selector or ""}
    )
    return _format_links(data if isinstance(data, dict) else {})
