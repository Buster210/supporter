from __future__ import annotations

import os
from typing import Any

import httpx

from ...logger import logger
from ..base import ToolError

_TAVILY_URL = "https://api.tavily.com/search"
_DEFAULT_MAX_RESULTS = 8
_MAX_RESULTS_CAP = 20
_SNIPPET_CAP = 500
_TIMEOUT = 20.0


async def fetch_tavily(api_key: str, query: str, max_results: int) -> dict[str, Any]:
    """POST to the Tavily search API and return the decoded JSON body.

    Isolated so tests can stub the network without faking httpx internals.
    """
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(_TAVILY_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, dict):
        raise ToolError("web_search got an unexpected (non-object) response")
    return data


def format_results(query: str, results: list[dict[str, Any]]) -> str:
    """Render Tavily results as a numbered, LLM-friendly frontier list.

    Each entry carries a real URL the caller can open with `browse`. Returns a
    plain-text block, not JSON, so it drops straight into the model's context.
    """
    if not results:
        return f"No web results for: {query}"
    lines = [f"Web results for: {query}", ""]
    for i, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        title = str(item.get("title", "")).strip() or "(untitled)"
        snippet = " ".join(str(item.get("content", "")).split())[:_SNIPPET_CAP]
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def web_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> str:
    """Real web search that returns actual result URLs to open and read.

    This is the FRONTIER SEEDER for research: it returns a ranked list of real
    pages (title, URL, snippet) from a live search index. It does NOT read the
    pages -- delegate the promising URLs to page-pilot to open, read, and emit
    claims. Use it to discover where to look, then go read there.

    Args:
        query: The search query string.
        max_results: How many results to return (1-20, default 8).

    Returns:
        A numbered text list of {title, URL, snippet} results.

    Raises:
        ToolError: Empty query, missing TAVILY_API_KEY, or a failed lookup.
    """
    query = (query or "").strip()
    if not query:
        raise ToolError("web_search requires a non-empty query")
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ToolError(
            "web_search unavailable: set TAVILY_API_KEY in the environment "
            "(.env) to enable real web search."
        )
    try:
        capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))
    except TypeError, ValueError:
        capped = _DEFAULT_MAX_RESULTS

    logger.info(f"Tool: web_search -- query='{query}' max_results={capped}")
    try:
        data = await fetch_tavily(api_key, query, capped)
    except httpx.HTTPStatusError as exc:
        raise ToolError(
            f"web_search failed ({exc.response.status_code}) for '{query}'"
        ) from exc
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"web_search failed for '{query}': {exc}") from exc

    results = data.get("results")
    results = results if isinstance(results, list) else []
    logger.info(f"Tool: web_search -- {len(results)} results for '{query}'")
    return format_results(query, results)
