from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any

from ...logger import logger
from ..base import ToolError
from .claims import ingest_claims, research_dir
from .loop import assess_research
from .search import _MAX_RESULTS_CAP, fetch_tavily
from .verify import build_report, verify_research

__all__ = [
    "_loads_lenient",
    "_reset_store",
    "deep_research",
    "ingest_claims",
    "question_id_for",
    "run_deep_research",
]

# Driver defaults. The loop is bounded by code, not by the model's discretion:
# rounds are capped, results-per-query are capped, and the stop decision comes
# from assess_research (saturation OR budget), never from a prompt.
_DEFAULT_ROUNDS = 3
_DEFAULT_RESULTS_PER_QUERY = 4
_DEFAULT_SUBQUERIES = 3
_MAX_ROUNDS_CAP = 8
_READ_CHARS_CAP = 8_000  # per-page text fed to the extractor
# Max concurrent browser-page reads per round; gated by browser_parallel_pilots.
_READ_CONCURRENCY = 3

# Fixed extraction/query-gen prompts. These are the only AI in the loop, and
# they are constant + schema-constrained, so drift is bounded to the NLP itself.
_QUERY_SYS = (
    "You generate web-search queries for a research question. Return diverse, "
    "specific queries that together cover the question from independent angles. "
    "No commentary -- only the structured list."
)
_EXTRACT_SYS = (
    "You extract atomic, checkable factual claims from one web page, relevant "
    "to a research question. Each claim is a single self-contained statement "
    "that the page asserts, plus a short verbatim snippet from the page that "
    "supports it. Do not invent claims the page does not make. Ignore opinion, "
    "navigation, and boilerplate. No commentary -- only the structured list."
)

_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
}
_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["statement"],
            },
        }
    },
    "required": ["claims"],
}


def question_id_for(question: str) -> str:
    """Stable, filesystem-safe id for a research question."""
    digest = hashlib.sha1(
        question.strip().encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    return f"q{digest}"


def _reset_store(question_id: str) -> None:
    """Clear any prior evidence/assess log so a run starts from a clean slate."""
    base = research_dir(question_id)
    for name in ("claims.jsonl", "assess_log.jsonl"):
        path = base / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug(f"deep_research: could not clear {path}: {exc}")


async def _structured(
    provider: Any, system: str, user: str, schema: dict[str, Any]
) -> Any:
    """One fixed-prompt, JSON-schema-constrained model call -- no tools.

    This is the only place the driver hands control to the model, and it is
    bounded: constant system prompt, a single user turn, structured output.
    Goes through the provider's generate() so it works with the pooled /
    fallback provider wrapper (no reaching into a raw client).
    """
    result = await provider.generate(
        user,
        {
            "system_instruction": system,
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    text = (getattr(result, "text", "") or "").strip()
    return _loads_lenient(text)


def _loads_lenient(text: str) -> Any:
    """Parse JSON from model output, tolerating stray markdown fences or prose
    around it (some models leak a trailing ``` even with structured output)."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError, TypeError:
        pass
    # Fall back to the outermost {...} slice.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug(f"deep_research: non-JSON model output: {exc}")
    return {}


async def _generate_queries(provider: Any, question: str, n: int) -> list[str]:
    user = f"Research question:\n{question}\n\nReturn up to {n} search queries."
    data = await _structured(provider, _QUERY_SYS, user, _QUERY_SCHEMA)
    raw = data.get("queries") if isinstance(data, dict) else None
    queries = [str(q).strip() for q in raw or [] if str(q).strip()]
    # Always anchor on the question itself; cap to n; dedup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for q in [question, *queries]:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= max(1, n):
            break
    return out


async def _extract_claims(
    provider: Any, question: str, url: str, text: str
) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    user = (
        f"Research question:\n{question}\n\nPage URL: {url}\n\n"
        f"Page content:\n{text[:_READ_CHARS_CAP]}"
    )
    data = await _structured(provider, _EXTRACT_SYS, user, _CLAIM_SCHEMA)
    raw = data.get("claims") if isinstance(data, dict) else None
    claims: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        claims.append(
            {
                "statement": statement,
                "url": url,
                "snippet": str(item.get("snippet", "")).strip(),
                "stance": "support",
            }
        )
    return claims


async def _search_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Live search -> ranked result dicts (url/title/content). Isolated for
    stubbing in tests."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ToolError(
            "deep_research unavailable: set TAVILY_API_KEY in the environment "
            "(.env) to enable real web search."
        )
    data = await fetch_tavily(api_key, query, max_results)
    results = data.get("results")
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


async def _read_page(url: str, full_page: bool) -> str:
    """Clean-read one page to markdown text. Isolated for stubbing in tests."""
    from ..browser.tool import browse

    return await browse("read", url=url, full_page=full_page)


async def _search_one(query: str, per_query: int) -> list[dict[str, Any]]:
    """Single search query, safe to gather. Re-raises ToolError; swallows others."""
    try:
        return await _search_results(query, per_query)
    except ToolError:
        raise
    except Exception as exc:
        logger.debug(f"deep_research: search failed for {query!r}: {exc}")
        return []


async def _read_extract_url(
    sem: asyncio.Semaphore,
    aid: str,
    url: str,
    provider: Any,
    question: str,
    full_page: bool,
) -> tuple[str, list[dict[str, Any]] | None]:
    """Read one page and extract claims under the concurrency semaphore.

    Returns (url, claims) where claims=None means the read failed (page not
    counted); claims=[] means the read succeeded but yielded no claims.
    Ingest is NOT called here — the caller serialises writes to avoid concurrent
    JSONL appends.

    Sets a distinct browser agent_id per call so concurrent reads each bind to
    their own tab (P1 per-agent page isolation). Semaphore caps concurrent tabs.
    """
    from ..browser.session import release_agent, reset_contextvar, set_agent_id

    token = set_agent_id(aid)
    text: str = ""
    try:
        async with sem:
            try:
                text = await _read_page(url, full_page)
            except Exception as exc:
                logger.debug(f"deep_research: read failed for {url}: {exc}")
                return url, None
            if not isinstance(text, str) or text.startswith("Error"):
                return url, None
    finally:
        import contextlib as _ctx

        with _ctx.suppress(Exception):
            await release_agent(aid)
        reset_contextvar(token)
    # Only reachable when read succeeded and text is valid; LLM call outside semaphore.
    try:
        return url, await _extract_claims(provider, question, url, text)
    except Exception as exc:
        logger.debug(f"deep_research: extract failed for {url}: {exc}")
        return url, []


async def run_deep_research(
    question: str,
    *,
    question_id: str | None = None,
    max_rounds: int = _DEFAULT_ROUNDS,
    results_per_query: int = _DEFAULT_RESULTS_PER_QUERY,
    subqueries: int = _DEFAULT_SUBQUERIES,
    full_page: bool = False,
    provider: Any = None,
    reset: bool = True,
) -> dict[str, Any]:
    """Code-driven research loop. The model is called ONLY for query
    generation, per-page claim extraction, and (by the caller) final synthesis;
    everything else -- search, read, ingest, the stop decision, verification,
    and report assembly -- is deterministic code.
    """
    question = (question or "").strip()
    if not question:
        raise ToolError("deep_research requires a non-empty question")
    if provider is None:
        from ...pool import get_provider

        provider = get_provider()

    qid = question_id or question_id_for(question)
    if reset:
        _reset_store(qid)
    rounds = max(1, min(int(max_rounds), _MAX_ROUNDS_CAP))
    per_query = max(1, min(int(results_per_query), _MAX_RESULTS_CAP))
    n_sub = max(1, int(subqueries))

    queries = await _generate_queries(provider, question, n_sub)
    trace: list[dict[str, Any]] = []
    visited: set[str] = set()
    round_no = 0
    from ...config import config as _cfg

    _concurrency = 1 if not _cfg.browser_parallel_pilots else _READ_CONCURRENCY
    read_sem = asyncio.Semaphore(_concurrency)

    while round_no < rounds:
        round_no += 1
        ingested = 0
        pages = 0

        # Parallel searches: all queries for this round run concurrently.
        try:
            search_lists = await asyncio.gather(
                *[_search_one(q, per_query) for q in queries]
            )
        except ToolError:
            raise

        # Dedup candidate URLs in the main loop before dispatch (constraint (c)).
        candidate_urls: list[str] = []
        for result_list in search_lists:
            for r in result_list[:per_query]:
                url = str(r.get("url", "")).strip()
                if url and url not in visited:
                    visited.add(url)
                    candidate_urls.append(url)

        # Parallel reads: each URL gets its own browser page via distinct agent_id.
        url_results = await asyncio.gather(
            *[
                _read_extract_url(
                    read_sem,
                    f"dr-r{round_no}-{i}",
                    url,
                    provider,
                    question,
                    full_page,
                )
                for i, url in enumerate(candidate_urls)
            ]
        )

        # Sequential ingest: JSONL writes must not interleave (constraint (c)).
        for _url, claims in url_results:
            if claims is None:
                continue
            pages += 1
            ingested += ingest_claims(
                qid,
                task_id=f"round-{round_no}",
                agent="deep_research",
                claims=claims,
                round=round_no,
            )

        assess = assess_research(qid, max_rounds=rounds)
        trace.append(
            {
                "round": round_no,
                "queries": queries,
                "pages_read": pages,
                "claims_ingested": ingested,
                "total_assertions": assess["total_assertions"],
                "new_this_round": assess["new_this_round"],
                "recommendation": assess["recommendation"],
                "reason": assess["reason"],
            }
        )
        logger.info(
            f"deep_research round={round_no} pages={pages} ingested={ingested} "
            f"total={assess['total_assertions']} rec={assess['recommendation']}"
        )
        if assess["recommendation"] == "stop":
            break
        # Code-driven gap closing: search the exact thin/contested statements.
        gaps = assess["uncorroborated_statements"] + assess["conflicted_statements"]
        queries = gaps[:n_sub] or [question]

    verify = verify_research(qid)
    report = build_report(qid)
    return {
        "question": question,
        "question_id": qid,
        "rounds_run": round_no,
        "stopped_because": trace[-1]["reason"] if trace else "no_rounds",
        "pages_read": len(visited),
        "verification": {
            "total_assertions": verify["total_assertions"],
            "corroborated": verify["corroborated"],
            "uncorroborated": verify["uncorroborated"],
            "conflicted": verify["conflicted"],
            "fully_verified": verify["fully_verified"],
        },
        "report": report,
        "trace": trace,
    }


async def deep_research(
    question: str,
    max_rounds: int = _DEFAULT_ROUNDS,
    results_per_query: int = _DEFAULT_RESULTS_PER_QUERY,
    full_page: bool = False,
) -> str:
    """Run a full, cross-verified web research loop and return a cited report.

    This is a self-contained deep researcher: from one question it searches the
    web, opens and reads the real pages, extracts atomic claims, and keeps going
    -- round after round, closing the thin/contested gaps -- until the evidence
    saturates or the round budget is hit. Every fact is cross-checked across
    independent sources: a claim is only "corroborated" when >=2 independent
    domains agree; single-source facts are flagged "uncorroborated" and
    disagreements "contested". Nothing is taken blindly.

    The loop is deterministic code -- search, read, ingest, the stop decision,
    verification, and report assembly all run without model discretion, so it
    cannot skip steps, stop early, or forget to cross-verify. Prefer this over
    driving web_search / browse / verify_claims by hand for any multi-step
    research question.

    Args:
        question: The research question to investigate.
        max_rounds: Hard ceiling on research rounds (1-8, default 3). The loop
            usually stops earlier once new findings saturate.
        results_per_query: Pages to open per search query (1-20, default 4).
        full_page: If True, auto-scroll each page to load lazy content before
            reading. Slower; use for infinite-scroll or JS-heavy sources.

    Returns:
        A JSON string with the cited report (assertions grouped corroborated /
        contested / uncorroborated), a verification summary, and a per-round
        trace. Write the final answer from this: state corroborated facts with
        citations, surface contested ones as disputed, flag single-source ones.

    Raises:
        ToolError: Empty question or missing TAVILY_API_KEY.
    """
    logger.info(f"Tool: deep_research -- question={question!r}")
    result = await run_deep_research(
        question,
        max_rounds=max_rounds,
        results_per_query=results_per_query,
        full_page=full_page,
    )
    logger.info(
        f"Tool: deep_research -- rounds={result['rounds_run']} "
        f"pages={result['pages_read']} "
        f"corroborated={result['verification']['corroborated']}"
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
