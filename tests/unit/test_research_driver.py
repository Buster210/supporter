from __future__ import annotations

from typing import Any

import pytest

from supporter.config import config
from supporter.tools.base import ToolError
from supporter.tools.research import driver as drv
from supporter.tools.research.claims import load_claims, research_dir
from supporter.tools.research.driver import (
    _extract_claims,
    _generate_queries,
    _loads_lenient,
    _reset_store,
    question_id_for,
    run_deep_research,
)


@pytest.fixture(autouse=True)
def isolate_research(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])


class _Provider:
    """Stand-in provider; the driver only touches it through _structured, which
    tests patch, so this is just an opaque sentinel."""


def test_loads_lenient_plain_json() -> None:
    assert _loads_lenient('{"a": 1}') == {"a": 1}


def test_loads_lenient_trailing_fence() -> None:
    # The live model emitted exactly this: valid JSON with a stray trailing ```
    raw = ' {\n  "claims": []\n}\n```'
    assert _loads_lenient(raw) == {"claims": []}


def test_loads_lenient_fenced_block_with_prose() -> None:
    raw = 'Here you go:\n```json\n{"x": [1, 2]}\n```\nDone.'
    assert _loads_lenient(raw) == {"x": [1, 2]}


def test_loads_lenient_garbage_returns_empty() -> None:
    assert _loads_lenient("not json at all") == {}
    assert _loads_lenient("") == {}


def test_question_id_stable_and_safe() -> None:
    a = question_id_for("What is X?")
    b = question_id_for("  What is X?  ")
    assert a == b
    assert a.startswith("q")
    assert "/" not in a and a == a.strip()


def test_reset_store_removes_files(tmp_path: Any) -> None:
    qid = question_id_for("z")
    base = research_dir(qid)
    base.mkdir(parents=True, exist_ok=True)
    (base / "claims.jsonl").write_text("{}\n")
    (base / "assess_log.jsonl").write_text("{}\n")
    _reset_store(qid)
    assert not (base / "claims.jsonl").exists()
    assert not (base / "assess_log.jsonl").exists()


async def test_generate_queries_anchors_and_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_structured(_p: Any, _sys: str, _user: str, _schema: Any) -> Any:
        return {"queries": ["sub a", "sub b", "sub c", "sub d"]}

    monkeypatch.setattr(drv, "_structured", fake_structured)
    out = await _generate_queries(_Provider(), "main question", 3)
    assert out[0] == "main question"  # always anchored on the question
    assert len(out) == 3


async def test_generate_queries_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_structured(*_a: Any) -> Any:
        return {"queries": ["main question", "other"]}

    monkeypatch.setattr(drv, "_structured", fake_structured)
    out = await _generate_queries(_Provider(), "main question", 5)
    assert out == ["main question", "other"]


async def test_extract_claims_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_structured(*_a: Any) -> Any:
        return {
            "claims": [
                {"statement": "Fact one", "snippet": "s1"},
                {"statement": "  ", "snippet": "drop-empty"},
                "not-a-dict",
                {"snippet": "no-statement"},
            ]
        }

    monkeypatch.setattr(drv, "_structured", fake_structured)
    claims = await _extract_claims(_Provider(), "q", "https://a.com/x", "page text")
    assert len(claims) == 1
    assert claims[0] == {
        "statement": "Fact one",
        "url": "https://a.com/x",
        "snippet": "s1",
        "stance": "support",
    }


async def test_extract_claims_empty_text_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_structured(*_a: Any) -> Any:
        nonlocal called
        called = True
        return {"claims": []}

    monkeypatch.setattr(drv, "_structured", fake_structured)
    assert await _extract_claims(_Provider(), "q", "u", "   ") == []
    assert called is False


async def test_run_deep_research_missing_question() -> None:
    with pytest.raises(ToolError):
        await run_deep_research("   ", provider=_Provider())


async def test_run_deep_research_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two independent domains assert the same statement -> corroborated.
    pages = {
        "https://a.com/x": "alpha page body",
        "https://b.org/y": "beta page body",
    }

    async def fake_search(query: str, n: int) -> list[dict[str, Any]]:
        return [{"url": "https://a.com/x"}, {"url": "https://b.org/y"}]

    async def fake_read(url: str, full_page: bool) -> str:
        return pages[url]

    async def fake_structured(_p: Any, system: str, _user: str, _schema: Any) -> Any:
        if "search queries" in system or "search-query" in system.lower():
            return {"queries": ["q1"]}
        # extraction: every page asserts the same fact
        return {"claims": [{"statement": "The sky is blue", "snippet": "snip"}]}

    monkeypatch.setattr(drv, "_search_results", fake_search)
    monkeypatch.setattr(drv, "_read_page", fake_read)
    monkeypatch.setattr(drv, "_structured", fake_structured)

    result = await run_deep_research(
        "why is the sky blue",
        provider=_Provider(),
        max_rounds=2,
        results_per_query=2,
        subqueries=1,
    )

    assert result["question_id"] == question_id_for("why is the sky blue")
    assert result["pages_read"] == 2
    assert result["verification"]["total_assertions"] == 1
    assert result["verification"]["corroborated"] == 1
    assert result["verification"]["fully_verified"] is True
    assert result["report"]["coverage"]["corroborated"] == 1
    # second round adds nothing new (URLs already visited) -> saturates/stops
    assert result["stopped_because"] in {"saturated", "budget"}
    # evidence actually persisted under the question id
    assert len(load_claims(result["question_id"])) == 2


async def test_run_deep_research_skips_error_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search(query: str, n: int) -> list[dict[str, Any]]:
        return [{"url": "https://a.com/x"}]

    async def fake_read(url: str, full_page: bool) -> str:
        return "Error: navigation failed"

    extract_called = False

    async def fake_structured(_p: Any, system: str, _user: str, _schema: Any) -> Any:
        nonlocal extract_called
        if "claim" in system.lower():
            extract_called = True
        return {"queries": ["q1"], "claims": []}

    monkeypatch.setattr(drv, "_search_results", fake_search)
    monkeypatch.setattr(drv, "_read_page", fake_read)
    monkeypatch.setattr(drv, "_structured", fake_structured)

    result = await run_deep_research(
        "q", provider=_Provider(), max_rounds=1, results_per_query=1, subqueries=1
    )
    assert result["pages_read"] == 1  # visited, but content was an error
    assert result["verification"]["total_assertions"] == 0
    assert extract_called is False  # never extract from an error page


# ---------------------------------------------------------------------------
# P3 Item 1 — parallel read/search focused tests
# ---------------------------------------------------------------------------


async def test_parallel_reads_use_distinct_agent_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N concurrent URL reads must each use a distinct browser agent_id."""
    urls = [f"https://src{i}.example.com/page" for i in range(3)]
    seen_ids: list[str] = []

    async def fake_search(query: str, n: int) -> list[dict[str, Any]]:
        return [{"url": u} for u in urls]

    async def fake_read(url: str, full_page: bool) -> str:
        return f"content of {url}"

    async def fake_structured(_p: Any, system: str, _user: str, _schema: Any) -> Any:
        if "generate" in system.lower():
            return {"queries": ["q1"]}
        return {"claims": [{"statement": "fact", "snippet": "s"}]}

    original_set_agent_id = None

    def capturing_set_agent_id(aid: str) -> Any:
        seen_ids.append(aid)
        if original_set_agent_id is not None:
            return original_set_agent_id(aid)
        return None

    import supporter.tools.browser.session as bsession

    original_set_agent_id = bsession.set_agent_id
    monkeypatch.setattr(bsession, "set_agent_id", capturing_set_agent_id)
    monkeypatch.setattr(drv, "_search_results", fake_search)
    monkeypatch.setattr(drv, "_read_page", fake_read)
    monkeypatch.setattr(drv, "_structured", fake_structured)

    result = await run_deep_research(
        "test question",
        provider=_Provider(),
        max_rounds=1,
        results_per_query=3,
        subqueries=1,
    )
    assert result["pages_read"] == 3
    # Each URL read used a distinct agent id
    assert len(seen_ids) == 3
    assert len(set(seen_ids)) == 3, f"agent ids not distinct: {seen_ids}"


async def test_one_failing_url_does_not_sink_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A _read_page exception on one URL must not fail the whole round."""

    async def fake_search(query: str, n: int) -> list[dict[str, Any]]:
        return [{"url": "https://good.com/p"}, {"url": "https://bad.com/p"}]

    async def fake_read(url: str, full_page: bool) -> str:
        if "bad" in url:
            raise OSError("network error")
        return "good page content"

    async def fake_structured(_p: Any, system: str, _user: str, _schema: Any) -> Any:
        if "generate" in system.lower():
            return {"queries": ["q1"]}
        return {"claims": [{"statement": "good fact", "snippet": "s"}]}

    monkeypatch.setattr(drv, "_search_results", fake_search)
    monkeypatch.setattr(drv, "_read_page", fake_read)
    monkeypatch.setattr(drv, "_structured", fake_structured)

    result = await run_deep_research(
        "question",
        provider=_Provider(),
        max_rounds=1,
        results_per_query=2,
        subqueries=1,
    )
    # bad URL visited but read failed; good URL read and claim extracted
    assert result["pages_read"] == 2  # both URLs in visited
    assert result["verification"]["total_assertions"] == 1
    trace = result["trace"][0]
    assert trace["pages_read"] == 1  # only good.com counted as a page
    assert trace["claims_ingested"] == 1


async def test_ingest_is_sequential_not_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest called in main loop sequentially, not inside concurrent read tasks."""
    ingest_calls: list[str] = []
    real_ingest = drv.ingest_claims

    def tracking_ingest(
        qid: str, *, task_id: str, agent: str, claims: list[Any], round: int = 0
    ) -> int:
        ingest_calls.append(task_id)
        return real_ingest(
            qid, task_id=task_id, agent=agent, claims=claims, round=round
        )

    monkeypatch.setattr(drv, "ingest_claims", tracking_ingest)

    urls = [f"https://s{i}.example.com/p" for i in range(3)]

    async def fake_search(query: str, n: int) -> list[dict[str, Any]]:
        return [{"url": u} for u in urls]

    async def fake_read(url: str, full_page: bool) -> str:
        return f"page {url}"

    async def fake_structured(_p: Any, system: str, _user: str, _schema: Any) -> Any:
        if "generate" in system.lower():
            return {"queries": ["q1"]}
        return {"claims": [{"statement": f"fact from {_user[:20]}", "snippet": "s"}]}

    monkeypatch.setattr(drv, "_search_results", fake_search)
    monkeypatch.setattr(drv, "_read_page", fake_read)
    monkeypatch.setattr(drv, "_structured", fake_structured)

    await run_deep_research(
        "q", provider=_Provider(), max_rounds=1, results_per_query=3, subqueries=1
    )
    # All 3 pages read; ingest called 3 times sequentially (same task_id each)
    assert len(ingest_calls) == 3
    assert all(tid == "round-1" for tid in ingest_calls)
