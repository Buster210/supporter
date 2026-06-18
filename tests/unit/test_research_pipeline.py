from __future__ import annotations

from typing import Any

import pytest

from supporter.config import config
from supporter.tools.base import ToolError
from supporter.tools.research import search as search_mod
from supporter.tools.research.claims import ingest_claims
from supporter.tools.research.loop import assess_research, research_assess
from supporter.tools.research.search import format_results, web_search
from supporter.tools.research.verify import (
    build_report,
    research_report,
    verify_claims,
    verify_research,
)


@pytest.fixture(autouse=True)
def isolate_research(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])


def _claim(statement: str, url: str, stance: str = "support") -> dict[str, Any]:
    return {
        "statement": statement,
        "url": url,
        "snippet": f"q:{statement}",
        "stance": stance,
    }


# --- P1: web_search -------------------------------------------------------


def test_format_results_numbers_and_urls() -> None:
    out = format_results(
        "q",
        [
            {"title": "A", "url": "https://a.com", "content": "alpha"},
            {"title": "B", "url": "https://b.com", "content": "beta"},
        ],
    )
    assert "1. A" in out and "URL: https://a.com" in out
    assert "2. B" in out and "URL: https://b.com" in out


def test_format_results_empty_is_explicit() -> None:
    assert "No web results" in format_results("q", [])


def test_format_results_skips_entries_without_url() -> None:
    out = format_results("q", [{"title": "no url", "content": "x"}])
    assert "no url" not in out


async def test_web_search_empty_query_raises() -> None:
    with pytest.raises(ToolError):
        await web_search("   ")


async def test_web_search_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        await web_search("rust async")


async def test_web_search_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    captured: dict[str, Any] = {}

    async def fake_fetch(api_key: str, query: str, max_results: int) -> dict[str, Any]:
        captured["max_results"] = max_results
        return {"results": [{"title": "T", "url": "https://x.com", "content": "c"}]}

    monkeypatch.setattr(search_mod, "fetch_tavily", fake_fetch)
    out = await web_search("rust async", max_results=99)
    assert "https://x.com" in out
    assert captured["max_results"] == 20  # capped


# --- P2: research_assess --------------------------------------------------


async def test_assess_empty_store() -> None:
    result = assess_research("empty_q")
    assert result["total_assertions"] == 0
    assert result["round"] == 1


async def test_assess_counts_by_status() -> None:
    ingest_claims(
        "q_counts",
        "t1",
        "page-pilot",
        [_claim("fact one", "https://a.com"), _claim("fact one", "https://b.com")],
    )
    ingest_claims("q_counts", "t2", "page-pilot", [_claim("lonely", "https://a.com")])
    result = assess_research("q_counts")
    assert result["corroborated"] == 1
    assert result["uncorroborated"] == 1
    assert "lonely" in result["uncorroborated_statements"]


async def test_assess_saturation_after_quiet_rounds() -> None:
    ingest_claims(
        "q_sat",
        "t1",
        "page-pilot",
        [_claim(f"f{i}", f"https://d{i}.com") for i in range(5)],
    )
    first = assess_research("q_sat")  # round 1: +5, not saturated
    assert first["recommendation"] == "continue"
    second = assess_research("q_sat")  # round 2: +0
    third = assess_research("q_sat")  # round 3: +0 -> saturated
    assert second["new_this_round"] == 0
    assert third["saturated"] is True
    assert third["recommendation"] == "stop"
    assert third["reason"] == "saturated"


async def test_assess_budget_ceiling() -> None:
    result = None
    for _ in range(6):
        result = assess_research("q_budget", min_new=0)  # never saturates on new>=0
    assert result is not None
    assert result["budget_exhausted"] is True
    assert result["reason"] == "budget"


async def test_research_assess_returns_json_string() -> None:
    out = await research_assess("q_json")
    assert out.strip().startswith("{")
    assert "recommendation" in out


# --- P3: verify_claims + research_report ----------------------------------


async def test_verify_labels_and_follow_ups() -> None:
    ingest_claims(
        "q_v",
        "t1",
        "page-pilot",
        [
            _claim("solid", "https://a.com"),
            _claim("solid", "https://b.com"),
            _claim("thin", "https://a.com"),
            _claim("disputed", "https://a.com"),
            _claim("disputed", "https://b.com", stance="refute"),
        ],
    )
    result = verify_research("q_v")
    assert result["corroborated"] == 1
    assert result["uncorroborated"] == 1
    assert result["conflicted"] == 1
    assert result["fully_verified"] is False
    assert set(result["follow_up_queries"]) == {"thin", "disputed"}


async def test_verify_fully_verified_when_all_corroborated() -> None:
    ingest_claims(
        "q_full",
        "t1",
        "page-pilot",
        [_claim("x", "https://a.com"), _claim("x", "https://b.com")],
    )
    assert verify_research("q_full")["fully_verified"] is True


async def test_build_report_groups_with_citations() -> None:
    ingest_claims(
        "q_rep",
        "t1",
        "page-pilot",
        [
            _claim("agreed", "https://a.com"),
            _claim("agreed", "https://b.com"),
            _claim("contested", "https://a.com"),
            _claim("contested", "https://b.com", stance="refute"),
            _claim("single", "https://a.com"),
        ],
    )
    report = build_report("q_rep")
    assert report["coverage"]["corroborated"] == 1
    assert report["coverage"]["contested"] == 1
    assert report["coverage"]["uncorroborated"] == 1
    [corr] = report["corroborated"]
    assert corr["statement"] == "agreed"
    assert {c["domain"] for c in corr["citations"]} == {"a.com", "b.com"}
    [cont] = report["contested"]
    assert cont["refuting"] and cont["supporting"]


async def test_verify_and_report_tools_return_json() -> None:
    ingest_claims("q_t", "t1", "page-pilot", [_claim("y", "https://a.com")])
    assert (await verify_claims("q_t")).strip().startswith("{")
    assert (await research_report("q_t")).strip().startswith("{")
