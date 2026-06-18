from __future__ import annotations

import json
from typing import Any

from ...logger import logger
from .claims import dedup_to_assertions, load_claims

_MAX_SOURCES_PER_ASSERTION = 5


def _first_snippet(sources: list[dict[str, Any]], stance: str | None = None) -> str:
    for s in sources:
        if stance is not None and s.get("stance") != stance:
            continue
        snippet = str(s.get("snippet", "")).strip()
        if snippet:
            return snippet
    return ""


def _citations(sources: list[dict[str, Any]], stance: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    cites: list[dict[str, str]] = []
    for s in sources:
        if s.get("stance") != stance:
            continue
        url = str(s.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        cites.append(
            {
                "url": url,
                "domain": str(s.get("domain", "")),
                "snippet": str(s.get("snippet", "")).strip(),
            }
        )
        if len(cites) >= _MAX_SOURCES_PER_ASSERTION:
            break
    return cites


def verify_research(question_id: str) -> dict[str, Any]:
    """Pure core of `verify_claims`: label every assertion and emit targeted
    follow-up queries for the thin/contested ones."""
    assertions = dedup_to_assertions(load_claims(question_id))
    labeled: list[dict[str, Any]] = []
    follow_ups: list[str] = []
    counts = {"corroborated": 0, "uncorroborated": 0, "conflicted": 0}
    for a in assertions:
        status = a.get("status", "uncorroborated")
        counts[status] = counts.get(status, 0) + 1
        entry = {
            "statement": a.get("statement", ""),
            "status": status,
            "supporting_domains": a.get("supporting_domains", []),
            "refuting_domains": a.get("refuting_domains", []),
            "claim_count": a.get("claim_count", 0),
        }
        labeled.append(entry)
        if status in ("uncorroborated", "conflicted"):
            stmt = a.get("statement", "")
            if stmt:
                follow_ups.append(stmt)
    verified = counts["corroborated"]
    total = len(assertions)
    return {
        "question_id": question_id,
        "total_assertions": total,
        "corroborated": counts["corroborated"],
        "uncorroborated": counts["uncorroborated"],
        "conflicted": counts["conflicted"],
        "fully_verified": total > 0 and verified == total,
        "assertions": labeled,
        "follow_up_queries": follow_ups,
    }


def build_report(question_id: str) -> dict[str, Any]:
    """Pure core of `research_report`: a synthesis-ready, cited dump grouped by
    verification status, for the main agent to turn into prose."""
    assertions = dedup_to_assertions(load_claims(question_id))
    corroborated: list[dict[str, Any]] = []
    contested: list[dict[str, Any]] = []
    uncorroborated: list[dict[str, Any]] = []
    for a in assertions:
        sources = a.get("sources", [])
        status = a.get("status", "uncorroborated")
        if status == "conflicted":
            contested.append(
                {
                    "statement": a.get("statement", ""),
                    "supporting": _citations(sources, "support"),
                    "refuting": _citations(sources, "refute"),
                }
            )
        elif status == "corroborated":
            corroborated.append(
                {
                    "statement": a.get("statement", ""),
                    "supporting_domains": a.get("supporting_domains", []),
                    "citations": _citations(sources, "support"),
                }
            )
        else:
            uncorroborated.append(
                {
                    "statement": a.get("statement", ""),
                    "citations": _citations(sources, "support"),
                }
            )
    return {
        "question_id": question_id,
        "coverage": {
            "total_assertions": len(assertions),
            "corroborated": len(corroborated),
            "contested": len(contested),
            "uncorroborated": len(uncorroborated),
        },
        "corroborated": corroborated,
        "contested": contested,
        "uncorroborated": uncorroborated,
    }


async def verify_claims(question_id: str) -> str:
    """Cross-verify every collected claim and label each assertion.

    Dedups the claim store into atomic assertions and labels each:
    - corroborated: >=2 independent domains agree, none refute.
    - conflicted: at least one source refutes -- needs adjudication.
    - uncorroborated: only one independent domain -- needs confirmation.

    For uncorroborated and conflicted assertions it returns `follow_up_queries`
    -- run these as targeted web_search + page-pilot rounds to seek independent
    confirmation or refutation, then re-verify. Nothing is taken on a single
    source; every fact ends labeled.

    Args:
        question_id: The research question id grouping the evidence store.

    Returns:
        A JSON string: per-assertion labels, status counts, and the list of
        follow-up queries for thin/contested facts.
    """
    logger.info(f"Tool: verify_claims -- question_id='{question_id}'")
    result = verify_research(question_id)
    logger.info(
        f"Tool: verify_claims -- {result['corroborated']}/{result['total_assertions']} "
        "corroborated"
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


async def research_report(question_id: str) -> str:
    """Synthesis-ready, cited evidence dump for the final report.

    Returns every assertion grouped by verification status, each with its
    source URLs and supporting snippets, plus a coverage summary. Read this to
    write the final answer: state corroborated facts with inline citations, put
    disagreements in a "Contested / Uncertain" section, and flag uncorroborated
    claims as single-source. Do NOT present uncorroborated or contested facts as
    settled.

    Args:
        question_id: The research question id grouping the evidence store.

    Returns:
        A JSON string with `coverage`, `corroborated`, `contested`, and
        `uncorroborated` assertion groups, each carrying citations.
    """
    logger.info(f"Tool: research_report -- question_id='{question_id}'")
    result = build_report(question_id)
    return json.dumps(result, ensure_ascii=False, indent=2)
