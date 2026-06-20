from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...logger import logger
from .claims import dedup_to_assertions, load_claims, research_dir

# Saturation defaults: stop once the last K assess rounds each added < N new
# atomic assertions. Budget: hard ceiling on rounds regardless of saturation.
_SATURATION_ROUNDS = 2
_MIN_NEW_ASSERTIONS = 2
_MAX_ROUNDS = 6
_GAP_SAMPLE = 12


def _assess_log_path(question_id: str) -> Path:
    return research_dir(question_id) / "assess_log.jsonl"


def _load_assess_log(question_id: str) -> list[dict[str, Any]]:
    path = _assess_log_path(question_id)
    if not path.exists():
        return []
    snapshots: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                snapshots.append(data)
    return snapshots


def _append_assess_log(question_id: str, snapshot: dict[str, Any]) -> None:
    path = _assess_log_path(question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def assess_research(
    question_id: str,
    *,
    saturation_rounds: int = _SATURATION_ROUNDS,
    min_new: int = _MIN_NEW_ASSERTIONS,
    max_rounds: int = _MAX_ROUNDS,
) -> dict[str, Any]:
    """Pure core of `research_assess`: compute the continue/stop signal and
    append a round snapshot to the assess log. Returns the assessment dict."""
    assertions = dedup_to_assertions(load_claims(question_id))
    total = len(assertions)
    by_status: dict[str, int] = {
        "corroborated": 0,
        "uncorroborated": 0,
        "conflicted": 0,
    }
    uncorroborated: list[str] = []
    conflicted: list[str] = []
    for a in assertions:
        status = a.get("status", "uncorroborated")
        by_status[status] = by_status.get(status, 0) + 1
        if status == "uncorroborated" and len(uncorroborated) < _GAP_SAMPLE:
            uncorroborated.append(a.get("statement", ""))
        elif status == "conflicted" and len(conflicted) < _GAP_SAMPLE:
            conflicted.append(a.get("statement", ""))

    prior = _load_assess_log(question_id)
    totals = [int(s.get("total", 0)) for s in prior] + [total]
    deltas = [totals[i] - totals[i - 1] for i in range(1, len(totals))]
    rounds_elapsed = len(prior) + 1
    new_this_round = deltas[-1] if deltas else total

    recent = deltas[-saturation_rounds:]
    saturated = len(recent) >= saturation_rounds and all(d < min_new for d in recent)
    budget_exhausted = rounds_elapsed >= max_rounds

    if saturated:
        recommendation, reason = "stop", "saturated"
    elif budget_exhausted:
        recommendation, reason = "stop", "budget"
    else:
        recommendation, reason = "continue", "gaps_remain"

    snapshot = {
        "round": rounds_elapsed,
        "total": total,
        "corroborated": by_status["corroborated"],
        "uncorroborated": by_status["uncorroborated"],
        "conflicted": by_status["conflicted"],
        "new_this_round": new_this_round,
    }
    _append_assess_log(question_id, snapshot)

    return {
        "question_id": question_id,
        "round": rounds_elapsed,
        "total_assertions": total,
        "corroborated": by_status["corroborated"],
        "uncorroborated": by_status["uncorroborated"],
        "conflicted": by_status["conflicted"],
        "new_this_round": new_this_round,
        "saturated": saturated,
        "budget_exhausted": budget_exhausted,
        "recommendation": recommendation,
        "reason": reason,
        "uncorroborated_statements": uncorroborated,
        "conflicted_statements": conflicted,
    }


async def research_assess(question_id: str) -> str:
    """Deterministic continue/stop signal for the research loop.

    Reads the shared claim store for `question_id`, dedups claims into atomic
    assertions, and reports how many are corroborated (>=2 independent
    domains), uncorroborated (thin -- need another independent source), or
    conflicted (sources disagree -- need adjudication). Records a per-round
    snapshot so it can detect saturation.

    Call this after each research round. Use it to decide the next move:
    - recommendation "continue": open the listed uncorroborated/conflicted
      gaps with targeted web_search + page-pilot reads, then assess again.
    - recommendation "stop": saturated (rounds stopped adding new assertions)
      or budget (round ceiling hit). Move to verify_claims + the report.

    Args:
        question_id: The research question id grouping the evidence store
            (the same id passed to delegate_tasks via question_id).

    Returns:
        A JSON string with counts, the saturation/budget flags, a
        continue/stop recommendation, and sampled gap statements to target.
    """
    logger.info(f"Tool: research_assess -- question_id='{question_id}'")
    result = assess_research(question_id)
    logger.info(
        f"Tool: research_assess -- round={result['round']} "
        f"total={result['total_assertions']} rec={result['recommendation']}"
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
