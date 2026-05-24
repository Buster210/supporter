from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.delegate.agents import _cache
from supporter.tools.delegate.api import (
    cancel_delegation,
    check_delegation,
    delegate_tasks,
)
from supporter.tools.delegate.bus import get_bus
from supporter.tools.delegate.capsule import capsule_path, load_capsule
from supporter.tools.delegate.capsule_query import (
    query_delegation,
    serialize_capsule_result,
)
from supporter.types import (
    LLMOptions,
    LLMResult,
    MilestoneCancelled,
    MilestoneCompleted,
)


class ScriptedProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_name(self) -> str:
        return "scripted-provider"

    async def generate(
        self, prompt: str, options: LLMOptions | None = None
    ) -> LLMResult:
        self.prompts.append(prompt)
        task_id = "synthesize" if "TASK:\nSummarize" in prompt else "map"
        result = {
            "summary": f"{task_id} summary",
            "evidence": {
                "files_read": [f"src/{task_id}.py"],
                "files_changed": [],
                "commands_run": ["pytest"],
                "sources": [],
            },
            "findings": [f"{task_id} finding"],
            "handoff": f"{task_id} handoff",
            "confidence": "high",
        }
        return LLMResult(
            text=f"{task_id} raw output\n\nDELEGATION_RESULT:\n{json.dumps(result)}",
            model="scripted-model",
            duration=0.01,
            usage={"total_tokens": 7},
        )


class BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def get_name(self) -> str:
        return "blocking-provider"

    async def generate(
        self, prompt: str, options: LLMOptions | None = None
    ) -> LLMResult:
        self.started.set()
        await self.release.wait()
        return LLMResult(text="released", model="blocking-model")


@pytest.fixture(autouse=True)
def isolate_delegation_flow(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    _cache.clear()


def _job_id(plan: str) -> str:
    return next(line for line in plan.splitlines() if "Job ID:" in line).split("`")[1]


async def _collect_until_complete(job_id: str) -> list[Any]:
    queue = get_bus(job_id).subscribe()
    events: list[Any] = []
    for _ in range(50):
        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        if event is None:
            break
        events.append(event)
        if isinstance(event, (MilestoneCompleted, MilestoneCancelled)):
            break
    return events


@pytest.mark.asyncio
async def test_delegation_lifecycle_completes_with_capsule_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ScriptedProvider()
    monkeypatch.setattr("supporter.pool.get_provider", lambda **_kwargs: provider)
    tasks_json = json.dumps(
        [
            {"id": "map", "task": "Map delegation flow"},
            {
                "id": "synthesize",
                "task": "Summarize delegation flow",
                "depends_on": ["map"],
            },
        ]
    )

    plan = await delegate_tasks("Integration lifecycle", tasks_json, max_parallel=2)
    job_id = _job_id(plan)
    snapshot = await check_delegation(job_id)
    events = await _collect_until_complete(job_id)

    completed = [event for event in events if isinstance(event, MilestoneCompleted)]
    capsule = load_capsule(job_id)
    payload = serialize_capsule_result(job_id)
    summary = query_delegation(job_id=job_id)
    tasks = query_delegation(job_id=job_id, detail="tasks")
    task_detail = query_delegation(job_id=job_id, task_id="synthesize")

    assert "Delegation started" in plan
    assert "after: map" in plan
    assert "Integration lifecycle" in snapshot or job_id in snapshot
    assert completed
    assert capsule_path(job_id).exists()
    assert capsule["status"] == "completed"
    assert capsule["tasks"]["map"]["summary"] == "map summary"
    assert capsule["tasks"]["synthesize"]["summary"] == "synthesize summary"
    assert "DEPENDENCY OUTPUTS" in provider.prompts[-1]
    assert "map raw output" in provider.prompts[-1]
    assert payload["totals"]["completed"] == 2
    assert payload["tasks"][1]["summary"] == "synthesize summary"
    assert "synthesize summary" in summary
    assert "synthesize handoff" in tasks
    assert "src/synthesize.py" in task_detail


@pytest.mark.asyncio
async def test_delegation_lifecycle_validates_bad_tasks() -> None:
    with pytest.raises(Exception, match="cannot be empty"):
        await delegate_tasks("Invalid lifecycle", "[]", max_parallel=1)


@pytest.mark.asyncio
async def test_delegation_lifecycle_cancellation_uses_real_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BlockingProvider()
    monkeypatch.setattr("supporter.pool.get_provider", lambda **_kwargs: provider)
    tasks_json = json.dumps([{"id": "slow", "task": "Wait until cancelled"}])

    plan = await delegate_tasks("Integration cancel", tasks_json, max_parallel=1)
    job_id = _job_id(plan)
    queue = get_bus(job_id).subscribe()
    await asyncio.wait_for(provider.started.wait(), timeout=2.0)

    cancellation = await cancel_delegation(job_id)
    events: list[Any] = []
    for _ in range(20):
        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        if event is None:
            break
        events.append(event)
        if isinstance(event, MilestoneCancelled):
            break

    capsule = load_capsule(job_id)

    assert "Cancellation requested" in cancellation
    assert any(isinstance(event, MilestoneCancelled) for event in events)
    assert capsule["status"] == "cancelled"
    assert "slow" in capsule["tasks"]
