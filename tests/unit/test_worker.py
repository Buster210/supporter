from __future__ import annotations

import io
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from supporter import worker
from supporter.prompts import DELEGATE_AGENT_ROSTER
from supporter.tools import file_ops
from supporter.tools.browser import guardrails


@pytest.fixture
def _restore_callbacks() -> Generator[None]:
    """Snapshot/restore the global confirmation callbacks so headless-callback
    installation in a test cannot leak into the rest of the suite."""
    prev_write = file_ops._CONFIRMATION_CALLBACK
    prev_browse = guardrails.browse_confirmation_callback
    try:
        yield
    finally:
        file_ops._CONFIRMATION_CALLBACK = prev_write
        guardrails.browse_confirmation_callback = prev_browse


class _FakeAgent:
    """Stand-in for the executor ChatAgent: writes the report on a chosen turn,
    optionally raising on specific turns to exercise per-turn fault tolerance."""

    def __init__(
        self,
        report_path: Path,
        *,
        write_on_turn: int = 1,
        fail_turns: tuple[int, ...] = (),
        provider: Any = None,
    ) -> None:
        self.report_path = report_path
        self.write_on_turn = write_on_turn
        self.fail_turns = set(fail_turns)
        self.calls = 0
        self.provider = provider if provider is not None else SimpleNamespace()

    async def execute(self, prompt: str) -> Any:
        self.calls += 1
        if self.calls in self.fail_turns:
            raise RuntimeError("transient turn failure")
        if self.calls >= self.write_on_turn:
            self.report_path.write_text("# Report\n\n" + "x" * 400)
        return SimpleNamespace(text="ok")


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report_path: Path,
    agent: _FakeAgent,
    plan: str = "PLAN BODY",
) -> AsyncMock:
    """Wire run_worker's collaborators to fakes; return the close_session mock."""
    monkeypatch.setattr(worker, "_install_headless_callbacks", lambda: lambda: None)
    monkeypatch.setattr(worker, "_report_path", lambda task, report_dir: report_path)

    async def _fake_plan(task: str) -> str:
        return plan

    monkeypatch.setattr(worker, "_make_plan", _fake_plan)
    monkeypatch.setattr(worker, "_build_executor_agent", lambda session_id: agent)
    close_mock = AsyncMock()
    monkeypatch.setattr(worker, "close_session", close_mock)
    return close_mock


def test_planner_role_in_roster() -> None:
    planner = DELEGATE_AGENT_ROSTER["planner"]
    assert planner["model"] == "gemma-4-31b-it"
    assert planner["tools"] == set()
    assert planner["live"] is False
    assert "PLAN" in planner["persona"]


def test_slugify_basic() -> None:
    assert (
        worker._slugify("Open HackerNews & get 50 news!")
        == "open-hackernews-get-50-news"
    )
    assert worker._slugify("   ") == "task"
    assert len(worker._slugify("a" * 200)) <= 50


async def test_make_plan_uses_planner_persona_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_provider = SimpleNamespace(calls=[])

    async def _generate(prompt: str, options: Any = None) -> Any:
        fake_provider.calls.append((prompt, options))
        return SimpleNamespace(text="PLAN BODY")

    fake_provider.generate = _generate
    captured: dict[str, Any] = {}

    def _fake_get_provider(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return fake_provider

    monkeypatch.setattr(worker, "get_provider", _fake_get_provider)

    plan = await worker._make_plan("open hackernews and get 50 news")

    assert plan == "PLAN BODY"
    assert captured["model_name"] == "gemma-4-31b-it"
    # Must not pollute the shared provider registry with a stale persona.
    assert captured["shared"] is False
    prompt, options = fake_provider.calls[0]
    assert "open hackernews and get 50 news" in prompt
    # REST ignores the factory system_instruction -> persona MUST ride options.
    assert options.system_instruction == DELEGATE_AGENT_ROSTER["planner"]["persona"]


async def test_run_worker_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"
    provider = SimpleNamespace(close=AsyncMock())
    agent = _FakeAgent(report_path, write_on_turn=1, provider=provider)
    close_mock = _patch_pipeline(monkeypatch, report_path=report_path, agent=agent)

    result = await worker.run_worker(
        "open hackernews and get 50 news", report_dir=tmp_path
    )

    assert result == report_path
    assert report_path.is_file()
    assert agent.calls == 1
    close_mock.assert_awaited_once()
    provider.close.assert_called_once()


async def test_run_worker_drives_multiple_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"
    agent = _FakeAgent(report_path, write_on_turn=3)
    _patch_pipeline(monkeypatch, report_path=report_path, agent=agent)

    result = await worker.run_worker("task", report_dir=tmp_path, max_executor_turns=5)

    assert result == report_path
    assert agent.calls == 3


async def test_run_worker_tolerates_turn_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"
    agent = _FakeAgent(report_path, write_on_turn=2, fail_turns=(1,))
    _patch_pipeline(monkeypatch, report_path=report_path, agent=agent)

    result = await worker.run_worker("task", report_dir=tmp_path)

    assert result == report_path
    assert agent.calls == 2


async def test_run_worker_raises_without_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"
    agent = _FakeAgent(report_path, write_on_turn=99)
    close_mock = _patch_pipeline(monkeypatch, report_path=report_path, agent=agent)

    with pytest.raises(RuntimeError, match="did not produce a report"):
        await worker.run_worker("task", report_dir=tmp_path, max_executor_turns=2)

    assert agent.calls == 2
    close_mock.assert_awaited_once()  # teardown still runs in finally


async def test_run_worker_rejects_empty_task(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await worker.run_worker("   ", report_dir=tmp_path)


async def test_run_worker_ignores_tiny_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.md"

    class _StubAgent(_FakeAgent):
        async def execute(self, prompt: str) -> Any:
            self.calls += 1
            self.report_path.write_text("tiny")  # below _MIN_REPORT_BYTES
            return SimpleNamespace(text="ok")

    agent = _StubAgent(report_path)
    _patch_pipeline(monkeypatch, report_path=report_path, agent=agent)

    with pytest.raises(RuntimeError, match="did not produce a report"):
        await worker.run_worker("task", report_dir=tmp_path, max_executor_turns=2)


async def test_install_headless_callbacks(_restore_callbacks: None) -> None:
    before_write = file_ops._CONFIRMATION_CALLBACK
    before_browse = guardrails.browse_confirmation_callback

    restore = worker._install_headless_callbacks()

    assert file_ops._CONFIRMATION_CALLBACK is not None
    assert file_ops._CONFIRMATION_CALLBACK(Path("x.md"), "diff") is True

    assert guardrails.browse_confirmation_callback is not None
    assert await guardrails.browse_confirmation_callback("title", "msg") is True

    # restore() reverts the global confirmation gates -> no leak into the TUI.
    restore()
    assert file_ops._CONFIRMATION_CALLBACK is before_write
    assert guardrails.browse_confirmation_callback is before_browse


def test_main_no_task_returns_2() -> None:
    assert worker.main([]) == 2


def test_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_path = tmp_path / "out.md"

    async def _fake_run(task: str, **kwargs: Any) -> Path:
        assert task == "open hackernews and get 50 news"
        return out_path

    monkeypatch.setattr(worker, "run_worker", _fake_run)
    buf = io.StringIO()
    monkeypatch.setattr(worker.sys, "stdout", buf)

    rc = worker.main(["open hackernews and get 50 news"])

    assert rc == 0
    assert str(out_path) in buf.getvalue()


def test_main_failure_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(task: str, **kwargs: Any) -> Path:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(worker, "run_worker", _boom)
    buf = io.StringIO()
    monkeypatch.setattr(worker.sys, "stderr", buf)

    rc = worker.main(["do something"])

    assert rc == 1
    assert "kaboom" in buf.getvalue()
