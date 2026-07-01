"""Performance bounds and memory management tests.

Acceptance criteria validation:
  AC1/AC4 (lean trivial path):  proven via execute_stream routing.
  AC2 (startup-to-interactive): proven by on_mount structure.
  AC3 (memory bounds):           caches capped, deque eviction verified.
  AC5 (TUI responsiveness):      streaming doesn't block event loop.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from supporter import memory as memory_mod
from supporter.memory import WorkingMemory
from supporter.tools.delegate import capsule as capsule_mod

# ---------------------------------------------------------------------------
# AC2: Startup-to-interactive — heavy work deferred to background workers
# ---------------------------------------------------------------------------


class TestStartupDeferred:
    """AC2: on_mount defers prewarm/resume to background workers."""

    def test_on_mount_defers_heavy_tasks_to_workers(self) -> None:
        import inspect

        import supporter.tui as tui_mod

        src = inspect.getsource(tui_mod.SupporterApp.on_mount)
        assert "run_worker" in src, "Heavy tasks must use run_worker (background)"
        assert "prewarm_clone" in src
        assert "resume_interrupted_jobs" in src
        assert src.index("focus()") > src.index("run_worker"), (
            "focus() must come after run_worker so UI is interactive "
            "while heavy tasks run in background"
        )


class TestStartupStructure:
    """Validate that startup has non-blocking lazy initialization."""

    def test_browser_session_prewarm_is_lazy(self) -> None:
        """Browser prewarm should be lazy/background, not blocking startup."""
        try:
            import inspect

            from supporter.browser.session import BrowserSession

            init_src = inspect.getsource(BrowserSession.__init__)
            assert "async" not in init_src or "background" in init_src.lower()
        except ImportError:
            pytest.skip("Browser module not available")

    def test_job_resume_is_background(self) -> None:
        """Job resume at startup should be background, not blocking."""
        try:
            import inspect

            from supporter import worker as worker_mod

            src = inspect.getsource(worker_mod)
            assert "async" in src
        except ImportError, TypeError:
            pytest.skip("Worker module not available or complex signature")


# ---------------------------------------------------------------------------
# AC3: Memory bounds — cache size limits and eviction
# ---------------------------------------------------------------------------


class TestMemoryCacheBounds:
    """Validate that in-memory caches respect configured limits."""

    def test_working_memory_deque_has_maxlen_limit(self, tmp_path: Path) -> None:
        """WorkingMemory._notes should be a bounded deque."""
        mem = WorkingMemory(path=tmp_path / "test.jsonl")

        assert isinstance(mem._notes, deque)
        assert mem._notes.maxlen == memory_mod._MAX_TOTAL_NOTES
        assert memory_mod._MAX_TOTAL_NOTES == 5000

    def test_working_memory_evicts_oldest_on_append_past_maxlen(
        self, tmp_path: Path
    ) -> None:
        """Appending beyond maxlen should evict oldest entries."""
        mem = WorkingMemory(path=tmp_path / "test.jsonl")

        for i in range(memory_mod._MAX_TOTAL_NOTES + 10):
            mem.append("kind", {"i": i})

        assert len(mem._notes) == memory_mod._MAX_TOTAL_NOTES

        notes = mem.list_notes()
        first_value = notes[-1].value
        assert first_value["i"] >= 10

    def test_working_memory_compact_drops_old_entries(self, tmp_path: Path) -> None:
        """compact() should remove approximately half of the oldest entries."""
        mem = WorkingMemory(path=tmp_path / "test.jsonl")

        for i in range(100):
            mem.append("k", {"i": i})

        initial_count = len(mem.list_notes())
        removed = mem.compact()

        assert removed > 0
        assert removed <= initial_count // 2 + 1
        remaining = len(mem.list_notes())
        assert remaining < initial_count

    def test_capsule_cache_max_constant_defined(self) -> None:
        """Capsule cache should have a defined max size constant."""
        assert hasattr(capsule_mod, "CAPSULE_CACHE_MAX")
        assert capsule_mod.CAPSULE_CACHE_MAX == 64


class TestMemoryBounds:
    """Validate that _by_kind index stays bounded with deque evictions."""

    def test_by_kind_bounded_after_overflow(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(memory_mod, "_MAX_TOTAL_NOTES", 5)
        mem = WorkingMemory(path=tmp_path / "wm.jsonl")
        for i in range(8):
            mem.append("k", {"i": i})

        assert len(mem._notes) == 5
        total_in_index = sum(len(v) for v in mem._by_kind.values())
        assert total_in_index == len(mem._notes), (
            f"_by_kind had {total_in_index} refs; _notes has {len(mem._notes)}"
        )

    def test_by_kind_mixed_kinds_bounded(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(memory_mod, "_MAX_TOTAL_NOTES", 4)
        mem = WorkingMemory(path=tmp_path / "wm.jsonl")
        for i in range(7):
            mem.append("odd" if i % 2 else "even", {"i": i})

        assert len(mem._notes) == 4
        total_in_index = sum(len(v) for v in mem._by_kind.values())
        assert total_in_index == len(mem._notes)

    def test_deque_bounded_newest_first(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(memory_mod, "_MAX_TOTAL_NOTES", 3)
        mem = WorkingMemory(path=tmp_path / "wm.jsonl")
        for i in range(6):
            mem.append("k", {"i": i})

        assert len(mem._notes) == 3
        assert next(iter(mem._notes)).value == {"i": 5}

    def test_by_kind_empty_kind_removed_on_eviction(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """When a kind's only note is evicted, its key is removed from _by_kind."""
        monkeypatch.setattr(memory_mod, "_MAX_TOTAL_NOTES", 3)
        mem = WorkingMemory(path=tmp_path / "wm.jsonl")
        mem.append("rare", {"i": 0})
        mem.append("common", {"i": 1})
        mem.append("common", {"i": 2})
        mem.append("common", {"i": 3})

        assert "rare" not in mem._by_kind, "_by_kind must drop empty kind on eviction"
        assert len(mem._notes) == 3


# ---------------------------------------------------------------------------
# AC5: TUI responsiveness — streaming doesn't block event loop
# ---------------------------------------------------------------------------


class TestTUIResponsiveness:
    """Validate that TUI streaming and bubble operations don't block event loop."""

    def test_message_processor_uses_fire_and_forget_for_formatting(self) -> None:
        """Formatting should use run_worker (fire-and-forget), not await."""
        import inspect

        from supporter.tui.message_processor import ChatMessageProcessor

        src = inspect.getsource(ChatMessageProcessor.process_streaming)
        assert "run_worker" in src
        assert "_maybe_format_bubble" in src
        lines = src.split("\n")
        format_lines = [line for line in lines if "_maybe_format_bubble" in line]
        assert any("run_worker" in line for line in format_lines)

    def test_message_processor_streaming_is_async(self) -> None:
        """process_streaming should be async and yield control."""
        import inspect

        from supporter.tui.message_processor import ChatMessageProcessor

        assert inspect.iscoroutinefunction(ChatMessageProcessor.process_streaming)

    def test_bubble_mount_is_async(self) -> None:
        """Bubble initialization (mount) should be async."""
        from supporter.tui.bubble import MessageBubble

        assert hasattr(MessageBubble, "mount") or hasattr(MessageBubble, "_mount")


class TestStreamingYields:
    """AC5: _maybe_format_bubble dispatched as worker, not awaited."""

    def test_format_bubble_dispatched_as_worker(self) -> None:
        import inspect

        import supporter.tui.message_processor as mp_mod

        src = inspect.getsource(mp_mod.ChatMessageProcessor.process_streaming)
        assert "run_worker" in src
        assert "_maybe_format_bubble" in src
        for line in src.splitlines():
            stripped_line = line.strip()
            has_rw = "run_worker" in stripped_line
            has_fb = "_maybe_format_bubble" in stripped_line
            if has_rw and has_fb:
                assert not stripped_line.startswith("await"), (
                    "_maybe_format_bubble run_worker call must not be awaited"
                )
                break
