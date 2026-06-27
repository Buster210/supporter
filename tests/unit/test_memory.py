"""Unit tests for the working-memory store and tool wrappers."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

from supporter import memory as memory_mod
from supporter.memory import (
    WorkingMemory,
    append_note,
    clear_memory,
    list_notes,
    memory_snapshot,
    search_notes,
)
from supporter.tools import memory_tools

# ---------------------------------------------------------------------------
# WorkingMemory direct
# ---------------------------------------------------------------------------


def _new_memory(tmp_path: Path) -> WorkingMemory:
    return WorkingMemory(path=tmp_path / "mem.jsonl")


def test_memory_creates_file_on_first_write(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("todo", {"task": "write tests"}, label="t1")
    assert mem.path.exists()


def test_memory_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "mem.jsonl"
    a = WorkingMemory(path=path)
    a.append("pref", {"theme": "dark"})

    b = WorkingMemory(path=path)
    notes = b.list_notes()
    assert len(notes) == 1
    assert notes[0].kind == "pref"
    assert notes[0].value == {"theme": "dark"}


def test_memory_rejects_empty_kind(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    with pytest.raises(ValueError):
        mem.append("", {"x": 1})


def test_memory_rejects_non_dict_value(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    with pytest.raises(ValueError):
        mem.append("k", "not-a-dict")  # type: ignore[arg-type]


def test_memory_rejects_huge_value(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    huge = {"blob": "x" * 10_000}
    with pytest.raises(ValueError):
        mem.append("k", huge)


def test_memory_list_filters_by_kind(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("a", {"i": 1})
    mem.append("b", {"i": 2})
    mem.append("a", {"i": 3})
    a_notes = mem.list_notes(kind="a")
    assert len(a_notes) == 2
    assert all(n.kind == "a" for n in a_notes)


def test_memory_list_respects_limit(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    for i in range(10):
        mem.append("k", {"i": i})
    assert len(mem.list_notes(limit=3)) == 3


def test_memory_search_matches_label_value_kind(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("todo", {"task": "deploy service"}, label="release")
    mem.append("todo", {"task": "write tests"})
    mem.append("note", {"x": "release checklist"})

    hits = mem.search("release")
    assert len(hits) == 2
    hits_label = mem.search("release", kind="todo")
    assert len(hits_label) == 1
    assert hits_label[0].label == "release"


def test_memory_search_case_insensitive(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("k", {"v": "UPPERCASE"})
    assert len(mem.search("uppercase")) == 1


def test_memory_search_empty_query(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("k", {"v": 1})
    assert mem.search("") == []


def test_memory_clear_empties_store(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("k", {"v": 1})
    mem.clear()
    assert mem.list_notes() == []


def test_memory_compact_drops_oldest_half(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    for i in range(20):
        mem.append("k", {"i": i})
    # Compact should remove half; the remaining notes should be the newest.
    removed = mem.compact()
    assert removed == 10
    remaining = mem.list_notes()
    assert len(remaining) == 10
    # Newest note has i=19
    assert remaining[0].value == {"i": 19}


def test_memory_compact_noop_when_small(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("k", {"i": 1})
    assert mem.compact() == 0


def test_memory_snapshot_shape(tmp_path: Path) -> None:
    mem = _new_memory(tmp_path)
    mem.append("a", {"i": 1})
    mem.append("b", {"i": 2})
    snap = mem.snapshot()
    assert snap["total"] == 2
    assert snap["kinds"] == {"a": 1, "b": 1}


def test_memory_tolerates_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "mem.jsonl"
    path.write_text(
        json.dumps({"timestamp": "2024", "kind": "ok", "value": {}})
        + "\n"
        + "garbage line\n"
        + json.dumps({"timestamp": "2024", "kind": "ok", "value": {"v": 2}})
        + "\n",
        encoding="utf-8",
    )
    mem = WorkingMemory(path=path)
    notes = mem.list_notes()
    assert len(notes) == 2


def test_memory_concurrent_writes_are_thread_safe(tmp_path: Path) -> None:
    import threading

    mem = _new_memory(tmp_path)

    def writer(start: int) -> None:
        for i in range(50):
            mem.append("k", {"i": start + i})

    threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    notes = mem.list_notes()
    assert len(notes) == 200  # 4 threads * 50


# ---------------------------------------------------------------------------
# Process-wide helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_memory_singleton() -> Generator[None, None, None]:
    memory_mod._MEMORY_SINGLETON = None
    yield
    memory_mod._MEMORY_SINGLETON = None


def test_append_note_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    note = append_note("k", {"v": 1}, label="L")
    assert note is not None
    assert note.kind == "k"
    assert list_notes() == [note]


def test_list_notes_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("a", {"i": 1})
    append_note("b", {"i": 2})
    assert {n.kind for n in list_notes()} == {"a", "b"}


def test_search_notes_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"task": "deploy"})
    assert len(search_notes("deploy")) == 1


def test_clear_memory_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"i": 1})
    clear_memory()
    assert list_notes() == []


def test_memory_snapshot_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"i": 1})
    snap = memory_snapshot()
    assert snap["available"] is True
    assert snap["total"] == 1


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_write_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    out = await memory_tools.memory_write("todo", '{"task": "ship"}', label="L")
    assert "ok" in out
    notes = list_notes()
    assert len(notes) == 1
    assert notes[0].value == {"task": "ship"}


@pytest.mark.asyncio
async def test_memory_write_rejects_bad_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    out = await memory_tools.memory_write("k", "not json")
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_memory_write_rejects_non_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    out = await memory_tools.memory_write("k", "[1, 2, 3]")
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_memory_read_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"v": 1}, label="L1")
    out = await memory_tools.memory_read(kind="k", limit=5)
    assert "L1" in out
    assert "k" in out


@pytest.mark.asyncio
async def test_memory_read_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    out = await memory_tools.memory_read()
    assert "no notes" in out


@pytest.mark.asyncio
async def test_memory_search_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"task": "deploy v2"})
    out = await memory_tools.memory_search("deploy")
    assert "deploy" in out


@pytest.mark.asyncio
async def test_memory_search_empty_query_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    out = await memory_tools.memory_search("")
    assert "ERROR" in out


@pytest.mark.asyncio
async def test_memory_list_kinds_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("a", {"i": 1})
    append_note("a", {"i": 2})
    append_note("b", {"i": 3})
    out = await memory_tools.memory_list_kinds()
    assert "a: 2" in out
    assert "b: 1" in out


@pytest.mark.asyncio
async def test_memory_compact_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    for i in range(20):
        append_note("k", {"i": i})
    out = await memory_tools.memory_compact()
    assert "compacted" in out
    assert len(list_notes()) == 10


@pytest.mark.asyncio
async def test_memory_clear_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"i": 1})
    out = await memory_tools.memory_clear()
    assert "cleared" in out
    assert list_notes() == []


@pytest.mark.asyncio
async def test_memory_status_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"i": 1})
    out = await memory_tools.memory_status()
    assert "total=1" in out


def test_memory_render_block_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    assert memory_tools.memory_render_block() == ""


def test_memory_render_block_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    append_note("k", {"v": 1}, label="L")
    block = memory_tools.memory_render_block(limit=5)
    assert "RECENT WORKING MEMORY" in block
    assert "[L]" in block
