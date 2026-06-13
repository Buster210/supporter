from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from supporter.session import HistoryStore


def _turn(role: str, text: str) -> Any:
    from google.genai.types import Content, Part

    return Content(role=role, parts=[Part(text=text)])


def _fn_call(name: str, args: dict[str, Any]) -> Any:
    from google.genai.types import Content, FunctionCall, Part

    return Content(
        role="model",
        parts=[Part(function_call=FunctionCall(name=name, args=args))],
    )


def _fn_response(name: str, response: dict[str, Any]) -> Any:
    from google.genai.types import Content, FunctionResponse, Part

    return Content(
        role="user",
        parts=[Part(function_response=FunctionResponse(name=name, response=response))],
    )


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir: Path) -> HistoryStore:
    return HistoryStore("test_session", tmp_dir)


def test_append_then_load_roundtrips_text_turns(store: HistoryStore) -> None:
    user = _turn("user", "hello")
    model = _turn("model", "hi there")
    store.append(user)
    store.append(model)

    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].role == "user"
    assert loaded[0].parts[0].text == "hello"
    assert loaded[1].role == "model"
    assert loaded[1].parts[0].text == "hi there"


def test_load_skips_torn_final_line(tmp_dir: Path) -> None:
    store = HistoryStore("torn", tmp_dir)
    store.append(_turn("user", "valid"))
    with open(store.path, "a") as f:
        f.write('{"role":')

    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].parts[0].text == "valid"


def test_tool_call_and_response_parts_survive_roundtrip(store: HistoryStore) -> None:
    fc = _fn_call("search", {"query": "test"})
    fr = _fn_response("search", {"results": [1, 2]})
    store.append(fc)
    store.append(fr)

    loaded = store.load()
    assert len(loaded) == 2
    fc_part = loaded[0].parts[0]
    assert fc_part.name == "search"
    assert fc_part.args == {"query": "test"}
    fr_part = loaded[1].parts[0]
    assert fr_part.name == "search"
    assert fr_part.response == {"results": [1, 2]}


def test_unsupported_part_degrades_not_crashes(store: HistoryStore) -> None:
    content = MagicMock()
    content.role = "model"

    class _Unsupported:
        pass

    content.parts = [_Unsupported()]
    store.append(content)

    loaded = store.load()
    assert len(loaded) == 0


def test_load_empty_file(store: HistoryStore) -> None:
    assert store.load() == []


def test_load_limit(store: HistoryStore) -> None:
    for i in range(5):
        store.append(_turn("user", f"msg{i}"))
    loaded = store.load(limit=3)
    assert len(loaded) == 3
    assert loaded[0].parts[0].text == "msg2"


def test_load_skips_blank_lines(tmp_dir: Path) -> None:
    store = HistoryStore("blank", tmp_dir)
    store.append(_turn("user", "hello"))
    with open(store.path, "a") as f:
        f.write("\n\n\n")
    loaded = store.load()
    assert len(loaded) == 1


def test_rotate_creates_new_session(store: HistoryStore) -> None:
    store.append(_turn("user", "old"))
    old_path = store.path
    store.rotate()
    assert store.path != old_path
    assert store._dir.exists()
    loaded = store.load()
    assert len(loaded) == 0


def test_images_dir_created(store: HistoryStore) -> None:
    assert (store._dir / "images").exists()


def test_close_is_noop(store: HistoryStore) -> None:
    store.close()


def test_new_session_id_format() -> None:
    from supporter.session import new_session_id

    sid = new_session_id()
    assert sid.startswith("session_")
    assert len(sid) > 8


# ---------------------------------------------------------------------------
# P3 Item 2 — batch fsync focused tests
# ---------------------------------------------------------------------------


def test_append_no_fsync_sync_does_fsync(store: HistoryStore) -> None:
    """append does NOT fsync; sync() does exactly one fsync per call."""
    fsync_calls: list[int] = []
    import os as _os

    real_fsync = _os.fsync

    def counting_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    import unittest.mock as mock

    with mock.patch("os.fsync", side_effect=counting_fsync):
        store.append(_turn("user", "msg1"))
        store.append(_turn("model", "reply1"))
        # No fsync yet
        assert len(fsync_calls) == 0
        store.sync()
        # Exactly one fsync for the whole turn
        assert len(fsync_calls) == 1
        store.append(_turn("user", "msg2"))
        store.append(_turn("model", "reply2"))
        store.sync()
        # One more fsync for second turn
        assert len(fsync_calls) == 2


def test_reloaded_history_equals_appended_sequence(store: HistoryStore) -> None:
    """History reloaded from disk equals the sequence that was appended."""
    msgs = [
        _turn("user", "hello"),
        _turn("model", "hi"),
        _turn("user", "how are you"),
        _turn("model", "fine"),
    ]
    for msg in msgs:
        store.append(msg)
    store.sync()
    loaded = store.load()
    assert len(loaded) == len(msgs)
    for orig, got in zip(msgs, loaded, strict=True):
        assert orig.role == got.role
        assert orig.parts[0].text == got.parts[0].text


def test_sync_noop_on_empty_store(store: HistoryStore) -> None:
    """sync() on a store with no history file does not raise."""
    assert not store.path.exists()
    store.sync()  # must not raise
