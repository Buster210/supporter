"""Test that trailing text chunks after generation_complete are not dropped.

Simulates the exact failure mode: the server sends
  model_turn text → generation_complete → model_turn text (trailing) → turn_complete

Before the fix, the loop broke on generation_complete and dropped the trailing
text chunk. After the fix, a 300 ms drain window captures it.

No network, no auth, no framework — stdlib + asyncio only.
"""

import asyncio
import time
import types as _types
from collections.abc import AsyncIterator, Sequence

# ---------------------------------------------------------------------------
# Minimal fakes that mirror the shape our loop body accesses
# ---------------------------------------------------------------------------

FakeMsg = _types.SimpleNamespace


def _make_response(
    *,
    text: str | None = None,
    generation_complete: bool = False,
    turn_complete: bool = False,
) -> FakeMsg:
    """Build a fake session.receive() message."""
    part = FakeMsg(text=text, thought=False) if text else None
    model_turn = FakeMsg(parts=[part] if part else [])
    content = FakeMsg(
        model_turn=model_turn,
        output_transcription=FakeMsg(text=None),
        generation_complete=generation_complete,
        turn_complete=turn_complete,
        grounding_metadata=None,
        session_resumption_update=None,
        go_away=None,
    )
    return FakeMsg(server_content=content)


async def _fake_receive(
    messages: Sequence[FakeMsg], block_at_end: bool = False
) -> AsyncIterator[FakeMsg]:
    """Async generator yielding pre-built fake messages.

    With block_at_end=True it never exhausts after the last message -- it
    blocks on a long sleep, reproducing the real silent-server case where
    the server stops sending after generation_complete. Only a working idle
    timeout can break a loop driven by this stream.
    """
    for msg in messages:
        await asyncio.sleep(0)  # yield control, simulate async I/O
        yield msg
    if block_at_end:
        await asyncio.sleep(3600)  # never returns; timeout must fire


# ---------------------------------------------------------------------------
# Extracted loop logic (mirrors the fixed generate()/generate_stream() body,
# including the asyncio.timeout idle-drain that fires even on a silent server)
# ---------------------------------------------------------------------------


async def _run_loop(
    messages: Sequence[FakeMsg],
    ends_turn_early: bool = True,
    drain_seconds: float = 0.3,
    block_at_end: bool = False,
) -> list[str]:
    """Run the fixed receive loop; return collected text chunks."""
    out: list[str] = []
    loop = asyncio.get_running_loop()
    _draining = False

    try:
        async with asyncio.timeout(None) as drain_cm:
            async for response in _fake_receive(messages, block_at_end):
                content: FakeMsg = response.server_content

                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.text and not part.thought:
                            out.append(part.text)

                if content.turn_complete:
                    break
                if ends_turn_early and content.generation_complete and not _draining:
                    _draining = True
                    drain_cm.reschedule(loop.time() + drain_seconds)
    except TimeoutError:
        pass

    return out


# Both real loops share the drain logic; one extracted loop covers both.
_run_generate_loop = _run_loop
_run_stream_loop = _run_loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trailing_chunk_after_generation_complete_is_captured_generate() -> None:
    """generate() must not drop text that arrives after generation_complete."""
    messages = [
        _make_response(text="Beyond managing "),
        _make_response(generation_complete=True),
        _make_response(text="files and shell commands, my capabilities include:"),
        _make_response(turn_complete=True),
    ]
    result = asyncio.run(_run_generate_loop(messages))
    full = "".join(result)
    assert "files and shell commands, my capabilities include:" in full, (
        f"Trailing chunk was dropped. Got: {result!r}"
    )


def test_trailing_chunk_after_generation_complete_is_captured_stream() -> None:
    """generate_stream() must not drop text that arrives after generation_complete."""
    messages = [
        _make_response(text="Beyond managing "),
        _make_response(generation_complete=True),
        _make_response(text="files and shell commands, my capabilities include:"),
        _make_response(turn_complete=True),
    ]
    result = asyncio.run(_run_stream_loop(messages))
    full = "".join(result)
    assert "files and shell commands, my capabilities include:" in full, (
        f"Trailing chunk was dropped. Got: {result!r}"
    )


def test_turn_complete_still_breaks_normally() -> None:
    """A normal turn_complete (no ends_turn_early path) still terminates cleanly."""
    messages = [
        _make_response(text="Hello world"),
        _make_response(turn_complete=True),
        _make_response(text="This should not appear"),
    ]
    result = asyncio.run(_run_generate_loop(messages, ends_turn_early=False))
    assert result == ["Hello world"], f"Expected only first chunk. Got: {result!r}"


def test_drain_timeout_exits_on_silent_server() -> None:
    """The hang regression: server goes SILENT after generation_complete.

    block_at_end=True makes the stream block forever instead of exhausting,
    reproducing the real gemini-3 case. Only a real idle timeout (not a
    deadline checked on the next message) can break the loop here.
    """
    messages = [
        _make_response(text="chunk one"),
        _make_response(generation_complete=True),
        # No turn_complete, and the stream then blocks forever.
    ]
    start = time.monotonic()
    result = asyncio.run(
        _run_generate_loop(messages, drain_seconds=0.05, block_at_end=True)
    )
    elapsed = time.monotonic() - start
    assert result == ["chunk one"], f"Expected only first chunk. Got: {result!r}"
    assert elapsed < 2.0, f"Idle timeout did not fire (hang): took {elapsed:.2f}s"


if __name__ == "__main__":
    test_trailing_chunk_after_generation_complete_is_captured_generate()
    test_trailing_chunk_after_generation_complete_is_captured_stream()
    test_turn_complete_still_breaks_normally()
    test_drain_timeout_exits_on_silent_server()
    print("All assertions passed.")
