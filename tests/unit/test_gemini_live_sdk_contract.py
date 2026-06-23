"""Contract tests pinning our usage of the installed google-genai SDK.

These do NOT hit the network. They assert that the kwargs and wire-field
behaviour our code relies on still hold for the *installed* SDK version, so a
silent SDK change (renamed kwarg, or a kwarg that serializes to a deprecated
wire field) fails here instead of only at runtime against the live API.

Motivated by a real outage: `send_realtime_input(media=...)` serialized to the
deprecated `mediaChunks` wire field, which the live API rejects with
APIError 1007. Unit tests mocked the session, so nothing caught it.

These guard the SDK side (the wire mapping stays as we expect). The provider
side -- that the provider actually passes `video=` and never `media=` -- is
guarded in tests/integration/test_gemini_live_provider.py. Both layers are
needed: one catches the provider choosing the wrong kwarg, the other catches
the SDK remapping the right kwarg to a deprecated field underneath us.
"""

import inspect

from google.genai import _live_converters as live_converters
from google.genai import types
from google.genai.live import AsyncSession


def _realtime_wire_keys(**kwargs: object) -> list[str]:
    # Mirrors the exact path AsyncSession.send_realtime_input takes for the
    # non-vertex (mldev) backend before writing to the websocket.
    params = types.LiveSendRealtimeInputParameters.model_validate(kwargs)
    wire = live_converters._LiveSendRealtimeInputParameters_to_mldev(from_object=params)
    return list(wire.keys())


def test_send_realtime_input_accepts_kwargs_we_use() -> None:
    params = inspect.signature(AsyncSession.send_realtime_input).parameters
    # gemini_live_provider passes exactly these two kwargs.
    assert "video" in params
    assert "text" in params


def test_video_kwarg_serializes_to_nondeprecated_wire_field() -> None:
    blob = types.Blob(data=b"png", mime_type="image/png")
    assert _realtime_wire_keys(video=blob) == ["video"]


def test_text_kwarg_serializes_to_text_wire_field() -> None:
    assert _realtime_wire_keys(text="hi") == ["text"]


def test_media_kwarg_still_maps_to_deprecated_mediachunks() -> None:
    # Locks in *why* the provider must use video= and never media=:
    # media= serializes to mediaChunks, the field the live API rejects (1007).
    # If a future SDK remaps this, revisit the provider's image-injection path.
    blob = types.Blob(data=b"png", mime_type="image/png")
    assert _realtime_wire_keys(media=blob) == ["mediaChunks"]


def test_other_live_methods_accept_kwargs_we_use() -> None:
    client_content = inspect.signature(AsyncSession.send_client_content).parameters
    assert "turns" in client_content
    assert "turn_complete" in client_content

    tool_response = inspect.signature(AsyncSession.send_tool_response).parameters
    assert "function_responses" in tool_response
