from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from textual.message import Message

if TYPE_CHECKING:
    from ..agent import ChatAgent


@dataclass
class ModeChanged(Message):
    mode: str
    enabled: bool


class StreamingState:
    def __init__(self) -> None:
        self.bubble: Any = None
        self.is_first_chunk = True
        self.actual_model: str | None = None


class ChatMessageProcessor:
    def __init__(self, app: Any) -> None:
        self._app = app

    async def process_streaming(
        self,
        text: str,
        target: Any,
        start_time: float,
        agent: ChatAgent,
    ) -> Any:
        from .widgets import MessageBubble

        state = StreamingState()

        try:
            async for chunk in agent.execute_stream(text):
                await self._handle_chunk(chunk, target, state, MessageBubble)
        finally:
            if state.bubble:
                duration = time.perf_counter() - start_time
                state.bubble.finalize(model=state.actual_model, duration=duration)

        return state.bubble

    async def _handle_chunk(
        self, chunk: Any, target: Any, state: StreamingState, bubble_class: type
    ) -> None:
        if chunk.is_tool_call:
            await self._handle_tool_chunk(chunk, target, state, bubble_class)
            return

        if chunk.is_last:
            self._update_status("Thinking")

        if chunk.model:
            state.actual_model = chunk.model

        if not state.is_first_chunk:
            self._append_to_existing_bubble(chunk, state)
            return

        if not chunk.text.strip() and not chunk.is_thought:
            return

        await self._create_and_append_first_chunk(chunk, target, state, bubble_class)

    def _append_to_existing_bubble(self, chunk: Any, state: StreamingState) -> None:
        if state.bubble:
            self._update_streaming_status()
            state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _create_and_append_first_chunk(
        self, chunk: Any, target: Any, state: StreamingState, bubble_class: type
    ) -> None:
        state.is_first_chunk = False
        state.bubble = await self._initialize_bubble(target, bubble_class)
        self._update_status("Streaming")
        state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _handle_tool_chunk(
        self, chunk: Any, target: Any, state: StreamingState, bubble_class: type
    ) -> None:
        self._handle_tool_call_status(chunk.tool_name)
        if state.is_first_chunk:
            state.is_first_chunk = False
            state.bubble = await self._initialize_bubble(target, bubble_class)

        if state.bubble:
            state.bubble.add_tool_call(
                chunk.tool_name or "unknown_tool", chunk.tool_args
            )

    def _update_streaming_status(self) -> None:
        status = self._app.status_label
        if status in ("Searching", "Thinking") or "Using" in status:
            self._update_status("Streaming")

    async def _initialize_bubble(self, target: Any, bubble_class: type) -> Any:
        from .widgets import ChatTurn

        bubble = bubble_class(role="agent", content="", streaming=True)
        if isinstance(target, ChatTurn):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)
        return bubble

    def _handle_tool_call_status(self, tool_name: str | None) -> None:
        name = (tool_name or "").lower()
        if "google_search" in name:
            self._update_status("Searching")
        else:
            self._update_status(f"Using {tool_name}")

    def _update_status(self, status: str) -> None:
        self._app.status_label = status
