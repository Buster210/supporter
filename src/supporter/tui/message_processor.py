from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..logger import logger

if TYPE_CHECKING:
    from ..agent import ChatAgent


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
        container: Any,
        start_time: float,
        agent: ChatAgent,
        exclude_from_history: bool = False,
    ) -> Any:
        from .bubble import MessageBubble

        state = StreamingState()

        try:
            logger.info(f"UI: starting stream process — text_len={len(text)}")
            async for chunk in agent.execute_stream(
                text, exclude_from_history=exclude_from_history
            ):
                await self._handle_chunk(chunk, container, state, MessageBubble)
        finally:
            if state.bubble:
                duration = time.perf_counter() - start_time
                logger.info(f"UI: stream process finalized — duration={duration:.2f}s")
                state.bubble.finalize(model=state.actual_model, duration=duration)

        return state.bubble

    async def _handle_chunk(
        self, chunk: Any, container: Any, state: StreamingState, bubble_class: type
    ) -> None:
        if chunk.is_tool_call:
            await self._handle_tool_chunk(chunk, container, state, bubble_class)
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

        logger.info("UI: creating first content bubble")
        await self._create_and_append_first_chunk(chunk, container, state, bubble_class)

    def _append_to_existing_bubble(self, chunk: Any, state: StreamingState) -> None:
        if state.bubble:
            self._update_streaming_status()
            state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _create_and_append_first_chunk(
        self, chunk: Any, container: Any, state: StreamingState, bubble_class: type
    ) -> None:
        state.is_first_chunk = False
        state.bubble = await self._initialize_bubble(container, bubble_class)
        self._update_status("Streaming")
        state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _handle_tool_chunk(
        self, chunk: Any, container: Any, state: StreamingState, bubble_class: type
    ) -> None:
        self._handle_tool_call_status(chunk.tool_name)
        if state.is_first_chunk:
            logger.info(f"UI: initializing tool bubble — tool={chunk.tool_name}")
            state.is_first_chunk = False
            state.bubble = await self._initialize_bubble(container, bubble_class)

        if state.bubble:
            state.bubble.add_tool_call(
                chunk.tool_name or "unknown_tool", chunk.tool_args
            )

    def _update_streaming_status(self) -> None:
        status = self._app.status_label
        if status in ("Searching", "Thinking") or "Using" in status:
            self._update_status("Streaming")

    async def _initialize_bubble(self, container: Any, bubble_class: type) -> Any:
        from .chat import ChatTurn

        bubble = bubble_class(role="agent", content="", streaming=True)
        if isinstance(container, ChatTurn):
            await container.mount_bubble(bubble)
        else:
            await container.mount(bubble)
        return bubble

    def _handle_tool_call_status(self, tool_name: str | None) -> None:
        name = (tool_name or "").lower()
        status = "Searching" if "google_search" in name else f"Using {tool_name}"
        self._update_status(status)

    def _update_status(self, status: str) -> None:
        self._app.status_label = status
