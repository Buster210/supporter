from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import ChatAgent, CrewAgent


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

        bubble = None
        is_first_chunk = True
        actual_model = None

        try:
            async for chunk in agent.execute_stream(text):
                if chunk.is_tool_call:
                    self._handle_tool_call_status(chunk.tool_name)

                    if is_first_chunk:
                        is_first_chunk = False
                        bubble = await self._initialize_bubble(target, MessageBubble)

                    if bubble:
                        bubble.add_tool_call(
                            chunk.tool_name or "unknown_tool", chunk.tool_args
                        )
                    continue

                if chunk.is_last:
                    self._update_status("Thinking")

                if chunk.model:
                    actual_model = chunk.model

                if is_first_chunk:
                    if not chunk.text.strip() and not chunk.is_thought:
                        continue

                    is_first_chunk = False
                    self._update_status("Streaming")
                    bubble = await self._initialize_bubble(target, MessageBubble)

                    if chunk.is_thought:
                        bubble.thoughts = chunk.text
                    else:
                        bubble.content = chunk.text
                elif bubble:
                    if (
                        self._app.status_label in ("Searching", "Thinking")
                        or "Using" in self._app.status_label
                    ):
                        self._update_status("Streaming")
                    bubble.append_token(chunk.text, is_thought=chunk.is_thought)
        finally:
            if bubble:
                bubble.finalize(
                    model=actual_model, duration=time.perf_counter() - start_time
                )

        return bubble

    async def _initialize_bubble(self, target: Any, bubble_class: type) -> Any:
        from .widgets import ChatTurn

        bubble = bubble_class(role="agent", content="", streaming=True)
        if isinstance(target, ChatTurn):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)
        return bubble

    def _handle_tool_call_status(self, tool_name: str | None) -> None:
        if "google_search" in (tool_name or "").lower():
            self._update_status("Searching")
        else:
            self._update_status(f"Using {tool_name}")

    def _update_status(self, status: str) -> None:
        self._app.status_label = status
        self._app._tick_spinner()

    async def process_crew(
        self,
        text: str,
        target: Any,
        start_time: float,
    ) -> Any:
        from .widgets import ChatTurn, MessageBubble

        if not isinstance(self._app.agent, CrewAgent):
            return None

        response = await self._app.agent.execute(text)
        agent_roles = response.usage.get("agents") if response.usage else None

        bubble = MessageBubble(
            role="agent",
            content=response.text,
            model=response.model,
            duration=time.perf_counter() - start_time,
            agents=agent_roles,
        )

        if isinstance(target, ChatTurn):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)

        return bubble
