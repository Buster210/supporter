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
        from .widgets import ChatTurn, MessageBubble

        bubble = None
        is_first_chunk = True
        actual_model = None

        async for chunk in agent.execute_stream(text):
            if chunk.is_tool_call:
                if "google_search" in (chunk.tool_name or "").lower():
                    self._app.status_label = "Searching"
                else:
                    self._app.status_label = f"Using {chunk.tool_name}"
                self._app._tick_spinner()

                if is_first_chunk:
                    is_first_chunk = False
                    bubble = MessageBubble(role="agent", content="", streaming=True)
                    if isinstance(target, ChatTurn):
                        await target.mount_bubble(bubble)
                    else:
                        await target.mount(bubble)

                if bubble:
                    bubble.add_tool_call(
                        chunk.tool_name or "unknown_tool", chunk.tool_args
                    )
                continue

            if chunk.is_last:
                self._app.status_label = "Thinking"
                self._app._tick_spinner()

            if chunk.model:
                actual_model = chunk.model

            if is_first_chunk:
                if not chunk.text.strip() and not chunk.is_thought:
                    continue

                is_first_chunk = False
                self._app.status_label = "Streaming"
                bubble = MessageBubble(role="agent", content="", streaming=True)

                if chunk.is_thought:
                    bubble.thoughts = chunk.text
                else:
                    bubble.content = chunk.text

                if isinstance(target, ChatTurn):
                    await target.mount_bubble(bubble)
                else:
                    await target.mount(bubble)
            else:
                if bubble:
                    if (
                        self._app.status_label == "Searching"
                        or "Using" in self._app.status_label
                    ):
                        self._app.status_label = "Streaming"
                    bubble.append_token(chunk.text, is_thought=chunk.is_thought)

        if bubble:
            bubble.finalize(
                model=actual_model, duration=time.perf_counter() - start_time
            )

        return bubble

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
