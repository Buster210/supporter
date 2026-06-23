from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..logger import logger

if TYPE_CHECKING:
    from ..agent import ChatAgent


class _StreamingState:
    def __init__(self) -> None:
        self.bubble: Any = None
        self.is_first_chunk = True
        self.actual_model: str | None = None


class ChatMessageProcessor:
    def __init__(self, app: Any) -> None:
        self._app = app

    def wire_recovery_observer(self, agent: Any) -> None:
        provider = getattr(agent, "provider", None)
        if provider is None:
            return
        live = getattr(provider, "primary", provider)
        if hasattr(live, "recovery_observer"):
            live.recovery_observer = self._on_recovery

    def _on_recovery(self, event: str, data: dict[str, Any]) -> None:
        if event in ("reconnecting", "replaying"):
            self._update_status("Reconnecting")
        elif event == "context_partial":
            self._update_status("Context may be partial")
            if not getattr(self._app, "_partial_banner_shown", False):
                self._app._partial_banner_shown = True
                from .chat import WelcomeBanner

                for widget in self._app.query(WelcomeBanner):
                    widget.message = (
                        "Connection recovered; earlier context may be partial."
                    )

    async def process_streaming(
        self,
        text: str,
        container: Any,
        start_time: float,
        agent: ChatAgent,
        exclude_from_history: bool = False,
    ) -> Any:
        from .bubble import MessageBubble

        state = _StreamingState()

        try:
            logger.info(f"UI: starting stream process — text_len={len(text)}")
            async for chunk in agent.execute_stream(
                text, exclude_from_history=exclude_from_history
            ):
                await self._handle_chunk(chunk, container, state, MessageBubble)
        finally:
            # Leave "Streaming" no matter how the stream ended. is_last only
            # fires on the clean text path; a tool-call/exception end would
            # otherwise strand the label on "Streaming".
            self._update_status("Thinking")
            if state.bubble:
                duration = time.perf_counter() - start_time
                logger.info(f"UI: stream process finalized — duration={duration:.2f}s")
                state.bubble.finalize(model=state.actual_model, duration=duration)
            # The answer is fully on screen now. Drop the thinking spinner BEFORE
            # the fallback formatter's LLM round-trip (and the cycle's later
            # verify call) — otherwise "Thinking" lingers long after the response
            # is visible. The cycle's own stop_thinking() in its finally is then
            # an idempotent no-op (active_queries clamps at 0).
            self._app.stop_thinking()
            if state.bubble:
                # Fire-and-forget: the answer is already painted, formatting only
                # swaps an already-finalized bubble. Awaiting it would hold the
                # cycle's _is_processing gate during the fallback model's round-trip
                # and wrongly queue the next message while only the formatter is busy.
                self._app.run_worker(self._maybe_format_bubble(state.bubble))

        return state.bubble

    async def _handle_chunk(
        self, chunk: Any, container: Any, state: _StreamingState, bubble_class: type
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

    def _append_to_existing_bubble(self, chunk: Any, state: _StreamingState) -> None:
        if state.bubble:
            self._update_streaming_status()
            state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _create_and_append_first_chunk(
        self, chunk: Any, container: Any, state: _StreamingState, bubble_class: type
    ) -> None:
        state.is_first_chunk = False
        state.bubble = await self._initialize_bubble(container, bubble_class)
        self._update_status("Streaming")
        state.bubble.append_token(chunk.text, is_thought=chunk.is_thought)

    async def _handle_tool_chunk(
        self, chunk: Any, container: Any, state: _StreamingState, bubble_class: type
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

    async def _maybe_format_bubble(self, bubble: Any) -> None:
        """Pass a finalized bubble's prose content through the fallback formatter.

        Always on (gemini_fallback_model defaults to a fast model). Formats the
        content of any bubble — mixed bubbles included; ``replace_content`` only
        swaps the content run and preserves tool_calls/thought/subagent_result
        elements. No-op when there is no model or no content. Never raises.
        """
        try:
            from ..config import config

            model = config.gemini_fallback_model
            if not model:
                return

            if not bubble.content.strip():
                return

            from ..worker import format_response

            formatted = await format_response(bubble.content, model)
            logger.debug(
                "formatter returned %d chars (original=%d)",
                len(formatted),
                len(bubble.content),
            )
            if not formatted or formatted == bubble.content:
                logger.debug("formatter returned unchanged content")
                return
            logger.info(
                "formatter changed content (%d -> %d chars), replacing bubble",
                len(bubble.content),
                len(formatted),
            )
            bubble.replace_content(formatted)
        except Exception:
            logger.warning("_maybe_format_bubble failed", exc_info=True)

    def _update_status(self, status: str) -> None:
        self._app.status_label = status
