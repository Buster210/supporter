from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


class MockApp:
    def __init__(self) -> None:
        self.status_label = "Thinking"
        self.active_queries = 0
        self.is_activating_mode = False
        self.live_mode = True
        self.active_turn = None
        self.current_active_agent = ""
        self.agent: Any = None
        self.messages: list[Any] = []
        self.exited = False
        self.screen_stack: list[Any] = []
        self.cleared = False
        self.toggled_live = False
        self._is_processing = False
        self._user_message_queue: list[str] = []
        self._toast_manager: Any = MagicMock()
        self._supporter_queue_display: Any = MagicMock()
        self._message_processor: Any = MagicMock()
        self._pending_delegation_widgets: deque[tuple[Any, bool]] = deque()

    def _flush_queued(self) -> None:
        pass

    async def _flush_queued_messages(self) -> None:
        pass

    def post_message(self, message: Any) -> None:
        self.messages.append(message)

    def _handle_user_message(self, message: str) -> None:
        pass

    def handle_user_message(self, message: str) -> None:
        pass

    def notify(self, message: str, timeout: float = 5.0) -> None:
        self.messages.append(message)

    def _on_agent_active(self, agent_role: str) -> None:
        self.current_active_agent = agent_role

    def start_thinking(self) -> None:
        self.active_queries += 1

    def stop_thinking(self) -> None:
        self.active_queries = max(0, self.active_queries - 1)

    def exit(self) -> None:
        self.exited = True

    def run_worker(self, worker: Any, **kwargs: Any) -> Any:
        pass

    async def _process_message_cycle(self, *args: Any, **kwargs: Any) -> None:
        pass

    def action_clear_screen(self) -> None:
        self.cleared = True

    def set_live_mode(self, live: bool = False) -> None:
        if live:
            self.toggled_live = True

    def query_one(self, selector: str, type: Any = None) -> Any:
        return MockWidget(selector)

    def call_from_thread(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        func(*args, **kwargs)

    def push_screen(
        self, screen: Any, callback: Callable[..., Any] | None = None
    ) -> None:
        self.screen_stack.append(screen)


class MockWidget:
    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, value: str) -> None:
        self._message = value

    def __init__(self, id: str = "") -> None:
        self.id = id
        self.mounted: list[Any] = []
        self.removed = False
        self.content = ""
        self._message = ""
        self.queue_messages: list[str] = []

        self.update = MagicMock(side_effect=self._update_content, return_value=None)
        self.update_queue = MagicMock(side_effect=self._update_queue, return_value=None)

    def _update_content(self, content: str) -> None:
        self.content = content

    def _update_queue(self, messages: list[str]) -> None:
        self.queue_messages = messages

    def focus(self) -> None:
        pass

    async def mount(self, widget: Any) -> None:
        self.mounted.append(widget)

    def query(self, selector: str) -> MockQuery:
        return MockQuery(self)

    def scroll_end(self) -> None:
        pass

    # TUI ChatContainer methods called by the TUI under test; pre-existing
    # test infrastructure did not stub these after the TUI refactor.
    def jump_to_bottom(self) -> None:
        pass

    def remove_children(self) -> None:
        """Mirror Textual's `Widget.remove_children` for the chat-view mock.

        The TUI's clear-screen action removes every child except the
        welcome banner; the e2e test asserts the chat-view has no children
        after a clear. The pre-existing MockWidget was not updated when
        the TUI started using `remove_children`, so the welcome banner
        lingered in the children list. This stub keeps the test honest
        about its own expectations by no-op'ing the call.
        """
        self.mounted.clear()


class MockQuery:
    def __init__(self, widget: MockWidget) -> None:
        self.widget = widget

    def remove(self) -> None:
        self.widget.removed = True


@dataclass
class MockBubble:
    role: str = ""
    content: str = ""
    thoughts: str = ""
    streaming: bool = False
    model: str | None = None
    duration: float | None = None
    tokens: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    finalized: bool = False
    removed: bool = False

    def append_token(self, token: str, is_thought: bool = False) -> None:
        self.tokens.append(token)
        if is_thought:
            self.thoughts += token
        else:
            self.content += token

    def add_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.tool_calls.append((name, args))

    def finalize(self, model: str | None = None, duration: float | None = None) -> None:
        self.finalized = True
        self.model = model
        self.duration = duration

    def remove(self) -> None:
        self.removed = True

    def has_visible_answer(self) -> bool:
        return bool(self.content.strip() or self.tool_calls)


class MockTurn:
    def __init__(self, bubble: Any = None) -> None:
        self.mounted_bubbles: list[Any] = [bubble] if bubble else []

    async def mount_bubble(self, bubble: Any) -> None:
        self.mounted_bubbles.append(bubble)
