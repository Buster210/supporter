from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


@dataclass
class MockMessage:
    pass


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

    def _start_thinking(self) -> None:
        self.active_queries += 1

    def _stop_thinking(self) -> None:
        self.active_queries = max(0, self.active_queries - 1)

    def exit(self) -> None:
        self.exited = True

    def action_clear_screen(self) -> None:
        self.cleared = True

    def _toggle_mode(self, live: bool = False) -> None:
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
    def __init__(self, id: str = "") -> None:
        self.id = id
        self.mounted: list[Any] = []
        self.removed = False
        self.content = ""
        self.queue_messages: list[str] = []
        from unittest.mock import MagicMock

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


class MockQuery:
    def __init__(self, widget: MockWidget) -> None:
        self.widget = widget

    def remove(self) -> None:
        self.widget.removed = True


@dataclass
class MockBubble:
    role: str = ""
    content: str = ""
    streaming: bool = False
    model: str | None = None
    duration: float | None = None
    tokens: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    finalized: bool = False

    def append_token(self, token: str, is_thought: bool = False) -> None:
        self.tokens.append(token)
        self.content += token

    def add_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.tool_calls.append((name, args))

    def finalize(self, model: str | None = None, duration: float | None = None) -> None:
        self.finalized = True
        self.model = model
        self.duration = duration


class MockTurn:
    def __init__(self, bubble: Any = None) -> None:
        self.mounted_bubbles: list[Any] = [bubble] if bubble else []

    async def mount_bubble(self, bubble: Any) -> None:
        self.mounted_bubbles.append(bubble)

    def auto_collapse(self) -> None:
        pass
