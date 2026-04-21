from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from textual.widgets import Label, Static

from ..config import config
from ..llm_types import DEFAULT_SYSTEM_INSTRUCTION

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SpinnerController:
    def __init__(self, app: Any) -> None:
        self._app = app
        self._timer: Any | None = None
        self._idx: int = 0

    def start(self) -> None:
        self._app.active_queries += 1
        if self._app.active_queries == 1:
            self._idx = 0
            if self._timer:
                self._timer.stop()
            self._timer = self._app.set_interval(0.15, self._tick_spinner)
            self._tick_spinner()

    def stop(self) -> None:
        self._app.active_queries = max(0, self._app.active_queries - 1)
        if self._app.active_queries == 0:
            if self._timer:
                self._timer.stop()
                self._timer = None
            self._clear_indicator()

    def _clear_indicator(self) -> None:
        indicator = self._app.query_one("#thinking-indicator", Static)
        indicator.update("")
        indicator.display = False
        indicator.refresh()

    def shutdown(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._idx % len(SPINNER_FRAMES)]
        dots = "." * (self._idx % 4)
        self._idx += 1

        if self._app.is_activating_mode:
            status = f"Activating Mode{dots}"
        else:
            agent_role = getattr(self._app, "current_active_agent", None)
            prefix = f"[{agent_role}] " if self._app.crew_mode and agent_role else ""
            status = f"{frame} {prefix}{self._app.status_label}{dots}"

        indicator = self._app.query_one("#thinking-indicator", Static)
        indicator.update(status)
        indicator.display = True


class ModeManager:
    def __init__(self, app: Any) -> None:
        self._app = app
        self._tasks: set[asyncio.Task[Any]] = set()

    async def setup_agent(self, use_crew: bool = False, use_live: bool = False) -> None:
        from .. import get_provider
        from ..agent import ChatAgent, CrewAgent
        from ..tools import list_dir, read_file, write_file

        tools_registry: dict[str, Callable[..., Any]] = {
            "read_file": read_file,
            "write_file": write_file,
            "list_dir": list_dir,
        }

        provider = get_provider(live=use_live, registry=tools_registry)

        if use_crew:
            self._app.agent = CrewAgent(
                provider=provider, status_callback=self._app._on_agent_active
            )
            return

        self._app.agent = ChatAgent(
            provider,
            registry=tools_registry,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            use_search=True,
            use_code_execution=True,
        )

    async def toggle_mode(self, crew: bool = False, live: bool = False) -> None:
        if crew:
            self._app.crew_mode = not self._app.crew_mode
            if self._app.crew_mode:
                self._app.live_mode = False
        elif live:
            self._app.live_mode = not self._app.live_mode
            if self._app.live_mode:
                self._app.crew_mode = False

        self._app._start_thinking()
        self._app.is_activating_mode = True

        try:
            await self.setup_agent(
                use_crew=self._app.crew_mode, use_live=self._app.live_mode
            )
            self._update_ui(crew, live)
        finally:
            self._app.is_activating_mode = False
            self._app._stop_thinking()

    def _update_ui(self, crew: bool, live: bool) -> None:
        self._update_indicator()
        task = asyncio.create_task(self._announce_change(crew, live))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _update_indicator(self) -> None:
        mode_text = "SINGLE"
        if self._app.crew_mode:
            mode_text = "CREW"
        elif self._app.live_mode:
            mode_text = "LIVE"

        indicator = self._app.query_one("#mode-indicator", Label)
        indicator.markup = False
        indicator.update(f"[{mode_text}]")

    async def _announce_change(self, crew: bool, live: bool) -> None:
        is_enabled = self._app.crew_mode or self._app.live_mode
        status = "ENABLED" if is_enabled else "DISABLED"

        if crew:
            label = "Multi-Agent Crew"
        else:
            model = config.gemini_live_model if self._app.live_mode else "Standard"
            label = f"Single Agent Live ({model})"

        from .widgets import MessageBubble

        target = getattr(self._app, "active_turn", self._app.query_one("#chat-view"))
        await target.mount(MessageBubble(role="agent", content=f"{label} {status}"))

    async def handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()

        handlers = {
            "/exit": self._app.exit,
            "/clear": self._app.action_clear_screen,
            "/crew": lambda: self._app._toggle_mode(crew=True),
            "/live": lambda: self._app._toggle_mode(live=True),
        }

        if cmd not in handlers:
            return False

        result = handlers[cmd]()
        if asyncio.iscoroutine(result):
            await result
        return True

    def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
