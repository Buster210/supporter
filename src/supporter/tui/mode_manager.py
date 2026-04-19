from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from textual.widgets import Label, Static

from ..config import config
from ..llm_types import DEFAULT_SYSTEM_INSTRUCTION
from ..logger import logger

if TYPE_CHECKING:
    pass


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class SpinnerController:
    def __init__(self, app: Any) -> None:
        self._app = app
        self._spinner_timer: Any | None = None
        self._spinner_idx: int = 0

    def start(self) -> None:
        self._app.active_queries += 1
        if self._app.active_queries == 1:
            self._spinner_idx = 0
            if self._spinner_timer:
                self._spinner_timer.stop()
            self._spinner_timer = self._app.set_interval(0.15, self._tick_spinner)

    def stop(self) -> None:
        self._app.active_queries = max(0, self._app.active_queries - 1)
        if self._app.active_queries == 0:
            if self._spinner_timer:
                self._spinner_timer.stop()
                self._spinner_timer = None
            indicator = self._app.query_one("#thinking-indicator", Static)
            indicator.update("")
            indicator.display = False
            indicator.refresh()

    def shutdown(self) -> None:
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * (self._spinner_idx % 4)
        self._spinner_idx += 1

        if self._app.is_activating_mode:
            status = f"Activating Mode{dots}"
        else:
            label = ""
            if self._app.crew_mode and self._app.current_active_agent:
                label = f"[{self._app.current_active_agent}] "

            status = f"{frame} {label}{self._app.status_label}{dots}"

        indicator = self._app.query_one("#thinking-indicator", Static)
        indicator.update(status)
        indicator.display = True


class ModeManager:
    def __init__(self, app: Any) -> None:
        self._app = app
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def setup_agent(self, use_crew: bool = False, use_live: bool = False) -> None:
        from .. import get_provider
        from ..agent import ChatAgent, CrewAgent
        from ..tools import list_dir, read_file, write_file

        registry: dict[str, Callable[..., Any]] = {
            "read_file": read_file,
            "write_file": write_file,
            "list_dir": list_dir,
        }

        provider = get_provider(live=use_live, registry=registry)
        if use_crew:
            self._app.agent = CrewAgent(
                provider=provider, status_callback=self._app._on_agent_active
            )
            return

        self._app.agent = ChatAgent(
            provider,
            registry=registry,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
            use_search=True,
            use_code_execution=True,
        )
        logger.info(f"Switched to standard chat agent (Live: {use_live})")

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

            self._update_mode_indicators()
            await self._announce_mode_change(crew, live)

        finally:
            self._app.is_activating_mode = False
            self._app._stop_thinking()

    def _update_mode_indicators(self) -> None:
        mode_text = "SINGLE"
        if self._app.crew_mode:
            mode_text = "CREW"
        elif self._app.live_mode:
            mode_text = "LIVE"

        indicator = self._app.query_one("#mode-indicator", Label)
        indicator.markup = False
        indicator.update(f"[{mode_text}]")

    async def _announce_mode_change(self, crew: bool, live: bool) -> None:
        is_enabled = self._app.crew_mode or self._app.live_mode
        status = "ENABLED" if is_enabled else "DISABLED"

        if crew:
            mode_label = "Multi-Agent Crew"
        else:
            model_name = config.gemini_live_model if self._app.live_mode else "Standard"
            mode_label = f"Single Agent Live ({model_name})"

        from .widgets import MessageBubble

        target = (
            self._app.active_turn
            if hasattr(self._app, "active_turn")
            else self._app.query_one("#chat-view", Any)
        )
        await target.mount(
            MessageBubble(role="agent", content=f"{mode_label} {status}")
        )

    async def handle_command(self, command: str) -> bool:
        command_map: dict[str, Callable[[], Any]] = {
            "/exit": self._app.exit,
            "/clear": self._app.action_clear_screen,
            "/crew": lambda: self._app._toggle_mode(crew=True),
            "/live": lambda: self._app._toggle_mode(live=True),
        }

        handler = command_map.get(command)
        if handler:
            result = handler()
            if asyncio.iscoroutine(result):
                await result
            return True
        return False

    def shutdown(self) -> None:
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
