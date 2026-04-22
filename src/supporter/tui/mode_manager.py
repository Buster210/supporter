from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..llm_types import DEFAULT_SYSTEM_INSTRUCTION
from .message_processor import ModeChanged


class ModeManager:
    def __init__(self, app: Any) -> None:
        self._app = app

    async def setup_agent(self, use_crew: bool = False, use_live: bool = False) -> None:
        from .. import get_provider
        from ..agent import ChatAgent, CrewAgent
        from ..tools import (
            check_bash_availability,
            execute_bash,
            list_dir,
            notify_bash_unavailable,
            read_file,
            write_file,
        )

        tools_registry: dict[str, Callable[..., Any]] = {
            "read_file": read_file,
            "write_file": write_file,
            "list_dir": list_dir,
        }

        if check_bash_availability():
            tools_registry["execute_bash"] = execute_bash
        else:
            notify_bash_unavailable()

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
            self._update_ui_state()
        finally:
            self._app.is_activating_mode = False
            self._app._stop_thinking()

    def _update_ui_state(self) -> None:
        mode = "SINGLE"
        if self._app.crew_mode:
            mode = "CREW"
        elif self._app.live_mode:
            mode = "LIVE"

        self._app.post_message(ModeChanged(mode=mode, enabled=True))

    async def handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        handlers = {
            "/exit": self._app.exit,
            "/clear": self._app.action_clear_screen,
            "/crew": lambda: self._app._toggle_mode(crew=True),
            "/live": lambda: self._app._toggle_mode(live=True),
        }

        handler = handlers.get(cmd)
        if not handler:
            return False

        result = handler()
        if asyncio.iscoroutine(result):
            await result
        return True
