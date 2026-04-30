from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..logger import logger
from ..types import ModeChanged


class ModeManager:
    def __init__(self, app: Any) -> None:
        self._app = app

    async def setup_agent(self, use_live: bool = False) -> None:
        from .. import get_provider
        from ..agent import ChatAgent
        from ..config import config
        from ..tools import (
            check_bash_availability,
            collect_delegation,
            delegate_tasks,
            execute_bash,
            notify_bash_unavailable,
            read_file,
            write_file,
        )

        tools_registry: dict[str, Callable[..., Any]] = {
            "read_file": read_file,
            "write_file": write_file,
            "delegate_tasks": delegate_tasks,
            "collect_delegation": collect_delegation,
        }

        if check_bash_availability():
            tools_registry["execute_bash"] = execute_bash
        else:
            notify_bash_unavailable()

        provider = get_provider(live=use_live, registry=tools_registry)

        logger.info(
            f"ModeManager: setting up agent — use_live={use_live}, "
            f"tools={list(tools_registry.keys())}"
        )
        self._app.agent = ChatAgent(
            provider,
            registry=tools_registry,
            use_search=True,
            use_code_execution=True,
            system_instruction=config.default_system_instruction,
        )

    async def toggle_mode(self, live: bool | None = None) -> None:
        if live is not None and self._app.live_mode == live:
            from .bubble import MessageBubble

            mode_name = "LIVE" if live else "SINGLE Agent"
            target = getattr(self._app, "active_turn", None) or self._app.query_one(
                "#chat-view"
            )
            await target.mount(
                MessageBubble(role="agent", content=f"Already in {mode_name} mode")
            )
            return

        if live is not None:
            self._app.live_mode = live
        else:
            self._app.live_mode = not self._app.live_mode

        logger.info(f"ModeManager: toggling mode — live={self._app.live_mode}")
        self._app._start_thinking()
        self._app.is_activating_mode = True

        try:
            await self.setup_agent(use_live=self._app.live_mode)
            self._update_ui_state()
        finally:
            self._app.is_activating_mode = False
            self._app._stop_thinking()

    def _update_ui_state(self) -> None:
        mode = "LIVE" if self._app.live_mode else "SINGLE"
        self._app.post_message(ModeChanged(mode=mode, enabled=True))

    async def handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        handlers = {
            "/exit": self._app.exit,
            "/clear": self._app.action_clear_screen,
            "/live": lambda: self._app._toggle_mode(live=True),
            "/agent": lambda: self._app._toggle_mode(live=False),
        }

        if cmd not in handlers:
            return False

        result = handlers[cmd]()
        if asyncio.iscoroutine(result):
            await result
        return True
