from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable
from typing import Any

from ..logger import logger
from ..types import ModeChanged


class ModeManager:
    def __init__(self, app: Any) -> None:
        self._app = app
        from ..config import config
        from ..providers.gemini_live_provider import GeminiLiveProvider

        self._greeting_provider: Any = None
        if config.gemini_api_keys:
            self._greeting_provider = GeminiLiveProvider(
                config.gemini_api_keys,
                model_name=config.gemini_live_model,
                system_instruction=(
                    "Greeting assistant. Output exactly one friendly sentence. "
                    "No preamble, no thoughts."
                ),
                include_thoughts=False,
            )
        self._warmup_task: asyncio.Task[Any] | None = None

    def start_warmup(self) -> None:
        if self._greeting_provider is None:
            return
        if self._warmup_task is None:
            self._warmup_task = asyncio.create_task(self._greeting_provider.warmup())

    async def close(self) -> None:
        if hasattr(self, "_warmup_task") and self._warmup_task:
            self._warmup_task.cancel()
        if hasattr(self, "_greeting_provider") and self._greeting_provider:
            await self._greeting_provider.close()

    async def setup_agent(self, use_live: bool = False) -> None:
        from .. import get_provider
        from ..agent import ChatAgent
        from ..config import config
        from ..tools import (
            cancel_delegation,
            check_bash_availability,
            check_delegation,
            delegate_tasks,
            execute_bash,
            notify_bash_unavailable,
            query_delegation,
            read_file,
            write_file,
        )

        tools_registry: dict[str, Callable[..., Any]] = {
            "read_file": read_file,
            "write_file": write_file,
            "delegate_tasks": delegate_tasks,
            "check_delegation": check_delegation,
            "cancel_delegation": cancel_delegation,
            "query_delegation": query_delegation,
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

        self._app.live_mode = live if live is not None else not self._app.live_mode

        logger.info(f"ModeManager: toggling mode — live={self._app.live_mode}")
        self._app._start_thinking()
        self._app.is_activating_mode = True

        try:
            await self.setup_agent(use_live=self._app.live_mode)
            self._update_ui_state()

            if self._app.live_mode:
                try:
                    await self.trigger_live_greeting()
                    logger.info("ModeManager: Successfully queued greeting worker")
                except Exception as e:
                    logger.error(f"ModeManager: Failed to queue greeting worker: {e}")
        finally:
            self._app.is_activating_mode = False
            self._app._stop_thinking()

    async def trigger_live_greeting(self) -> None:
        import datetime
        import getpass

        username = getpass.getuser()
        now = datetime.datetime.now().strftime("%I:%M %p")
        prompt = (
            f"Give a short, unique, and friendly one-sentence greeting to {username}. "
            f"Current time is {now}. Include the exact username `{username}` "
            "in the sentence."
        )

        banner = self._app.query_one("#welcome-banner")
        if self._greeting_provider is None:
            banner.message = self._bold_username(f"Hello {username}!", username)
            logger.info("ModeManager: No greeting provider; using fallback message")
            return
        logger.info(f"ModeManager: Starting persistent live greeting for {username}")
        loading_stop = asyncio.Event()
        loading_task: asyncio.Task[None] | None = asyncio.create_task(
            self._animate_loading_banner(banner, loading_stop)
        )

        try:
            self.start_warmup()
            if self._warmup_task:
                await self._warmup_task

            full_text = ""
            async for chunk in self._greeting_provider.generate_stream(prompt):
                if chunk.text:
                    if not loading_stop.is_set():
                        loading_stop.set()
                        if loading_task is not None:
                            with contextlib.suppress(asyncio.CancelledError):
                                await loading_task
                            loading_task = None
                    full_text += chunk.text
                    banner.message = self._bold_username(full_text.strip(), username)
            if not full_text.strip():
                banner.message = self._bold_username(f"Hello {username}!", username)

            logger.info("ModeManager: Persistent live greeting complete")
        except Exception as e:
            logger.error(f"ModeManager: Greeting failed: {e}")
            banner.message = self._bold_username(f"Hello {username}!", username)
        finally:
            loading_stop.set()
            if loading_task is not None:
                loading_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await loading_task

    async def _animate_loading_banner(
        self, banner: Any, stop_event: asyncio.Event
    ) -> None:
        frames = ("wait, Loading.", "wait, Loading..", "wait, Loading...")
        idx = 0
        while not stop_event.is_set():
            banner.message = frames[idx % len(frames)]
            idx += 1
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.18)
            except TimeoutError:
                continue

    def _update_ui_state(self) -> None:
        mode = "LIVE" if self._app.live_mode else "SINGLE"
        self._app.post_message(ModeChanged(mode=mode, enabled=True))

    @staticmethod
    def _bold_username(text: str, username: str) -> str:
        pattern = re.escape(username)
        return re.sub(pattern, f"[b]{username}[/b]", text, count=1)

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
