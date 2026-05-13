from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from .config import config
from .logger import logger
from .types import LLMChunk, LLMOptions, LLMProvider, LLMResult


class ChatAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Any] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
        use_search: bool = False,
        use_code_execution: bool = False,
        system_instruction: str | None = None,
    ):
        self.provider = provider
        self.history: list[Any] = []
        self.current_interaction_id: str | None = None
        self.tools = tools
        self.registry = registry
        self.use_search = use_search
        self.use_code_execution = use_code_execution
        self.system_instruction = system_instruction
        logger.info(f"ChatAgent initialized with provider: {provider.get_name()}")

    def _prepare_execution_context(self) -> LLMOptions:
        return {
            "history": self.history,
            "interaction_id": self.current_interaction_id,
            "tools": self.tools or [],
            "registry": self.registry or {},
            "use_search": self.use_search,
            "use_code_execution": self.use_code_execution,
            "system_instruction": self.system_instruction,
        }

    def _trim_history(self) -> None:
        cap = config.history_max_turns
        if cap and len(self.history) > cap:
            del self.history[: len(self.history) - cap]

    async def execute(self, prompt: str) -> LLMResult:
        logger.info(f"Agent: executing prompt — length={len(prompt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Agent: full prompt: {prompt!r}")

        user_message = self.provider.build_user_message(prompt)
        options = self._prepare_execution_context()
        options["user_content"] = user_message
        result = await self.provider.generate(prompt, options)

        self.current_interaction_id = result.interaction_id
        self._sync_history(user_message, result)

        duration_str = (
            f"{result.duration:.3f}s" if result.duration is not None else "unknown"
        )
        logger.info(
            f"Agent: execution complete — duration={duration_str}, "
            f"history_size={len(self.history)}"
        )
        return result

    def _sync_history(self, user_message: Any, result: LLMResult) -> None:
        if result.automatic_function_calling_history:
            logger.info("Agent: syncing history from automatic function calling")
            self.history = result.automatic_function_calling_history
            self._trim_history()
            return

        self.history.append(user_message)

        assistant_message = self.provider.extract_assistant_message(result)
        if assistant_message is None:
            self._trim_history()
            return

        self.history.append(assistant_message)
        self._trim_history()
        logger.info(f"Agent: history synced — new size={len(self.history)}")

    async def execute_stream(
        self, prompt: str, exclude_from_history: bool = False
    ) -> AsyncIterator[LLMChunk]:
        logger.info(f"Agent: executing streaming prompt — length={len(prompt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Agent: full streaming prompt: {prompt!r}")

        user_message = self.provider.build_user_message(prompt)
        options = self._prepare_execution_context()
        options["user_content"] = user_message
        text_parts: list[str] = []

        async for chunk in self.provider.generate_stream(prompt, options):
            text_parts.append(chunk.text)
            yield chunk

        if not exclude_from_history:
            self.history.append(user_message)
            self.history.append(
                self.provider.build_assistant_message("".join(text_parts))
            )
            self._trim_history()
        logger.info(f"Agent: stream complete — history_size={len(self.history)}")

    def clear_history(self) -> None:
        logger.info("Clearing agent session history")
        self.history = []
        self.current_interaction_id = None
