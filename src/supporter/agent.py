from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai.types import Content, Tool

from .config import config
from .logger import logger
from .types import LLMChunk, LLMOptions, LLMProvider, LLMResult


class ChatAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
        use_search: bool = False,
        use_code_execution: bool = False,
        system_instruction: str | None = None,
    ):
        self.provider = provider
        self.history: list[Content] = []
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
        from google.genai.types import Content, Part

        user_message = Content(role="user", parts=[Part(text=prompt)])
        result = await self.provider.generate(prompt, self._prepare_execution_context())

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

    def _sync_history(self, user_message: Content, result: LLMResult) -> None:
        if result.automatic_function_calling_history:
            logger.info("Agent: syncing history from automatic function calling")
            self.history = result.automatic_function_calling_history
            self._trim_history()
            return

        self.history.append(user_message)

        if not result.candidates or not result.candidates[0].content:
            self._trim_history()
            return

        from google.genai.types import Content

        self.history.append(
            Content(role="model", parts=result.candidates[0].content.parts)
        )
        self._trim_history()
        logger.info(f"Agent: history synced — new size={len(self.history)}")

    async def execute_stream(
        self, prompt: str, exclude_from_history: bool = False
    ) -> AsyncIterator[LLMChunk]:
        logger.info(f"Agent: executing streaming prompt — length={len(prompt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Agent: full streaming prompt: {prompt!r}")
        from google.genai.types import Content, Part

        user_message = Content(role="user", parts=[Part(text=prompt)])
        text_parts: list[str] = []

        async for chunk in self.provider.generate_stream(
            prompt, self._prepare_execution_context()
        ):
            text_parts.append(chunk.text)
            yield chunk

        if not exclude_from_history:
            self.history.append(user_message)
            from google.genai.types import Content, Part

            self.history.append(
                Content(role="model", parts=[Part(text="".join(text_parts))])
            )
            self._trim_history()
        logger.info(f"Agent: stream complete — history_size={len(self.history)}")

    def clear_history(self) -> None:
        logger.info("Clearing agent session history")
        self.history = []
        self.current_interaction_id = None
