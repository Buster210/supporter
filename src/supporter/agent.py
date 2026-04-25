from collections.abc import AsyncIterator, Callable
from typing import Any

from google.genai.types import Content, Part, Tool

from .llm_types import LLMChunk, LLMOptions, LLMProvider, LLMResult
from .logger import logger


class ChatAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
        system_instruction: str | None = None,
        use_search: bool = False,
        use_code_execution: bool = False,
    ):
        self.provider = provider
        self.history: list[Content] = []
        self.current_interaction_id: str | None = None
        self.tools = tools
        self.registry = registry
        self.system_instruction = system_instruction
        self.use_search = use_search
        self.use_code_execution = use_code_execution
        logger.debug(f"ChatAgent initialized with provider: {provider.get_name()}")

    def _prepare_execution_context(self) -> LLMOptions:
        return {
            "history": self.history,
            "interaction_id": self.current_interaction_id,
            "tools": self.tools or [],
            "registry": self.registry or {},
            "system_instruction": self.system_instruction,
            "use_search": self.use_search,
            "use_code_execution": self.use_code_execution,
        }

    async def execute(self, prompt: str) -> LLMResult:
        user_message = Content(role="user", parts=[Part(text=prompt)])
        result = await self.provider.generate(prompt, self._prepare_execution_context())

        self.current_interaction_id = result.interaction_id
        self._sync_history(user_message, result)

        return result

    def _sync_history(self, user_message: Content, result: LLMResult) -> None:
        if result.automatic_function_calling_history:
            self.history = result.automatic_function_calling_history
            return

        self.history.append(user_message)

        if not result.candidates or not result.candidates[0].content:
            return

        self.history.append(
            Content(role="model", parts=result.candidates[0].content.parts)
        )

    async def execute_stream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        user_message = Content(role="user", parts=[Part(text=prompt)])
        accumulated_text = ""

        async for chunk in self.provider.generate_stream(
            prompt, self._prepare_execution_context()
        ):
            accumulated_text += chunk.text
            yield chunk

        self.history.append(user_message)
        self.history.append(Content(role="model", parts=[Part(text=accumulated_text)]))

    def clear_history(self) -> None:
        logger.info("Clearing agent session history")
        self.history = []
        self.current_interaction_id = None
