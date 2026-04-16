import time
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

    def _get_execution_options(self) -> LLMOptions:
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
        result = await self.provider.generate(prompt, self._get_execution_options())

        self.current_interaction_id = result.interaction_id

        if result.automatic_function_calling_history:
            self.history = result.automatic_function_calling_history
        else:
            self.history.append(user_message)
            model_parts = (
                result.candidates[0].content.parts if result.candidates else []
            )
            self.history.append(Content(role="model", parts=model_parts))

        return result

    async def execute_stream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        user_message = Content(role="user", parts=[Part(text=prompt)])
        accumulated_text = ""

        async for chunk in self.provider.generate_stream(
            prompt, self._get_execution_options()
        ):
            accumulated_text += chunk.text
            yield chunk

        self.history.append(user_message)
        self.history.append(Content(role="model", parts=[Part(text=accumulated_text)]))

    def get_history(self) -> list[Content]:
        return self.history

    def clear_history(self) -> None:
        logger.info("Clearing agent session history")
        self.history = []
        self.current_interaction_id = None


class CrewAgent:
    def __init__(self, provider: LLMProvider, status_callback: Any = None):
        from .crew_agent import CrewManager

        self.manager = CrewManager(provider=provider, status_callback=status_callback)

    async def execute(self, prompt: str) -> LLMResult:
        start_time = time.perf_counter()
        result = await self.manager.coordinate_execution(prompt)
        result.duration = time.perf_counter() - start_time
        result.model = "CrewAI (Multi-Agent)"
        return result

    async def execute_stream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        raise NotImplementedError(
            "Streaming is not yet supported for multi-agent workflows"
        )

    def get_history(self) -> list[Content]:
        return []

    def clear_history(self) -> None:
        pass
