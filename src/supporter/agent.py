import time
from collections.abc import Callable
from typing import Any

from google.genai.types import Content, Part, Tool

from .index import LLMOptions, LLMProvider, LLMResult
from .logger import logger


class ChatAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        registry: dict[str, Callable] | None = None,
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

    async def execute(self, prompt: str) -> LLMResult:
        logger.debug(f"Executing prompt: {prompt}")
        user_message = Content(role="user", parts=[Part(text=prompt)])
        options: LLMOptions = {
            "history": self.history,
            "interaction_id": self.current_interaction_id,
            "tools": self.tools,
            "registry": self.registry,
            "system_instruction": self.system_instruction,
            "use_search": self.use_search,
            "use_code_execution": self.use_code_execution,
        }
        result = await self.provider.generate(prompt, options)
        self.current_interaction_id = result.interaction_id
        if result.automatic_function_calling_history:
            logger.debug("Updating history with automatic function calling results")
            self.history = result.automatic_function_calling_history
        else:
            self.history.append(user_message)
            self.history.append(
                Content(
                    role="model",
                    parts=result.candidates[0].content.parts
                    if result.candidates
                    else [],
                )
            )
        response_len = len(result.text) if result.text else 0
        logger.debug(f"Execution complete. Response length: {response_len}")
        return result

    def get_history(self) -> list[Content]:
        return self.history

    def clear_history(self) -> None:
        logger.info("Clearing agent history")
        self.history = []
        self.current_interaction_id = None


class CrewAgent:
    def __init__(self, provider: LLMProvider, status_callback: Any = None):
        from .crew_agent import CrewManager

        self.manager = CrewManager(provider=provider, status_callback=status_callback)

    async def execute(self, prompt: str) -> LLMResult:
        start_time = time.perf_counter()
        logger.info(f"Executing crew for prompt: {prompt}")
        result = await self.manager.coordinate_execution(prompt)
        end_time = time.perf_counter()
        result.duration = end_time - start_time
        result.model = "CrewAI (Multi-Agent)"
        return result

    def get_history(self) -> list[Content]:
        return []

    def clear_history(self) -> None:
        pass
