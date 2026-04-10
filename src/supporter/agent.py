import time
from collections.abc import Callable
from typing import Any, TypedDict
from google.genai.types import Content, Part, Tool
from .index import LLMProvider
from .logger import logger
from .crew_agent import CrewManager


class ToolRegistry(dict[str, Callable]):
    pass


class AgentOptions(TypedDict, total=False):
    tools: list[Tool]
    registry: ToolRegistry
    system_instruction: str
    use_search: bool
    use_code_execution: bool


class AgentResult(TypedDict):
    text: str
    model: str | None
    duration: float | None
    agents: list[str] | None


class ChatAgent:
    def __init__(self, provider: LLMProvider, options: AgentOptions | None = None):
        self.provider = provider
        self.history: list[Content] = []
        self.current_interaction_id: str | None = None
        opts = options or {}
        self.tools = opts.get("tools")
        self.registry = opts.get("registry")
        self.system_instruction = opts.get("system_instruction")
        self.use_search = opts.get("use_search", False)
        self.use_code_execution = opts.get("use_code_execution", False)

    async def execute(self, prompt: str) -> AgentResult:
        logger.debug(f"Executing prompt: {prompt}")
        user_message = Content(role="user", parts=[Part(text=prompt)])
        result = await self.provider.generate(
            prompt,
            {
                "history": self.history,
                "interaction_id": self.current_interaction_id,
                "tools": self.tools,
                "registry": self.registry,
                "system_instruction": self.system_instruction,
                "use_search": self.use_search,
                "use_code_execution": self.use_code_execution,
            },
        )
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
        logger.debug(
            f"Execution complete. Response length: {(len(result.text) if result.text else 0)}"
        )
        return {
            "text": result.text or "",
            "model": result.model,
            "duration": result.duration,
            "agents": None,
        }

    def get_history(self) -> list[Content]:
        return self.history

    def clear_history(self) -> None:
        logger.info("Clearing agent history")
        self.history = []
        self.current_interaction_id = None


class CrewAgent:
    def __init__(self, status_callback: Any = None):
        self.manager = CrewManager(status_callback=status_callback)

    async def execute(self, prompt: str) -> AgentResult:
        start_time = time.perf_counter()
        logger.info(f"Executing crew for prompt: {prompt}")
        (result_text, agent_roles) = await self.manager.coordinate_execution(prompt)
        end_time = time.perf_counter()
        return {
            "text": result_text,
            "model": "CrewAI (Multi-Agent)",
            "duration": end_time - start_time,
            "agents": agent_roles,
        }

    def get_history(self) -> list[Content]:
        return []

    def clear_history(self) -> None:
        pass
