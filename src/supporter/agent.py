from collections.abc import Callable
from typing import TypedDict

from google.genai.types import Content, Part, Tool

from .index import LLMProvider
from .logger import logger


class ToolRegistry(dict[str, Callable]):
    pass


class AgentOptions(TypedDict, total=False):
    tools: list[Tool]
    registry: ToolRegistry
    system_instruction: str


class AgentResult(TypedDict):
    text: str
    model: str | None
    duration: float | None


class ChatAgent:
    """
    A unified chat agent interface that manages conversation history,
    tool execution, and LLM provider interactions.
    """

    def __init__(self, provider: LLMProvider, options: AgentOptions | None = None):
        """
        Initializes the agent with a provider and optional tools/instructions.

        Args:
            provider: The LLMProvider instance to use for generation.
            options: Configuration including tools, registry, and system instructions.
        """
        self.provider = provider
        self.history: list[Content] = []
        self.current_interaction_id: str | None = None

        opts = options or {}
        self.tools = opts.get("tools")
        self.registry = opts.get("registry")
        self.system_instruction = opts.get("system_instruction")

    async def execute(self, prompt: str) -> AgentResult:
        """
        Executes a prompt against the configured provider.

        Handles history synchronization between simple text responses and
        automatic function calling results.

        Args:
            prompt: The user input string.

        Returns:
            An AgentResult containing the response text and metadata.
        """
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
            f"Execution complete. Response length: {len(result.text) if result.text else 0}"
        )
        return {
            "text": result.text or "",
            "model": result.model,
            "duration": result.duration,
        }

    def get_history(self) -> list[Content]:
        return self.history

    def clear_history(self) -> None:
        logger.info("Clearing agent history")
        self.history = []
        self.current_interaction_id = None
