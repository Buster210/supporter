from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from google.genai.types import Content, GenerateContentConfig, Tool

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an elite technical strategist and principal software architect. "
    "Your objective is to provide rigorous, high-fidelity, and "
    "architecturally sound guidance. Analyze complex problems through "
    "the lens of scalability, maintainability, and efficiency. Always "
    "anticipate edge cases and performance bottlenecks before "
    "formulating a response."
)


class LLMOptions(TypedDict, total=False):
    history: list[Content]
    model: str
    tools: list[Tool]
    registry: dict[str, Callable[..., Any]]
    interaction_id: str | None
    use_search: bool
    use_code_execution: bool
    system_instruction: str | None
    thinking_level: str | None
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    config: GenerateContentConfig


@dataclass
class LLMResult:
    text: str
    model: str | None = None
    duration: float | None = None
    interaction_id: str | None = None
    thoughts: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    automatic_function_calling_history: list[Content] | None = None
    candidates: list[Any] = field(default_factory=list)


@dataclass
class LLMChunk:
    text: str
    is_last: bool
    is_thought: bool = False
    is_tool_call: bool = False
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    model: str | None = None
    raw: Any = None


class LLMProvider(Protocol):
    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult: ...

    def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]: ...

    def get_name(self) -> str: ...
