from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from google.genai.types import Content, GenerateContentConfig, Tool


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
