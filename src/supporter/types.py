from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from google.genai.types import Content, GenerateContentConfig, Tool
from textual.message import Message


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
class AppConfig:
    log_level: str
    provider: str
    gemini_api_keys: list[str]
    gemini_model: str
    gemini_live_model: str
    gemini_live_fallback_model: str
    gemini_fallback_model: str | None
    log_file: str
    voice_name: str
    default_system_instruction: str
    allowed_directories: list[str]
    require_write_confirmation: bool
    live_thinking_level: str
    retriable_codes: set[str]
    google_5xx_errors: set[str]
    transient_signals: set[str]
    http_errors_5xx: set[int]
    rate_limit_signals: set[str]
    drain_timeout: float
    context_trigger_tokens: int
    context_target_tokens: int


@dataclass
class MockCandidate:
    grounding_metadata: Any


@dataclass
class MockRaw:
    candidates: list[MockCandidate]


@dataclass
class ModeChanged(Message):
    mode: str
    enabled: bool


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
