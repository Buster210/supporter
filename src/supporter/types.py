from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypedDict

from google.genai.types import Content, GenerateContentConfig, Tool
from textual.message import Message


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    ERROR = "error"
    STARTED = "started"
    PENDING = "pending"


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
    retriable_error_strings: set[str]
    google_api_5xx_exceptions: set[str]
    transient_error_strings: set[str]
    http_5xx_status_codes: set[int]
    rate_limit_error_strings: set[str]
    drain_timeout: float
    context_trigger_tokens: int
    context_target_tokens: int
    http_retry_attempts: int
    delegate_max_hard_cap: int
    delegate_default_parallel: int
    delegate_default_timeout: int
    delegate_max_timeout: int
    delegate_max_tasks: int
    delegate_max_output_chars: int
    delegate_allowed_tools: set[str]
    delegate_default_persona: str
    delegate_agent_roster: dict[str, dict[str, Any]]
    delegate_max_retries: int
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3
    history_max_turns: int = 200


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


@dataclass(frozen=True)
class DelegationEvent:
    job_id: str


@dataclass(frozen=True)
class MilestoneStarted(DelegationEvent):
    milestone: str
    task_ids: list[str]
    parallel_cap: int


@dataclass(frozen=True)
class MilestoneCompleted(DelegationEvent):
    milestone: str
    results: list[dict[str, Any]]
    total_duration: float


@dataclass(frozen=True)
class MilestoneCancelled(DelegationEvent):
    milestone: str
    total_duration: float


@dataclass(frozen=True)
class TaskStarted(DelegationEvent):
    task_id: str
    agent_label: str
    started_at: float
    timeout: float


@dataclass(frozen=True)
class TaskCompleted(DelegationEvent):
    task_id: str
    duration: float
    output: str
    model: str
    summary: str = ""
    confidence: str = "unknown"
    findings_count: int = 0
    evidence_counts: dict[str, int] = field(default_factory=dict)
    handoff: str = ""


@dataclass(frozen=True)
class TaskFailed(DelegationEvent):
    task_id: str
    duration: float
    error: str


@dataclass(frozen=True)
class TaskTimedOut(DelegationEvent):
    task_id: str
    duration: float


@dataclass(frozen=True)
class TaskSkipped(DelegationEvent):
    task_id: str
    reason: str


@dataclass(frozen=True)
class TaskRetrying(DelegationEvent):
    task_id: str
    attempt: int
    reason: str


@dataclass(frozen=True)
class HeartbeatTick(DelegationEvent):
    milestone: str
    snapshot: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TaskAnomaly(DelegationEvent):
    task_id: str
    agent_label: str
    elapsed_seconds: float
    timeout: float
