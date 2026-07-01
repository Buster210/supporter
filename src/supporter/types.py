from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from textual.message import Message as TextualMessage

from .llm.types import GenOptions, Message


class TaskStatus(StrEnum):
    """Task execution status enumeration."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    ERROR = "error"
    STARTED = "started"
    PENDING = "pending"


@dataclass
class AppConfig:
    """Application configuration loaded from environment variables.

    Contains LLM provider settings, tool configuration, browser automation
    options, history management, and delegation/verification parameters.
    """

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
    delegate_default_persona: str
    delegate_agent_roster: dict[str, dict[str, Any]]
    delegate_max_retries: int
    delegate_correction_rounds: int = 3
    delegate_min_confidence: str = "medium"
    delegate_persist_noncode: bool = True
    delegate_result_repair: bool = True
    delegate_tier1_commands: list[list[str]] = field(default_factory=list)
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3
    history_max_turns: int = 200
    history_compaction_enabled: bool = True
    history_compaction_trigger: int = 160
    history_summary_keep_recent: int = 80
    browser_profile_path: str | None = None
    browser_profile_name: str | None = None
    browser_debug_overlay: bool = False
    browser_parallel_pilots: bool = True
    browser_diff_threshold: int = 40
    # Auto-close the browser after this many seconds with no interaction.
    # Any browser interaction resets the clock; 0 disables idle auto-close
    # (browser persists until explicit close).
    browser_idle_close_seconds: int = 600
    # D1: Browser output caps for page read, batch read, links, and eval results
    browse_page_chars_cap: int = 50_000
    browse_batch_chars_cap: int = 150_000
    browse_max_links: int = 100
    browse_eval_chars_cap: int = 16_000
    browse_fullpage_shot_max_px: int = 12_000
    durable_history_enabled: bool = True
    history_dir: str = ".supporter/history"
    replay_image_count: int = 2
    replay_tool_summary_max_chars: int = 200
    reconnect_attempts_max: int = 5
    reconnect_backoff_base: float = 0.5
    reconnect_backoff_cap: float = 8.0
    prewarm_safety_margin: float = 5.0
    idle_monitor_enabled: bool = True
    empty_resume_policy: str = "trust"

    browser_trusted_hosts: str = ""
    browser_micro_behavior_rate: float = 0.06
    browser_promotion_threshold: int = 5
    browser_auto_approve: bool = True
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-oss-120b:free"
    # G2: Plan → Implement → Verify → Replan loop: max replan cycles on verify failure
    replan_max_cycles: int = 3


@dataclass(frozen=True)
class SubtaskVerificationResult:
    task_id: str
    passed: bool
    reason: str = ""
    marker: str = ""


@dataclass
class ModeChanged(TextualMessage):
    """Event fired when an agent mode is toggled (live vs. offline, etc)."""

    mode: str
    enabled: bool


@dataclass
class LLMResult:
    """Result from an LLM generation call, including metadata and full history."""

    text: str
    model: str | None = None
    duration: float | None = None
    interaction_id: str | None = None
    thoughts: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    automatic_function_calling_history: list[Any] | None = None
    candidates: list[Any] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)


@dataclass
class LLMChunk:
    """A single chunk from streaming LLM generation (text, tool call, or thought)."""

    text: str
    is_last: bool
    is_thought: bool = False
    is_tool_call: bool = False
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    model: str | None = None
    raw: Any = None


class LLMProvider(Protocol):
    """Abstract protocol for LLM providers (Gemini, OpenRouter, etc)."""

    async def generate(
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> LLMResult:
        """Generate a single LLM response."""
        ...

    def generate_stream(
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        """Stream LLM response chunks."""
        ...

    def get_name(self) -> str:
        """Return the provider's name."""
        ...


@dataclass(frozen=True)
class DelegationEvent:
    """Base event emitted during task delegation (job_id scopes all events)."""

    job_id: str


@dataclass(frozen=True)
class MilestoneStarted(DelegationEvent):
    """Event fired when a delegation milestone starts."""

    milestone: str
    task_ids: list[str]
    parallel_cap: int


@dataclass(frozen=True)
class MilestoneCompleted(DelegationEvent):
    """Event fired when a delegation milestone completes successfully."""

    milestone: str
    results: list[dict[str, Any]]
    total_duration: float


@dataclass(frozen=True)
class MilestoneCancelled(DelegationEvent):
    """Event fired when a delegation milestone is cancelled."""

    milestone: str
    total_duration: float


@dataclass(frozen=True)
class TaskStarted(DelegationEvent):
    """Event fired when a task starts execution."""

    task_id: str
    agent_label: str
    started_at: float
    timeout: float


@dataclass(frozen=True)
class TaskCompleted(DelegationEvent):
    """Event fired when a task completes successfully."""

    task_id: str
    duration: float
    output: str
    model: str
    summary: str = ""
    confidence: str = "unknown"
    findings_count: int = 0
    evidence_counts: dict[str, int] = field(default_factory=dict)
    handoff: str = ""
    tokens: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0


@dataclass(frozen=True)
class TaskFailed(DelegationEvent):
    """Event fired when a task fails with an error."""

    task_id: str
    duration: float
    error: str


@dataclass(frozen=True)
class TaskTimedOut(DelegationEvent):
    """Event fired when a task exceeds its timeout."""

    task_id: str
    duration: float


@dataclass(frozen=True)
class TaskOutputChunk(DelegationEvent):
    """Event fired when a task emits a chunk of streaming output."""

    task_id: str
    chunk: str
    seq: int


@dataclass(frozen=True)
class TaskSkipped(DelegationEvent):
    """Event fired when a task is skipped."""

    task_id: str
    reason: str


@dataclass(frozen=True)
class TaskRetrying(DelegationEvent):
    """Event fired when a task is retried after a transient failure."""

    task_id: str
    attempt: int
    reason: str


@dataclass(frozen=True)
class HeartbeatTick(DelegationEvent):
    """Periodic event emitted during long-running milestone execution."""

    milestone: str
    snapshot: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TaskAnomaly(DelegationEvent):
    """Event fired when a task shows signs of hanging (exceeded time threshold)."""

    task_id: str
    agent_label: str
    elapsed_seconds: float
    timeout: float


@dataclass(frozen=True)
class TaskUpdateSent(DelegationEvent):
    task_id: str
    message: str


@dataclass(frozen=True)
class VerificationVerdict(DelegationEvent):
    """A verification outcome for the UI.

    scope='task' for Phase A per-subtask QA, scope='objective' for Phase B
    verify_plan.
    """

    task_id: str  # subtask id, or the objective label for scope='objective'
    scope: str  # "task" | "objective"
    passed: bool
    reason: str = ""
    round: int = 0  # Phase A correction round (0 for objective)
