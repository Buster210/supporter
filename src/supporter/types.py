from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from textual.message import Message as TextualMessage

from .llm.types import GenOptions, Message


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    ERROR = "error"
    STARTED = "started"
    PENDING = "pending"


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

    # WI-3: Generalized trust gating
    browser_trusted_hosts: str = ""
    browser_micro_behavior_rate: float = 0.06
    browser_promotion_threshold: int = 5
    # Auto-approve browser actions (files remain gated)
    browser_auto_approve: bool = True
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-oss-120b:free"
    # G2: Plan → Implement → Verify → Replan loop: max replan cycles on verify failure
    replan_max_cycles: int = 3


@dataclass
class ModeChanged(TextualMessage):
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
    automatic_function_calling_history: list[Any] | None = None
    candidates: list[Any] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)


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
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> LLMResult: ...

    def generate_stream(
        self, prompt: str | list[Message], options: GenOptions | None = None
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
    tokens: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0


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
class TaskOutputChunk(DelegationEvent):
    task_id: str
    chunk: str
    seq: int


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
