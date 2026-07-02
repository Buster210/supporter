import dataclasses
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .prompts import (
    DEFAULT_SYSTEM_INSTRUCTION,
    DELEGATE_AGENT_ROSTER,
    DELEGATE_DEFAULT_PERSONA,
    MODEL_GEMINI_LIVE,
    MODEL_GEMINI_LIVE_FALLBACK,
    MODEL_GEMMA_26B,
    MODEL_GEMMA_31B,
)
from .types import AppConfig

__all__ = ["AppConfig", "config", "reload_config"]

HTTP_RATE_LIMIT = 429
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_RETRY_ATTEMPTS = 2

INTERNAL_BLACKLIST = [
    ".env",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
]

DEFAULT_MODEL = MODEL_GEMMA_31B

RATE_LIMIT_ERROR_STRINGS = {"quota", "too many requests", "429"}

RETRIABLE_ERROR_STRINGS = {
    "1000",
    "1006",
    "1011",
    "1007",
    "1008",
    "exhausted",
} | RATE_LIMIT_ERROR_STRINGS

GOOGLE_API_5XX_EXCEPTIONS = {
    "InternalServerError",
    "ServiceUnavailable",
    "BadGateway",
    "GatewayTimeout",
    "APIError",
}

TRANSIENT_ERROR_STRINGS = {
    "unavailable",
    "overloaded",
    "internal error",
    "service level",
    "cooldown",
}
HTTP_5XX_STATUS_CODES = {HTTP_SERVICE_UNAVAILABLE, HTTP_INTERNAL_ERROR, 502, 504}

DRAIN_TIMEOUT = 2.0
CONTEXT_TRIGGER_TOKENS = 100_000
CONTEXT_TARGET_TOKENS = 4_000

DELEGATE_MAX_HARD_CAP = 5
DELEGATE_DEFAULT_PARALLEL = 3
DELEGATE_DEFAULT_TIMEOUT = 180
DELEGATE_MAX_TIMEOUT = 600
DELEGATE_MAX_TASKS = 10
DELEGATE_MAX_OUTPUT_CHARS = 30000
DELEGATE_MAX_RETRIES = 2

DELEGATE_RESULT_REPAIR = True
DELEGATE_CORRECTION_ROUNDS = 3
DELEGATE_MIN_CONFIDENCE = "medium"
DELEGATE_PERSIST_NONCODE = True
DELEGATE_HEARTBEAT_INTERVAL = 30
DELEGATE_ANOMALY_THRESHOLD = 0.8
DELEGATE_JOB_ID_LEN = 8
DELEGATE_RETRY_BACKOFF = [1.0, 3.0]
DELEGATE_TIER1_COMMANDS: list[list[str]] = []


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"${name} must be an integer, got: {raw!r}") from exc


def _confidence_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.lower()
    allowed = {"low", "medium", "high"}
    if value not in allowed:
        raise ValueError(f"${name} must be one of {allowed}, got: {raw!r}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("true", "1", "yes")


def _cmd_list_env(name: str, default: list[list[str]]) -> list[list[str]]:
    """Parse a JSON array-of-string-arrays env var into ``list[list[str]]``.

    An empty/unset value returns ``default``. A non-empty value that is not
    shaped as a list of lists of strings raises ``ValueError`` so the bad
    config is caught at startup, not at the first tier-1 run.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"${name} must be a JSON array of string arrays: {exc}"
        ) from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, list) and all(isinstance(tok, str) for tok in item)
        for item in parsed
    ):
        raise ValueError(f"${name} must be a JSON array of string arrays")
    return [list(cmd) for cmd in parsed]


def _get_project_root() -> str:
    current = Path(__file__).resolve().parent
    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return str(parent)
    return os.getcwd()


def load_config() -> AppConfig:
    load_dotenv()
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    stripped = raw_keys.replace("\n", " ").replace("\r", "").strip()
    if stripped.startswith("["):
        try:
            keys = [k for k in json.loads(stripped) if isinstance(k, str) and k.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"GEMINI_API_KEYS is not valid JSON array: {e}") from e
    else:
        keys = [k.strip() for k in stripped.split(",") if k.strip()]
    project_root = _get_project_root()

    return AppConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_keys=keys,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_live_model=os.getenv("GEMINI_LIVE_MODEL", MODEL_GEMINI_LIVE),
        gemini_live_fallback_model=os.getenv(
            "GEMINI_LIVE_FALLBACK_MODEL", MODEL_GEMINI_LIVE_FALLBACK
        ),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", MODEL_GEMMA_26B),
        log_file=os.getenv("LOG_FILE", "logs/app.log"),
        voice_name=os.getenv("GEMINI_VOICE_NAME", "Puck"),
        default_system_instruction=os.getenv(
            "DEFAULT_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION
        ),
        allowed_directories=[project_root],
        require_write_confirmation=os.getenv(
            "REQUIRE_WRITE_CONFIRMATION", "true"
        ).lower()
        == "true",
        live_thinking_level=os.getenv("GEMINI_LIVE_THINKING_LEVEL", "high").lower(),
        retriable_error_strings=RETRIABLE_ERROR_STRINGS,
        google_api_5xx_exceptions=GOOGLE_API_5XX_EXCEPTIONS,
        transient_error_strings=TRANSIENT_ERROR_STRINGS,
        http_5xx_status_codes=HTTP_5XX_STATUS_CODES,
        rate_limit_error_strings=RATE_LIMIT_ERROR_STRINGS,
        drain_timeout=DRAIN_TIMEOUT,
        context_trigger_tokens=CONTEXT_TRIGGER_TOKENS,
        context_target_tokens=CONTEXT_TARGET_TOKENS,
        http_retry_attempts=_int_env("HTTP_RETRY_ATTEMPTS", HTTP_RETRY_ATTEMPTS),
        delegate_max_hard_cap=DELEGATE_MAX_HARD_CAP,
        delegate_default_parallel=DELEGATE_DEFAULT_PARALLEL,
        delegate_default_timeout=DELEGATE_DEFAULT_TIMEOUT,
        delegate_max_timeout=DELEGATE_MAX_TIMEOUT,
        delegate_max_tasks=DELEGATE_MAX_TASKS,
        delegate_max_output_chars=_int_env(
            "DELEGATE_MAX_OUTPUT_CHARS", DELEGATE_MAX_OUTPUT_CHARS
        ),
        delegate_default_persona=DELEGATE_DEFAULT_PERSONA,
        delegate_agent_roster=DELEGATE_AGENT_ROSTER,
        delegate_max_retries=_int_env("DELEGATE_MAX_RETRIES", DELEGATE_MAX_RETRIES),
        delegate_correction_rounds=_int_env(
            "DELEGATE_CORRECTION_ROUNDS", DELEGATE_CORRECTION_ROUNDS
        ),
        delegate_min_confidence=_confidence_env(
            "DELEGATE_MIN_CONFIDENCE", DELEGATE_MIN_CONFIDENCE
        ),
        delegate_persist_noncode=_bool_env(
            "DELEGATE_PERSIST_NONCODE", DELEGATE_PERSIST_NONCODE
        ),
        delegate_result_repair=_bool_env(
            "DELEGATE_RESULT_REPAIR", DELEGATE_RESULT_REPAIR
        ),
        delegate_tier1_commands=_cmd_list_env(
            "DELEGATE_TIER1_COMMANDS", DELEGATE_TIER1_COMMANDS
        ),
        log_max_bytes=_int_env("LOG_MAX_BYTES", 5_000_000),
        log_backup_count=_int_env("LOG_BACKUP_COUNT", 3),
        history_max_turns=_int_env("HISTORY_MAX_TURNS", 200),
        history_compaction_enabled=_bool_env("HISTORY_COMPACTION", True),
        history_compaction_trigger=_int_env("HISTORY_COMPACTION_TRIGGER", 160),
        history_summary_keep_recent=_int_env("HISTORY_SUMMARY_KEEP_RECENT", 80),
        browser_profile_path=os.getenv("BROWSER_PROFILE_PATH"),
        browser_profile_name=os.getenv("BROWSER_PROFILE_NAME"),
        browser_debug_overlay=_bool_env("BROWSER_DEBUG_OVERLAY", False),
        browser_parallel_pilots=_bool_env("BROWSER_PARALLEL_PILOTS", True),
        browser_diff_threshold=_int_env("BROWSER_DIFF_THRESHOLD", 40),
        browser_idle_close_seconds=_int_env("BROWSER_IDLE_CLOSE_SECONDS", 600),
        # D1: Browser output caps - env overrides
        browse_page_chars_cap=_int_env("BROWSE_PAGE_CHARS_CAP", 50_000),
        browse_batch_chars_cap=_int_env("BROWSE_BATCH_CHARS_CAP", 150_000),
        browse_max_links=_int_env("BROWSE_MAX_LINKS", 100),
        browse_eval_chars_cap=_int_env("BROWSE_EVAL_CHARS_CAP", 16_000),
        durable_history_enabled=_bool_env("DURABLE_HISTORY", True),
        history_dir=str(Path(project_root) / ".supporter" / "history"),
        replay_image_count=_int_env("REPLAY_IMAGE_COUNT", 2),
        replay_tool_summary_max_chars=_int_env("REPLAY_TOOL_SUMMARY_MAX_CHARS", 200),
        reconnect_attempts_max=_int_env("RECONNECT_ATTEMPTS_MAX", 5),
        reconnect_backoff_base=float(os.getenv("RECONNECT_BACKOFF_BASE", "0.5")),
        reconnect_backoff_cap=float(os.getenv("RECONNECT_BACKOFF_CAP", "8.0")),
        prewarm_safety_margin=float(os.getenv("PREWARM_SAFETY_MARGIN", "5.0")),
        idle_monitor_enabled=_bool_env(
            "IDLE_MONITOR_ENABLED",
            _bool_env("KEEPALIVE_ENABLED", True),
        ),
        empty_resume_policy=os.getenv("EMPTY_RESUME_POLICY", "trust").lower(),
        browser_trusted_hosts=os.getenv("BROWSER_TRUSTED_HOSTS", ""),
        browser_micro_behavior_rate=float(os.getenv("MICRO_BEHAVIOR_RATE", "0.06")),
        browser_promotion_threshold=_int_env("BROWSER_PROMOTION_THRESHOLD", 5),
        browser_auto_approve=_bool_env("BROWSER_AUTO_APPROVE", True),
    )


config = load_config()


def reload_config() -> AppConfig:
    """Reload config from env, mutating the existing module-global ``config``
    object in place so all importers that did ``from .config import config``
    automatically see the updated values without rebinding.
    """
    global config
    fresh = load_config()
    for f in dataclasses.fields(AppConfig):
        setattr(config, f.name, getattr(fresh, f.name))
    return config
