import os
from pathlib import Path

from dotenv import load_dotenv

from .prompts import (
    DEFAULT_SYSTEM_INSTRUCTION,
    DELEGATE_AGENT_ROSTER,
    DELEGATE_DEFAULT_PERSONA,
)
from .types import AppConfig

__all__ = ["AppConfig", "config"]

load_dotenv()

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

DEFAULT_MODEL = "gemma-4-31b-it"

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
DELEGATE_MAX_OUTPUT_CHARS = 10000
DELEGATE_ALLOWED_TOOLS = {
    "read_file",
    "write_file",
    "execute_bash",
    "google_search",
}
DELEGATE_MAX_RETRIES = 2

DELEGATE_HEARTBEAT_INTERVAL = 30
DELEGATE_ANOMALY_THRESHOLD = 0.8
DELEGATE_JOB_ID_LEN = 8
DELEGATE_RETRY_BACKOFF = [1.0, 3.0]


def _get_project_root() -> str:
    current = Path(__file__).resolve().parent
    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return str(parent)
    return os.getcwd()


def load_config() -> AppConfig:
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    project_root = _get_project_root()

    return AppConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_keys=keys,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_live_model=os.getenv(
            "GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview"
        ),
        gemini_live_fallback_model=os.getenv(
            "GEMINI_LIVE_FALLBACK_MODEL",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL"),
        log_file=os.getenv("LOG_FILE", "app.log"),
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
        http_retry_attempts=int(
            os.getenv("HTTP_RETRY_ATTEMPTS", str(HTTP_RETRY_ATTEMPTS))
        ),
        delegate_max_hard_cap=DELEGATE_MAX_HARD_CAP,
        delegate_default_parallel=DELEGATE_DEFAULT_PARALLEL,
        delegate_default_timeout=DELEGATE_DEFAULT_TIMEOUT,
        delegate_max_timeout=DELEGATE_MAX_TIMEOUT,
        delegate_max_tasks=DELEGATE_MAX_TASKS,
        delegate_max_output_chars=DELEGATE_MAX_OUTPUT_CHARS,
        delegate_allowed_tools=DELEGATE_ALLOWED_TOOLS,
        delegate_default_persona=DELEGATE_DEFAULT_PERSONA,
        delegate_agent_roster=DELEGATE_AGENT_ROSTER,
        delegate_max_retries=int(
            os.getenv("DELEGATE_MAX_RETRIES", str(DELEGATE_MAX_RETRIES))
        ),
        log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "5000000")),
        log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "3")),
        history_max_turns=int(os.getenv("HISTORY_MAX_TURNS", "200")),
    )


config = load_config()
