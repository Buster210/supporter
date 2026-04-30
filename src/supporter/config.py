import os
from pathlib import Path

from dotenv import load_dotenv

from .types import AppConfig

__all__ = ["AppConfig", "config"]

load_dotenv()

HTTP_RATE_LIMIT = 429
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_RETRY_ATTEMPTS = 2

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

THEME = {
    "background": "#121212",
    "bubble_bg": "#1e1e1e",
    "header_teal": "#00ffcc",
    "magenta": "#ff06b5",
    "green": "#00ff00",
    "blue": "#0080ff",
    "yellow": "#ffeb3b",
    "meta_gray": "#999999",
}

CRYSTAL_GRADIENT_STOPS: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (0, 255, 180),
    (0, 180, 255),
    (100, 200, 255),
]

TOOL_ARG_MAX_LEN = 40
TOOL_ARG_TRUNC_LEN = 37
MODAL_WIDTH_SCALE = 1.3
MODAL_MAX_WIDTH_PERCENT = 0.9
MODAL_PADDING = 6
BASH_MODAL_MAX_WIDTH = 80
SCROLL_STEP = 5
COLLAPSED_SUMMARY_LEN = 50
MARKDOWN_SYNTAX_MARKERS = [
    r"[*+-]\s",
    r"\d+\.\s",
    r"#+\s",
    r"\*\*.*?\*\*",
    r"\*.*?\*",
    r"`.*?`",
    r"\[.*?\]\(.*?\)",
    r">\s",
]

INTERNAL_BLACKLIST = [
    ".env",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
]

DEFAULT_MODEL = "gemma-4-31b-it"

RETRIABLE_CODES = {
    "1000",
    "1006",
    "1011",
    "1007",
    "1008",
    "429",
    "quota",
    "exhausted",
}

GOOGLE_5XX_ERRORS = {
    "InternalServerError",
    "ServiceUnavailable",
    "BadGateway",
    "GatewayTimeout",
    "APIError",
}

TRANSIENT_SIGNALS = {"unavailable", "overloaded", "internal error", "service level"}
HTTP_ERRORS_5XX = {HTTP_SERVICE_UNAVAILABLE, HTTP_INTERNAL_ERROR}
RATE_LIMIT_SIGNALS = {"quota", "too many requests", "429"}

DRAIN_TIMEOUT = 2.0
CONTEXT_TRIGGER_TOKENS = 100_000
CONTEXT_TARGET_TOKENS = 4_000

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an elite technical strategist and principal software architect. "
    "Your objective is to provide rigorous, high-fidelity, and "
    "architecturally sound guidance. Analyze complex problems through "
    "the lens of scalability, maintainability, and efficiency. Always "
    "anticipate edge cases and performance bottlenecks before "
    "formulating a response."
)


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
        live_thinking_level=os.getenv("GEMINI_LIVE_THINKING_LEVEL", "medium").lower(),
        retriable_codes=RETRIABLE_CODES,
        google_5xx_errors=GOOGLE_5XX_ERRORS,
        transient_signals=TRANSIENT_SIGNALS,
        http_errors_5xx=HTTP_ERRORS_5XX,
        rate_limit_signals=RATE_LIMIT_SIGNALS,
        drain_timeout=DRAIN_TIMEOUT,
        context_trigger_tokens=CONTEXT_TRIGGER_TOKENS,
        context_target_tokens=CONTEXT_TARGET_TOKENS,
        http_retry_attempts=int(os.getenv("HTTP_RETRY_ATTEMPTS", HTTP_RETRY_ATTEMPTS)),
    )


config = load_config()
