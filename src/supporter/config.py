import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HTTP_NOT_FOUND = 404
HTTP_RATE_LIMIT = 429
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

DEFAULT_MODEL = "supporter-gemini"
DEFAULT_AGENT_ROLE = "Analyzing"

RESEARCHER_ROLE = "Senior Research Analyst"
WRITER_ROLE = "Technical Content Strategist"


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
        gemini_model=os.getenv("GEMINI_MODEL", "gemma-4-31b-it"),
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
            "DEFAULT_SYSTEM_INSTRUCTION",
            (
                "You are a helpful assistant. "
                "Prioritize quality and clarity in every response."
            ),
        ),
        allowed_directories=[project_root],
        require_write_confirmation=os.getenv(
            "REQUIRE_WRITE_CONFIRMATION", "true"
        ).lower()
        == "true",
    )


config = load_config()
