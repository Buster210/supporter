import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    log_level: str
    provider: str
    gemini_api_keys: list[str]
    gemini_model: str
    gemini_fallback_model: str | None
    log_file: str
    default_system_instruction: str


def load_config() -> AppConfig:
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return AppConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_keys=keys,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL"),
        log_file=os.getenv("LOG_FILE", "app.log"),
        default_system_instruction=os.getenv(
            "DEFAULT_SYSTEM_INSTRUCTION",
            "You are a helpful assistant. Prioritize quality and clarity in every response.",
        ),
    )


config = load_config()
