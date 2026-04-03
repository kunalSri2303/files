from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Text Summarizer API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM — Gemini
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.3
    LLM_PROVIDER: str = "gemini"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # seconds (1 hour)
    CACHE_MAX_SIZE: int = 500

    # Input constraints
    MAX_TEXT_LENGTH: int = 50_000
    MIN_TEXT_LENGTH: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()