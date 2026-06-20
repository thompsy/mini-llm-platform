from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / a .env file.

    All variables are prefixed with ``APP_`` (e.g. ``APP_MODEL``).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    ollama_base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    request_timeout: float = 60.0
    default_temperature: float = 0.7


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (so the .env is read once)."""
    return Settings()
