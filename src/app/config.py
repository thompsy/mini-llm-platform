from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / a .env file.

    All variables are prefixed with ``APP_`` (e.g. ``APP_MODEL``).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Ollama / model
    ollama_base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    request_timeout: float = 60.0
    default_temperature: float = 0.7

    # RAG
    embed_model: str = "nomic-embed-text"
    vector_store_dir: str = ".chroma"
    rag_top_k: int = 4
    rag_min_score: float = 0.5
    chunk_size: int = 200
    chunk_overlap: int = 40

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (so the .env is read once)."""
    return Settings()
