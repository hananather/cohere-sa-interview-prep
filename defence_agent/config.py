"""Configuration for the ADK-first Defence Agent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Small settings object for retrieval, Cohere, and local storage."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cohere_api_key: str = Field(alias="COHERE_API_KEY")
    cohere_chat_model: str = Field(default="command-a-03-2025", alias="COHERE_CHAT_MODEL")
    cohere_embed_model: str = Field(default="embed-v4.0", alias="COHERE_EMBED_MODEL")
    cohere_rerank_model: str = Field(default="rerank-v4.0-pro", alias="COHERE_RERANK_MODEL")
    cohere_embed_output_dimension: int = Field(default=1536, alias="COHERE_EMBED_OUTPUT_DIMENSION")
    cohere_embed_page_batch_size: int = Field(default=1, alias="COHERE_EMBED_PAGE_BATCH_SIZE")
    cohere_embed_page_batch_max_bytes: int = Field(default=6_000_000, alias="COHERE_EMBED_PAGE_BATCH_MAX_BYTES")
    cohere_requests_per_minute: float = Field(default=20.0, alias="COHERE_REQUESTS_PER_MINUTE")
    cohere_max_retries: int = Field(default=6, alias="COHERE_MAX_RETRIES")
    cohere_retry_max_wait_seconds: float = Field(default=90.0, alias="COHERE_RETRY_MAX_WAIT_SECONDS")
    cohere_timeout_seconds: float = Field(default=60.0, alias="COHERE_TIMEOUT_SECONDS")
    cohere_thinking_token_budget: int | None = Field(default=None, alias="COHERE_THINKING_TOKEN_BUDGET")
    tool_cache_enabled: bool = Field(default=True, alias="DEFENCE_AGENT_TOOL_CACHE_ENABLED")
    tool_cache_max_entries: int = Field(default=24, alias="DEFENCE_AGENT_TOOL_CACHE_MAX_ENTRIES")

    data_dir: Path = Field(default=Path("./defence_agent/data"), alias="DEFENCE_AGENT_DATA_DIR")
    corpus_dir: Path = Field(default=Path("./defence_agent/data/corpus"), alias="DEFENCE_AGENT_CORPUS_DIR")

    @field_validator("data_dir", "corpus_dir", mode="before")
    @classmethod
    def expand_paths(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("cohere_thinking_token_budget", mode="before")
    @classmethod
    def blank_thinking_budget_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("cohere_api_key")
    @classmethod
    def require_api_key(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("COHERE_API_KEY is required for all Defence Agent runs")
        return value

    @model_validator(mode="after")
    def validate_live_settings(self) -> "Settings":
        if self.cohere_embed_output_dimension <= 0:
            raise ValueError("COHERE_EMBED_OUTPUT_DIMENSION must be positive")
        if self.cohere_embed_page_batch_size < 1:
            raise ValueError("COHERE_EMBED_PAGE_BATCH_SIZE must be at least 1")
        if self.cohere_embed_page_batch_max_bytes < 1:
            raise ValueError("COHERE_EMBED_PAGE_BATCH_MAX_BYTES must be positive")
        if self.cohere_requests_per_minute <= 0:
            raise ValueError("COHERE_REQUESTS_PER_MINUTE must be positive")
        if self.cohere_max_retries < 1:
            raise ValueError("COHERE_MAX_RETRIES must be at least 1")
        if self.cohere_retry_max_wait_seconds <= 0:
            raise ValueError("COHERE_RETRY_MAX_WAIT_SECONDS must be positive")
        if self.cohere_timeout_seconds <= 0:
            raise ValueError("COHERE_TIMEOUT_SECONDS must be positive")
        if self.cohere_thinking_token_budget is not None and self.cohere_thinking_token_budget <= 0:
            raise ValueError("COHERE_THINKING_TOKEN_BUDGET must be positive when set")
        if self.tool_cache_max_entries < 1:
            raise ValueError("DEFENCE_AGENT_TOOL_CACHE_MAX_ENTRIES must be at least 1")
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    return settings
