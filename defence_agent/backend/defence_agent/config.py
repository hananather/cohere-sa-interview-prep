from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_chat_model: str = Field(default="command-a-03-2025", alias="COHERE_CHAT_MODEL")
    cohere_embed_model: str = Field(default="embed-v4.0", alias="COHERE_EMBED_MODEL")
    cohere_rerank_model: str = Field(default="rerank-v3.5", alias="COHERE_RERANK_MODEL")
    use_mock_cohere: bool = Field(default=True, alias="USE_MOCK_COHERE")
    cohere_embed_output_dimension: int = Field(default=1024, alias="COHERE_EMBED_OUTPUT_DIMENSION")

    database_url: str = Field(default="sqlite:///./defence_agent/data/defence_agent.db", alias="DATABASE_URL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="defence_agent_chunks", alias="QDRANT_COLLECTION")
    generated_corpus_dir: Path = Field(default=Path("./defence_agent/data/corpus"), alias="GENERATED_CORPUS_DIR")
    trace_jsonl_path: Path = Field(default=Path("./defence_agent/data/traces.jsonl"), alias="TRACE_JSONL_PATH")

    max_agent_steps: int = Field(default=6, alias="MAX_AGENT_STEPS")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")
    vector_size: int = Field(default=64, alias="VECTOR_SIZE")
    cohere_timeout_seconds: int = Field(default=20, alias="COHERE_TIMEOUT_SECONDS")

    @field_validator("generated_corpus_dir", "trace_jsonl_path", mode="before")
    @classmethod
    def expand_paths(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @model_validator(mode="after")
    def validate_real_cohere(self) -> "Settings":
        if not self.use_mock_cohere and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when USE_MOCK_COHERE=false")
        return self

    @property
    def data_dir(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url.replace("sqlite:///", "", 1))
            return db_path.parent
        return Path("./defence_agent/data")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.generated_corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
