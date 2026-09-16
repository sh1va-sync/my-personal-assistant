from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    vector_db_path: str = "./data/vectorstore"
    knowledge_path: str = "./data/knowledge"

    frontend_url: str = "http://localhost:5173"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    session_ttl_hours: int = Field(default=48, ge=1, le=168)
    max_history_messages: int = Field(default=20, ge=2, le=50)
    max_message_length: int = Field(default=2000, ge=1, le=8000)
    retrieval_top_k: int = Field(default=4, ge=1, le=10)

    default_timezone: str = "Asia/Kolkata"
    environment: Literal["development", "production", "test"] = "development"
    port: int = 8000
    request_timeout_seconds: float = Field(default=30.0, ge=5.0, le=120.0)
    rate_limit_per_minute: int = Field(default=40, ge=5, le=300)

    persist_conversations: bool = False

    @field_validator("gemini_model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("GEMINI_MODEL cannot be empty")
        return cleaned

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origins(self) -> list[str]:
        origins = [item.strip() for item in self.allowed_origins.split(",") if item.strip()]
        if self.frontend_url.strip() and self.frontend_url.strip() not in origins:
            origins.append(self.frontend_url.strip())
        if self.is_production:
            origins = [origin for origin in origins if origin != "*"]
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
