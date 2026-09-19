from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config.settings import get_settings

ResponseType = Literal[
    "text",
    "greeting",
    "personal_info",
    "project",
    "skill",
    "general",
    "error",
]

MessageRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty")
        settings = get_settings()
        if len(cleaned) > settings.max_message_length:
            raise ValueError(f"Message exceeds {settings.max_message_length} characters")
        return cleaned


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=80)
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        settings = get_settings()
        if len(cleaned) > settings.max_message_length:
            raise ValueError(f"Message exceeds {settings.max_message_length} characters")
        return cleaned

    @field_validator("history")
    @classmethod
    def cap_history(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        settings = get_settings()
        if len(value) > settings.max_history_messages:
            return value[-settings.max_history_messages :]
        return value


class AgentResponse(BaseModel):
    conversation_id: str
    message: str
    response_type: ResponseType
    sources: list[str] = Field(default_factory=list)
    timestamp: datetime
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    knowledge_ready: bool = True
    document_count: int = 0


class SessionResponse(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    messages: list[ChatMessage] = Field(default_factory=list)


class KnowledgeStatusResponse(BaseModel):
    ready: bool
    document_count: int
    sources: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    conversation_id: str | None = None
    message: str
    response_type: Literal["error"] = "error"
    sources: list[str] = Field(default_factory=list)
    timestamp: datetime
