from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock

from app.config.settings import get_settings
from app.models.requests import ChatMessage
from app.utils.time import now_tz


@dataclass
class ConversationSession:
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)

    def is_expired(self, at: datetime | None = None) -> bool:
        moment = at or now_tz()
        return moment >= self.expires_at

    def as_public_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "messages": [message.model_dump() for message in self.messages],
        }


class InMemorySessionStore:
    """Short-lived session metadata. Visitor transcripts stay in the browser."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = Lock()

    def _ttl(self) -> timedelta:
        return timedelta(hours=get_settings().session_ttl_hours)

    def create(self, conversation_id: str) -> ConversationSession:
        now = now_tz()
        session = ConversationSession(
            conversation_id=conversation_id,
            created_at=now,
            updated_at=now,
            expires_at=now + self._ttl(),
        )
        with self._lock:
            self._sessions[conversation_id] = session
        return session

    def get(self, conversation_id: str) -> ConversationSession | None:
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return None
            if session.is_expired():
                self._sessions.pop(conversation_id, None)
                return None
            return session

    def touch(self, conversation_id: str) -> ConversationSession:
        existing = self.get(conversation_id)
        if existing is None:
            return self.create(conversation_id)
        existing.updated_at = now_tz()
        with self._lock:
            self._sessions[conversation_id] = existing
        return existing

    def purge_expired(self) -> int:
        now = now_tz()
        removed = 0
        with self._lock:
            expired = [key for key, session in self._sessions.items() if session.is_expired(now)]
            for key in expired:
                self._sessions.pop(key, None)
                removed += 1
        return removed
