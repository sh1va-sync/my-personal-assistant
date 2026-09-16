from dataclasses import dataclass, field
from enum import Enum

from app.models.requests import ChatMessage, ResponseType
from app.utils.security import looks_like_injection


class Intent(str, Enum):
    GREETING = "greeting"
    PERSONAL = "personal"
    GENERAL = "general"
    TIME = "time"
    INJECTION = "injection"


GREETING_MARKERS = {
    "hi",
    "hey",
    "hello",
    "yo",
    "sup",
    "hiya",
    "good morning",
    "good evening",
    "what's up",
    "whats up",
}

TIME_MARKERS = (
    "what time",
    "current time",
    "what's the time",
    "whats the time",
    "date today",
    "today's date",
    "todays date",
    "what day is it",
    "timezone",
    "clock",
)

PERSONAL_MARKERS = (
    "shiva",
    "boss",
    "your boss",
    "about you",
    "who are you",
    "who is he",
    "his ",
    "he ",
    "project",
    "projects",
    "tech stack",
    "skills",
    "skill",
    "education",
    "college",
    "university",
    "experience",
    "resume",
    "cv",
    "hire",
    "contact",
    "email",
    "github",
    "linkedin",
    "certification",
    "achievement",
    "interest",
    "hobby",
    "goal",
    "portfolio",
    "metaconnect",
    "pneumo",
    "webrtc",
)


@dataclass
class AgentState:
    conversation_id: str
    user_message: str
    history: list[ChatMessage] = field(default_factory=list)
    intent: Intent = Intent.GENERAL
    retrieved_context: str = ""
    sources: list[str] = field(default_factory=list)
    tool_context: str = ""
    response_type: ResponseType = "text"


def classify_intent(message: str) -> Intent:
    lowered = message.lower().strip()
    if looks_like_injection(lowered):
        return Intent.INJECTION
    if lowered in GREETING_MARKERS or (len(lowered) <= 24 and any(lowered.startswith(item) for item in GREETING_MARKERS)):
        if not any(marker in lowered for marker in PERSONAL_MARKERS):
            return Intent.GREETING
    if any(marker in lowered for marker in TIME_MARKERS) or lowered in {"time", "date", "today"}:
        return Intent.TIME
    if any(marker in lowered for marker in PERSONAL_MARKERS):
        return Intent.PERSONAL
    return Intent.GENERAL


def classify_response_type(intent: Intent, sources: list[str], used_time_tool: bool) -> ResponseType:
    if intent == Intent.GREETING:
        return "greeting"
    if intent == Intent.INJECTION:
        return "text"
    if used_time_tool:
        return "general"
    joined = " ".join(sources).lower()
    if "project" in joined:
        return "project"
    if "skill" in joined:
        return "skill"
    if sources:
        return "personal_info"
    if intent == Intent.PERSONAL:
        return "personal_info"
    return "general"
