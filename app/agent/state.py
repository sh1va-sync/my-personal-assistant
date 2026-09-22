from dataclasses import dataclass, field
from enum import Enum

from app.models.requests import ChatMessage, ResponseType
from app.utils.security import looks_like_injection


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    PERSONAL = "personal"
    GENERAL = "general"
    TIME = "time"
    INJECTION = "injection"
    OUT_OF_SCOPE = "out_of_scope"


CONTEXTUAL_MARKERS = (
    "it",
    "that",
    "this",
    "they",
    "them",
    "he",
    "him",
    "she",
    "his",
    "her",
    "more",
    "why",
    "how so",
    "what about",
    "and ",
    "also ",
    "what do you mean",
    "tell me more",
    "okay ",
    "ok ",
    "sure ",
    "fair ",
)


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

FAREWELL_MARKERS = {
    "bye",
    "goodbye",
    "good bye",
    "see ya",
    "see you",
    "see you later",
    "catch you later",
    "talk later",
    "gotta go",
    "i gotta go",
    "i have to go",
    "good night",
    "goodnight",
    "signing off",
    "i'm leaving",
    "im leaving",
    "i'm off",
    "im off",
    "peace out",
    "later",
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
    "family",
    "friend",
    "personality",
    "relationship",
    "love",
    "free time",
    "routine",
    "likes",
    "dislikes",
)

SUPPORTED_MARKERS = (
    "what can you do",
    "how do you work",
    "your capabilities",
    "this portfolio",
    "this website",
)

CASUAL_MARKERS = {
    "ok",
    "okay",
    "k",
    "kk",
    "got it",
    "i get it",
    "understood",
    "thanks",
    "thank you",
    "cool",
    "nice",
    "wow",
    "really",
    "what",
    "wt",
    "huh",
    "why",
    "i don't understand",
    "i dont understand",
    "i am confused",
    "im confused",
    "confused",
}

WORK_MARKERS = ("project", "projects", "skill", "skills", "experience", "resume", "cv", "achievement", "certification", "work")
ABOUT_MARKERS = ("about", "education", "college", "university", "interest", "hobby", "goal", "family", "friend", "personality", "love", "routine", "free time")
CONTACT_MARKERS = ("contact", "email", "github", "linkedin", "hire", "reach")


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
    if lowered in FAREWELL_MARKERS or (
        len(lowered) <= 48
        and any(
            lowered.startswith(f"{marker} ") or lowered.endswith(f" {marker}")
            for marker in FAREWELL_MARKERS
        )
    ):
        return Intent.FAREWELL
    if lowered in GREETING_MARKERS or (len(lowered) <= 24 and any(lowered.startswith(item) for item in GREETING_MARKERS)):
        if not any(marker in lowered for marker in PERSONAL_MARKERS):
            return Intent.GREETING
    if any(marker in lowered for marker in TIME_MARKERS) or lowered in {"time", "date", "today"}:
        return Intent.TIME
    if any(marker in lowered for marker in SUPPORTED_MARKERS):
        return Intent.GENERAL
    if lowered in CASUAL_MARKERS:
        return Intent.GENERAL
    if any(marker in lowered for marker in PERSONAL_MARKERS):
        return Intent.PERSONAL
    return Intent.OUT_OF_SCOPE


def is_contextual_follow_up(message: str, history: list[ChatMessage]) -> bool:
    if not history:
        return False
    lowered = message.lower().strip()
    if lowered in CASUAL_MARKERS:
        return True
    if len(lowered.split()) > 14:
        return False
    if any(
        word in lowered.split()
        for word in {"it", "that", "this", "they", "them", "he", "him", "she", "her"}
    ):
        return True
    return any(
        lowered == marker.strip() or lowered.startswith(marker)
        for marker in CONTEXTUAL_MARKERS
    )


def infer_page_topic(text: str) -> str | None:
    lowered = text.lower()
    if any(marker in lowered for marker in CONTACT_MARKERS):
        return "contact"
    if any(marker in lowered for marker in WORK_MARKERS):
        return "work"
    if any(marker in lowered for marker in ABOUT_MARKERS):
        return "about"
    return None


def topic_count(history: list[ChatMessage], message: str, topic: str) -> int:
    entries = [item.content for item in history if item.role == "user"] + [message]
    markers = {
        "work": WORK_MARKERS,
        "about": ABOUT_MARKERS,
        "contact": CONTACT_MARKERS,
    }[topic]
    return sum(1 for entry in entries if any(marker in entry.lower() for marker in markers))


def classify_response_type(intent: Intent, sources: list[str], used_time_tool: bool) -> ResponseType:
    if intent in {Intent.GREETING, Intent.FAREWELL}:
        return "greeting"
    if intent in {Intent.INJECTION, Intent.OUT_OF_SCOPE}:
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
