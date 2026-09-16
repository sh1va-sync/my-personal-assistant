import re

INJECTION_PATTERNS = (
    r"ignore (all|any|previous|your) (instructions|prompts)",
    r"reveal (your )?(system prompt|hidden instructions)",
    r"show me (your )?(system prompt|instructions)",
    r"print (the )?(system prompt|api key)",
    r"pretend (you are|shiva is) someone else",
    r"override (the )?(system|safety)",
    r"jailbreak",
)


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)
