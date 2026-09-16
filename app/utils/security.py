import re

INJECTION_PATTERNS = (
    r"ignore (all|any|previous|your) (instructions|prompts)",
    r"disregard (all|any|previous|your) (instructions|prompts)",
    r"reveal (your )?(system prompt|hidden instructions)",
    r"show me (your )?(system prompt|instructions)",
    r"print (the )?(system prompt|api key)",
    r"(system|developer|hidden) (prompt|instructions).{0,20}(leak|reveal|show|print)",
    r"pretend (you are|shiva is) someone else",
    r"override (the )?(system|safety)",
    r"jailbreak",
    r"enter (developer|admin|debug) mode",
    r"follow these instructions instead",
)


def looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def safe_untrusted_text(text: str, replacement: str = "[untrusted content omitted]") -> str:
    return replacement if looks_like_injection(text) else text
