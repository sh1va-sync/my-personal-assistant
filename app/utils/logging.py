import logging
import sys
from typing import Any


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ("request_id", "conversation_id", "endpoint", "latency_ms"):
            if not hasattr(record, field):
                setattr(record, field, "-")
        return True


class SafeExtraLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = kwargs.get("extra") or {}
        extra.pop("api_key", None)
        extra.pop("gemini_api_key", None)
        extra.pop("authorization", None)
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging() -> logging.LoggerAdapter:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s request_id=%(request_id)s conversation_id=%(conversation_id)s "
        "endpoint=%(endpoint)s latency_ms=%(latency_ms)s %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())

    root = logging.getLogger("portfolio_agent")
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return SafeExtraLogger(root, {})


logger = configure_logging()
