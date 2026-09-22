from __future__ import annotations

from collections.abc import Iterator
import time
from typing import Any

from app.config.settings import get_settings
from app.utils.logging import logger


class GeminiClient:
    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        self._types = types
        timeout_ms = int(settings.request_timeout_seconds * 1000)
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self._model = settings.gemini_model

    def generate(
        self,
        *,
        system_instruction: str,
        contents: list[Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": 0.7,
            "max_output_tokens": 800,
        }
        if tools:
            config_kwargs["tools"] = [
                self._types.Tool(
                    function_declarations=[
                        self._types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                        )
                        for tool in tools
                    ]
                )
            ]
        return self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._types.GenerateContentConfig(**config_kwargs),
            ),
            endpoint="generation",
        )

    def _with_retry(self, operation, *, endpoint: str) -> Any:
        attempts = get_settings().provider_retry_attempts
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                if attempt + 1 >= attempts or not _is_transient_provider_error(exc):
                    raise
                logger.warning(
                    "Transient Gemini failure endpoint=%s retry=%s category=%s",
                    endpoint,
                    attempt + 1,
                    _provider_error_category(exc),
                    extra={"request_id": "-", "conversation_id": "-", "endpoint": endpoint, "latency_ms": "-"},
                )
                time.sleep(2**attempt)
        raise RuntimeError("Gemini operation did not return a result")

    def stream(
        self,
        *,
        system_instruction: str,
        contents: list[Any],
    ) -> Iterator[str]:
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=800,
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            blobs = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
            if blobs:
                return "\n".join(blobs).strip()
        logger.info(
            "Empty model output",
            extra={"request_id": "-", "conversation_id": "-", "endpoint": "gemini", "latency_ms": "-"},
        )
        return ""


def _is_transient_provider_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in {408, 429, 500, 502, 503, 504} or isinstance(
        exc,
        TimeoutError,
    )


def _provider_error_category(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return {
        408: "timeout",
        429: "quota",
        500: "provider_error",
        502: "provider_error",
        503: "provider_unavailable",
        504: "timeout",
    }.get(status, "timeout" if isinstance(exc, TimeoutError) else "provider_error")
