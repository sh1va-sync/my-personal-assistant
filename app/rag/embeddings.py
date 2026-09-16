from __future__ import annotations

import hashlib
from typing import Protocol

from app.config.settings import get_settings
from app.utils.logging import logger


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashEmbeddings:
    """Deterministic local embeddings for tests and offline ingest dry-runs."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        seed = digest
        while len(values) < self.dimensions:
            seed = hashlib.sha256(seed).digest()
            values.extend(byte / 255.0 for byte in seed)
        vector = values[: self.dimensions]
        norm = sum(item * item for item in vector) ** 0.5 or 1.0
        return [item / norm for item in vector]


class GeminiEmbeddings:
    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._types = types
        self._model = settings.gemini_embedding_model.removeprefix("models/")
        if self._model == "text-embedding-004":
            raise RuntimeError(
                "GEMINI_EMBEDDING_MODEL=text-embedding-004 is no longer supported; "
                "use gemini-embedding-001"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=self._types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            for embedding in response.embeddings:
                values = embedding.values
                if not values:
                    raise RuntimeError("Gemini returned an empty document embedding")
                vectors.append(list(values))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini returned an empty query embedding")
        return list(response.embeddings[0].values)


def build_embeddings(force_hash: bool = False) -> Embeddings:
    settings = get_settings()
    if force_hash or settings.environment == "test" or not settings.gemini_api_key:
        logger.info("Using hash embeddings", extra={"endpoint": "embeddings", "request_id": "-", "conversation_id": "-", "latency_ms": "-"})
        return HashEmbeddings()
    return GeminiEmbeddings()
