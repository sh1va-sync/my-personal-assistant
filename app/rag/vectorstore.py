from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.api.models.Collection import Collection

from app.config.settings import get_settings
from app.rag.embeddings import Embeddings, build_embeddings


@dataclass
class RetrievedChunk:
    text: str
    source: str
    category: str
    visibility: str = "public"
    distance: float | None = None


class VectorStore(Protocol):
    def reset(self) -> None:
        ...

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        ...

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        ...

    def count(self) -> int:
        ...

    def all_chunks(self) -> list[RetrievedChunk]:
        ...

    def sources(self) -> list[str]:
        ...


class ChromaVectorStore:
    collection_name = "shiva_knowledge"

    def __init__(self, embeddings: Embeddings | None = None, persist_path: str | None = None) -> None:
        settings = get_settings()
        self._embeddings = embeddings or build_embeddings()
        if settings.environment == "test":
            self._client = chromadb.Client()
        else:
            path = persist_path or settings.vector_db_path
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=path)
        self._collection = self._get_or_create()

    def _get_or_create(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._get_or_create()

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        embeddings = self._embeddings.embed_documents(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [self._embeddings.embed_query(query)],
            "n_results": min(k, max(self.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if category:
            query_kwargs["where"] = {"category": category}
        result = self._collection.query(**query_kwargs)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            if not text:
                continue
            meta = metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    category=str(meta.get("category", "general")),
                    visibility=str(meta.get("visibility", "public")),
                    distance=float(distance) if distance is not None else None,
                )
            )
        return chunks

    def count(self) -> int:
        return int(self._collection.count())

    def all_chunks(self) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[RetrievedChunk] = []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for text, metadata in zip(documents, metadatas, strict=False):
            if not text:
                continue
            meta = metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    category=str(meta.get("category", "general")),
                    visibility=str(meta.get("visibility", "public")),
                )
            )
        return chunks

    def sources(self) -> list[str]:
        names = {chunk.source for chunk in self.all_chunks()}
        return sorted(names)
