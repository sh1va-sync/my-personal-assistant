from functools import lru_cache

from app.agent.agent import PortfolioAgent
from app.config.settings import Settings, get_settings
from app.memory.session import InMemorySessionStore
from app.rag.retriever import KnowledgeRetriever
from app.rag.vectorstore import ChromaVectorStore
from app.utils.logging import logger

session_store = InMemorySessionStore()


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@lru_cache
def get_retriever() -> KnowledgeRetriever:
    try:
        store = get_vector_store()
        return KnowledgeRetriever(store)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Vector store unavailable: %s",
            type(exc).__name__,
            extra={"request_id": "-", "conversation_id": "-", "endpoint": "rag", "latency_ms": "-"},
        )
        raise


@lru_cache
def get_agent() -> PortfolioAgent:
    retriever: KnowledgeRetriever | None
    try:
        retriever = get_retriever()
    except Exception:  # noqa: BLE001
        if get_settings().is_production:
            raise
        retriever = None
    return PortfolioAgent(retriever=retriever)


def reset_runtime_caches() -> None:
    get_vector_store.cache_clear()
    get_retriever.cache_clear()
    get_agent.cache_clear()
    get_settings.cache_clear()


def current_settings() -> Settings:
    return get_settings()
