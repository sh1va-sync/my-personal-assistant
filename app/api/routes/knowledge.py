from fastapi import APIRouter

from app.api.dependencies import get_vector_store
from app.models.requests import KnowledgeStatusResponse
from app.utils.logging import logger

router = APIRouter(tags=["knowledge"])


@router.get("/knowledge", response_model=KnowledgeStatusResponse)
async def knowledge_status() -> KnowledgeStatusResponse:
    try:
        store = get_vector_store()
        sources = store.sources()
        return KnowledgeStatusResponse(ready=store.count() > 0, document_count=store.count(), sources=sources)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Knowledge status failed: %s",
            type(exc).__name__,
            extra={"request_id": "-", "conversation_id": "-", "endpoint": "/api/knowledge", "latency_ms": "-"},
        )
        return KnowledgeStatusResponse(ready=False, document_count=0, sources=[])
