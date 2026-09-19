from fastapi import APIRouter
from app.api.dependencies import get_vector_store
from app.config.settings import get_settings

from app.models.requests import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        store = get_vector_store()
        document_count = store.count()
        ready = document_count > 0
        return HealthResponse(
            status="ok" if ready else "degraded",
            knowledge_ready=ready,
            document_count=document_count,
        )
    except Exception:
        if get_settings().is_production:
            return HealthResponse(status="degraded", knowledge_ready=False, document_count=0)
        return HealthResponse(status="ok", knowledge_ready=False, document_count=0)
