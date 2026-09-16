from fastapi import APIRouter, HTTPException, Request
import json

from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent.agent import FRIENDLY_ERROR, PortfolioAgent
from app.api.dependencies import get_agent, get_settings, session_store
from app.memory.session import ConversationSession
from app.models.requests import AgentResponse, ChatRequest, SessionResponse
from app.utils.ids import new_id
from app.utils.logging import logger
from app.utils.time import now_tz

router = APIRouter(tags=["chat"])
limiter = Limiter(key_func=get_remote_address)


def _resolve_session(conversation_id: str | None) -> ConversationSession:
    if conversation_id:
        session = session_store.touch(conversation_id)
        return session
    return session_store.create(new_id())


@router.post("/chat", response_model=AgentResponse)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def chat(request: Request, payload: ChatRequest) -> AgentResponse:
    started = now_tz()
    session = _resolve_session(payload.conversation_id)
    request_id = getattr(request.state, "request_id", "-")
    agent: PortfolioAgent = getattr(request.app.state, "agent", None) or get_agent()

    try:
        result = agent.reply(session.conversation_id, payload.message, payload.history)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Chat failed: %s",
            type(exc).__name__,
            extra={
                "request_id": request_id,
                "conversation_id": session.conversation_id,
                "endpoint": "/api/chat",
                "latency_ms": int((now_tz() - started).total_seconds() * 1000),
            },
        )
        raise HTTPException(status_code=502, detail=FRIENDLY_ERROR) from None

    result.conversation_id = session.conversation_id
    result.created_at = session.created_at
    result.updated_at = session.updated_at
    result.expires_at = session.expires_at
    result.timestamp = now_tz()
    logger.info(
        "Chat completed",
        extra={
            "request_id": request_id,
            "conversation_id": session.conversation_id,
            "endpoint": "/api/chat",
            "latency_ms": int((now_tz() - started).total_seconds() * 1000),
        },
    )
    return result


@router.post("/chat/stream")
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    session = _resolve_session(payload.conversation_id)
    agent: PortfolioAgent = getattr(request.app.state, "agent", None) or get_agent()

    def events():
        try:
            for token in agent.stream_text(session.conversation_id, payload.message, payload.history):
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Chat stream failed: %s",
                type(exc).__name__,
                extra={
                    "request_id": getattr(request.state, "request_id", "-"),
                    "conversation_id": session.conversation_id,
                    "endpoint": "/api/chat/stream",
                    "latency_ms": "-",
                },
            )
            yield f"data: {json.dumps({'type': 'error', 'message': FRIENDLY_ERROR})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Conversation-Id": session.conversation_id,
            "X-Expires-At": session.expires_at.isoformat(),
        },
    )


@router.post("/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    session = session_store.create(new_id())
    return SessionResponse(
        conversation_id=session.conversation_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        expires_at=session.expires_at,
        messages=[],
    )
