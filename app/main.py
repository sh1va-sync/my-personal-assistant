import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.dependencies import get_agent, get_settings
from app.api.routes.chat import limiter, router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.utils.ids import new_id
from app.utils.logging import logger
from app.utils.time import now_tz

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

FRIENDLY_ERROR = "My brain just hit a tiny server-side speed bump. Try that again."


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.agent = get_agent()
    logger.info(
        "Portfolio agent ready model=%s env=%s",
        settings.gemini_model,
        settings.environment,
        extra={"request_id": "-", "conversation_id": "-", "endpoint": "startup", "latency_ms": "-"},
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Shiva's Personal AI Portfolio Agent",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.limiter = limiter
    application.add_middleware(SlowAPIMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or new_id()
        started = perf_counter()
        response = await call_next(request)
        latency_ms = int((perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request completed status=%s",
            response.status_code,
            extra={
                "request_id": request.state.request_id,
                "conversation_id": "-",
                "endpoint": request.url.path,
                "latency_ms": latency_ms,
            },
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "message": "That message didn't look usable. Keep it short and try again.",
                "response_type": "error",
                "sources": [],
                "timestamp": now_tz().isoformat(),
                "details": exc.errors(),
            },
        )

    @application.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(_: Request, __: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "message": "Whoa, slow down a second. Give me a beat, then ask again.",
                "response_type": "error",
                "timestamp": now_tz().isoformat(),
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled error: %s",
            type(exc).__name__,
            extra={
                "request_id": getattr(request.state, "request_id", "-"),
                "conversation_id": "-",
                "endpoint": request.url.path,
                "latency_ms": "-",
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "message": FRIENDLY_ERROR,
                "response_type": "error",
                "timestamp": now_tz().isoformat(),
            },
        )

    application.include_router(health_router, prefix="/api")
    application.include_router(chat_router, prefix="/api")
    application.include_router(knowledge_router, prefix="/api")
    return application


app = create_app()
