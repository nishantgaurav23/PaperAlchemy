"""FastAPI application factory with async lifespan management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, get_settings
from src.db import init_database
from src.middlewares import RequestLoggingMiddleware, register_error_handlers
from src.routers.analysis import router as analysis_router
from src.routers.ask import router as ask_router
from src.routers.chat import router as chat_router
from src.routers.collections import router as collections_router
from src.routers.ingest import router as ingest_router
from src.routers.papers import router as papers_router
from src.routers.ping import router as ping_router
from src.routers.search import router as search_router
from src.routers.upload import router as upload_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = getattr(app, "_settings_override", None) or get_settings()
    logger.info("Starting %s v%s (debug=%s)", settings.app.title, settings.app.version, settings.app.debug)
    db = init_database(database_url=settings.postgres.url, echo=settings.app.debug)
    logger.info("Database engine created (pool_size=5)")

    # Initialize Redis cache + conversation memory (best-effort — None if Redis unavailable)
    await _init_redis_services(settings)

    yield
    await db.close()
    logger.info("Shutting down %s", settings.app.title)


async def _init_redis_services(settings: Settings) -> None:
    """Try to connect Redis for caching and conversation memory. Non-fatal if unavailable."""
    from src.dependency import set_cache_client, set_conversation_memory

    try:
        from src.services.cache.factory import make_cache_client

        cache = await make_cache_client()
        set_cache_client(cache)
        if cache:
            logger.info("Redis cache connected")
        else:
            logger.warning("Redis cache unavailable — caching disabled")
    except Exception as exc:
        logger.warning("Redis cache unavailable (%s) — caching disabled", exc)
        set_cache_client(None)

    try:
        from src.services.chat.memory import make_conversation_memory

        memory = await make_conversation_memory()
        set_conversation_memory(memory)
        if memory:
            logger.info("Conversation memory connected")
        else:
            logger.warning("Conversation memory unavailable — memory disabled")
    except Exception as exc:
        logger.warning("Conversation memory unavailable (%s) — memory disabled", exc)
        set_conversation_memory(None)


def create_app(settings_override: Settings | None = None) -> FastAPI:
    settings = settings_override or get_settings()

    app = FastAPI(
        title=settings.app.title,
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
    )
    if settings_override:
        app._settings_override = settings_override  # type: ignore[attr-defined]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(ping_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(ask_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(upload_router, prefix="/api/v1")
    app.include_router(papers_router, prefix="/api/v1")
    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(collections_router, prefix="/api/v1")

    return app


app = create_app()
