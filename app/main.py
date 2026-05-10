"""
app/main.py
════════════
FastAPI application factory for Orqestra.

Mounts all routers, configures middleware, and manages lifespan events
for DB initialization, Redis connection, and ChromaDB setup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import evals, prompts, query, trace
from app.config import settings
from app.database.session import init_db
from app.logging.logger import configure_logging, get_logger
from app.logging.middleware import StructuredLoggingMiddleware
from app.streaming.publisher import publisher

log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup → serve → shutdown.
    All initialization happens here — never in module globals.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging(settings.api_log_level)
    log.info("orqestra_starting", version=settings.app_version, env=settings.app_env)

    # Initialize PostgreSQL tables
    await init_db()
    log.info("database_initialized")

    # Connect Redis publisher
    await publisher.connect()
    log.info("redis_connected", url=settings.redis_url)

    log.info("orqestra_ready", host=settings.api_host, port=settings.api_port)

    yield  # ← Application serves requests here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await publisher.disconnect()
    log.info("orqestra_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Orqestra",
        description=(
            "Production-grade Multi-Agent LLM Orchestration & Evaluation Platform. "
            "Dynamic routing, typed shared context, streaming SSE, and full eval harness."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Structured logging middleware ─────────────────────────────────────────
    app.add_middleware(StructuredLoggingMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(query.router)
    app.include_router(trace.router)
    app.include_router(evals.router)
    app.include_router(prompts.router)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "healthy", "version": settings.app_version}

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred.",
                    "detail": str(exc) if settings.app_env == "development" else None,
                }
            },
        )

    return app


app = create_app()
