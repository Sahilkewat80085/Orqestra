"""
app/logging/middleware.py
═════════════════════════
FastAPI middleware for request-level structured logging.
Attaches a correlation ID to every request and logs:
  - method, path, status, latency_ms, correlation_id
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

import structlog

log = structlog.get_logger("http")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        # Bind correlation_id for the duration of this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        response: Response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            "http_request",
            event_type="HTTP_REQUEST",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            correlation_id=correlation_id,
        )

        response.headers["X-Correlation-ID"] = correlation_id
        return response
