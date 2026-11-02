"""
app/api/routes/query.py
═══════════════════════
Endpoint 1: POST /api/v1/query — Submit query and stream SSE
Endpoint 2: GET /api/v1/query/{query_id}/stream — SSE stream
"""
from __future__ import annotations

import uuid
from typing import Final

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.logging.logger import get_logger
from app.schemas.context import SharedContext
from app.streaming.subscriber import stream_query_events
from app.worker import run_query_task

router = APIRouter(prefix="/api/v1", tags=["Query"])
log = get_logger("api.query")

_MAX_QUERY_LENGTH: Final[int] = 4096
_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    stream: bool = Field(default=True, description="Whether to enable SSE streaming")


class QueryResponse(BaseModel):
    query_id: str
    status: str
    stream_url: str
    message: str


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a query to the multi-agent pipeline",
    description=(
        "Submits a user query to the Orqestra pipeline. "
        "Returns a query_id and SSE stream URL. "
        "Connect to the stream URL to receive real-time agent activity."
    ),
)
async def submit_query(request: QueryRequest) -> QueryResponse:
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EMPTY_QUERY",
                "message": "Query must not be empty",
            },
        )

    if len(request.query) > _MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "QUERY_TOO_LONG",
                "message": f"Query exceeds maximum length of {_MAX_QUERY_LENGTH} characters",
            },
        )

    # Generate the query_id immediately via SharedContext
    ctx = SharedContext(user_query=request.query)
    query_id = str(ctx.query_id)

    # Dispatch to the background Worker (Dramatiq)
    run_query_task.send(request.query, query_id)

    log.info("query_submitted", query_id=query_id)

    return QueryResponse(
        query_id=query_id,
        status="accepted",
        stream_url=f"/api/v1/query/{query_id}/stream",
        message="Query accepted. Connect to stream_url for real-time updates.",
    )


@router.get(
    "/query/{query_id}/stream",
    summary="Stream SSE events for a running query",
    description=(
        "Server-Sent Events stream for a query. "
        "Emits: agent_started, agent_completed, tool_call_started, "
        "tool_call_completed, budget_update, policy_violation, final_answer, "
        "pipeline_complete events."
    ),
)
async def stream_query(query_id: str) -> StreamingResponse:
    try:
        uuid.UUID(query_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_QUERY_ID",
                "message": f"'{query_id}' is not a valid UUID format",
            },
        )

    return StreamingResponse(
        stream_query_events(query_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )

