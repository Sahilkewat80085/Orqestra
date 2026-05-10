"""
app/api/routes/query.py
═══════════════════════
Endpoint 1: POST /api/v1/query — Submit query and stream SSE
Endpoint 2: GET /api/v1/query/{query_id}/stream — SSE stream
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.logging.logger import get_logger
from app.services.query_service import QueryService
from app.streaming.subscriber import stream_query_events

router = APIRouter(prefix="/api/v1", tags=["Query"])
log = get_logger("api.query")


class QueryRequest(BaseModel):
    query: str
    stream: bool = True


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
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EMPTY_QUERY",
                "message": "Query must not be empty",
            },
        )

    if len(request.query) > 4096:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "QUERY_TOO_LONG",
                "message": "Query exceeds maximum length of 4096 characters",
            },
        )

    service = QueryService(db)

    # Run pipeline in background — SSE stream delivers results
    background_tasks.add_task(service.run_query, request.query)

    # We need the query_id before the pipeline runs
    # Generate it deterministically from the service
    from app.schemas.context import SharedContext
    ctx = SharedContext(user_query=request.query)
    query_id = str(ctx.query_id)

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
async def stream_query(query_id: str):
    return StreamingResponse(
        stream_query_events(query_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
