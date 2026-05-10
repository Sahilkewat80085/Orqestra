"""
app/api/routes/trace.py
════════════════════════
Endpoint: GET /api/v1/trace/{query_id}
Returns the full execution trace for any completed query.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.services.query_service import QueryService

router = APIRouter(prefix="/api/v1", tags=["Traces"])


@router.get(
    "/trace/{query_id}",
    summary="Retrieve full execution trace",
    description=(
        "Returns the complete execution trace for a query including: "
        "routing decisions, all agent outputs, tool calls with latencies, "
        "policy violations, token usage, and provenance metadata. "
        "Supports full replay and diff between runs."
    ),
)
async def get_trace(query_id: str, db: AsyncSession = Depends(get_db)):
    service = QueryService(db)
    trace = await service.get_trace(query_id)

    if not trace:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TRACE_NOT_FOUND",
                "message": f"No execution trace found for query_id: {query_id}",
                "detail": "The query may still be running or the ID is invalid.",
            },
        )

    return trace
