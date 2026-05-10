"""
app/services/query_service.py
═════════════════════════════
Orchestrates the full query lifecycle:
  query → SharedContext → LangGraph → persist trace → publish SSE events
"""
from __future__ import annotations

import time
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories.trace_repository import TraceRepository
from app.logging.logger import get_logger
from app.orchestrator.graph import graph
from app.schemas.context import ExecutionStatus, SharedContext
from app.streaming.events import final_answer, pipeline_complete, pipeline_error
from app.streaming.publisher import publisher

log = get_logger("services.query")


class QueryService:
    def __init__(self, db: AsyncSession):
        self._db = db
        self._repo = TraceRepository(db)

    async def run_query(self, query: str, query_id: str | None = None) -> SharedContext:
        """
        Execute the full multi-agent pipeline for a query.
        Persists trace and publishes SSE events.
        """
        context = SharedContext(user_query=query)
        if query_id:
            context.query_id = UUID(query_id)
        
        query_id = str(context.query_id)
        start = time.perf_counter()

        log.info("query_started", query_id=query_id, query=query[:100])

        # Persist initial trace record
        await self._repo.create(context)
        await self._db.commit()

        try:
            # Run LangGraph pipeline
            initial_state = {"context": context.model_dump(mode="json")}
            final_state = await graph.ainvoke(initial_state)
            final_context = SharedContext.model_validate(final_state["context"])

            # Publish final answer to SSE
            confidence = 0.0
            if final_context.agent_outputs.get("synthesis"):
                confidence = final_context.agent_outputs["synthesis"].confidence or 0.0

            await publisher.publish(
                final_answer(query_id, final_context.final_answer or "", confidence)
            )

        except Exception as e:
            log.error("pipeline_failed", query_id=query_id, error=str(e))
            final_context = context.model_copy(update={"status": ExecutionStatus.FAILED})
            await publisher.publish(
                pipeline_error(query_id, str(e))
            )

        total_ms = (time.perf_counter() - start) * 1000
        await publisher.publish(pipeline_complete(query_id, total_ms))

        # Update persisted trace with final state
        await self._repo.update(final_context)
        await self._db.commit()

        log.info("query_completed", query_id=query_id, latency_ms=round(total_ms, 2))
        return final_context

    async def get_trace(self, query_id: str) -> dict | None:
        trace = await self._repo.get_by_query_id(query_id)
        if not trace:
            return None
        return {
            "query_id": trace.query_id,
            "user_query": trace.user_query,
            "status": trace.status,
            "final_answer": trace.final_answer,
            "routing_decision": trace.routing_decision,
            "agent_outputs": trace.agent_outputs,
            "tool_calls": trace.tool_calls,
            "policy_violations": trace.policy_violations,
            "total_tokens_used": trace.total_tokens_used,
            "total_latency_ms": trace.total_latency_ms,
            "retry_count": trace.retry_count,
            "created_at": trace.created_at.isoformat() if trace.created_at else None,
            "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
        }


