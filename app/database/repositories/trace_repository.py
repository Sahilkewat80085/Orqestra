"""
app/database/repositories/trace_repository.py
══════════════════════════════════════════════
Repository for ExecutionTrace — no raw SQL in application code.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ExecutionTrace, ToolCallLog
from app.schemas.context import SharedContext


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class TraceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, context: SharedContext) -> ExecutionTrace:
        """Persist a new execution trace from SharedContext."""
        trace = ExecutionTrace(
            query_id=str(context.query_id),
            user_query=context.user_query,
            status=context.status,
            final_answer=context.final_answer,
            context_snapshot=context.model_dump(mode="json"),
            routing_decision=context.routing_decision,
            agent_outputs={
                k: v.model_dump(mode="json")
                for k, v in context.agent_outputs.items()
            },
            tool_calls=[tc.model_dump(mode="json") for tc in context.tool_calls],
            policy_violations=[pv.model_dump(mode="json") for pv in context.policy_violations],
            total_tokens_used=sum(
                b.used for b in context.token_usage.values()
            ),
            retry_count=len(context.retry_history),
        )
        self._session.add(trace)
        await self._session.flush()

        # Batch persist individual tool call logs for analytics
        if context.tool_calls:
            logs = [
                ToolCallLog(
                    call_id=tc.call_id,
                    trace_id=trace.id,
                    tool_name=tc.tool_name,
                    agent_id=tc.agent_id,
                    input_data=tc.input,
                    output_data=tc.output,
                    status="success" if tc.accepted else "failed",
                    latency_ms=tc.latency_ms or 0.0,
                    retry_count=tc.retry_count,
                    accepted=tc.accepted,
                    failure_type=tc.failure_type,
                )
                for tc in context.tool_calls
            ]
            self._session.add_all(logs)

        return trace

    async def update(self, context: SharedContext) -> ExecutionTrace | None:
        """Update an existing trace with final context state."""
        result = await self._session.execute(
            select(ExecutionTrace).where(
                ExecutionTrace.query_id == str(context.query_id)
            )
        )
        trace = result.scalar_one_or_none()
        if not trace:
            return None

        trace.status = context.status
        trace.final_answer = context.final_answer
        trace.context_snapshot = context.model_dump(mode="json")
        trace.agent_outputs = {
            k: v.model_dump(mode="json")
            for k, v in context.agent_outputs.items()
        }
        trace.tool_calls = [tc.model_dump(mode="json") for tc in context.tool_calls]
        trace.policy_violations = [
            pv.model_dump(mode="json") for pv in context.policy_violations
        ]
        trace.total_tokens_used = sum(b.used for b in context.token_usage.values())

        # Persist new tool call logs for analytics in batch
        if context.tool_calls:
            existing_result = await self._session.execute(
                select(ToolCallLog.call_id).where(ToolCallLog.trace_id == trace.id)
            )
            existing_call_ids = set(existing_result.scalars().all())

            new_logs = [
                ToolCallLog(
                    call_id=tc.call_id,
                    trace_id=trace.id,
                    tool_name=tc.tool_name,
                    agent_id=tc.agent_id,
                    input_data=tc.input,
                    output_data=tc.output,
                    status="success" if tc.accepted else "failed",
                    latency_ms=tc.latency_ms or 0.0,
                    retry_count=tc.retry_count,
                    accepted=tc.accepted,
                    failure_type=tc.failure_type,
                )
                for tc in context.tool_calls
                if tc.call_id not in existing_call_ids
            ]
            if new_logs:
                self._session.add_all(new_logs)

        trace.completed_at = _utc_now()
        return trace

    async def get_by_query_id(self, query_id: str) -> ExecutionTrace | None:
        result = await self._session.execute(
            select(ExecutionTrace).where(ExecutionTrace.query_id == query_id)
        )
        return result.scalar_one_or_none()
