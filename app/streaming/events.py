"""
app/streaming/events.py
════════════════════════
Typed SSE event models for all observable activity in Orqestra.

Every event emitted over SSE is one of these typed models.
The client sees the full execution in real-time.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Agent lifecycle
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"

    # Token streaming
    TOKEN_CHUNK = "token_chunk"

    # Tool lifecycle
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_RETRY = "tool_retry"

    # Budget
    BUDGET_UPDATE = "budget_update"
    POLICY_VIOLATION = "policy_violation"

    # Pipeline
    ROUTING_DECISION = "routing_decision"
    FINAL_ANSWER = "final_answer"
    PIPELINE_ERROR = "pipeline_error"
    PIPELINE_COMPLETE = "pipeline_complete"


class StreamEvent(BaseModel):
    """Base event published to Redis and streamed via SSE."""
    query_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    sequence: int = 0  # For ordering on the client side


# ── Typed constructors ────────────────────────────────────────────────────────

def agent_started(query_id: str, agent_id: str, seq: int = 0) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.AGENT_STARTED,
        payload={"agent_id": agent_id},
        sequence=seq,
    )


def agent_completed(
    query_id: str, agent_id: str, latency_ms: float,
    token_count: int, seq: int = 0
) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.AGENT_COMPLETED,
        payload={
            "agent_id": agent_id,
            "latency_ms": round(latency_ms, 2),
            "token_count": token_count,
        },
        sequence=seq,
    )


def tool_call_started(
    query_id: str, tool_name: str, agent_id: str, seq: int = 0
) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.TOOL_CALL_STARTED,
        payload={"tool_name": tool_name, "agent_id": agent_id},
        sequence=seq,
    )


def tool_call_completed(
    query_id: str, tool_name: str, status: str,
    latency_ms: float, retry_count: int, seq: int = 0
) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.TOOL_CALL_COMPLETED,
        payload={
            "tool_name": tool_name,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "retry_count": retry_count,
        },
        sequence=seq,
    )


def budget_update(
    query_id: str, agent_id: str, used: int,
    remaining: int, allocated: int, seq: int = 0
) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.BUDGET_UPDATE,
        payload={
            "agent_id": agent_id,
            "used": used,
            "remaining": remaining,
            "allocated": allocated,
            "pct_used": round(used / allocated * 100, 1) if allocated else 0,
        },
        sequence=seq,
    )


def routing_decision(query_id: str, decision: dict, seq: int = 0) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.ROUTING_DECISION,
        payload={"decision": decision},
        sequence=seq,
    )


def final_answer(query_id: str, answer: str, confidence: float, seq: int = 0) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.FINAL_ANSWER,
        payload={"answer": answer, "confidence": confidence},
        sequence=seq,
    )


def policy_violation(
    query_id: str, agent_id: str, violation_type: str, detail: str, seq: int = 0
) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.POLICY_VIOLATION,
        payload={
            "agent_id": agent_id,
            "violation_type": violation_type,
            "detail": detail,
        },
        sequence=seq,
    )


def pipeline_complete(query_id: str, total_latency_ms: float, seq: int = 0) -> StreamEvent:
    return StreamEvent(
        query_id=query_id,
        event_type=EventType.PIPELINE_COMPLETE,
        payload={"total_latency_ms": round(total_latency_ms, 2)},
        sequence=seq,
    )
