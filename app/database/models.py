"""
app/database/models.py
══════════════════════
SQLAlchemy 2.0 ORM models for all persistent data in Orqestra.
Uses async-compatible mapped column syntax.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Execution Traces ──────────────────────────────────────────────────────────

class ExecutionTrace(Base):
    """
    Stores the full execution trace for every query.
    Supports complete replay and diff between runs.
    """
    __tablename__ = "execution_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    query_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full SharedContext snapshot (JSON) — enables complete replay
    context_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Structured sub-records (denormalized for fast querying)
    routing_decision: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    agent_outputs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    policy_violations: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Hashes for content-addressed audit
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    output_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Metrics
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    tool_call_logs: Mapped[list["ToolCallLog"]] = relationship(
        "ToolCallLog", back_populates="trace", cascade="all, delete-orphan"
    )


# ── Tool Call Logs ────────────────────────────────────────────────────────────

class ToolCallLog(Base):
    """
    Immutable log of every tool invocation.
    Separate table enables efficient tool-level analytics.
    """
    __tablename__ = "tool_call_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    call_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    trace_id: Mapped[str] = mapped_column(String(36), ForeignKey("execution_traces.id"), index=True)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False)

    input_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="success")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    accepted: Mapped[bool] = mapped_column(Boolean, default=True)
    failure_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    trace: Mapped["ExecutionTrace"] = relationship("ExecutionTrace", back_populates="tool_call_logs")


# ── Eval Runs ─────────────────────────────────────────────────────────────────

class EvalRunRecord(Base):
    """Stores metadata for each evaluation run."""
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    triggered_by: Mapped[str] = mapped_column(String(64), default="manual")
    case_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    prompt_versions_used: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, default=0)
    avg_total_score: Mapped[float] = mapped_column(Float, default=0.0)

    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    case_results: Mapped[list["EvalCaseRecord"]] = relationship(
        "EvalCaseRecord", back_populates="run", cascade="all, delete-orphan"
    )


class EvalCaseRecord(Base):
    """Stores the result for each individual eval case."""
    __tablename__ = "eval_case_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("eval_runs.id"), index=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(32))
    query: Mapped[str] = mapped_column(Text)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full score JSON
    score: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)

    agent_outputs_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tool_calls_used: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    execution_trace_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    run: Mapped["EvalRunRecord"] = relationship("EvalRunRecord", back_populates="case_results")


# ── Prompt Versions ───────────────────────────────────────────────────────────

class PromptVersionRecord(Base):
    """
    Versioned prompt storage with approval workflow.
    Agents always load the latest 'active' version.
    Pending rewrites sit here until human approval via API.
    """
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|pending|rejected|deprecated

    approved_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    rewrite_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parent_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    failed_case_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ── Policy Violation Log ──────────────────────────────────────────────────────

class PolicyViolationRecord(Base):
    """Separate table for fast violation queries and alerting."""
    __tablename__ = "policy_violations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    query_id: Mapped[str] = mapped_column(String(36), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    violation_type: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str] = mapped_column(Text)
    tokens_over_budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
