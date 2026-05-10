"""
app/schemas/tools.py
════════════════════
Typed schemas for tool invocations, results, and failure contracts.
Every tool must conform to these contracts for observability and retryability.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MALFORMED_INPUT = "malformed_input"
    MALFORMED_OUTPUT = "malformed_output"
    EMPTY_RESULT = "empty_result"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMITED = "rate_limited"
    EXECUTION_ERROR = "execution_error"


class RetryEligibility(str, Enum):
    """Determines whether the orchestrator should retry a failed tool call."""
    ELIGIBLE = "eligible"           # Retry immediately
    ELIGIBLE_BACKOFF = "backoff"    # Retry after exponential backoff
    NOT_ELIGIBLE = "not_eligible"   # Do not retry (e.g. invalid input)


# ── Tool Result Base ──────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Base result returned by every tool."""
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    status: ToolStatus
    latency_ms: float = 0.0
    retry_count: int = 0
    retry_eligible: RetryEligibility = RetryEligibility.NOT_ELIGIBLE
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_output: Optional[Any] = None
    error_message: Optional[str] = None


# ── Web Search ────────────────────────────────────────────────────────────────

class WebSearchInput(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    language: str = "en"


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    published_date: Optional[str] = None


class WebSearchOutput(ToolResult):
    results: List[WebSearchResult] = Field(default_factory=list)
    query_used: str = ""
    total_found: int = 0


# ── Python Sandbox ────────────────────────────────────────────────────────────

class PythonSandboxInput(BaseModel):
    code: str
    timeout_secs: int = Field(default=10, ge=1, le=30)
    allowed_imports: List[str] = Field(
        default_factory=lambda: ["math", "json", "datetime", "re", "collections"]
    )


class PythonSandboxOutput(ToolResult):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time_ms: float = 0.0


# ── NL to SQL ─────────────────────────────────────────────────────────────────

class NL2SQLInput(BaseModel):
    natural_language_query: str
    target_schema: Optional[str] = None    # Schema hint for SQL generation
    max_rows: int = Field(default=100, ge=1, le=1000)


class NL2SQLOutput(ToolResult):
    generated_sql: str = ""
    validated: bool = False
    result_rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)
    sql_validated: bool = False


# ── Self Reflection ───────────────────────────────────────────────────────────

class SelfReflectionInput(BaseModel):
    focus_agent_ids: List[str]       # Which agents' outputs to compare
    reflection_prompt: Optional[str] = None   # Optional focused question


class ContradictionFound(BaseModel):
    agent_a: str
    agent_b: str
    claim_a: str
    claim_b: str
    severity: str    # "low" | "medium" | "high"
    suggested_resolution: str


class SelfReflectionOutput(ToolResult):
    contradictions: List[ContradictionFound] = Field(default_factory=list)
    reasoning_gaps: List[str] = Field(default_factory=list)
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reflection_summary: str = ""
    recommended_actions: List[str] = Field(default_factory=list)


# ── Tool Failure Contract (explicit per tool) ─────────────────────────────────

class ToolFailureContract(BaseModel):
    """
    Explicit failure contract that every tool must declare.
    The orchestrator reads this to determine fallback strategy.
    """
    tool_name: str
    timeout_secs: int
    timeout_retry_eligible: bool
    malformed_input_retry_eligible: bool
    empty_result_retry_eligible: bool
    rate_limit_retry_eligible: bool
    max_retries: int
    fallback_strategy: str    # e.g. "skip", "use_cache", "fallback_agent"
