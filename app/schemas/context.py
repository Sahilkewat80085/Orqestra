"""
app/schemas/context.py
══════════════════════
The SharedContext is the SINGLE typed contract for all inter-agent
communication in Orqestra.

RULES:
  - Agents MUST only read/write state through this object.
  - Agents MUST NOT call each other directly.
  - All writes are append-only for list fields (provenance, tool_calls, etc.)
    to preserve a complete audit trail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


# ── Enumerations ──────────────────────────────────────────────────────────────

class AgentID(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DECOMPOSITION = "decomposition"
    RETRIEVAL = "retrieval"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    META = "meta"


class ToolName(str, Enum):
    WEB_SEARCH = "web_search"
    PYTHON_SANDBOX = "python_sandbox"
    NL2SQL = "nl2sql"
    SELF_REFLECTION = "self_reflection"


class FailureType(str, Enum):
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    EMPTY = "empty"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ── Sub-models ────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """Tracks which retrieved chunk contributed to which claim."""
    chunk_id: str
    source_url: str | None = None
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    contributing_to_claim: str  # The claim this citation supports


class ProvenanceEntry(BaseModel):
    """Sentence-level provenance: maps each output sentence to its origin."""
    sentence: str
    agent_id: AgentID
    tool_calls: list[str] = Field(default_factory=list)  # call_ids
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TokenBudget(BaseModel):
    """Per-agent token budget enforced by ContextBudgetManager."""
    allocated: int
    used: int = 0
    remaining: int
    compressed: bool = False

    def consume(self, tokens: int) -> TokenBudget:
        new_used = self.used + tokens
        return self.model_copy(update={
            "used": new_used,
            "remaining": self.allocated - new_used,
        })


class ToolCallRecord(BaseModel):
    """Immutable audit record for every tool invocation."""
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: ToolName
    agent_id: AgentID
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    accepted: bool = True
    failure_type: FailureType | None = None
    timestamp: datetime = Field(default_factory=_utc_now)


class RetryEvent(BaseModel):
    """Logged every time the orchestrator triggers a retry."""
    agent_id: AgentID
    tool_name: ToolName | None = None
    attempt: int
    reason: str
    failure_type: FailureType
    timestamp: datetime = Field(default_factory=_utc_now)


class PolicyViolation(BaseModel):
    """Raised when an agent exceeds its context budget or violates policy."""
    agent_id: AgentID
    violation_type: str
    detail: str
    tokens_over_budget: int | None = None
    timestamp: datetime = Field(default_factory=_utc_now)


class AgentOutput(BaseModel):
    """Typed container for any agent's output stored in shared context."""
    agent_id: AgentID
    raw_output: str
    structured_output: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    token_count: int = 0
    timestamp: datetime = Field(default_factory=_utc_now)


# ── SharedContext ─────────────────────────────────────────────────────────────

class SharedContext(BaseModel):
    """
    The ONLY channel for inter-agent communication.

    All agents receive this object, mutate a copy, and return the updated copy.
    The orchestrator merges updates back into the authoritative context.
    List fields are append-only — nothing is ever deleted to preserve audit trails.
    """

    model_config = ConfigDict(use_enum_values=True)

    # Identity
    query_id: UUID = Field(default_factory=uuid4)
    user_query: str

    # Agent outputs — keyed by AgentID value (string)
    agent_outputs: dict[str, AgentOutput] = Field(default_factory=dict)

    # Tool call audit log (append-only)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # Retrieved citations (append-only)
    citations: list[Citation] = Field(default_factory=list)

    # Per-agent token budgets — keyed by AgentID value
    token_usage: dict[str, TokenBudget] = Field(default_factory=dict)

    # Sentence-level provenance map (append-only)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)

    # Retry history (append-only)
    retry_history: list[RetryEvent] = Field(default_factory=list)

    # Policy violations (append-only)
    policy_violations: list[PolicyViolation] = Field(default_factory=list)

    # Lifecycle timestamps — e.g. {"started": datetime, "orchestrator_done": datetime}
    timestamps: dict[str, datetime] = Field(default_factory=dict)

    # Orchestrator routing decision (set once per run)
    routing_decision: dict[str, Any] | None = None

    # Decomposition task graph
    task_graph: dict[str, Any] | None = None

    # Final synthesized answer
    final_answer: str | None = None

    # Execution lifecycle
    status: ExecutionStatus = ExecutionStatus.PENDING

    # ── Helpers ───────────────────────────────────────────────────────────────

    def mark_timestamp(self, key: str) -> SharedContext:
        updated = dict(self.timestamps)
        updated[key] = _utc_now()
        return self.model_copy(update={"timestamps": updated})

    def add_tool_call(self, record: ToolCallRecord) -> SharedContext:
        return self.model_copy(update={"tool_calls": [*self.tool_calls, record]})

    def add_citation(self, citation: Citation) -> SharedContext:
        return self.model_copy(update={"citations": [*self.citations, citation]})

    def add_provenance(self, entry: ProvenanceEntry) -> SharedContext:
        return self.model_copy(update={"provenance": [*self.provenance, entry]})

    def add_retry(self, event: RetryEvent) -> SharedContext:
        return self.model_copy(update={"retry_history": [*self.retry_history, event]})

    def add_violation(self, violation: PolicyViolation) -> SharedContext:
        return self.model_copy(update={"policy_violations": [*self.policy_violations, violation]})

    def set_agent_output(self, output: AgentOutput) -> SharedContext:
        updated = dict(self.agent_outputs)
        agent_key = output.agent_id.value if isinstance(output.agent_id, Enum) else str(output.agent_id)
        updated[agent_key] = output
        return self.model_copy(update={"agent_outputs": updated})

    def update_budget(self, agent_id: str, budget: TokenBudget) -> SharedContext:
        updated = dict(self.token_usage)
        updated[agent_id] = budget
        return self.model_copy(update={"token_usage": updated})
