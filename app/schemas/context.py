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

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
    source_url: Optional[str] = None
    excerpt: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    contributing_to_claim: str  # The claim this citation supports


class ProvenanceEntry(BaseModel):
    """Sentence-level provenance: maps each output sentence to its origin."""
    sentence: str
    agent_id: AgentID
    tool_calls: List[str] = Field(default_factory=list)   # call_ids
    chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TokenBudget(BaseModel):
    """Per-agent token budget enforced by ContextBudgetManager."""
    allocated: int
    used: int = 0
    remaining: int
    compressed: bool = False

    def consume(self, tokens: int) -> "TokenBudget":
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
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    retry_count: int = 0
    accepted: bool = True
    failure_type: Optional[FailureType] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RetryEvent(BaseModel):
    """Logged every time the orchestrator triggers a retry."""
    agent_id: AgentID
    tool_name: Optional[ToolName] = None
    attempt: int
    reason: str
    failure_type: FailureType
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PolicyViolation(BaseModel):
    """Raised when an agent exceeds its context budget or violates policy."""
    agent_id: AgentID
    violation_type: str
    detail: str
    tokens_over_budget: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentOutput(BaseModel):
    """Typed container for any agent's output stored in shared context."""
    agent_id: AgentID
    raw_output: str
    structured_output: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    token_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── SharedContext ─────────────────────────────────────────────────────────────

class SharedContext(BaseModel):
    """
    The ONLY channel for inter-agent communication.

    All agents receive this object, mutate a copy, and return the updated copy.
    The orchestrator merges updates back into the authoritative context.
    List fields are append-only — nothing is ever deleted to preserve audit trails.
    """

    # Identity
    query_id: UUID = Field(default_factory=uuid4)
    user_query: str

    # Agent outputs — keyed by AgentID value (string)
    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)

    # Tool call audit log (append-only)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)

    # Retrieved citations (append-only)
    citations: List[Citation] = Field(default_factory=list)

    # Per-agent token budgets — keyed by AgentID value
    token_usage: Dict[str, TokenBudget] = Field(default_factory=dict)

    # Sentence-level provenance map (append-only)
    provenance: List[ProvenanceEntry] = Field(default_factory=list)

    # Retry history (append-only)
    retry_history: List[RetryEvent] = Field(default_factory=list)

    # Policy violations (append-only)
    policy_violations: List[PolicyViolation] = Field(default_factory=list)

    # Lifecycle timestamps — e.g. {"started": datetime, "orchestrator_done": datetime}
    timestamps: Dict[str, datetime] = Field(default_factory=dict)

    # Orchestrator routing decision (set once per run)
    routing_decision: Optional[Dict[str, Any]] = None

    # Decomposition task graph
    task_graph: Optional[Dict[str, Any]] = None

    # Final synthesized answer
    final_answer: Optional[str] = None

    # Execution lifecycle
    status: ExecutionStatus = ExecutionStatus.PENDING

    class Config:
        use_enum_values = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def mark_timestamp(self, key: str) -> "SharedContext":
        updated = dict(self.timestamps)
        updated[key] = datetime.utcnow()
        return self.model_copy(update={"timestamps": updated})

    def add_tool_call(self, record: ToolCallRecord) -> "SharedContext":
        return self.model_copy(update={"tool_calls": [*self.tool_calls, record]})

    def add_citation(self, citation: Citation) -> "SharedContext":
        return self.model_copy(update={"citations": [*self.citations, citation]})

    def add_provenance(self, entry: ProvenanceEntry) -> "SharedContext":
        return self.model_copy(update={"provenance": [*self.provenance, entry]})

    def add_retry(self, event: RetryEvent) -> "SharedContext":
        return self.model_copy(update={"retry_history": [*self.retry_history, event]})

    def add_violation(self, violation: PolicyViolation) -> "SharedContext":
        return self.model_copy(update={"policy_violations": [*self.policy_violations, violation]})

    def set_agent_output(self, output: AgentOutput) -> "SharedContext":
        updated = dict(self.agent_outputs)
        updated[output.agent_id] = output
        return self.model_copy(update={"agent_outputs": updated})

    def update_budget(self, agent_id: str, budget: TokenBudget) -> "SharedContext":
        updated = dict(self.token_usage)
        updated[agent_id] = budget
        return self.model_copy(update={"token_usage": updated})
