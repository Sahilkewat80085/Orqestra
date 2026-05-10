"""
app/schemas/agents.py
═════════════════════
Typed schemas for agent inputs, outputs, and routing decisions.
All structures are validated by Pydantic v2.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Orchestrator Routing ──────────────────────────────────────────────────────

class ExecutionStep(BaseModel):
    """A single step in the orchestrator's execution plan."""
    agent: str                          # AgentID value
    priority: int                       # Lower = run first
    depends_on: List[str] = Field(default_factory=list)  # Other agent IDs
    context_budget: Optional[int] = None  # Override default budget


class FallbackSpec(BaseModel):
    """What to do when a step fails."""
    agent: str
    condition: str     # e.g. "retrieval_failed", "tool_timeout"
    max_retries: int = 1


class RoutingDecision(BaseModel):
    """
    Structured output from the OrchestratorAgent.
    This is the ONLY way the orchestrator communicates routing to the graph.
    """
    reasoning: str
    selected_agents: List[str]
    execution_plan: List[ExecutionStep]
    fallback: Optional[FallbackSpec] = None
    estimated_total_tokens: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)


# ── Decomposition ─────────────────────────────────────────────────────────────

class TaskType(str):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPUTATIONAL = "computational"
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"


class SubTask(BaseModel):
    """A single decomposed sub-task."""
    id: str
    task: str
    type: str           # TaskType value
    depends_on: List[str] = Field(default_factory=list)
    estimated_tokens: int = 512
    requires_tools: List[str] = Field(default_factory=list)


class TaskGraph(BaseModel):
    """Output of the DecompositionAgent."""
    tasks: List[SubTask]
    reasoning: str
    total_estimated_tokens: int = 0

    def get_ready_tasks(self, completed_ids: List[str]) -> List[SubTask]:
        """Returns tasks whose dependencies are all completed."""
        return [
            t for t in self.tasks
            if all(dep in completed_ids for dep in t.depends_on)
            and t.id not in completed_ids
        ]


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single chunk returned from the vector store."""
    chunk_id: str
    content: str
    source_url: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    hop: int = 1    # 1 = first hop, 2 = second hop (multi-hop required)


class RetrievalResult(BaseModel):
    """Output of the RetrievalAgent."""
    query_used: str
    chunks: List[RetrievedChunk]
    total_hops: int
    provenance_map: Dict[str, List[str]]  # claim → [chunk_ids]
    reasoning: str


# ── Critique ──────────────────────────────────────────────────────────────────

class CritiqueFinding(BaseModel):
    """A single span-level critique finding."""
    claim: str
    confidence: float = Field(ge=0.0, le=1.0)
    issue: str
    affected_span: str           # The exact text span being critiqued
    severity: Literal["low", "medium", "high"] = "medium"
    suggested_fix: Optional[str] = None


class CritiqueReport(BaseModel):
    """Output of the CritiqueAgent — span-level, not whole-output."""
    findings: List[CritiqueFinding]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    passed: bool       # True = no high-severity issues
    reasoning: str


# ── Synthesis ─────────────────────────────────────────────────────────────────

class SentenceProvenance(BaseModel):
    """Provenance entry for a single sentence in the final answer."""
    sentence: str
    source_agent_ids: List[str]
    chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesisOutput(BaseModel):
    """Output of the SynthesisAgent."""
    final_answer: str
    sentence_provenance: List[SentenceProvenance]
    contradictions_resolved: List[str]    # Description of each resolved contradiction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ── Meta Agent ────────────────────────────────────────────────────────────────

class PromptRewriteDiff(BaseModel):
    """A proposed prompt rewrite — NOT auto-applied, requires human approval."""
    target_agent_id: str
    original_prompt_version: str
    proposed_prompt: str
    reasoning: str
    expected_improvement: str
    failed_case_ids: List[str]    # Eval cases that motivated this rewrite
    diff_summary: str             # Human-readable diff description
    status: Literal["pending", "approved", "rejected"] = "pending"


class MetaAgentOutput(BaseModel):
    """Output of the MetaPromptAgent."""
    analyzed_failures: List[str]   # Case IDs
    root_cause_analysis: str
    proposed_rewrites: List[PromptRewriteDiff]
    priority_order: List[str]      # Agent IDs ordered by improvement priority
    reasoning: str


# ── Generic API models ────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str           # Machine-readable error code e.g. "BUDGET_EXCEEDED"
    message: str
    detail: Optional[Any] = None


class APIError(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None
