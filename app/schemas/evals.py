"""
app/schemas/evals.py
════════════════════
Typed schemas for the evaluation harness.
All 15 test cases, scoring rubrics, and eval run storage use these models.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    BASELINE = "baseline"         # Factual, well-defined queries
    AMBIGUOUS = "ambiguous"       # Requires decomposition and clarification
    ADVERSARIAL = "adversarial"   # Prompt injection, false premises, attacks


class AdversarialType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    FALSE_PREMISE = "false_premise"
    CONTRADICTION_INDUCTION = "contradiction_induction"
    INSTRUCTION_OVERRIDE = "instruction_override"
    SQL_INJECTION = "sql_injection"


# ── Test Case Definition ──────────────────────────────────────────────────────

class EvalCase(BaseModel):
    """A single evaluation test case with ground truth."""
    case_id: str
    category: EvalCategory
    adversarial_type: Optional[AdversarialType] = None
    query: str
    expected_answer_keywords: List[str]    # Key concepts that must appear
    expected_citations_min: int = 0        # Minimum citations required
    should_reject_premise: bool = False    # For false-premise cases
    should_detect_injection: bool = False  # For injection cases
    notes: str = ""


# ── Scoring ───────────────────────────────────────────────────────────────────

class ScoreDimension(BaseModel):
    """
    A single scoring dimension with numeric value and written justification.
    NO black-box evaluation — every score includes a human-readable explanation.
    """
    name: str
    score: float = Field(ge=0.0, le=1.0)
    justification: str
    max_score: float = 1.0


class EvalScore(BaseModel):
    """
    Full score for one eval case.
    Six dimensions, each with numeric value + justification.
    """
    case_id: str
    correctness: ScoreDimension
    citation_accuracy: ScoreDimension
    contradiction_handling: ScoreDimension
    tool_efficiency: ScoreDimension
    context_compliance: ScoreDimension
    critique_agreement: ScoreDimension

    # Derived
    total_score: float = 0.0
    passed: bool = False

    def compute_total(self) -> "EvalScore":
        dims = [
            self.correctness,
            self.citation_accuracy,
            self.contradiction_handling,
            self.tool_efficiency,
            self.context_compliance,
            self.critique_agreement,
        ]
        total = sum(d.score for d in dims) / len(dims)
        return self.model_copy(update={"total_score": total, "passed": total >= 0.6})


# ── Eval Run ──────────────────────────────────────────────────────────────────

class EvalCaseResult(BaseModel):
    """Full result for a single case: inputs, outputs, trace, score."""
    case_id: str
    query: str
    category: EvalCategory
    final_answer: Optional[str] = None
    tool_calls_used: List[str] = Field(default_factory=list)   # tool names
    agent_outputs_snapshot: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[EvalScore] = None
    execution_trace_id: Optional[str] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvalRun(BaseModel):
    """A complete evaluation run across all (or selected) cases."""
    run_id: UUID = Field(default_factory=uuid4)
    triggered_by: str = "manual"     # "manual" | "meta_agent" | "scheduled"
    case_ids: List[str] = Field(default_factory=list)
    results: List[EvalCaseResult] = Field(default_factory=list)
    summary: Optional["EvalSummary"] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    prompt_versions_used: Dict[str, str] = Field(default_factory=dict)  # agent → version


class EvalSummary(BaseModel):
    """Aggregate statistics across all cases in an eval run."""
    run_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_total_score: float
    avg_by_category: Dict[str, float]   # category → avg score
    avg_by_dimension: Dict[str, float]  # dimension name → avg score
    worst_cases: List[str]             # case_ids with lowest scores
    best_cases: List[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Prompt Rewrite Storage ────────────────────────────────────────────────────

class PromptVersion(BaseModel):
    """Versioned prompt record stored in the database."""
    version_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    version: str                     # e.g. "v1.0", "v1.1"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = None
    status: str = "active"          # "active" | "pending" | "rejected" | "deprecated"
    rewrite_reasoning: Optional[str] = None
    parent_version_id: Optional[str] = None
