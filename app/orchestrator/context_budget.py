"""
app/orchestrator/context_budget.py
════════════════════════════════════
ContextBudgetManager: Per-agent token budget enforcement.

Rules:
  - Each agent has a pre-allocated token budget from settings
  - Usage is tracked as agents consume tokens
  - Budget overruns trigger PolicyViolation events (never silent truncation)
  - Overflow triggers a compression pass (summarize conversational filler)
  - Structured data is preserved losslessly during compression
"""
from __future__ import annotations

from typing import Dict, Optional

from app.config import settings
from app.logging.logger import get_logger
from app.schemas.context import (
    AgentID, PolicyViolation, SharedContext, TokenBudget,
)
from app.utils.token_counter import count_tokens

log = get_logger("orchestrator.budget")


class BudgetExceededError(Exception):
    """Raised when an agent exceeds its token budget without permission."""
    def __init__(self, agent_id: str, tokens_over: int):
        self.agent_id = agent_id
        self.tokens_over = tokens_over
        super().__init__(
            f"Agent '{agent_id}' exceeded budget by {tokens_over} tokens"
        )


class ContextBudgetManager:
    """
    Manages token budgets for all agents.

    Usage:
        manager = ContextBudgetManager(context)
        manager.initialize_budgets()

        # Before agent call:
        manager.check_budget(AgentID.RETRIEVAL, prompt_text)

        # After agent completes:
        manager.consume(AgentID.RETRIEVAL, tokens_used)
    """

    def __init__(self, context: SharedContext):
        self._context = context
        self._budgets: Dict[str, TokenBudget] = {}

    def initialize_budgets(self) -> SharedContext:
        """
        Allocate token budgets for all agents based on settings.
        Returns the updated SharedContext with budgets set.
        """
        agent_budgets = settings.agent_budgets
        ctx = self._context

        for agent_id, allocated in agent_budgets.items():
            budget = TokenBudget(
                allocated=allocated,
                used=0,
                remaining=allocated,
            )
            self._budgets[agent_id] = budget
            ctx = ctx.update_budget(agent_id, budget)

        self._context = ctx
        log.info("budgets_initialized", agent_budgets=agent_budgets)
        return ctx

    def check_budget(
        self, agent_id: str, text: str, raise_on_exceed: bool = True
    ) -> tuple[bool, int]:
        """
        Check if agent has enough budget for the given text.

        Args:
            agent_id: The agent requesting budget
            text: The text to count tokens for
            raise_on_exceed: If True, raise BudgetExceededError

        Returns:
            (within_budget, token_count)
        """
        token_count = count_tokens(text)
        budget = self._budgets.get(agent_id)

        if budget is None:
            log.warning("no_budget_for_agent", agent_id=agent_id)
            return True, token_count

        if token_count > budget.remaining:
            over = token_count - budget.remaining
            log.error(
                "budget_exceeded",
                agent_id=agent_id,
                requested=token_count,
                remaining=budget.remaining,
                over_by=over,
            )
            if raise_on_exceed:
                raise BudgetExceededError(agent_id, over)
            return False, token_count

        return True, token_count

    def consume(
        self, agent_id: str, text: str
    ) -> tuple[SharedContext, TokenBudget]:
        """
        Consume tokens from an agent's budget.
        Returns the updated context and new budget state.
        Emits a PolicyViolation if budget is exceeded.
        """
        token_count = count_tokens(text)
        budget = self._budgets.get(agent_id)

        if budget is None:
            return self._context, TokenBudget(allocated=0, used=0, remaining=0)

        new_budget = budget.consume(token_count)
        self._budgets[agent_id] = new_budget
        ctx = self._context.update_budget(agent_id, new_budget)

        if new_budget.remaining < 0:
            # Policy violation — NEVER silent truncation
            violation = PolicyViolation(
                agent_id=agent_id,
                violation_type="BUDGET_EXCEEDED",
                detail=(
                    f"Agent consumed {token_count} tokens but only "
                    f"{budget.remaining} remained"
                ),
                tokens_over_budget=abs(new_budget.remaining),
            )
            ctx = ctx.add_violation(violation)
            log.error(
                "budget_policy_violation",
                agent_id=agent_id,
                tokens_over=abs(new_budget.remaining),
            )

        self._context = ctx
        return ctx, new_budget

    def compress_context(
        self, agent_id: str, text: str, preserve_structured: bool = True
    ) -> str:
        """
        Compress conversational filler when budget is tight.
        Structured data (JSON, tables, citations) is NEVER compressed.

        In a full implementation this would call a summarization LLM.
        Here we truncate non-structured segments as a safe fallback.
        """
        budget = self._budgets.get(agent_id)
        if budget is None or budget.remaining <= 0:
            return text

        if preserve_structured:
            # Simple heuristic: preserve any line containing JSON-like content
            lines = text.splitlines()
            important = [l for l in lines if any(
                c in l for c in ["{", "}", "[", "]", '"id"', '"chunk"']
            )]
            filler = [l for l in lines if l not in important]

            # Truncate filler to fit budget
            budget_chars = budget.remaining * 4  # ~4 chars per token
            filler_budget = budget_chars - sum(len(l) for l in important)
            filler_text = "\n".join(filler)[:max(0, filler_budget)]

            compressed = "\n".join(important) + "\n" + filler_text
        else:
            budget_chars = budget.remaining * 4
            compressed = text[:budget_chars]

        log.info(
            "context_compressed",
            agent_id=agent_id,
            original_len=len(text),
            compressed_len=len(compressed),
        )

        # Mark budget as compressed
        if agent_id in self._budgets:
            updated = self._budgets[agent_id].model_copy(update={"compressed": True})
            self._budgets[agent_id] = updated
            self._context = self._context.update_budget(agent_id, updated)

        return compressed

    @property
    def context(self) -> SharedContext:
        return self._context

    def get_budget_summary(self) -> Dict[str, dict]:
        return {
            agent_id: {
                "allocated": b.allocated,
                "used": b.used,
                "remaining": b.remaining,
                "pct_used": round(b.used / b.allocated * 100, 1) if b.allocated else 0,
            }
            for agent_id, b in self._budgets.items()
        }
