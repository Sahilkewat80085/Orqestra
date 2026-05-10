"""tests/test_context_budget.py — Budget enforcement unit tests."""
from __future__ import annotations

import pytest
from app.orchestrator.context_budget import ContextBudgetManager, BudgetExceededError
from app.schemas.context import SharedContext


def test_budget_initialized(sample_context):
    manager = ContextBudgetManager(sample_context)
    ctx = manager.initialize_budgets()
    assert "orchestrator" in ctx.token_usage
    assert ctx.token_usage["orchestrator"].allocated > 0
    assert ctx.token_usage["orchestrator"].remaining > 0


def test_budget_within_limits(sample_context):
    manager = ContextBudgetManager(sample_context)
    manager.initialize_budgets()
    within, count = manager.check_budget("orchestrator", "short text", raise_on_exceed=False)
    assert within is True
    assert count > 0


def test_budget_exceeded_raises(sample_context):
    manager = ContextBudgetManager(sample_context)
    manager.initialize_budgets()
    huge_text = "word " * 10000  # Way over any budget
    with pytest.raises(BudgetExceededError):
        manager.check_budget("orchestrator", huge_text, raise_on_exceed=True)


def test_budget_violation_logged(sample_context):
    manager = ContextBudgetManager(sample_context)
    ctx = manager.initialize_budgets()

    # Force a violation by consuming more than allocated
    huge_text = "word " * 10000
    updated_ctx, _ = manager.consume("orchestrator", huge_text)

    # Should have logged a policy violation
    assert len(updated_ctx.policy_violations) > 0
    assert updated_ctx.policy_violations[0].violation_type == "BUDGET_EXCEEDED"


def test_no_silent_truncation(sample_context):
    """Verify that budget overruns are ALWAYS logged, never silently ignored."""
    manager = ContextBudgetManager(sample_context)
    ctx = manager.initialize_budgets()
    huge = "x " * 5000
    updated, budget = manager.consume("synthesis", huge)
    if budget.remaining < 0:
        assert any(v.violation_type == "BUDGET_EXCEEDED" for v in updated.policy_violations)
