"""tests/test_agents.py — Agent unit tests with mocked SharedContext."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.context import AgentID, AgentOutput, SharedContext


def test_shared_context_immutable_helpers(sample_context):
    """Verify that helper methods return new copies, preserving audit trail."""
    from app.schemas.context import Citation
    original = sample_context
    updated = original.add_citation(
        Citation(
            chunk_id="chunk_test",
            excerpt="test excerpt",
            relevance_score=0.9,
            contributing_to_claim="test claim",
        )
    )
    assert len(original.citations) == 0   # Original unchanged
    assert len(updated.citations) == 1    # New copy has it


def test_agent_output_stored_in_context(sample_context):
    output = AgentOutput(
        agent_id=AgentID.RETRIEVAL,
        raw_output="retrieved content",
        token_count=50,
    )
    updated = sample_context.set_agent_output(output)
    assert AgentID.RETRIEVAL in updated.agent_outputs
    assert updated.agent_outputs[AgentID.RETRIEVAL].raw_output == "retrieved content"


def test_retry_event_appended(sample_context):
    from app.schemas.context import RetryEvent, FailureType
    event = RetryEvent(
        agent_id=AgentID.RETRIEVAL,
        attempt=1,
        reason="empty result",
        failure_type=FailureType.EMPTY,
    )
    updated = sample_context.add_retry(event)
    assert len(updated.retry_history) == 1
    assert len(sample_context.retry_history) == 0  # Original unchanged


@pytest.mark.asyncio
async def test_decomposition_fallback_on_llm_error(sample_context):
    """DecompositionAgent must return a valid context even on LLM failure."""
    from app.agents.decomposition import DecompositionAgent
    agent = DecompositionAgent()
    with patch("app.agents.decomposition.ChatGoogleGenerativeAI") as mock_llm:
        mock_llm.return_value.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
        result = await agent.invoke(sample_context)
    # Must not raise, must return context with some output
    assert result is not None
    assert AgentID.DECOMPOSITION in result.agent_outputs


def test_routing_decision_structure():
    from app.schemas.agents import RoutingDecision, ExecutionStep
    decision = RoutingDecision(
        reasoning="Complex query needs decomposition",
        selected_agents=["decomposition", "retrieval", "synthesis"],
        execution_plan=[
            ExecutionStep(agent="decomposition", priority=1),
            ExecutionStep(agent="retrieval", priority=2, depends_on=["decomposition"]),
            ExecutionStep(agent="synthesis", priority=3, depends_on=["retrieval"]),
        ],
    )
    assert decision.confidence == 0.9
    assert len(decision.execution_plan) == 3
    assert decision.execution_plan[1].depends_on == ["decomposition"]
