"""
app/agents/base.py
════════════════════
Abstract BaseAgent for all Orqestra agents.

Enforces:
  - Context-only communication (no direct agent-to-agent calls)
  - Budget check before every LLM call
  - Automatic timing and token tracking
  - Structured logging with AgentLogger
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.logging.logger import AgentLogger
from app.orchestrator.context_budget import ContextBudgetManager
from app.schemas.context import SharedContext


class BaseAgent(ABC):
    """
    All agents extend this class.

    Subclasses implement:
      - `agent_id` property
      - `_run(context)` — core agent logic
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        ...

    @abstractmethod
    async def _run(self, context: SharedContext) -> SharedContext:
        """Core agent logic. Receives and returns SharedContext."""
        ...

    async def invoke(self, context: SharedContext) -> SharedContext:
        """
        Public entry point.
        Handles timing, logging, and error recovery.
        Agents MUST only communicate through the returned SharedContext.
        """
        logger = AgentLogger(self.agent_id, str(context.query_id))
        logger.started()

        start = time.perf_counter()
        try:
            updated_context = await self._run(context)
        except Exception as e:
            logger.error(str(e))
            # On unhandled error, return context with status updated
            return context.mark_timestamp(f"{self.agent_id}_failed").model_copy(
                update={"status": "partial"}
            )

        latency_ms = (time.perf_counter() - start) * 1000

        # Determine token count from agent's output if available
        output = updated_context.agent_outputs.get(self.agent_id)
        token_count = output.token_count if output else 0

        logger.completed(latency_ms=latency_ms, token_count=token_count)
        return updated_context.mark_timestamp(f"{self.agent_id}_completed")
