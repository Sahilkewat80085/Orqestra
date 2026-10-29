"""
app/orchestrator/router.py
════════════════════════════
OrchestratorAgent: The dynamic routing brain of Orqestra.

Responsibilities:
  - Analyze the user query and current context
  - Decide which agents to invoke and in what order
  - Produce a structured RoutingDecision
  - Log all reasoning
  - Handle conditional execution and fallback behavior

RULES:
  - NO static chains — routing is always decided at runtime
  - Routing decision is stored in SharedContext.routing_decision
  - The graph reads this decision to set conditional edges
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.logging.logger import AgentLogger
from app.schemas.agents import ExecutionStep, FallbackSpec, RoutingDecision
from app.schemas.context import AgentID, AgentOutput, SharedContext
from app.utils.hashing import hash_content
from app.utils.token_counter import count_tokens

_ORCHESTRATOR_PROMPT = """You are the Orchestrator of a multi-agent AI system called Orqestra.

Your job is to analyze the user's query and decide:
1. Which agents should be invoked
2. The optimal execution order
3. Fallback behavior if agents fail

Available agents:
- decomposition: Breaks complex queries into typed sub-tasks with dependency graphs
- retrieval: Multi-hop retrieval from a knowledge base (needs ≥2 chunks)
- critique: Reviews agent outputs for contradictions and weak reasoning (span-level)
- synthesis: Merges all outputs into a final answer with provenance

Guidelines:
- Simple factual queries: retrieval → synthesis (skip decomposition)
- Complex or ambiguous queries: decomposition → retrieval → critique → synthesis
- Adversarial/suspicious queries: add critique BEFORE synthesis
- Computational queries: add python_sandbox tool usage in retrieval stage
- ALWAYS include synthesis as the final step

User query: {query}

Return a JSON object with EXACTLY this structure:
{{
  "reasoning": "<your step-by-step reasoning for this routing decision>",
  "selected_agents": ["<agent1>", "<agent2>", ...],
  "execution_plan": [
    {{"agent": "<name>", "priority": <int>, "depends_on": [], "context_budget": <optional int>}},
    ...
  ],
  "fallback": {{"agent": "synthesis", "condition": "<failure_condition>", "max_retries": 1}},
  "estimated_total_tokens": <int>,
  "confidence": <0.0-1.0>
}}

Return ONLY the JSON object, no markdown fences."""

_JSON_BLOCK_RE: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)


def _extract_json_block(raw: str) -> str:
    """Extract clean JSON text even when surrounded by markdown code blocks."""
    text = raw.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return text


class OrchestratorRouter:
    """
    Calls the LLM once to produce a dynamic RoutingDecision.
    This decision drives all subsequent graph edges.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        self._llm = llm or ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.gemini_temperature,
        )

    async def route(self, context: SharedContext) -> SharedContext:
        """
        Analyze the query and produce a routing decision.
        Mutates the context with the decision and logs reasoning.
        """
        agent_logger = AgentLogger(AgentID.ORCHESTRATOR, str(context.query_id))
        agent_logger.started(query=context.user_query[:100])

        start = time.perf_counter()
        prompt = _ORCHESTRATOR_PROMPT.format(query=context.user_query)

        raw = ""
        try:
            response = await self._llm.ainvoke(prompt)
            clean_json = _extract_json_block(str(response.content))
            raw = clean_json
            data: dict[str, Any] = json.loads(clean_json)

            decision = RoutingDecision(
                reasoning=str(data.get("reasoning", "")),
                selected_agents=list(data.get("selected_agents", ["synthesis"])),
                execution_plan=[
                    ExecutionStep(**step) for step in data.get("execution_plan", [])
                ],
                fallback=FallbackSpec(**data["fallback"]) if data.get("fallback") else None,
                estimated_total_tokens=int(data.get("estimated_total_tokens", 0)),
                confidence=float(data.get("confidence", 0.9)),
            )
        except Exception as e:
            # Fallback routing on parse failure — still structured, not hardcoded
            agent_logger.error(f"Routing LLM failed: {e}, using fallback plan")
            decision = RoutingDecision(
                reasoning=f"Fallback routing due to error: {e}",
                selected_agents=["retrieval", "synthesis"],
                execution_plan=[
                    ExecutionStep(agent="retrieval", priority=1),
                    ExecutionStep(agent="synthesis", priority=2, depends_on=["retrieval"]),
                ],
                fallback=FallbackSpec(agent="synthesis", condition="retrieval_failed"),
                confidence=0.5,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        token_count = count_tokens(raw if raw else str(decision))

        agent_logger.completed(
            latency_ms=latency_ms,
            token_count=token_count,
            selected_agents=decision.selected_agents,
            confidence=decision.confidence,
        )

        output = AgentOutput(
            agent_id=AgentID.ORCHESTRATOR,
            raw_output=decision.reasoning,
            structured_output=decision.model_dump(),
            confidence=decision.confidence,
            token_count=token_count,
        )

        return (
            context
            .set_agent_output(output)
            .mark_timestamp("orchestrator_completed")
            .model_copy(update={"routing_decision": decision.model_dump()})
        )
