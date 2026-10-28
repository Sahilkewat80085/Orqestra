"""
app/agents/decomposition.py
═════════════════════════════
DecompositionAgent: Breaks ambiguous queries into typed sub-tasks
with explicit dependency graphs.

Output: TaskGraph stored in SharedContext.task_graph
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import BaseAgent
from app.config import settings
from app.schemas.agents import SubTask, TaskGraph
from app.schemas.context import AgentID, AgentOutput, SharedContext
from app.utils.token_counter import count_tokens

_PROMPT = """You are a task decomposition expert. Break the following query into typed sub-tasks.

Query: {query}

Rules:
- Each task must have a unique id (task_1, task_2, ...)
- task type must be one of: factual, analytical, computational, retrieval, synthesis
- depends_on lists IDs of tasks that must complete before this task starts
- Identify execution ordering from dependencies
- Be precise — list only tasks that are actually needed

Return a JSON object:
{{
  "tasks": [
    {{
      "id": "task_1",
      "task": "<description>",
      "type": "<type>",
      "depends_on": [],
      "estimated_tokens": <int>,
      "requires_tools": ["web_search"]
    }}
  ],
  "reasoning": "<why you chose this decomposition>",
  "total_estimated_tokens": <int>
}}

Return ONLY the JSON object."""

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


class DecompositionAgent(BaseAgent):

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=settings.gemini_temperature,
            )
        return self._llm

    @property
    def agent_id(self) -> str:
        return AgentID.DECOMPOSITION

    async def _run(self, context: SharedContext) -> SharedContext:
        llm = self._get_llm()
        prompt = _PROMPT.format(query=context.user_query)

        try:
            response = await llm.ainvoke(prompt)
            clean_json = _extract_json_block(str(response.content))
            data: dict[str, Any] = json.loads(clean_json)
            task_graph = TaskGraph(
                tasks=[SubTask(**t) for t in data.get("tasks", [])],
                reasoning=str(data.get("reasoning", "")),
                total_estimated_tokens=int(data.get("total_estimated_tokens", 0)),
            )
            raw = clean_json
        except Exception as e:
            # Minimal fallback: single retrieval + synthesis task
            task_graph = TaskGraph(
                tasks=[
                    SubTask(
                        id="task_1",
                        task=context.user_query,
                        type="retrieval",
                        depends_on=[],
                    ),
                    SubTask(
                        id="task_2",
                        task="Synthesize findings",
                        type="synthesis",
                        depends_on=["task_1"],
                    ),
                ],
                reasoning=f"Fallback decomposition due to error: {e}",
            )
            raw = str(task_graph.model_dump())

        token_count = count_tokens(raw)
        output = AgentOutput(
            agent_id=AgentID.DECOMPOSITION,
            raw_output=raw,
            structured_output=task_graph.model_dump(),
            confidence=0.85,
            token_count=token_count,
        )

        return (
            context
            .set_agent_output(output)
            .model_copy(update={"task_graph": task_graph.model_dump()})
        )
