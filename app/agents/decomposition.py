"""
app/agents/decomposition.py
═════════════════════════════
DecompositionAgent: Breaks ambiguous queries into typed sub-tasks
with explicit dependency graphs.

Output: TaskGraph stored in SharedContext.task_graph
"""
from __future__ import annotations

import json

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


class DecompositionAgent(BaseAgent):

    @property
    def agent_id(self) -> str:
        return AgentID.DECOMPOSITION

    async def _run(self, context: SharedContext) -> SharedContext:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.gemini_temperature,
        )

        prompt = _PROMPT.format(query=context.user_query)

        try:
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            task_graph = TaskGraph(
                tasks=[SubTask(**t) for t in data.get("tasks", [])],
                reasoning=data.get("reasoning", ""),
                total_estimated_tokens=data.get("total_estimated_tokens", 0),
            )
        except Exception as e:
            # Minimal fallback: single retrieval + synthesis task
            task_graph = TaskGraph(
                tasks=[
                    SubTask(id="task_1", task=context.user_query,
                            type="retrieval", depends_on=[]),
                    SubTask(id="task_2", task="Synthesize findings",
                            type="synthesis", depends_on=["task_1"]),
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
