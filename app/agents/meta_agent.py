"""
app/agents/meta_agent.py
══════════════════════════
MetaPromptAgent: Analyzes failed evals and proposes prompt rewrites.

IMPORTANT RULES:
  - Rewrites are NEVER auto-applied
  - Human approval required via POST /api/v1/prompts/approve
  - All proposals persisted with status="pending"
  - After approval, ONLY failed cases are re-run
  - All changes are fully auditable
"""
from __future__ import annotations

import json
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import BaseAgent
from app.config import settings
from app.schemas.agents import MetaAgentOutput, PromptRewriteDiff
from app.schemas.context import AgentID, AgentOutput, SharedContext
from app.schemas.evals import EvalCaseResult
from app.utils.token_counter import count_tokens

_PROMPT = """You are a meta-learning agent that improves AI agent prompts.

Analyze these failed evaluation cases and propose targeted prompt rewrites.

Failed cases:
{failed_cases}

Current agent prompts (abbreviated):
{current_prompts}

For each failure, identify:
1. Which agent's prompt caused the failure
2. What was wrong with the prompt
3. A specific, improved rewrite

Return a JSON object:
{{
  "analyzed_failures": ["case_id_1", "case_id_2"],
  "root_cause_analysis": "<what common pattern caused these failures>",
  "proposed_rewrites": [
    {{
      "target_agent_id": "<agent_id>",
      "original_prompt_version": "<version>",
      "proposed_prompt": "<full new prompt text>",
      "reasoning": "<why this rewrite will fix the failure>",
      "expected_improvement": "<what metrics should improve>",
      "failed_case_ids": ["case_id_1"],
      "diff_summary": "<human-readable description of what changed>"
    }}
  ],
  "priority_order": ["<agent_id_1>", "<agent_id_2>"],
  "reasoning": "<overall analysis>"
}}

Return ONLY the JSON object."""


class MetaPromptAgent(BaseAgent):
    """
    Analyzes eval failures and proposes prompt rewrites for human review.
    Never auto-applies changes — all rewrites require explicit approval.
    """

    def __init__(self, failed_cases: List[EvalCaseResult], current_prompts: dict):
        self._failed_cases = failed_cases
        self._current_prompts = current_prompts

    @property
    def agent_id(self) -> str:
        return AgentID.META

    async def _run(self, context: SharedContext) -> SharedContext:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )

        # Format failed cases for analysis
        cases_text = json.dumps(
            [
                {
                    "case_id": c.case_id,
                    "query": c.query,
                    "category": c.category,
                    "final_answer": (c.final_answer or "")[:300],
                    "score": c.score.total_score if c.score else 0.0,
                    "error": c.error,
                }
                for c in self._failed_cases
            ],
            indent=2,
        )

        prompts_text = json.dumps(
            {k: v[:200] for k, v in self._current_prompts.items()}, indent=2
        )

        prompt = _PROMPT.format(
            failed_cases=cases_text,
            current_prompts=prompts_text,
        )

        try:
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)

            meta_output = MetaAgentOutput(
                analyzed_failures=data.get("analyzed_failures", []),
                root_cause_analysis=data.get("root_cause_analysis", ""),
                proposed_rewrites=[
                    PromptRewriteDiff(**r)
                    for r in data.get("proposed_rewrites", [])
                ],
                priority_order=data.get("priority_order", []),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            meta_output = MetaAgentOutput(
                analyzed_failures=[c.case_id for c in self._failed_cases],
                root_cause_analysis=f"Meta-analysis failed: {e}",
                proposed_rewrites=[],
                priority_order=[],
                reasoning=str(e),
            )

        token_count = count_tokens(str(meta_output.model_dump()))
        output = AgentOutput(
            agent_id=AgentID.META,
            raw_output=meta_output.root_cause_analysis,
            structured_output=meta_output.model_dump(),
            confidence=0.7,
            token_count=token_count,
        )

        return context.set_agent_output(output)
