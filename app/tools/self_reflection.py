"""
app/tools/self_reflection.py
════════════════════════════
Self-Reflection Tool.

Allows agents to re-read previous agent outputs from SharedContext,
identify contradictions, and compare reasoning states.

Failure contract:
  - no outputs to compare: NOT retry eligible
  - LLM parse error: retry eligible once
"""
from __future__ import annotations

import json

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.schemas.context import SharedContext, ToolName
from app.schemas.tools import (
    ContradictionFound, RetryEligibility, SelfReflectionInput,
    SelfReflectionOutput, ToolFailureContract, ToolStatus,
)
from app.tools.base import BaseTool


class SelfReflectionTool(BaseTool):
    """
    Reads previous agent outputs from SharedContext and uses the LLM
    to identify contradictions, reasoning gaps, and improvement opportunities.
    """

    def __init__(self, context: SharedContext):
        self._context = context

    @property
    def tool_name(self) -> ToolName:
        return ToolName.SELF_REFLECTION

    @property
    def failure_contract(self) -> ToolFailureContract:
        return ToolFailureContract(
            tool_name="self_reflection",
            timeout_secs=30,
            timeout_retry_eligible=True,
            malformed_input_retry_eligible=False,
            empty_result_retry_eligible=False,
            rate_limit_retry_eligible=False,
            max_retries=1,
            fallback_strategy="skip",
        )

    def _validate_input(self, input_data: SelfReflectionInput) -> str | None:
        available = list(self._context.agent_outputs.keys())
        for agent_id in input_data.focus_agent_ids:
            if agent_id not in available:
                return (
                    f"Agent '{agent_id}' has no output in context. "
                    f"Available: {available}"
                )
        if len(input_data.focus_agent_ids) < 2:
            return "At least two agent IDs required for contradiction detection"
        return None

    async def _execute(self, input_data: SelfReflectionInput) -> SelfReflectionOutput:
        # Gather the outputs to compare
        outputs_to_compare = {
            agent_id: self._context.agent_outputs[agent_id].raw_output
            for agent_id in input_data.focus_agent_ids
            if agent_id in self._context.agent_outputs
        }

        outputs_text = "\n\n".join(
            f"[{agent_id}]:\n{output}"
            for agent_id, output in outputs_to_compare.items()
        )

        reflection_focus = (
            input_data.reflection_prompt
            or "Identify contradictions and reasoning gaps between these outputs."
        )

        prompt = f"""You are a critical reasoning auditor. Compare these agent outputs carefully.

{outputs_text}

Task: {reflection_focus}

Return a JSON object with this exact structure:
{{
  "contradictions": [
    {{
      "agent_a": "<agent_id>",
      "agent_b": "<agent_id>",
      "claim_a": "<exact claim from agent_a>",
      "claim_b": "<exact claim from agent_b>",
      "severity": "low|medium|high",
      "suggested_resolution": "<how to resolve>"
    }}
  ],
  "reasoning_gaps": ["<gap 1>", "<gap 2>"],
  "consistency_score": <0.0-1.0>,
  "reflection_summary": "<one paragraph summary>",
  "recommended_actions": ["<action 1>"]
}}

Return ONLY the JSON object, no markdown."""

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )

        try:
            response = await llm.ainvoke(prompt)
            data = json.loads(response.content.strip())
        except (json.JSONDecodeError, Exception) as e:
            return SelfReflectionOutput(
                tool_name="self_reflection",
                status=ToolStatus.MALFORMED_OUTPUT,
                retry_eligible=RetryEligibility.ELIGIBLE,
                error_message=f"Failed to parse reflection output: {e}",
            )

        contradictions = [
            ContradictionFound(**c) for c in data.get("contradictions", [])
        ]

        return SelfReflectionOutput(
            tool_name="self_reflection",
            status=ToolStatus.SUCCESS,
            contradictions=contradictions,
            reasoning_gaps=data.get("reasoning_gaps", []),
            consistency_score=float(data.get("consistency_score", 1.0)),
            reflection_summary=data.get("reflection_summary", ""),
            recommended_actions=data.get("recommended_actions", []),
            raw_output=data,
        )
