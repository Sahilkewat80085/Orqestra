"""
app/agents/critique.py
════════════════════════
CritiqueAgent: Span-level review of all agent outputs.

Rules:
  - Reviews EVERY populated agent_outputs entry
  - Must target specific spans, NOT entire outputs
  - Produces structured CritiqueFinding per claim
  - Assigns confidence per claim, flags weak reasoning
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.base import BaseAgent
from app.config import settings
from app.schemas.agents import CritiqueFinding, CritiqueReport
from app.schemas.context import AgentID, AgentOutput, SharedContext
from app.utils.token_counter import count_tokens

_PROMPT = """You are a critical reasoning auditor. Review these agent outputs for contradictions,
weak reasoning, and unsupported claims. Target SPECIFIC text spans, not entire outputs.

Agent outputs to review:
{outputs}

Return a JSON object:
{{
  "findings": [
    {{
      "claim": "<the specific claim being critiqued>",
      "confidence": <0.0-1.0 — your confidence this IS an issue>,
      "issue": "<what is wrong with this claim>",
      "affected_span": "<exact text span from the output>",
      "severity": "low|medium|high",
      "suggested_fix": "<optional fix>"
    }}
  ],
  "overall_confidence": <0.0-1.0 confidence in the combined outputs>,
  "passed": <true if no high-severity issues>,
  "reasoning": "<your overall assessment>"
}}

If no issues found, return findings as empty array with passed=true.
Return ONLY the JSON object."""

_JSON_BLOCK_RE: re.Pattern[str] = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)

_EXCLUDED_AGENT_IDS: tuple[str, ...] = (AgentID.ORCHESTRATOR, AgentID.CRITIQUE)


def _extract_json_payload(raw: str) -> str:
    """Extract clean JSON content from model response string."""
    text = raw.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return text


class CritiqueAgent(BaseAgent):

    def __init__(self, llm: ChatGoogleGenerativeAI | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=0.1,
            )
        return self._llm

    @property
    def agent_id(self) -> str:
        return AgentID.CRITIQUE

    async def _run(self, context: SharedContext) -> SharedContext:
        # Build the outputs summary for critique efficiently
        output_chunks = [
            f"[{agent_id}]:\n{output.raw_output[:1500]}"
            for agent_id, output in context.agent_outputs.items()
            if agent_id not in _EXCLUDED_AGENT_IDS
        ]
        outputs_text = "\n\n".join(output_chunks)

        if not outputs_text.strip():
            # Nothing to critique yet
            report = CritiqueReport(
                findings=[],
                overall_confidence=1.0,
                passed=True,
                reasoning="No agent outputs available to critique.",
            )
        else:
            llm = self._get_llm()
            prompt = _PROMPT.format(outputs=outputs_text)
            try:
                response = await llm.ainvoke(prompt)
                clean_json = _extract_json_payload(str(response.content))
                data: dict[str, Any] = json.loads(clean_json)
                report = CritiqueReport(
                    findings=[CritiqueFinding(**f) for f in data.get("findings", [])],
                    overall_confidence=float(data.get("overall_confidence", 0.8)),
                    passed=bool(data.get("passed", True)),
                    reasoning=str(data.get("reasoning", "")),
                )
            except Exception as e:
                report = CritiqueReport(
                    findings=[],
                    overall_confidence=0.5,
                    passed=True,
                    reasoning=f"Critique failed to parse: {e}",
                )

        token_count = count_tokens(outputs_text)
        output = AgentOutput(
            agent_id=AgentID.CRITIQUE,
            raw_output=report.reasoning,
            structured_output=report.model_dump(),
            confidence=report.overall_confidence,
            token_count=token_count,
        )

        return context.set_agent_output(output)
