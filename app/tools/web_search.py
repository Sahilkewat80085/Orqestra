"""
app/tools/web_search.py
═══════════════════════
Web Search Tool.

Failure contract:
  - timeout (5s default): retry eligible with backoff
  - malformed response: retry eligible (once)
  - empty results: retry eligible with broadened query
  - rate limit: NOT retry eligible (escalate to orchestrator)

By default runs in mock/stub mode (WEB_SEARCH_MOCK=true).
Set WEB_SEARCH_MOCK=false and provide WEB_SEARCH_API_KEY for real search.
"""
from __future__ import annotations

import asyncio
from typing import List

import httpx

from app.config import settings
from app.schemas.context import ToolName
from app.schemas.tools import (
    RetryEligibility,
    ToolFailureContract,
    ToolStatus,
    WebSearchInput,
    WebSearchOutput,
    WebSearchResult,
)
from app.tools.base import BaseTool


# ── Stub data ─────────────────────────────────────────────────────────────────
# Realistic mock results used when WEB_SEARCH_MOCK=true.
# In production, replace with Tavily/SerpAPI integration.

_MOCK_RESULTS: List[WebSearchResult] = [
    WebSearchResult(
        title="Introduction to Large Language Models",
        url="https://arxiv.org/abs/2307.06435",
        snippet="Large language models (LLMs) are neural networks trained on vast text corpora...",
        relevance_score=0.93,
        published_date="2023-07-12",
    ),
    WebSearchResult(
        title="Multi-Agent Systems: Survey and Applications",
        url="https://arxiv.org/abs/2402.01680",
        snippet="Multi-agent systems coordinate multiple autonomous agents through shared state...",
        relevance_score=0.87,
        published_date="2024-02-03",
    ),
    WebSearchResult(
        title="LangGraph: Building Stateful Agentic Applications",
        url="https://blog.langchain.dev/langgraph",
        snippet="LangGraph enables building cyclical agent workflows with explicit state management...",
        relevance_score=0.82,
        published_date="2024-01-17",
    ),
    WebSearchResult(
        title="Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
        url="https://arxiv.org/abs/2005.11401",
        snippet="RAG combines parametric memory of pretrained models with non-parametric retrieval...",
        relevance_score=0.79,
        published_date="2020-05-22",
    ),
    WebSearchResult(
        title="Prompt Engineering: Techniques for Reliable LLM Outputs",
        url="https://www.promptingguide.ai",
        snippet="Chain-of-thought, few-shot, and self-consistency prompting improve LLM accuracy...",
        relevance_score=0.74,
        published_date="2023-11-01",
    ),
]


class WebSearchTool(BaseTool):
    """
    Searches the web and returns structured results with relevance scores.
    Mock mode returns curated realistic results filtered by query terms.
    """

    @property
    def tool_name(self) -> ToolName:
        return ToolName.WEB_SEARCH

    @property
    def failure_contract(self) -> ToolFailureContract:
        return ToolFailureContract(
            tool_name="web_search",
            timeout_secs=settings.web_search_timeout_secs,
            timeout_retry_eligible=True,
            malformed_input_retry_eligible=False,
            empty_result_retry_eligible=True,
            rate_limit_retry_eligible=False,
            max_retries=settings.tool_max_retries,
            fallback_strategy="skip",
        )

    def _validate_input(self, input_data: WebSearchInput) -> str | None:
        if not input_data.query or not input_data.query.strip():
            return "Query string must not be empty"
        if len(input_data.query) > 512:
            return "Query exceeds maximum length of 512 characters"
        return None

    async def _execute(self, input_data: WebSearchInput) -> WebSearchOutput:
        if settings.web_search_mock:
            return await self._mock_search(input_data)
        return await self._real_search(input_data)

    async def _mock_search(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Return stub results filtered/scored against the query."""
        await asyncio.sleep(0.05)  # Simulate network latency

        query_lower = input_data.query.lower()
        scored = []
        for result in _MOCK_RESULTS:
            # Simple keyword overlap scoring
            overlap = sum(
                1 for word in query_lower.split()
                if word in result.snippet.lower() or word in result.title.lower()
            )
            adjusted_score = min(1.0, result.relevance_score + overlap * 0.05)
            scored.append(result.model_copy(update={"relevance_score": adjusted_score}))

        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        results = scored[: input_data.max_results]

        if not results:
            return WebSearchOutput(
                tool_name="web_search",
                status=ToolStatus.EMPTY_RESULT,
                retry_eligible=RetryEligibility.ELIGIBLE,
                error_message="No results found for query",
                query_used=input_data.query,
            )

        return WebSearchOutput(
            tool_name="web_search",
            status=ToolStatus.SUCCESS,
            results=results,
            query_used=input_data.query,
            total_found=len(results),
            raw_output=[r.model_dump() for r in results],
        )

    async def _real_search(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Call real search API (Tavily). Requires WEB_SEARCH_API_KEY."""
        if not settings.web_search_api_key:
            return WebSearchOutput(
                tool_name="web_search",
                status=ToolStatus.VALIDATION_ERROR,
                retry_eligible=RetryEligibility.NOT_ELIGIBLE,
                error_message="WEB_SEARCH_API_KEY not configured",
            )

        async with httpx.AsyncClient(timeout=settings.web_search_timeout_secs) as client:
            try:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.web_search_api_key,
                        "query": input_data.query,
                        "max_results": input_data.max_results,
                    },
                )
                response.raise_for_status()
                data = response.json()
            except httpx.TimeoutException:
                raise asyncio.TimeoutError()
            except Exception as e:
                return WebSearchOutput(
                    tool_name="web_search",
                    status=ToolStatus.MALFORMED_OUTPUT,
                    retry_eligible=RetryEligibility.ELIGIBLE,
                    error_message=str(e),
                )

        results = [
            WebSearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                relevance_score=r.get("score", 0.5),
            )
            for r in data.get("results", [])
        ]

        return WebSearchOutput(
            tool_name="web_search",
            status=ToolStatus.SUCCESS,
            results=results,
            query_used=input_data.query,
            total_found=len(results),
            raw_output=data,
        )
