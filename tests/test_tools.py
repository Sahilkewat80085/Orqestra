"""tests/test_tools.py — Tool failure contract unit tests."""
from __future__ import annotations

import pytest
import asyncio

from app.tools.web_search import WebSearchTool
from app.tools.python_sandbox import PythonSandboxTool
from app.schemas.tools import WebSearchInput, PythonSandboxInput, ToolStatus, RetryEligibility


@pytest.mark.asyncio
async def test_web_search_mock_returns_results():
    tool = WebSearchTool()
    inp = WebSearchInput(query="retrieval augmented generation", max_results=3)
    result, record = await tool.run(inp, agent_id="retrieval", query_id="test-123")
    assert result.status == ToolStatus.SUCCESS
    assert len(result.results) > 0
    assert record.accepted is True


@pytest.mark.asyncio
async def test_web_search_empty_query_validation():
    tool = WebSearchTool()
    inp = WebSearchInput(query="   ", max_results=3)
    result, record = await tool.run(inp, agent_id="retrieval", query_id="test-124")
    assert result.status == ToolStatus.VALIDATION_ERROR
    assert record.accepted is False


@pytest.mark.asyncio
async def test_python_sandbox_success():
    tool = PythonSandboxTool()
    inp = PythonSandboxInput(code='print("hello world")', timeout_secs=5)
    result, record = await tool.run(inp, agent_id="retrieval", query_id="test-125")
    assert result.status == ToolStatus.SUCCESS
    assert "hello world" in result.stdout


@pytest.mark.asyncio
async def test_python_sandbox_blocked_import():
    tool = PythonSandboxTool()
    inp = PythonSandboxInput(code="import os; print(os.getcwd())", timeout_secs=5)
    result, record = await tool.run(inp, agent_id="retrieval", query_id="test-126")
    assert result.status == ToolStatus.VALIDATION_ERROR
    assert record.accepted is False


@pytest.mark.asyncio
async def test_python_sandbox_syntax_error():
    tool = PythonSandboxTool()
    inp = PythonSandboxInput(code="def broken(", timeout_secs=5)
    result, record = await tool.run(inp, agent_id="retrieval", query_id="test-127")
    assert result.status == ToolStatus.VALIDATION_ERROR


def test_tool_call_record_has_hash():
    """Tool records must include hashes for audit trails."""
    import asyncio
    tool = WebSearchTool()
    inp = WebSearchInput(query="test")
    result, record = asyncio.get_event_loop().run_until_complete(
        tool.run(inp, agent_id="retrieval", query_id="test-128")
    )
    assert record.call_id is not None
    assert record.timestamp is not None
