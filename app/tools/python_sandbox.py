"""
app/tools/python_sandbox.py
═══════════════════════════
Python Sandbox Tool.

Runs Python code in a separate subprocess with resource limits (timeout)
and restricted imports (AST validation).
"""
from __future__ import annotations

import ast
import asyncio
import sys
import time
from typing import Any

from app.config import settings
from app.schemas.context import ToolName
from app.schemas.tools import (
    PythonSandboxInput,
    PythonSandboxOutput,
    RetryEligibility,
    ToolFailureContract,
    ToolStatus,
)
from app.tools.base import BaseTool


def _validate_code_imports(code: str, allowed: list[str]) -> str | None:
    """Parse the code and traverse AST to ensure no blocked imports or syntax errors."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in allowed:
                    return f"Import of module '{top_level}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level = node.module.split(".")[0]
                if top_level not in allowed:
                    return f"Import of module '{top_level}' is not allowed"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                return "Dynamic imports via __import__ are not allowed"
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                return "Imports via importlib are not allowed"
    return None


class PythonSandboxTool(BaseTool):
    """
    Asynchronously executes python code in a separate subprocess.
    Before execution, checks the code for syntax errors and banned imports.
    """

    @property
    def tool_name(self) -> ToolName:
        return ToolName.PYTHON_SANDBOX

    @property
    def failure_contract(self) -> ToolFailureContract:
        return ToolFailureContract(
            tool_name="python_sandbox",
            timeout_secs=settings.python_sandbox_timeout_secs,
            timeout_retry_eligible=False,
            malformed_input_retry_eligible=False,
            empty_result_retry_eligible=False,
            rate_limit_retry_eligible=False,
            max_retries=0,
            fallback_strategy="skip",
        )

    def _validate_input(self, input_data: PythonSandboxInput) -> str | None:
        if not input_data.code or not input_data.code.strip():
            return "Code must not be empty"

        # Check syntax and banned imports
        import_error = _validate_code_imports(input_data.code, input_data.allowed_imports)
        if import_error:
            return import_error

        return None

    async def _execute(self, input_data: PythonSandboxInput) -> PythonSandboxOutput:
        start_time = time.perf_counter()

        # Run code asynchronously in a subprocess
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            input_data.code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=input_data.timeout_secs,
            )
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        status = ToolStatus.SUCCESS if exit_code == 0 else ToolStatus.EXECUTION_ERROR

        return PythonSandboxOutput(
            tool_name="python_sandbox",
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=latency_ms,
            raw_output={"stdout": stdout, "stderr": stderr, "exit_code": exit_code},
        )
