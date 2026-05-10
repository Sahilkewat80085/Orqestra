"""
app/tools/base.py
═════════════════
Abstract BaseTool that all tools must extend.

Enforces:
  - Input validation before execution
  - Automatic retry with configurable policy
  - Structured logging for every call (latency, retry count, hashes)
  - Explicit failure contracts (timeout, malformed, empty, rate_limit)
  - Acceptance/rejection decisions by the calling agent
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from app.logging.logger import get_logger
from app.schemas.context import FailureType, ToolCallRecord, ToolName
from app.schemas.tools import RetryEligibility, ToolFailureContract, ToolResult, ToolStatus
from app.utils.hashing import hash_content

log = get_logger("tools.base")


class BaseTool(ABC):
    """
    Abstract base for all Orqestra tools.

    Subclasses must implement:
      - `failure_contract` property
      - `_execute(input_data)` — the actual tool logic

    The base class handles:
      - Input validation
      - Retry loop (up to max_retries, different strategies per failure type)
      - Structured logging with latency and content hashes
      - ToolCallRecord creation for SharedContext audit trail
    """

    @property
    @abstractmethod
    def tool_name(self) -> ToolName:
        ...

    @property
    @abstractmethod
    def failure_contract(self) -> ToolFailureContract:
        ...

    @abstractmethod
    async def _execute(self, input_data: Any) -> ToolResult:
        """Core execution logic — does NOT handle retries."""
        ...

    def _validate_input(self, input_data: Any) -> Optional[str]:
        """
        Validate input before execution.
        Return an error string if invalid, None if valid.
        Override in subclasses for tool-specific validation.
        """
        return None

    async def run(
        self,
        input_data: Any,
        agent_id: str,
        query_id: str,
    ) -> tuple[ToolResult, ToolCallRecord]:
        """
        Public entry point. Handles validation, retry loop, and audit logging.

        Returns:
            (result, call_record) — result is the tool output,
            call_record is the immutable audit entry for SharedContext.
        """
        input_hash = hash_content(input_data)
        max_retries = self.failure_contract.max_retries
        last_result: Optional[ToolResult] = None
        retry_count = 0

        # ── Input validation ──────────────────────────────────────────────────
        validation_error = self._validate_input(input_data)
        if validation_error:
            result = ToolResult(
                tool_name=self.tool_name,
                status=ToolStatus.VALIDATION_ERROR,
                retry_eligible=RetryEligibility.NOT_ELIGIBLE,
                error_message=validation_error,
            )
            record = self._build_record(
                result, input_data, agent_id, query_id, input_hash, retry_count=0
            )
            log.warning(
                "tool_validation_failed",
                tool=self.tool_name,
                agent_id=agent_id,
                error=validation_error,
            )
            return result, record

        # ── Retry loop ────────────────────────────────────────────────────────
        while retry_count <= max_retries:
            start = time.perf_counter()
            try:
                result = await self._execute(input_data)
            except asyncio.TimeoutError:
                result = ToolResult(
                    tool_name=self.tool_name,
                    status=ToolStatus.TIMEOUT,
                    retry_eligible=RetryEligibility.ELIGIBLE_BACKOFF
                    if self.failure_contract.timeout_retry_eligible
                    else RetryEligibility.NOT_ELIGIBLE,
                    error_message=f"Tool timed out after {self.failure_contract.timeout_secs}s",
                )
            except Exception as exc:
                result = ToolResult(
                    tool_name=self.tool_name,
                    status=ToolStatus.EXECUTION_ERROR,
                    retry_eligible=RetryEligibility.NOT_ELIGIBLE,
                    error_message=str(exc),
                )

            result = result.model_copy(
                update={
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "retry_count": retry_count,
                }
            )
            last_result = result

            # Success — break out immediately
            if result.status == ToolStatus.SUCCESS:
                break

            # Determine if we should retry
            should_retry = (
                retry_count < max_retries
                and result.retry_eligible
                in (RetryEligibility.ELIGIBLE, RetryEligibility.ELIGIBLE_BACKOFF)
            )

            if not should_retry:
                break

            # Backoff before retry
            backoff = 2 ** retry_count  # 1s, 2s
            log.warning(
                "tool_retry",
                tool=self.tool_name,
                agent_id=agent_id,
                attempt=retry_count + 1,
                status=result.status,
                backoff_secs=backoff,
            )
            await asyncio.sleep(backoff)
            retry_count += 1

        output_hash = hash_content(last_result.raw_output if last_result else None)
        record = self._build_record(
            last_result, input_data, agent_id, query_id, input_hash,
            retry_count=retry_count, output_hash=output_hash
        )

        log.info(
            "tool_completed",
            tool=self.tool_name,
            agent_id=agent_id,
            status=last_result.status,
            latency_ms=round(last_result.latency_ms, 2),
            retry_count=retry_count,
            input_hash=input_hash[:8],
            output_hash=output_hash[:8],
        )

        return last_result, record

    def _build_record(
        self,
        result: ToolResult,
        input_data: Any,
        agent_id: str,
        query_id: str,
        input_hash: str,
        retry_count: int = 0,
        output_hash: str = "",
    ) -> ToolCallRecord:
        """Build the immutable audit record for SharedContext."""
        failure_map = {
            ToolStatus.TIMEOUT: FailureType.TIMEOUT,
            ToolStatus.MALFORMED_INPUT: FailureType.MALFORMED,
            ToolStatus.MALFORMED_OUTPUT: FailureType.MALFORMED,
            ToolStatus.EMPTY_RESULT: FailureType.EMPTY,
            ToolStatus.VALIDATION_ERROR: FailureType.VALIDATION,
            ToolStatus.RATE_LIMITED: FailureType.RATE_LIMIT,
        }
        return ToolCallRecord(
            tool_name=self.tool_name,
            agent_id=agent_id,
            input=input_data if isinstance(input_data, dict)
                  else (input_data.model_dump() if hasattr(input_data, "model_dump") else {"raw": str(input_data)}),
            output=result.model_dump() if result else None,
            latency_ms=result.latency_ms if result else 0.0,
            retry_count=retry_count,
            accepted=result.status == ToolStatus.SUCCESS if result else False,
            failure_type=failure_map.get(result.status) if result else FailureType.UNKNOWN,
        )
