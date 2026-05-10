"""
app/logging/logger.py
═════════════════════
Structured JSON logging via structlog.

Every log event includes:
  - timestamp (ISO 8601)
  - agent_id
  - event_type
  - latency_ms
  - token_count
  - input_hash
  - output_hash
  - policy_violations flag

Usage:
    from app.logging.logger import get_logger
    log = get_logger("orchestrator")
    log.info("agent_started", query_id="...", budget=4096)
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_severity(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Map structlog level names to Cloud Logging severity strings."""
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
    }
    event_dict["severity"] = level_map.get(method, "DEFAULT")
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for structured JSON output.
    Call once at application startup.
    """
    # Configure stdlib logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        _add_severity,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """
    Returns a structlog logger pre-bound with the given context values.

    Args:
        name:           Component name (e.g. "orchestrator", "tool.web_search")
        **initial_values: Key-value pairs permanently bound to all log events
                          from this logger instance.
    """
    return structlog.get_logger(name).bind(**initial_values)


class AgentLogger:
    """
    Convenience wrapper that auto-binds agent_id and query_id.
    Use this inside every agent to ensure consistent log fields.
    """

    def __init__(self, agent_id: str, query_id: str):
        self._log = get_logger(f"agent.{agent_id}", agent_id=agent_id, query_id=query_id)

    def started(self, **kwargs: Any) -> None:
        self._log.info("agent_started", event_type="AGENT_START", **kwargs)

    def completed(self, latency_ms: float, token_count: int, **kwargs: Any) -> None:
        self._log.info(
            "agent_completed",
            event_type="AGENT_COMPLETE",
            latency_ms=latency_ms,
            token_count=token_count,
            **kwargs,
        )

    def tool_called(
        self, tool_name: str, input_hash: str, output_hash: str,
        latency_ms: float, retry_count: int, accepted: bool, **kwargs: Any
    ) -> None:
        self._log.info(
            "tool_called",
            event_type="TOOL_CALL",
            tool_name=tool_name,
            input_hash=input_hash,
            output_hash=output_hash,
            latency_ms=latency_ms,
            retry_count=retry_count,
            accepted=accepted,
            **kwargs,
        )

    def retry_triggered(self, tool_name: str, attempt: int, reason: str, **kwargs: Any) -> None:
        self._log.warning(
            "retry_triggered",
            event_type="RETRY",
            tool_name=tool_name,
            attempt=attempt,
            reason=reason,
            **kwargs,
        )

    def policy_violation(self, violation_type: str, detail: str, **kwargs: Any) -> None:
        self._log.error(
            "policy_violation",
            event_type="POLICY_VIOLATION",
            violation_type=violation_type,
            detail=detail,
            **kwargs,
        )

    def error(self, message: str, **kwargs: Any) -> None:
        self._log.error("agent_error", event_type="AGENT_ERROR", message=message, **kwargs)
