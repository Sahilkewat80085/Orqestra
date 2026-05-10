"""
app/orchestrator/retry_policy.py
══════════════════════════════════
Typed retry policy definitions.

Maps FailureType → RetryStrategy so the orchestrator can choose
the right fallback without hardcoded conditional chains.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from app.schemas.context import FailureType


class RetryStrategy(str, Enum):
    """What the orchestrator does when a tool or agent fails."""
    IMMEDIATE_RETRY = "immediate_retry"       # Retry at once, same input
    BACKOFF_RETRY = "backoff_retry"           # Retry after 2^n seconds
    MODIFY_AND_RETRY = "modify_and_retry"     # Broaden/rewrite input, then retry
    FALLBACK_AGENT = "fallback_agent"         # Route to a different agent
    SKIP = "skip"                             # Mark as non-critical, continue
    ABORT = "abort"                           # Fatal — stop the pipeline


@dataclass
class RetryDecision:
    strategy: RetryStrategy
    max_attempts: int
    backoff_base_secs: float = 1.0
    modify_input: Optional[Callable] = None   # Optional input transformer
    fallback_agent: Optional[str] = None      # Used with FALLBACK_AGENT


# ── Default Failure → Strategy mapping ───────────────────────────────────────
# The orchestrator uses this table to select a strategy without hardcoded logic.

DEFAULT_POLICY: Dict[FailureType, RetryDecision] = {
    FailureType.TIMEOUT: RetryDecision(
        strategy=RetryStrategy.BACKOFF_RETRY,
        max_attempts=2,
        backoff_base_secs=2.0,
    ),
    FailureType.MALFORMED: RetryDecision(
        strategy=RetryStrategy.MODIFY_AND_RETRY,
        max_attempts=1,
        backoff_base_secs=0.0,
    ),
    FailureType.EMPTY: RetryDecision(
        strategy=RetryStrategy.MODIFY_AND_RETRY,
        max_attempts=2,
        backoff_base_secs=0.0,
    ),
    FailureType.RATE_LIMIT: RetryDecision(
        strategy=RetryStrategy.BACKOFF_RETRY,
        max_attempts=2,
        backoff_base_secs=5.0,
    ),
    FailureType.VALIDATION: RetryDecision(
        strategy=RetryStrategy.SKIP,
        max_attempts=0,
    ),
    FailureType.UNKNOWN: RetryDecision(
        strategy=RetryStrategy.FALLBACK_AGENT,
        max_attempts=1,
        fallback_agent="synthesis",  # Synthesize from what we have
    ),
}


class RetryPolicyEngine:
    """
    Stateless engine that determines retry decisions given a failure type
    and current attempt count.

    Usage:
        engine = RetryPolicyEngine()
        decision = engine.decide(FailureType.TIMEOUT, attempt=1)
        if decision.strategy == RetryStrategy.BACKOFF_RETRY:
            await asyncio.sleep(decision.backoff_base_secs ** attempt)
    """

    def __init__(self, custom_policy: Optional[Dict[FailureType, RetryDecision]] = None):
        self._policy = custom_policy or DEFAULT_POLICY

    def decide(
        self, failure_type: FailureType, attempt: int
    ) -> Optional[RetryDecision]:
        """
        Returns a RetryDecision if another attempt should be made,
        or None if the failure is terminal.

        Args:
            failure_type: The type of failure that occurred
            attempt: Current attempt number (0-indexed)
        """
        decision = self._policy.get(failure_type)
        if decision is None:
            return RetryDecision(strategy=RetryStrategy.SKIP, max_attempts=0)

        if attempt >= decision.max_attempts:
            return None  # Terminal — no more retries

        return decision

    def should_retry(self, failure_type: FailureType, attempt: int) -> bool:
        return self.decide(failure_type, attempt) is not None

    def get_backoff(self, failure_type: FailureType, attempt: int) -> float:
        """Calculate backoff delay in seconds for this failure + attempt."""
        decision = self._policy.get(failure_type)
        if not decision:
            return 0.0
        return decision.backoff_base_secs * (2 ** attempt)
