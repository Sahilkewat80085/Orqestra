"""
app/utils/token_counter.py
══════════════════════════
Token counting using tiktoken.
Used by the ContextBudgetManager to enforce per-agent token budgets.
"""
from __future__ import annotations

import functools
from typing import List

import tiktoken


@functools.lru_cache(maxsize=4)
def _get_encoder(model: str = "gpt-4") -> tiktoken.Encoding:
    """
    Cache the tiktoken encoder. We use the gpt-4 encoder as a close approximation
    for Gemini token counts (both use similar BPE tokenization).
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a string."""
    if not text:
        return 0
    enc = _get_encoder(model)
    return len(enc.encode(text))


def count_messages_tokens(messages: List[dict], model: str = "gpt-4") -> int:
    """
    Estimate token count for a list of chat messages.
    Adds ~4 tokens per message for role/formatting overhead.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        for value in msg.values():
            total += count_tokens(str(value), model)
    total += 2  # reply priming
    return total


def estimate_remaining(allocated: int, used: int) -> int:
    """Return remaining token budget, floored at 0."""
    return max(0, allocated - used)
