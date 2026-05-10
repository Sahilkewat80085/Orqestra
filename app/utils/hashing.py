"""
app/utils/hashing.py
════════════════════
SHA-256 hashing utilities for content-addressed audit trails.
Used to generate input_hash and output_hash for all tool calls and agent outputs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_content(content: Any) -> str:
    """
    Returns a 64-character hex SHA-256 hash of the serialized content.
    Handles dicts, lists, strings, and Pydantic models.
    """
    if hasattr(content, "model_dump"):
        # Pydantic model
        serialized = json.dumps(content.model_dump(mode="json"), sort_keys=True)
    elif isinstance(content, (dict, list)):
        serialized = json.dumps(content, sort_keys=True, default=str)
    else:
        serialized = str(content)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_string(text: str) -> str:
    """Hash a plain string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(content: Any, length: int = 8) -> str:
    """Return a truncated hash for human-readable IDs."""
    return hash_content(content)[:length]
