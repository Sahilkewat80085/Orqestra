"""
app/streaming/subscriber.py
════════════════════════════
FastAPI SSE generator backed by Redis pub/sub.
Yields text/event-stream formatted events until PIPELINE_COMPLETE.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import redis.asyncio as aioredis

from app.config import settings
from app.streaming.events import EventType, StreamEvent

_CHANNEL_PREFIX = "orqestra:stream:"
_TIMEOUT_SECS = 120  # Max time to wait for pipeline completion


async def stream_query_events(query_id: str) -> AsyncGenerator[str, None]:
    """
    AsyncGenerator that subscribes to a query's Redis channel and
    yields SSE-formatted event strings.

    Usage in FastAPI:
        return StreamingResponse(
            stream_query_events(query_id),
            media_type="text/event-stream"
        )
    """
    redis = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    pubsub = redis.pubsub()
    channel = f"{_CHANNEL_PREFIX}{query_id}"

    await pubsub.subscribe(channel)

    # Send a connection acknowledgment
    yield _format_sse({"type": "connected", "query_id": query_id})

    try:
        elapsed = 0.0
        while elapsed < _TIMEOUT_SECS:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )

            if message and message["type"] == "message":
                try:
                    event = StreamEvent.model_validate_json(message["data"])
                    yield _format_sse(event.model_dump(mode="json"))

                    # Terminal event — close stream
                    if event.event_type in (
                        EventType.PIPELINE_COMPLETE,
                        EventType.PIPELINE_ERROR,
                        EventType.FINAL_ANSWER,
                    ):
                        break
                except Exception:
                    pass  # Malformed event — continue listening

            await asyncio.sleep(0.05)
            elapsed += 0.05

        if elapsed >= _TIMEOUT_SECS:
            yield _format_sse({"type": "timeout", "query_id": query_id})

    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        await redis.aclose()


def _format_sse(data: dict) -> str:
    """Format a dict as a proper SSE data field."""
    return f"data: {json.dumps(data)}\n\n"
