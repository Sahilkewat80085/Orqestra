"""
app/streaming/publisher.py
═══════════════════════════
Publishes StreamEvents to Redis pub/sub channels.
Channel key: orqestra:stream:{query_id}
"""
from __future__ import annotations

import json

import redis.asyncio as aioredis

from app.config import settings
from app.streaming.events import StreamEvent

_CHANNEL_PREFIX = "orqestra:stream:"
_EXPIRY_SECS = 3600  # Events expire after 1 hour


class EventPublisher:
    """
    Publishes typed StreamEvent objects to Redis.
    Each query gets its own channel so SSE subscribers only receive
    events for their specific query.
    """

    def __init__(self):
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    def _channel(self, query_id: str) -> str:
        return f"{_CHANNEL_PREFIX}{query_id}"

    async def publish(self, event: StreamEvent) -> None:
        """Publish an event to the query's Redis channel."""
        if not self._redis:
            return  # Silently skip if Redis not connected (e.g. tests)
        payload = event.model_dump_json()
        await self._redis.publish(self._channel(event.query_id), payload)

        # Also store in a list for trace reconstruction
        list_key = f"orqestra:events:{event.query_id}"
        await self._redis.rpush(list_key, payload)
        await self._redis.expire(list_key, _EXPIRY_SECS)

    async def get_all_events(self, query_id: str) -> list[StreamEvent]:
        """Retrieve all stored events for a query (for trace reconstruction)."""
        if not self._redis:
            return []
        list_key = f"orqestra:events:{query_id}"
        raw_events = await self._redis.lrange(list_key, 0, -1)
        return [StreamEvent.model_validate_json(e) for e in raw_events]


# Module-level singleton
publisher = EventPublisher()
