"""
app/streaming/publisher.py
═══════════════════════════
Publishes StreamEvents to Redis pub/sub channels.
Channel key: orqestra:stream:{query_id}
"""
from __future__ import annotations

from typing import Final

import redis.asyncio as aioredis

from app.config import settings
from app.logging.logger import get_logger
from app.streaming.events import StreamEvent

log = get_logger("streaming.publisher")

_CHANNEL_PREFIX: Final[str] = "orqestra:stream:"
_EVENTS_PREFIX: Final[str] = "orqestra:events:"
_EXPIRY_SECS: Final[int] = 3600  # Events expire after 1 hour


class EventPublisher:
    """
    Publishes typed StreamEvent objects to Redis.
    Each query gets its own channel so SSE subscribers only receive
    events for their specific query.
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    async def disconnect(self) -> None:
        if self._redis is not None:
            client, self._redis = self._redis, None
            try:
                await client.aclose()
            except Exception as e:
                log.warning("redis_disconnect_error", error=str(e))

    @staticmethod
    def _channel(query_id: str) -> str:
        return f"{_CHANNEL_PREFIX}{query_id}"

    async def publish(self, event: StreamEvent) -> None:
        """
        Publish an event to the query's Redis channel and append to trace history.
        Uses a pipeline to reduce Redis network round-trips from 3 to 1.
        """
        if self._redis is None:
            return  # Silently skip if Redis not connected (e.g. tests)

        payload = event.model_dump_json()
        channel = self._channel(event.query_id)
        list_key = f"{_EVENTS_PREFIX}{event.query_id}"

        try:
            pipe = self._redis.pipeline()
            pipe.publish(channel, payload)
            pipe.rpush(list_key, payload)
            pipe.expire(list_key, _EXPIRY_SECS)
            await pipe.execute()
        except Exception as e:
            log.error("redis_publish_error", query_id=event.query_id, error=str(e))

    async def get_all_events(self, query_id: str) -> list[StreamEvent]:
        """Retrieve all stored events for a query (for trace reconstruction)."""
        if self._redis is None:
            return []

        list_key = f"{_EVENTS_PREFIX}{query_id}"
        try:
            raw_events = await self._redis.lrange(list_key, 0, -1)
            return [StreamEvent.model_validate_json(e) for e in raw_events]
        except Exception as e:
            log.error("redis_get_events_error", query_id=query_id, error=str(e))
            return []


# Module-level singleton
publisher = EventPublisher()

