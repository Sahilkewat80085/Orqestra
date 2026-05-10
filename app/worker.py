"""
app/worker.py
═════════════
Background worker entrypoint for Dramatiq.
Processes background tasks like multi-agent orchestration.
"""
import asyncio
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from app.config import settings
from app.database.session import AsyncSessionFactory
from app.services.query_service import QueryService

# Initialize the broker for background tasks
redis_broker = RedisBroker(
    host=settings.redis_host, 
    port=settings.redis_port,
    db=settings.redis_db
)
dramatiq.set_broker(redis_broker)

# Export for Dramatiq CLI
broker = redis_broker

@dramatiq.actor(max_retries=3)
def run_query_task(query: str, query_id: str):
    """
    Background task to run the multi-agent pipeline.
    Runs the async QueryService in a synchronous Dramatiq worker.
    """
    async def _run():
        async with AsyncSessionFactory() as db:
            service = QueryService(db)
            await service.run_query(query, query_id=query_id)
    
    # Run the async loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run())
