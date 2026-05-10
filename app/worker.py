"""
app/worker.py
═════════════
Background worker entrypoint for Dramatiq.
Processes background tasks like logging and trace updates.
"""
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from app.config import settings

# Initialize the broker for background tasks
# Note: Dramatiq uses a separate connection from the SSE publisher
redis_broker = RedisBroker(
    host=settings.redis_host, 
    port=settings.redis_port,
    db=settings.redis_db
)
dramatiq.set_broker(redis_broker)

# Export for Dramatiq CLI
broker = redis_broker
