"""
app/database/session.py
═══════════════════════
Async SQLAlchemy engine and session factory.
Uses asyncpg driver for PostgreSQL.
"""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings
from app.database.models import Base


def _build_dsn() -> str:
    """Construct the async PostgreSQL DSN from environment settings."""
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    _build_dsn(),
    echo=False,          # Set True for SQL debug logging
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Validate connections before use
)

# Separate engine for test isolation (NullPool disables connection pooling)
test_engine = create_async_engine(
    _build_dsn(),
    echo=False,
    poolclass=NullPool,
)

# ── Session Factory ───────────────────────────────────────────────────────────

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent expired instance errors post-commit
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables. Called once at application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop all tables. Used in test teardown ONLY."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
