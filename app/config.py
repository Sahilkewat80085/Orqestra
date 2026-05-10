"""
app/config.py
═════════════
Centralized configuration via environment variables using pydantic-settings.
All configuration MUST come from here — no hardcoded values anywhere else.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_version: str = "0.1.0"
    secret_key: str = "change_me_in_production"

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── Gemini ───────────────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.2

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "orqestra"
    postgres_user: str = "orqestra"
    postgres_password: str = Field(..., description="PostgreSQL password")
    database_url: Optional[str] = Field(None, description="Complete PostgreSQL DSN")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "orqestra_docs"

    # ── Context Budgets (tokens per agent) ────────────────────────────────────
    budget_orchestrator: int = 4096
    budget_decomposition: int = 2048
    budget_retrieval: int = 8192
    budget_critique: int = 4096
    budget_synthesis: int = 8192
    budget_meta: int = 4096

    @property
    def agent_budgets(self) -> dict:
        return {
            "orchestrator": self.budget_orchestrator,
            "decomposition": self.budget_decomposition,
            "retrieval": self.budget_retrieval,
            "critique": self.budget_critique,
            "synthesis": self.budget_synthesis,
            "meta": self.budget_meta,
        }

    # ── Tools ─────────────────────────────────────────────────────────────────
    web_search_mock: bool = True
    web_search_api_key: str = ""
    web_search_timeout_secs: int = 5
    python_sandbox_timeout_secs: int = 10
    tool_max_retries: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — import and call this everywhere."""
    return Settings()


# Module-level singleton for convenience
settings = get_settings()
