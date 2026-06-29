"""Supabase asynchronous client manager.

Wraps the official ``supabase`` async client behind a small, lazily
initialised manager so the rest of the codebase never imports the SDK
directly and degrades gracefully when credentials are absent (e.g. local
development or unit tests).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

try:  # The SDK is optional at import time so tests run without credentials.
    from supabase import AClient, acreate_client  # type: ignore
except Exception:  # pragma: no cover - exercised only when SDK missing
    AClient = Any  # type: ignore
    acreate_client = None  # type: ignore


class SupabaseManager:
    """Lazily creates and caches a single async Supabase client."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client: Optional[AClient] = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def is_configured(self) -> bool:
        return self._settings.supabase_configured and acreate_client is not None

    async def client(self) -> AClient:
        """Return a connected async client, creating it on first use."""
        if not self._settings.supabase_configured:
            raise RuntimeError(
                "Supabase is not configured: set SUPABASE_URL and SUPABASE_KEY."
            )
        if acreate_client is None:  # pragma: no cover
            raise RuntimeError(
                "supabase package is not installed; run `pip install supabase`."
            )
        if self._client is None:
            logger.info("Initialising Supabase async client")
            self._client = await acreate_client(
                self._settings.supabase_url,
                self._settings.supabase_key,
            )
        return self._client

    async def fetch_listings(
        self, property_type: Optional[str] = None, limit: int = 50_000
    ) -> list[dict[str, Any]]:
        """Fetch raw listing rows, optionally filtered by ``property_type``.

        Returns a list of plain dictionaries ready for the cleaning and
        regression pipeline.
        """
        client = await self.client()
        query = client.table(self._settings.supabase_listings_table).select("*")
        if property_type:
            query = query.eq("property_type", property_type)
        response = await query.limit(limit).execute()
        return list(response.data or [])

    async def upsert_listings(self, rows: list[dict[str, Any]]) -> int:
        """Upsert cleaned listing rows; returns the number of rows written."""
        if not rows:
            return 0
        client = await self.client()
        response = (
            await client.table(self._settings.supabase_listings_table)
            .upsert(rows)
            .execute()
        )
        return len(response.data or [])

    async def aclose(self) -> None:
        """Best-effort shutdown hook for the cached client."""
        self._client = None


# Module level singleton used by the application.
supabase_manager = SupabaseManager()


def get_supabase_manager() -> SupabaseManager:
    """FastAPI dependency accessor for the shared manager."""
    return supabase_manager
