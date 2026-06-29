"""Asynchronous crawl + ingest orchestration.

Ties the per-site scrapers to network fetching (via ``httpx``) and persistence
(via :class:`~app.database.SupabaseManager`). Kept separate from
``parser.py`` so the pure parsing/normalization logic stays free of I/O and is
trivially unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.database import SupabaseManager, supabase_manager
from app.scraper.parser import (
    APARTMENT,
    HOUSE,
    SCRAPERS,
    BaseScraper,
    NormalizedListing,
)

logger = logging.getLogger(__name__)

# Base URLs per registered scraper source.
SOURCE_BASE_URLS: dict[str, str] = {
    "999.md": "https://999.md",
    "makler.md": "https://makler.md",
    "proimobil.md": "https://proimobil.md",
    "oximobil.md": "https://oximobil.md",
    "lara.md": "https://lara.md",
}


def _urls_for(scraper: BaseScraper, property_type: str) -> list[str]:
    base = SOURCE_BASE_URLS.get(scraper.source, "").rstrip("/")
    paths = (
        scraper.apartment_paths
        if property_type == APARTMENT
        else scraper.house_paths
    )
    return [f"{base}{path}" for path in paths]


async def _fetch(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text
    except Exception:  # pragma: no cover - network faults are expected.
        logger.warning("Fetch failed for %s", url, exc_info=True)
        return None


async def crawl_source(
    client: httpx.AsyncClient,
    scraper: BaseScraper,
    property_type: str,
) -> list[NormalizedListing]:
    listings: list[NormalizedListing] = []
    for url in _urls_for(scraper, property_type):
        html = await _fetch(client, url)
        if not html:
            continue
        listings.extend(scraper.parse_normalized(html, property_type))
    return listings


async def crawl_all(
    settings: Optional[Settings] = None,
) -> list[NormalizedListing]:
    """Crawl every source for both property types concurrently."""
    settings = settings or get_settings()
    headers = {"User-Agent": settings.scraper_user_agent}
    timeout = httpx.Timeout(settings.scraper_request_timeout)

    async with httpx.AsyncClient(
        headers=headers, timeout=timeout, follow_redirects=True
    ) as client:
        tasks = [
            crawl_source(client, scraper, property_type)
            for scraper in SCRAPERS.values()
            for property_type in (APARTMENT, HOUSE)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    listings: list[NormalizedListing] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Crawl task failed: %s", result)
            continue
        listings.extend(result)
    return listings


async def ingest(manager: Optional[SupabaseManager] = None) -> int:
    """Crawl all sources and upsert normalized rows into Supabase.

    Returns the number of rows written (0 when Supabase is not configured).
    """
    manager = manager or supabase_manager
    listings = await crawl_all(manager.settings)
    rows = [item.to_dict() for item in listings]
    if not manager.is_configured:
        logger.warning("Supabase not configured; %d rows not persisted.", len(rows))
        return 0
    written = await manager.upsert_listings(rows)
    logger.info("Ingested %d listings into Supabase", written)
    return written


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(ingest())
