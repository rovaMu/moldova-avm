"""GET endpoint returning the live dual-formula coefficient payload.

The mobile client calls ``GET /api/active-formula`` once and then runs
valuations entirely offline using the returned weights, so this controller is
intentionally light: it serves a cached payload and only recomputes when the
cache is stale (or a refresh is forced).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query

from app.analytics.regression_engine import (
    TrainingResult,
    build_formula_payload,
    train_models,
)
from app.config import Settings, get_settings
from app.database import SupabaseManager, get_supabase_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["formula"])

# Simple process-local cache. Coefficients change slowly (new crawl cycles),
# so a short TTL keeps the endpoint instant without serving stale math.
_CACHE_TTL_SECONDS = 15 * 60
_cache: dict[str, Any] = {"payload": None, "ts": 0.0}


async def _load_records(
    manager: SupabaseManager, settings: Settings
) -> list[dict[str, Any]]:
    """Load normalized listing rows from Supabase, or [] when unavailable."""
    if not manager.is_configured:
        logger.warning("Supabase not configured; serving baseline coefficients.")
        return []
    try:
        return await manager.fetch_listings()
    except Exception:  # pragma: no cover - network/DB faults must not 500.
        logger.exception("Failed to fetch listings from Supabase")
        return []


async def compute_payload(
    manager: SupabaseManager, settings: Settings
) -> dict[str, Any]:
    records = await _load_records(manager, settings)
    result: TrainingResult = train_models(records)
    return build_formula_payload(result)


@router.get("/active-formula", summary="Live dual-formula coefficient payload")
async def active_formula(
    refresh: bool = Query(
        default=False, description="Force a recompute, bypassing the cache."
    ),
    manager: SupabaseManager = Depends(get_supabase_manager),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return the computed apartment + house valuation weights.

    Response shape (abridged)::

        {
          "status": "success",
          "last_updated": "...ISO-8601...",
          "ai_analytics_meta": {...},
          "models": {"apartment": {...}, "house": {...}}
        }
    """
    now = time.time()
    cached: Optional[dict[str, Any]] = _cache["payload"]
    if not refresh and cached is not None and (now - _cache["ts"]) < _CACHE_TTL_SECONDS:
        return cached

    payload = await compute_payload(manager, settings)
    _cache["payload"] = payload
    _cache["ts"] = now
    return payload


def clear_cache() -> None:
    """Test/ops helper to drop the cached payload."""
    _cache["payload"] = None
    _cache["ts"] = 0.0
