"""IQR outlier filtering + scikit-learn multi-variable regression engine.

The engine consumes normalized listing records (dicts or
:class:`~app.scraper.parser.NormalizedListing`) and produces two independent
Multiple Linear Regression models - one for apartments, one for houses -
together with data-driven per-sector multipliers and an R² confidence score.

Pipeline per property type:

1. Coerce + validate records (drop rows lacking price or area).
2. Compute price-per-m² and apply IQR filtering on the 10th/90th percentiles
   to discard junk (1 EUR test ads, 999,999 EUR speculative outliers, ...).
3. Compute a linear time-recency weight (1.0 for the last 30 days, decaying
   linearly for older listings).
4. Fit ``sklearn.linear_model.LinearRegression`` with sample weights.
5. Extract the coefficients into the public payload shape.

The engine never raises on sparse/dirty data: when there are too few usable
rows it returns documented baseline coefficients so the API always responds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from app.scraper.parser import (
    APARTMENT,
    HOUSE,
    HOUSE_TOWNHOUSE,
    HOUSE_VILLA,
    SECTOR_LABELS,
    NormalizedListing,
)

logger = logging.getLogger(__name__)

# Tunables -------------------------------------------------------------------
IQR_LOWER_PERCENTILE = 10
IQR_UPPER_PERCENTILE = 90
RECENCY_FULL_WEIGHT_DAYS = 30
RECENCY_HORIZON_DAYS = 365
RECENCY_MIN_WEIGHT = 0.1
MIN_TRAINING_ROWS = 8
MOLDOVA_PPSQM_BASELINE = 1720.0  # EUR/m² sanity anchor

# Hard sanity bounds to drop obvious junk before percentile filtering.
ABS_MIN_PRICE = 1_000.0
ABS_MAX_PRICE = 5_000_000.0
ABS_MIN_PPSQM = 100.0
ABS_MAX_PPSQM = 20_000.0


# Baseline / fallback coefficients (used when data is too sparse to fit).
_DEFAULT_APARTMENT = {
    "intercept": 24_500.50,
    "sqm_coefficient": 1_150.20,
    "new_building_premium": 18_200.00,
    "renovation_step_value": 7_500.00,
    "middle_floor_premium": 4_200.00,
}
_DEFAULT_HOUSE = {
    "intercept": 45_000.00,
    "sqm_construction_coefficient": 950.40,
    "per_are_land_coefficient": 6_200.00,
    "renovation_step_value": 11_200.00,
    "townhouse_penalty": -8_500.00,
    "villa_penalty": -15_000.00,
    "levels_coefficient": 5_000.00,
}
_DEFAULT_APARTMENT_SECTORS = {
    "Centru": 1.0, "Rascani": 0.92, "Botanica": 0.89, "Buiucani": 0.87,
    "Ciocana": 0.84, "Telecentru": 0.94, "Suburbs": 0.65,
}
_DEFAULT_HOUSE_SECTORS = {
    "Centru": 1.2, "Rascani": 1.0, "Botanica": 0.95, "Buiucani": 0.98,
    "Ciocana": 0.88, "Telecentru": 1.1, "Suburbs": 0.60,
}


@dataclass
class ModelResult:
    """Fitted coefficients + metadata for a single property type."""

    property_type: str
    coefficients: dict[str, float]
    sector_multipliers: dict[str, float]
    n_records: int
    r2: float
    used_fallback: bool = False


@dataclass
class TrainingResult:
    apartment: ModelResult
    house: ModelResult
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def market_confidence_r2(self) -> float:
        """Record-weighted average R² across both models."""
        total = self.apartment.n_records + self.house.n_records
        if total <= 0:
            return 0.0
        weighted = (
            self.apartment.r2 * self.apartment.n_records
            + self.house.r2 * self.house.n_records
        )
        return round(weighted / total, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, NormalizedListing):
        return record.to_dict()
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _listing_age_days(listed_at: Any, now: datetime) -> float:
    if not listed_at:
        return 0.0
    text = str(listed_at).strip()
    parsed: Optional[datetime] = None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = now - parsed
    return max(0.0, delta.total_seconds() / 86_400.0)


def _recency_weight(age_days: float) -> float:
    """Linear decay: 1.0 within the recency window, down to a small floor."""
    if age_days <= RECENCY_FULL_WEIGHT_DAYS:
        return 1.0
    span = RECENCY_HORIZON_DAYS - RECENCY_FULL_WEIGHT_DAYS
    if span <= 0:
        return RECENCY_MIN_WEIGHT
    decayed = 1.0 - (age_days - RECENCY_FULL_WEIGHT_DAYS) / span
    return float(max(RECENCY_MIN_WEIGHT, min(1.0, decayed)))


def _split_by_type(
    records: Iterable[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    apartments: list[dict[str, Any]] = []
    houses: list[dict[str, Any]] = []
    for record in records:
        row = _as_dict(record)
        ptype = str(row.get("property_type", "")).lower()
        if ptype == HOUSE:
            houses.append(row)
        elif ptype == APARTMENT:
            apartments.append(row)
        else:
            # Discriminator missing: infer from presence of land footprint.
            if _to_float(row.get("land_ari")) is not None:
                houses.append(row)
            else:
                apartments.append(row)
    return apartments, houses


def _iqr_filter(
    rows: list[dict[str, Any]], prices: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """Return a boolean mask keeping rows within the price/m² IQR band."""
    ppsqm = np.divide(
        prices, areas, out=np.full_like(prices, np.nan), where=areas > 0
    )
    finite = np.isfinite(ppsqm)
    # Absolute sanity bounds first.
    sane = (
        finite
        & (prices >= ABS_MIN_PRICE)
        & (prices <= ABS_MAX_PRICE)
        & (ppsqm >= ABS_MIN_PPSQM)
        & (ppsqm <= ABS_MAX_PPSQM)
    )
    if sane.sum() < MIN_TRAINING_ROWS:
        return sane
    lower = np.percentile(ppsqm[sane], IQR_LOWER_PERCENTILE)
    upper = np.percentile(ppsqm[sane], IQR_UPPER_PERCENTILE)
    return sane & (ppsqm >= lower) & (ppsqm <= upper)


def _sector_multipliers(
    rows: list[dict[str, Any]],
    prices: np.ndarray,
    areas: np.ndarray,
    fallback: dict[str, float],
) -> dict[str, float]:
    """Data-driven multipliers = sector mean(price/m²) / overall mean."""
    sectors = np.array([int(r.get("sector", 2)) for r in rows])
    ppsqm = np.divide(
        prices, areas, out=np.full_like(prices, np.nan), where=areas > 0
    )
    valid = np.isfinite(ppsqm)
    if valid.sum() == 0:
        return dict(fallback)
    overall = float(np.mean(ppsqm[valid]))
    if overall <= 0:
        return dict(fallback)
    multipliers: dict[str, float] = {}
    for code, label in SECTOR_LABELS.items():
        mask = valid & (sectors == code)
        if mask.sum() == 0:
            multipliers[label] = fallback.get(label, 1.0)
        else:
            multipliers[label] = round(float(np.mean(ppsqm[mask]) / overall), 4)
    return multipliers


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def _train_apartments(rows: list[dict[str, Any]]) -> ModelResult:
    clean = [
        r
        for r in rows
        if _to_float(r.get("price_eur")) is not None
        and _to_float(r.get("sqm")) not in (None, 0.0)
    ]
    if len(clean) < MIN_TRAINING_ROWS:
        return ModelResult(
            property_type=APARTMENT,
            coefficients=dict(_DEFAULT_APARTMENT),
            sector_multipliers=dict(_DEFAULT_APARTMENT_SECTORS),
            n_records=len(clean),
            r2=0.0,
            used_fallback=True,
        )

    prices = np.array([_to_float(r["price_eur"]) for r in clean], dtype=float)
    areas = np.array([_to_float(r["sqm"]) for r in clean], dtype=float)
    mask = _iqr_filter(clean, prices, areas)

    kept = [clean[i] for i in range(len(clean)) if mask[i]]
    if len(kept) < MIN_TRAINING_ROWS:
        return ModelResult(
            property_type=APARTMENT,
            coefficients=dict(_DEFAULT_APARTMENT),
            sector_multipliers=_sector_multipliers(
                clean, prices, areas, _DEFAULT_APARTMENT_SECTORS
            ),
            n_records=len(kept),
            r2=0.0,
            used_fallback=True,
        )

    y = prices[mask]
    sqm = areas[mask]
    is_new = np.array([float(r.get("is_new_building", 1)) for r in kept])
    renovation = np.array([float(r.get("renovation_level", 3)) for r in kept])
    is_middle = np.array([float(r.get("is_middle_floor", 1)) for r in kept])

    now = datetime.now(timezone.utc)
    weights = np.array(
        [_recency_weight(_listing_age_days(r.get("listed_at"), now)) for r in kept]
    )

    # Sector dummies de-confound the core coefficients from location.
    sectors = np.array([int(r.get("sector", 2)) for r in kept])
    sector_dummies = _one_hot(sectors, drop_first=True)

    X = np.column_stack([sqm, is_new, renovation, is_middle, sector_dummies])
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    r2 = float(r2_score(y, model.predict(X), sample_weight=weights))

    coef = model.coef_
    coefficients = {
        "intercept": round(float(model.intercept_), 2),
        "sqm_coefficient": round(float(coef[0]), 2),
        "new_building_premium": round(float(coef[1]), 2),
        "renovation_step_value": round(float(coef[2]), 2),
        "middle_floor_premium": round(float(coef[3]), 2),
    }
    return ModelResult(
        property_type=APARTMENT,
        coefficients=coefficients,
        sector_multipliers=_sector_multipliers(
            kept, y, sqm, _DEFAULT_APARTMENT_SECTORS
        ),
        n_records=len(kept),
        r2=round(max(0.0, r2), 4),
    )


def _train_houses(rows: list[dict[str, Any]]) -> ModelResult:
    clean = [
        r
        for r in rows
        if _to_float(r.get("price_eur")) is not None
        and _to_float(r.get("sqm")) not in (None, 0.0)
    ]
    if len(clean) < MIN_TRAINING_ROWS:
        return ModelResult(
            property_type=HOUSE,
            coefficients=dict(_DEFAULT_HOUSE),
            sector_multipliers=dict(_DEFAULT_HOUSE_SECTORS),
            n_records=len(clean),
            r2=0.0,
            used_fallback=True,
        )

    prices = np.array([_to_float(r["price_eur"]) for r in clean], dtype=float)
    areas = np.array([_to_float(r["sqm"]) for r in clean], dtype=float)
    mask = _iqr_filter(clean, prices, areas)

    kept = [clean[i] for i in range(len(clean)) if mask[i]]
    if len(kept) < MIN_TRAINING_ROWS:
        return ModelResult(
            property_type=HOUSE,
            coefficients=dict(_DEFAULT_HOUSE),
            sector_multipliers=_sector_multipliers(
                clean, prices, areas, _DEFAULT_HOUSE_SECTORS
            ),
            n_records=len(kept),
            r2=0.0,
            used_fallback=True,
        )

    y = prices[mask]
    sqm = areas[mask]
    land = np.array(
        [_to_float(r.get("land_ari")) or 0.0 for r in kept], dtype=float
    )
    renovation = np.array([float(r.get("renovation_level", 3)) for r in kept])
    sub_type = np.array([int(r.get("house_sub_type", 0)) for r in kept])
    is_townhouse = (sub_type == HOUSE_TOWNHOUSE).astype(float)
    is_villa = (sub_type == HOUSE_VILLA).astype(float)
    levels = np.array([_to_float(r.get("levels")) or 1.0 for r in kept], dtype=float)

    now = datetime.now(timezone.utc)
    weights = np.array(
        [_recency_weight(_listing_age_days(r.get("listed_at"), now)) for r in kept]
    )

    sectors = np.array([int(r.get("sector", 2)) for r in kept])
    sector_dummies = _one_hot(sectors, drop_first=True)

    X = np.column_stack(
        [sqm, land, renovation, is_townhouse, is_villa, levels, sector_dummies]
    )
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    r2 = float(r2_score(y, model.predict(X), sample_weight=weights))

    coef = model.coef_
    coefficients = {
        "intercept": round(float(model.intercept_), 2),
        "sqm_construction_coefficient": round(float(coef[0]), 2),
        "per_are_land_coefficient": round(float(coef[1]), 2),
        "renovation_step_value": round(float(coef[2]), 2),
        "townhouse_penalty": round(float(coef[3]), 2),
        "villa_penalty": round(float(coef[4]), 2),
        "levels_coefficient": round(float(coef[5]), 2),
    }
    return ModelResult(
        property_type=HOUSE,
        coefficients=coefficients,
        sector_multipliers=_sector_multipliers(
            kept, y, sqm, _DEFAULT_HOUSE_SECTORS
        ),
        n_records=len(kept),
        r2=round(max(0.0, r2), 4),
    )


def _one_hot(values: np.ndarray, drop_first: bool = True) -> np.ndarray:
    """One-hot encode sector codes across the full known sector space."""
    codes = sorted(SECTOR_LABELS.keys())
    if drop_first:
        codes = codes[1:]
    if not codes:
        return np.zeros((len(values), 0))
    return np.column_stack([(values == c).astype(float) for c in codes])


def train_models(records: Iterable[Any]) -> TrainingResult:
    """Train both regression models from a single record iterable."""
    apartments, houses = _split_by_type(records)
    logger.info(
        "Training models: %d apartment rows, %d house rows",
        len(apartments),
        len(houses),
    )
    return TrainingResult(
        apartment=_train_apartments(apartments),
        house=_train_houses(houses),
    )


def build_formula_payload(result: TrainingResult) -> dict[str, Any]:
    """Render a :class:`TrainingResult` into the public API payload shape."""
    return {
        "status": "success",
        "last_updated": result.generated_at,
        "ai_analytics_meta": {
            "apartment_dataset_records": result.apartment.n_records,
            "house_dataset_records": result.house.n_records,
            "market_confidence_r2": result.market_confidence_r2,
        },
        "models": {
            "apartment": {
                **result.apartment.coefficients,
                "sector_multipliers": result.apartment.sector_multipliers,
            },
            "house": {
                **result.house.coefficients,
                "sector_multipliers": result.house.sector_multipliers,
            },
        },
    }
