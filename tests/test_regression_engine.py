"""Tests for IQR filtering, recency weighting and dual regression training."""

import random
from datetime import datetime, timedelta, timezone

from app.analytics import regression_engine as eng
from app.scraper.parser import APARTMENT, HOUSE


def _make_apartment(sqm, is_new, reno, mid, sector, noise=0.0, listed_at=None):
    price = (
        20000
        + sqm * 1100
        + is_new * 18000
        + reno * 7000
        + mid * 4000
        + sector * -1500
        + noise
    )
    return {
        "property_type": APARTMENT,
        "price_eur": price,
        "sqm": sqm,
        "is_new_building": is_new,
        "renovation_level": reno,
        "is_middle_floor": mid,
        "sector": sector,
        "listed_at": listed_at
        or datetime.now(timezone.utc).isoformat(),
    }


def _make_house(sqm, land, reno, sub, levels, sector, noise=0.0):
    price = (
        40000
        + sqm * 950
        + land * 6000
        + reno * 11000
        + (-8000 if sub == 1 else 0)
        + (-15000 if sub == 2 else 0)
        + levels * 5000
        + sector * -2000
        + noise
    )
    return {
        "property_type": HOUSE,
        "price_eur": price,
        "sqm": sqm,
        "land_ari": land,
        "renovation_level": reno,
        "house_sub_type": sub,
        "levels": levels,
        "sector": sector,
        "listed_at": datetime.now(timezone.utc).isoformat(),
    }


def _synthetic_dataset(n=200, seed=42):
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        records.append(
            _make_apartment(
                sqm=rng.uniform(35, 120),
                is_new=rng.choice([0, 1]),
                reno=rng.choice([1, 2, 3, 4]),
                mid=rng.choice([0, 1]),
                sector=rng.choice([0, 1, 2, 3, 4, 5, 6]),
                noise=rng.uniform(-1500, 1500),
            )
        )
        records.append(
            _make_house(
                sqm=rng.uniform(80, 300),
                land=rng.uniform(2, 12),
                reno=rng.choice([1, 2, 3, 4]),
                sub=rng.choice([0, 1, 2]),
                levels=rng.choice([1, 2, 3]),
                sector=rng.choice([0, 1, 2, 3, 4, 5, 6]),
                noise=rng.uniform(-3000, 3000),
            )
        )
    return records


def test_train_recovers_known_coefficients():
    records = _synthetic_dataset()
    result = eng.train_models(records)

    apt = result.apartment
    assert not apt.used_fallback
    assert apt.r2 > 0.9
    # Coefficients should be close to the synthetic ground truth.
    assert abs(apt.coefficients["sqm_coefficient"] - 1100) < 120
    assert abs(apt.coefficients["new_building_premium"] - 18000) < 3000
    assert abs(apt.coefficients["renovation_step_value"] - 7000) < 1500

    house = result.house
    assert not house.used_fallback
    assert house.r2 > 0.9
    assert abs(house.coefficients["sqm_construction_coefficient"] - 950) < 120
    assert abs(house.coefficients["per_are_land_coefficient"] - 6000) < 1500
    assert house.coefficients["villa_penalty"] < 0
    assert house.coefficients["townhouse_penalty"] < 0


def test_iqr_drops_junk_listings():
    records = _synthetic_dataset(n=120)
    # Inject obvious junk: 1 EUR test ads and speculative 999,999 prices.
    junk = [
        _make_apartment(50, 1, 3, 1, 2),
        _make_apartment(50, 1, 3, 1, 2),
    ]
    junk[0]["price_eur"] = 1.0
    junk[1]["price_eur"] = 999999.0
    result = eng.train_models(records + junk)
    # Junk must not wreck the fit.
    assert result.apartment.r2 > 0.85


def test_recency_weight_decay():
    assert eng._recency_weight(0) == 1.0
    assert eng._recency_weight(30) == 1.0
    w200 = eng._recency_weight(200)
    assert eng.RECENCY_MIN_WEIGHT <= w200 < 1.0
    assert eng._recency_weight(10_000) == eng.RECENCY_MIN_WEIGHT


def test_listing_age_parsing():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    iso = (now - timedelta(days=10)).isoformat()
    assert abs(eng._listing_age_days(iso, now) - 10) < 0.01
    assert eng._listing_age_days("31.01.2026", now) >= 0
    assert eng._listing_age_days(None, now) == 0.0
    assert eng._listing_age_days("garbage", now) == 0.0


def test_fallback_on_sparse_data():
    result = eng.train_models([])
    assert result.apartment.used_fallback
    assert result.house.used_fallback
    payload = eng.build_formula_payload(result)
    assert payload["status"] == "success"
    assert "sqm_coefficient" in payload["models"]["apartment"]
    assert "per_are_land_coefficient" in payload["models"]["house"]


def test_payload_structure_matches_contract():
    result = eng.train_models(_synthetic_dataset())
    payload = eng.build_formula_payload(result)

    assert set(payload.keys()) == {
        "status", "last_updated", "ai_analytics_meta", "models"
    }
    meta = payload["ai_analytics_meta"]
    assert set(meta.keys()) == {
        "apartment_dataset_records",
        "house_dataset_records",
        "market_confidence_r2",
    }
    apt = payload["models"]["apartment"]
    for key in (
        "intercept",
        "sqm_coefficient",
        "new_building_premium",
        "renovation_step_value",
        "middle_floor_premium",
        "sector_multipliers",
    ):
        assert key in apt
    house = payload["models"]["house"]
    for key in (
        "intercept",
        "sqm_construction_coefficient",
        "per_are_land_coefficient",
        "renovation_step_value",
        "townhouse_penalty",
        "villa_penalty",
        "sector_multipliers",
    ):
        assert key in house
    # Sector multipliers expose all seven labels.
    assert set(apt["sector_multipliers"].keys()) == {
        "Centru", "Rascani", "Botanica", "Buiucani",
        "Ciocana", "Telecentru", "Suburbs",
    }


def test_type_discriminator_inference():
    # A record with no property_type but with land -> treated as house.
    rec = {"price_eur": 100000, "sqm": 120, "land_ari": 5, "sector": 2}
    apts, houses = eng._split_by_type([rec])
    assert len(houses) == 1 and len(apts) == 0
