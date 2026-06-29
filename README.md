# Moldova AVM — Automated Valuation Model backend

Production-grade, modular **FastAPI** backend for an automated valuation model
(AVM) of the Moldovan property market. It crawls five primary listing sources,
normalizes multilingual (Romanian/Russian) features into a numeric space,
trains two independent Multiple Linear Regression engines (apartments &
houses), and exposes the resulting coefficients to a mobile client through a
single GET endpoint so valuations can run **offline on-device**.

## Architecture

```
app/
├── main.py                  # FastAPI core, CORS, global exception handler
├── config.py                # Pydantic BaseSettings (env-driven secrets)
├── database.py              # Supabase async client manager
├── scraper/
│   ├── parser.py            # Site scrapers + multilingual regex normalizers
│   └── ingest.py            # Async crawl → parse → upsert orchestration
├── analytics/
│   └── regression_engine.py # IQR filtering, recency weighting, dual MLR
└── routes/
    └── formula.py           # GET /api/active-formula
tests/                       # pytest suite (normalizers, scrapers, engine, API)
```

### Data sources
| Source | Role | Parsing strategy |
|--------|------|------------------|
| **999.md** | High-volume board | Feature grid (`is-wrapped` / `m-value`); floor split `3/9`, `5 de la 10` |
| **Makler.md** | Historical secondary market | List grids; Soviet series tags (`Serie 143`, `MS`, `Hrușciovka`, `Breznevka`) |
| **Proimobil.md** / **Oximobil.md** | Premium/agency | `<script>` JSON metadata for residential complexes; DOM fallback |
| **Lara.md** | Conservative appraisal baseline | Classic `<tr>/<td>` index tables |

### Feature normalization
Multilingual strings are folded (Romanian diacritics → ASCII, Cyrillic
preserved) and mapped to numeric features: `sector` (0–6), `is_new_building`
(0/1), `renovation_level` (1–4 ordinal), `sqm`, `land_ari` (normalized to ari),
`is_middle_floor`, `house_sub_type` and structural `levels`. Every normalizer
is defensive and falls back to documented defaults on dirty input.

### Regression engine
1. **IQR filtering** on price/m² (10th–90th percentile) plus absolute sanity
   bounds drop junk (1 EUR test ads, 999,999 EUR speculation).
2. **Recency weighting** — listings ≤ 30 days old weight `1.0`, decaying
   linearly to a floor for older points.
3. Two independent `sklearn.linear_model.LinearRegression` fits with sample
   weights and sector dummies (to de-confound core coefficients).

When data is too sparse the engine returns documented baseline coefficients so
the API always responds.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in SUPABASE_URL / SUPABASE_KEY
uvicorn app.main:app --reload
```

Run the crawler/ingestion job:

```bash
python -m app.scraper.ingest
```

Fetch the live formula payload:

```bash
curl http://localhost:8000/api/active-formula | jq
```

## API

`GET /api/active-formula` → dual-formula coefficient payload:

```json
{
  "status": "success",
  "last_updated": "ISO-8601",
  "ai_analytics_meta": {
    "apartment_dataset_records": 0,
    "house_dataset_records": 0,
    "market_confidence_r2": 0.0
  },
  "models": {
    "apartment": { "intercept": 0, "sqm_coefficient": 0, "new_building_premium": 0,
      "renovation_step_value": 0, "middle_floor_premium": 0, "sector_multipliers": {…} },
    "house": { "intercept": 0, "sqm_construction_coefficient": 0,
      "per_are_land_coefficient": 0, "renovation_step_value": 0,
      "townhouse_penalty": 0, "villa_penalty": 0, "levels_coefficient": 0,
      "sector_multipliers": {…} }
  }
}
```

Pass `?refresh=true` to bypass the 15-minute server cache.

### On-device valuation
```
apartment_price ≈ (intercept + sqm·sqm_coefficient + is_new·new_building_premium
                   + renovation·renovation_step_value + is_middle·middle_floor_premium)
                   · sector_multiplier[sector]

house_price     ≈ (intercept + sqm·sqm_construction_coefficient
                   + land_ari·per_are_land_coefficient + renovation·renovation_step_value
                   + is_townhouse·townhouse_penalty + is_villa·villa_penalty
                   + levels·levels_coefficient) · sector_multiplier[sector]
```

## Tests

```bash
pytest
```
