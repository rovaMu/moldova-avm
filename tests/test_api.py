"""End-to-end API tests using FastAPI's TestClient (no DB required)."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.routes import formula


def _client() -> TestClient:
    formula.clear_cache()
    return TestClient(create_app())


def test_root_and_health():
    client = _client()
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/health").json()["status"] == "healthy"


def test_active_formula_endpoint_serves_payload():
    client = _client()
    resp = client.get("/api/active-formula")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "apartment" in body["models"]
    assert "house" in body["models"]
    assert "sqm_coefficient" in body["models"]["apartment"]
    assert "sector_multipliers" in body["models"]["house"]


def test_active_formula_is_cached_then_refreshable():
    client = _client()
    first = client.get("/api/active-formula").json()
    second = client.get("/api/active-formula").json()
    assert first["last_updated"] == second["last_updated"]  # served from cache
    refreshed = client.get("/api/active-formula?refresh=true").json()
    assert refreshed["status"] == "success"


def test_unknown_route_returns_structured_error():
    client = _client()
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == 404
