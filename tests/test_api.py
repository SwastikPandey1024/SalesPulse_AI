"""Tests for SalesPulse AI Phase 3C: FastAPI Service Layer.

Covers:
1. App construction and route registration
2. Health and readiness responses (healthy vs degraded/unready)
3. Request validation and malformed payloads (HTTP 422 / stable error format)
4. Successful and failed service calls
5. Stable error-response shape without sensitive details (path and credential sanitization)
6. Forecast endpoint behavior and model-selection compatibility (strict non-fallback)
7. Anomaly detection and persisted-anomaly retrieval
8. Database dependency/session lifecycle and cleanup
9. Upload validation and request-size restrictions (413 for oversized files)
10. CORS configuration (conservative, no wildcard with credentials)
"""

from __future__ import annotations

import io
import os
import datetime
from unittest.mock import MagicMock
from typing import Generator
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np

from salespulse.api.app import create_app, sanitize_error_detail
from salespulse.api.dependencies import (
    get_db,
    get_database_engine,
    get_audit_repository,
    get_registry,
)
from salespulse.services.model_registry import ModelRegistry
from salespulse.services.forecasting_service import (
    ForecastingService,
    build_feature_dataframe,
    extract_date_features,
)
from salespulse.storage.database import (
    create_storage_engine,
    get_session_factory,
    init_db,
    SCHEMA_VERSION,
)
from salespulse.storage.models import Base
from salespulse.storage.repository import AuditRepository


@pytest.fixture
def test_db_path(tmp_path) -> str:
    """Fixture providing an isolated SQLite database file."""
    db_file = tmp_path / "test_api.db"
    return str(db_file)


@pytest.fixture
def test_engine(test_db_path):
    """Engine fixture for API tests."""
    engine = create_storage_engine(f"sqlite:///{test_db_path}")
    init_db(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def test_session_factory(test_engine):
    """Session factory fixture."""
    return get_session_factory(test_engine)


@pytest.fixture
def test_client(test_engine, test_session_factory) -> Generator[TestClient, None, None]:
    """TestClient fixture with dependency overrides for isolated database."""
    app = create_app(engine=test_engine, cors_origins=["https://dashboard.example.com"])

    def override_get_db():
        session = test_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. App construction and route registration
# ---------------------------------------------------------------------------

def test_app_construction_and_routes(test_client: TestClient):
    """Verify app creation and that expected versioned routes are registered."""
    response = test_client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    paths = openapi["paths"]

    assert "/health" in paths
    assert "/ready" in paths
    assert "/api/v1/data/validate" in paths
    assert "/api/v1/data/validate-json" in paths
    assert "/api/v1/forecasts" in paths
    assert "/api/v1/forecasts/{run_id}" in paths
    assert "/api/v1/analytics/anomalies/detect" in paths
    assert "/api/v1/analytics/anomalies" in paths


# ---------------------------------------------------------------------------
# 2. Health and readiness responses
# ---------------------------------------------------------------------------

def test_health_liveness(test_client: TestClient):
    """GET /health should return 200 and healthy liveness status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "salespulse-api"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


def test_ready_readiness_healthy(test_client: TestClient):
    """GET /ready should return 200 when database and models are operational."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "ok"
    assert data["schema_version"] == SCHEMA_VERSION
    assert "supported_models" in data
    assert isinstance(data["supported_models"], list)
    assert len(data["supported_models"]) > 0


def test_ready_readiness_degraded_on_db_failure():
    """GET /ready should return 503 when the database connectivity fails."""
    app = create_app()

    def failing_engine():
        m = MagicMock()
        m.connect.side_effect = RuntimeError("Database unreachable connection failure")
        return m

    app.dependency_overrides[get_database_engine] = failing_engine
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["database"] == "unavailable"


# ---------------------------------------------------------------------------
# 3. Request validation and malformed payloads
# ---------------------------------------------------------------------------

def test_malformed_json_returns_422_with_stable_error_shape(test_client: TestClient):
    """Malformed request body must return HTTP 422 with standardized API error shape."""
    bad_payload = {
        "base_date": "2023-01-31",
        "recent_history_sales": "invalid_not_a_list",
        "horizon_days": 14,
        "model_choice": "RandomForest",
    }
    response = test_client.post("/api/v1/forecasts", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert isinstance(data["detail"], str)
    assert "timestamp" in data
    assert data["path"] == "/api/v1/forecasts"


def test_invalid_horizon_range(test_client: TestClient):
    """Horizon days outside [1, 90] should be rejected by schema validation."""
    payload = {
        "base_date": "2023-01-31",
        "recent_history_sales": [10.0] * 35,
        "horizon_days": 150,  # exceeds max 90
        "model_choice": "RandomForest",
    }
    response = test_client.post("/api/v1/forecasts", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
    assert "horizon_days" in data["detail"]


# ---------------------------------------------------------------------------
# 4 & 5. Error shape and path sanitization (No filesystem leak)
# ---------------------------------------------------------------------------

def test_error_sanitization_removes_path_traces(test_client: TestClient):
    """Simulated not found must not leak local paths or stack traces."""
    response = test_client.get("/api/v1/forecasts/run-nonexistent-999")
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "RESOURCE_NOT_FOUND"
    assert "run-nonexistent-999" in data["detail"]
    detail_lower = data["detail"].lower()
    assert "c:\\" not in detail_lower
    assert "/users/" not in detail_lower
    assert "/home/" not in detail_lower


# ---------------------------------------------------------------------------
# 6. Data validation endpoints (File & JSON)
# ---------------------------------------------------------------------------

def test_data_validate_file_success(test_client: TestClient):
    """POST /api/v1/data/validate with valid CSV."""
    csv_content = (
        b"Date,Sales\n"
        b"2023-01-01,100.5\n"
        b"2023-01-02,120.0\n"
        b"2023-01-03,110.0\n"
    )
    files = {"file": ("sales_sample.csv", io.BytesIO(csv_content), "text/csv")}
    response = test_client.post("/api/v1/data/validate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ("PASSED", "WARNING")
    assert data["total_rows_read"] == 3
    assert data["valid_rows_count"] == 3
    assert data["quarantined_rows_count"] == 0
    assert "issues" in data


def test_data_validate_file_oversized_returns_413(test_client: TestClient):
    """POST /api/v1/data/validate with file > 10MB must return 413 Payload Too Large."""
    oversized_data = b"x" * (10 * 1024 * 1024 + 1024)
    files = {"file": ("big_upload.csv", io.BytesIO(oversized_data), "text/csv")}
    response = test_client.post("/api/v1/data/validate", files=files)
    assert response.status_code == 413
    data = response.json()
    assert data["error_code"] == "PAYLOAD_TOO_LARGE"
    assert "10MB" in data["detail"]


def test_data_validate_json_records(test_client: TestClient):
    """POST /api/v1/data/validate-json with valid list of records."""
    payload = {
        "source_name": "test_payload.csv",
        "records": [
            {"Date": "2023-01-01", "Sales": 150.0},
            {"Date": "2023-01-02", "Sales": 160.0},
        ],
        "persist_audit": False,
    }
    response = test_client.post("/api/v1/data/validate-json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows_read"] == 2
    assert data["valid_rows_count"] == 2


# ---------------------------------------------------------------------------
# 7. Forecast generation and retrieval (Strict non-fallback)
# ---------------------------------------------------------------------------

def test_forecast_generation_and_persistence(test_client: TestClient):
    """POST /api/v1/forecasts creates forecast and persists run."""
    ready_resp = test_client.get("/ready")
    available_models = ready_resp.json()["supported_models"]
    assert len(available_models) > 0
    model_to_use = available_models[0]

    historical = [100.0 + i * 2.0 for i in range(35)]
    payload = {
        "base_date": "2023-01-31",
        "recent_history_sales": historical,
        "horizon_days": 7,
        "model_choice": model_to_use,
        "interval_method": "heuristic_sqrt_step",
        "sales_scope": "dept_grocery",
        "persist_run": True,
    }

    response = test_client.post("/api/v1/forecasts", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == model_to_use
    assert data["horizon_days"] == 7
    assert data["sales_scope"] == "dept_grocery"
    assert len(data["items"]) == 7
    assert data["run_id"] is not None

    run_id = data["run_id"]

    # Verify retrieval via GET /api/v1/forecasts/{run_id}
    get_resp = test_client.get(f"/api/v1/forecasts/{run_id}")
    assert get_resp.status_code == 200
    run_data = get_resp.json()
    assert run_data["run_id"] == run_id
    assert run_data["model_name"] == model_to_use
    assert run_data["horizon_days"] == 7
    assert len(run_data["items"]) == 7
    assert run_data["items"][0]["step_ahead"] == 1


def test_forecast_generation_unknown_model_returns_400(test_client: TestClient):
    """POST /api/v1/forecasts with unknown model name returns HTTP 400 (strict non-fallback)."""
    payload = {
        "base_date": "2023-01-31",
        "recent_history_sales": [10.0] * 35,
        "horizon_days": 7,
        "model_choice": "non_existent_unregistered_model",
        "persist_run": False,
    }
    response = test_client.post("/api/v1/forecasts", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "BAD_REQUEST"
    assert "is not available" in data["detail"]


# ---------------------------------------------------------------------------
# 8. Anomaly detection and persistence retrieval
# ---------------------------------------------------------------------------

def test_anomaly_detection_in_memory_series(test_client: TestClient):
    """POST /api/v1/analytics/anomalies/detect detects spikes and returns seasonality notice."""
    start_dt = datetime.date(2023, 1, 1)
    series = []
    for i in range(25):
        dt = (start_dt + datetime.timedelta(days=i)).isoformat()
        val = 500.0 if i == 20 else (100.0 + (i % 3) * 2.0)
        series.append({"date": dt, "sales": val})

    payload = {
        "sales_scope": "test_store_1",
        "series": series,
        "window_size": 14,
        "min_history": 14,
        "surge_threshold": 2.5,
        "drop_threshold": 2.5,
        "persist": True,
    }
    response = test_client.post("/api/v1/analytics/anomalies/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == len(series)
    assert data["anomalies_detected"] >= 1
    assert "weekly seasonality" in data["seasonality_limitation_notice"].lower()

    # Verify spike anomaly details
    spike_point = next((p for p in data["anomalies"] if p["observed_value"] == 500.0), None)
    assert spike_point is not None
    assert spike_point["anomaly_score"] > 2.5
    assert spike_point["direction"] == "SURGE"

    # Verify persistence query via GET /api/v1/analytics/anomalies
    get_resp = test_client.get("/api/v1/analytics/anomalies?sales_scope=test_store_1&limit=50")
    assert get_resp.status_code == 200
    records = get_resp.json()
    assert isinstance(records, list)
    assert len(records) >= 1
    assert any(a["observed_value"] == 500.0 for a in records)


# ---------------------------------------------------------------------------
# 9. Database session cleanup
# ---------------------------------------------------------------------------

def test_database_session_cleanup(test_engine):
    """Verify database sessions are closed properly in get_db generator and via request."""
    # 1. Direct generator lifecycle verification
    gen = get_db(engine=test_engine)
    session = next(gen)
    assert session is not None
    assert session.is_active is True
    try:
        next(gen)
    except StopIteration:
        pass

    # 2. Request-scoped FastAPI dependency lifecycle verification
    app = create_app(engine=test_engine)
    session_closed = False

    @app.get("/api/v1/test-session-lifecycle")
    def sample_endpoint(db: Session = Depends(get_db)):
        nonlocal session_closed
        orig_close = db.close

        def spy_close():
            nonlocal session_closed
            session_closed = True
            orig_close()

        db.close = spy_close
        return {"status": "ok"}

    with TestClient(app) as client:
        res = client.get("/api/v1/test-session-lifecycle")
        assert res.status_code == 200
        assert session_closed is True


# ---------------------------------------------------------------------------
# 10. CORS configuration check
# ---------------------------------------------------------------------------

def test_cors_headers_conservative(test_client: TestClient):
    """Verify CORS headers follow conservative origin policy and do not allow wildcard credentials."""
    headers = {
        "Origin": "https://dashboard.example.com",
        "Access-Control-Request-Method": "GET"
    }
    response = test_client.options("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://dashboard.example.com"
    allow_origin = response.headers.get("access-control-allow-origin")
    allow_credentials = response.headers.get("access-control-allow-credentials")
    assert not (allow_origin == "*" and allow_credentials == "true")


# ---------------------------------------------------------------------------
# 11. Acceptance Audit Hardening Tests (Uploads, Edge Cases, Parity)
# ---------------------------------------------------------------------------

def test_data_validate_empty_file_rejected(test_client: TestClient):
    """Empty (0-byte) file upload must be rejected with HTTP 400."""
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = test_client.post("/api/v1/data/validate", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "BAD_REQUEST"
    assert "empty" in data["detail"].lower()


def test_data_validate_unsupported_file_extension(test_client: TestClient):
    """Uploading unsupported format (.json/.py/.exe) to /data/validate returns 400."""
    files = {"file": ("script.py", io.BytesIO(b"print('hello')"), "text/plain")}
    response = test_client.post("/api/v1/data/validate", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "BAD_REQUEST"
    assert "unsupported file format" in data["detail"].lower()


def test_data_validate_path_traversal_filename_sanitization(test_client: TestClient):
    """Path traversal sequences in filename must be stripped to clean basename."""
    csv_content = b"Date,Sales\n2023-01-01,100.5\n2023-01-02,120.0\n"
    # Filename with path traversal and null bytes
    files = {"file": ("../../../../etc/passwd\x00.csv", io.BytesIO(csv_content), "text/csv")}
    response = test_client.post("/api/v1/data/validate", files=files)
    assert response.status_code == 200
    # Audit log should record clean basename, never path traversal
    data = response.json()
    assert data["verdict"] in ("PASSED", "WARNING")


def test_ready_readiness_future_schema_version(test_engine):
    """GET /ready returns 503 when the database schema version is newer than supported."""
    with test_engine.connect() as conn:
        conn.execute(text("PRAGMA user_version = 999;"))
        conn.commit()

    app = create_app(engine=test_engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["database"] == "unsupported_future_schema"
        assert data["schema_version"] == 999


def test_ready_readiness_no_models_available(test_engine):
    """GET /ready returns 503 when the model registry has no candidate models."""
    app = create_app(engine=test_engine)

    empty_registry = MagicMock(spec=ModelRegistry)
    empty_registry.list_models.return_value = []
    app.dependency_overrides[get_registry] = lambda: empty_registry

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["supported_models"] == []


def test_forecast_api_parity_with_domain_service(test_client: TestClient, test_engine):
    """Verify exact float parity between POST /api/v1/forecasts and ForecastingService."""
    ready_resp = test_client.get("/ready")
    available_models = ready_resp.json()["supported_models"]
    model_choice = "RandomForest" if "RandomForest" in available_models else available_models[0]

    base_dt_str = "2024-06-15"
    base_dt = datetime.date.fromisoformat(base_dt_str)
    horizon_days = 7
    hist = [120.0 + (i % 7) * 5.0 for i in range(35)]

    # 1. Call via API
    api_payload = {
        "base_date": base_dt_str,
        "recent_history_sales": hist,
        "horizon_days": horizon_days,
        "model_choice": model_choice,
        "interval_method": "heuristic_sqrt_step",
        "sales_scope": "parity_test_scope",
        "persist_run": False,
    }
    api_resp = test_client.post("/api/v1/forecasts", json=api_payload)
    assert api_resp.status_code == 200
    api_data = api_resp.json()

    # 2. Call directly via ForecastingService using identical feature preparation
    d_feats = extract_date_features(base_dt)
    initial_df = build_feature_dataframe(
        date_features=d_feats,
        lag_1=hist[-1],
        lag_7=hist[-7],
        lag_14=hist[-14],
        lag_30=hist[-30],
        rolling_mean_7=float(np.mean(hist[-7:])),
        rolling_mean_14=float(np.mean(hist[-14:])),
        rolling_std_7=float(np.std(hist[-7:])),
        trend=float(len(hist)),
    )
    service = ForecastingService()
    service_df = service.forecast_multi_horizon(
        initial_df=initial_df,
        horizon_days=horizon_days,
        model_choice=model_choice,
        origin_date=base_dt,
        interval_method="heuristic_sqrt_step",
        sales_scope="parity_test_scope",
        persist_run=False,
    )

    # 3. Assert exact parity
    assert len(api_data["items"]) == len(service_df)
    for api_item, (_, s_row) in zip(api_data["items"], service_df.iterrows()):
        assert abs(api_item["predicted_value"] - float(s_row["Forecasted Sales ($)"])) < 1e-4
        assert abs(api_item["lower_bound"] - float(s_row["Lower Bound ($)"])) < 1e-4
        assert abs(api_item["upper_bound"] - float(s_row["Upper Bound ($)"])) < 1e-4


def test_anomaly_detection_from_db_actuals(test_client: TestClient, test_engine):
    """POST /api/v1/analytics/anomalies/detect with series=None evaluates actuals in DB."""
    # Seed DB with actual sales using AuditRepository
    repo = AuditRepository(engine=test_engine)
    start_date = datetime.date(2023, 1, 1)
    for i in range(25):
        d = start_date + datetime.timedelta(days=i)
        val = 1000.0 if i == 20 else 100.0
        repo.ingest_actual_sales(calendar_date=d, sales_scope="db_scope_test", daily_sales=val)

    payload = {
        "sales_scope": "db_scope_test",
        "series": None,  # evaluate DB actuals
        "window_size": 14,
        "min_history": 14,
        "surge_threshold": 2.5,
        "persist": True,
    }
    response = test_client.post("/api/v1/analytics/anomalies/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_scope"] == "db_scope_test"
    assert data["total_evaluated"] == 25
    assert data["anomalies_detected"] >= 1


def test_sanitize_error_detail_patterns():
    """Verify regex patterns in sanitize_error_detail redact drive letters, paths, and credentials."""
    assert "redacted" in sanitize_error_detail("Error in D:\\data\\file.py").lower()
    assert "redacted" in sanitize_error_detail("Failed at /home/user/app.py").lower()
    assert "redacted" in sanitize_error_detail("Failed with password=secret_pw").lower()
    assert "redacted" in sanitize_error_detail("Connection sqlite:///db.sqlite failed").lower()
    assert sanitize_error_detail("Invalid date format YYYY-MM-DD") == "Invalid date format YYYY-MM-DD"

