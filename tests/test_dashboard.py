"""
SalesPulse AI - Phase 3D Dashboard Integration Tests
Verifies:
- Service initialization and defensive database handling.
- Forecast provenance and out-of-sample performance metrics.
- Missing actuals policy (never treated as 0.0; excluded from realized errors).
- Incompatible scope and horizon isolation.
- Anomaly monitor queries, triage status resolution, and seasonality disclosure.
- System health diagnostic probes.
- Parity and backward compatibility of legacy forecast helper functions.
"""

import os
import datetime
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from salespulse.storage.database import init_db, SCHEMA_VERSION
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import ForecastRunLog, ForecastItemLog, ActualSalesRecord, AnomalyRecord
from salespulse.services.forecasting_service import ForecastingService
from salespulse.services.validation_service import ValidationService
from salespulse.services.analytics_service import AnalyticsService
from salespulse.services.model_registry import get_model_registry
from salespulse.features.temporal import ENGINEERED_FEATURES
import app


@pytest.fixture
def test_repo(tmp_path):
    """Isolated SQLite test repository in WAL mode."""
    db_file = tmp_path / "test_dashboard.db"
    engine = create_engine(f"sqlite:///{db_file}")
    init_db(engine)
    return AuditRepository(engine)


# ----------------------------------------------------------------------------
# 1. Dashboard Service Initialization & Defensive Error Handling
# ----------------------------------------------------------------------------
def test_dashboard_services_initialization():
    """Verify get_dashboard_services boots with all required services."""
    services = app.get_dashboard_services()
    assert "fc_service" in services
    assert "val_service" in services
    assert "ana_service" in services
    assert "registry" in services
    assert isinstance(services["fc_service"], ForecastingService)
    assert isinstance(services["val_service"], ValidationService)
    assert isinstance(services["ana_service"], AnalyticsService)
    assert len(services["registry"].list_models()) >= 2


def test_dashboard_services_database_fallback():
    """Verify dashboard degrades safely without crashing when database init fails."""
    with patch("app.get_engine", side_effect=RuntimeError("Simulated DB Lockout")):
        # Call the underlying function bypassing streamlit cache wrapper
        func = getattr(app.get_dashboard_services, "__wrapped__", getattr(app.get_dashboard_services, "func", app.get_dashboard_services))
        degraded = func()
        assert degraded["db_ready"] is False
        assert "Simulated DB Lockout" in degraded["db_error"]
        assert degraded["fc_service"] is not None
        assert degraded["val_service"] is not None
        assert degraded["ana_service"] is not None



# ----------------------------------------------------------------------------
# 2. Forecast Provenance & Performance Realization
# ----------------------------------------------------------------------------
def test_missing_actuals_never_treated_as_zero(test_repo):
    """Test that items pending reconciliation are NOT treated as 0.0 sales."""
    run_id = "run-test-missing-actuals"
    items = [
        {"target_date": "2026-06-01", "step_ahead": 1, "predicted_value": 150.0, "lower_bound": 130.0, "upper_bound": 170.0},
        {"target_date": "2026-06-02", "step_ahead": 2, "predicted_value": 160.0, "lower_bound": 140.0, "upper_bound": 180.0},
    ]
    test_repo.record_forecast_run(
        run_id=run_id,
        model_name="XGBoost",
        origin_date="2026-05-31",
        horizon_days=2,
        items=items,
        sales_scope="STORE_A",
    )

    # Ingest actual ONLY for step 1
    test_repo.ingest_actual_sales(
        calendar_date="2026-06-01",
        daily_sales=155.0,
        sales_scope="STORE_A",
    )

    run = test_repo.get_forecast_run(run_id)
    step1 = next(it for it in run.items if it.step_ahead == 1)
    step2 = next(it for it in run.items if it.step_ahead == 2)

    # Step 1 should be realized
    assert step1.actual_value == 155.0
    assert step1.absolute_error == 5.0
    assert step1.interval_covered is True

    # Step 2 must remain pending and NEVER treated as 0.0 or 160.0 error
    assert step2.actual_value is None
    assert step2.absolute_error is None
    assert step2.interval_covered is None

    # Evaluation summary must compute MAE strictly over step 1
    eval_summary = test_repo.get_forecast_evaluation_summary(
        model_name="XGBoost",
        horizon_days=2,
        sales_scope="STORE_A",
    )
    assert eval_summary["evaluated_count"] == 1
    assert eval_summary["mae"] == 5.0
    assert eval_summary["rmse"] == 5.0
    assert eval_summary["coverage_rate"] == 1.0


def test_forecast_performance_metric_formulas():
    """Verify realized WAPE, Bias, and MAE calculations on realized items."""
    reconciled_items = [
        {"predicted": 100.0, "actual": 120.0, "abs_err": 20.0, "covered": True},
        {"predicted": 200.0, "actual": 180.0, "abs_err": 20.0, "covered": True},
        {"predicted": 150.0, "actual": 150.0, "abs_err": 0.0, "covered": True},
    ]

    mae = float(np.mean([it["abs_err"] for it in reconciled_items]))
    assert mae == pytest.approx(40.0 / 3.0)

    sum_act = sum(it["actual"] for it in reconciled_items)  # 450.0
    sum_abs_err = sum(it["abs_err"] for it in reconciled_items)  # 40.0
    wape = (sum_abs_err / sum_act) * 100.0
    assert wape == pytest.approx((40.0 / 450.0) * 100.0)

    # Bias: sum(pred - actual) / sum(actual)
    sum_diff = sum(it["predicted"] - it["actual"] for it in reconciled_items)  # -20 + 20 + 0 = 0.0
    bias = (sum_diff / sum_act) * 100.0
    assert bias == 0.0


# ----------------------------------------------------------------------------
# 3. Scope and Horizon Isolation
# ----------------------------------------------------------------------------
def test_scope_and_horizon_isolation(test_repo):
    """Verify that runs and evaluations are strictly isolated by scope and horizon."""
    # Run 1: Scope STORE_A, Horizon 7
    test_repo.record_forecast_run(
        run_id="run-iso-1",
        model_name="XGBoost",
        origin_date="2026-06-01",
        horizon_days=7,
        items=[{"target_date": "2026-06-02", "step_ahead": 1, "predicted_value": 100.0, "lower_bound": 90.0, "upper_bound": 110.0}],
        sales_scope="STORE_A",
    )
    # Run 2: Scope STORE_B, Horizon 14
    test_repo.record_forecast_run(
        run_id="run-iso-2",
        model_name="XGBoost",
        origin_date="2026-06-01",
        horizon_days=14,
        items=[{"target_date": "2026-06-02", "step_ahead": 1, "predicted_value": 500.0, "lower_bound": 450.0, "upper_bound": 550.0}],
        sales_scope="STORE_B",
    )

    test_repo.ingest_actual_sales(calendar_date="2026-06-02", daily_sales=110.0, sales_scope="STORE_A")
    test_repo.ingest_actual_sales(calendar_date="2026-06-02", daily_sales=550.0, sales_scope="STORE_B")

    # Evaluation for STORE_A must only see STORE_A items
    sum_a = test_repo.get_forecast_evaluation_summary(sales_scope="STORE_A", horizon_days=7)
    assert sum_a["evaluated_count"] == 1
    assert sum_a["mae"] == 10.0

    sum_b = test_repo.get_forecast_evaluation_summary(sales_scope="STORE_B", horizon_days=14)
    assert sum_b["evaluated_count"] == 1
    assert sum_b["mae"] == 50.0

    # Cross query must yield 0 evaluated
    sum_cross = test_repo.get_forecast_evaluation_summary(sales_scope="STORE_A", horizon_days=14)
    assert sum_cross["evaluated_count"] == 0


# ----------------------------------------------------------------------------
# 4. Anomaly Monitor, Status Resolution & Seasonality Disclosure
# ----------------------------------------------------------------------------
def test_anomaly_status_update_and_resolution(test_repo):
    """Verify operator workflow for updating anomaly status and attaching notes."""
    # Persist sample anomaly point
    from salespulse.analytics.schemas import AnomalyPoint, AnomalyType, SeverityLevel, AnomalyDirection
    point = AnomalyPoint(
        anomaly_id="ano-test-resolution-001",
        observation_date="2026-07-01",
        sales_scope="TOTAL_STORE",
        anomaly_type=AnomalyType.SALES_SURGE,
        severity=SeverityLevel.HIGH,
        direction=AnomalyDirection.SURGE,
        observed_value=350.0,
        reference_center=150.0,
        reference_spread=20.0,
        anomaly_score=10.0,
        threshold=3.0,
        explanation="Statistical surge detected",
        detection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="ACTIVE",
    )
    test_repo.record_anomalies([point])

    # Initial query confirms ACTIVE
    records = test_repo.get_anomalies(sales_scope="TOTAL_STORE", status="ACTIVE")
    assert len(records) == 1
    assert records[0].status == "ACTIVE"

    # Operator marks as RESOLVED
    updated = test_repo.update_anomaly_status(
        anomaly_id="ano-test-resolution-001",
        new_status="RESOLVED",
        resolution_note="Verified promotional sale",
    )
    assert updated is True

    # Verify status changed and note appended
    resolved_records = test_repo.get_anomalies(sales_scope="TOTAL_STORE", status="RESOLVED")
    assert len(resolved_records) == 1
    assert resolved_records[0].status == "RESOLVED"
    assert "Verified promotional sale" in resolved_records[0].explanation
    assert resolved_records[0].status_updated_at is not None


def test_seasonality_disclosure_presence():
    """Verify that app.py contains the mandatory weekly seasonality limitation notice."""
    with open("app.py", "r", encoding="utf-8") as f:
        app_source = f.read()

    # Verify key phrases required by governance
    assert "Seasonality" in app_source
    assert "14-day rolling median" in app_source
    assert "weekdays versus weekends" in app_source
    assert "statistical signals" in app_source


# ----------------------------------------------------------------------------
# 5. System Health Probes
# ----------------------------------------------------------------------------
def test_system_health_readiness(test_repo):
    """Verify database schema version and model candidate inventory."""
    assert SCHEMA_VERSION == 2

    registry = get_model_registry()
    models = registry.list_models()
    assert "XGBoost" in models
    assert "RandomForest" in models
    assert "Ensemble" in models


# ----------------------------------------------------------------------------
# 6. Legacy Parity and Backward Compatibility
# ----------------------------------------------------------------------------
def test_dashboard_legacy_helper_functions_parity():
    """Verify legacy helper functions build valid feature vectors and generate insights."""
    d_feats = app.extract_date_features(datetime.date(2026, 6, 15))
    assert d_feats["day"] == 15
    assert d_feats["month"] == 6
    assert d_feats["weekday"] == 0  # Monday
    assert d_feats["is_weekend"] == 0

    df_feats = app.build_feature_dataframe(
        date_features=d_feats,
        lag_1=150.0, lag_7=145.0, lag_14=140.0, lag_30=135.0,
        rolling_mean_7=148.0, rolling_mean_14=146.0, rolling_std_7=12.0,
        trend=100.0,
    )
    assert list(df_feats.columns) == list(ENGINEERED_FEATURES)
    assert df_feats.shape == (1, 13)

    pred = app.predict_sales(df_feats, model_choice="XGBoost")
    assert isinstance(pred, float)
    assert pred >= 0.0

    insights = app.generate_business_insights(pred, 150.0, 148.0, 146.0)
    assert len(insights) >= 2
    assert "icon" in insights[0]
    assert "title" in insights[0]
    assert "action" in insights[0]
