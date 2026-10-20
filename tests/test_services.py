"""
SalesPulse AI - Tests for Domain Services Module (Phase 3A)
Verifies:
- ModelRegistry standalone loading without Streamlit runtime dependencies.
- Explicit model selection with no silent fallback.
- Exact numerical parity between ForecastingService and legacy dashboard forecasting.
- Ensemble prediction arithmetic parity.
- Explicit interval methodology separation ('heuristic_sqrt_step' vs 'calibrated_oof_residual').
- ForecastingService and ValidationService audit persistence integration.
"""

import os
import tempfile
import datetime
import pytest
import numpy as np
import pandas as pd

from salespulse.services.model_registry import ModelRegistry, compute_file_sha256
from salespulse.services.forecasting_service import (
    ForecastingService,
    extract_date_features,
    build_feature_dataframe,
    validate_numeric_scalar,
)
from salespulse.services.validation_service import ValidationService
from salespulse.storage.database import create_storage_engine, init_db
from salespulse.storage.repository import AuditRepository
from salespulse.features.temporal import ENGINEERED_FEATURES


@pytest.fixture
def standalone_repo():
    """Temporary storage repository for service testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()

    engine = create_storage_engine(db_url=f"sqlite:///{tmp_path}")
    init_db(engine=engine)
    repo = AuditRepository(engine=engine)

    yield {"repo": repo, "engine": engine, "path": tmp_path}

    engine.dispose()
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except PermissionError:
            pass
    for ext in ("-wal", "-shm"):
        if os.path.exists(tmp_path + ext):
            try:
                os.remove(tmp_path + ext)
            except PermissionError:
                pass


def test_model_registry_standalone_loading():
    """Verify registry loads models from model_data/ without Streamlit runtime."""
    registry = ModelRegistry()
    available = registry.list_models()
    assert "XGBoost" in available
    assert "RandomForest" in available
    assert "Ensemble" in available

    meta_xgb = registry.get_metadata("XGBoost")
    assert meta_xgb["artifact_sha256"] is not None
    assert meta_xgb["artifact_filename"] == "model.pkl"
    assert len(meta_xgb["feature_schema"]) == 13


def test_model_registry_explicit_selection_no_silent_fallback():
    """Verify requesting an unverified or nonexistent model raises KeyError without silent fallback."""
    registry = ModelRegistry()
    with pytest.raises(KeyError) as exc_info:
        registry.get_model("CatBoostRegressor")
    assert "CatBoostRegressor" in str(exc_info.value)
    assert "Loaded models:" in str(exc_info.value)


def test_forecasting_service_numerical_parity_with_legacy_dashboard():
    """Verify ForecastingService reproduces the legacy dashboard calculations with exact float parity."""
    # 1. Setup fixture input
    date_feats = extract_date_features("2024-06-15")
    initial_df = build_feature_dataframe(
        date_feats=date_feats,
        lag_1=160.0,
        lag_7=155.0,
        lag_14=150.0,
        lag_30=145.0,
        rolling_mean_7=158.0,
        rolling_mean_14=152.0,
        rolling_std_7=15.0,
        trend=100.0,
    )

    registry = ModelRegistry()
    service = ForecastingService(registry=registry)
    origin_dt = datetime.datetime(2024, 6, 15, 0, 0, 0)

    # 2. Execute via ForecastingService
    service_result = service.forecast_multi_horizon(
        initial_df=initial_df,
        horizon_days=7,
        model_choice="XGBoost",
        origin_date=origin_dt,
        interval_method="heuristic_sqrt_step",
    )

    # 3. Execute legacy dashboard logic directly (as extracted from app.py:L327-L373)
    xgb_model = registry.get_model("XGBoost")
    legacy_dates = []
    legacy_preds = []
    legacy_lowers = []
    legacy_uppers = []

    curr_df = initial_df.copy()
    history = [curr_df['lag_1'].iloc[0]]
    std_7 = curr_df['rolling_std_7'].iloc[0]

    for i in range(1, 8):
        target_date = origin_dt + datetime.timedelta(days=i)
        legacy_dates.append(target_date)

        target_dt = pd.Timestamp(target_date)
        d_feats = {
            "day": int(target_dt.day),
            "month": int(target_dt.month),
            "weekday": int(target_dt.weekday()),
            "is_weekend": 1 if target_dt.weekday() >= 5 else 0,
            "weekofyear": int(target_dt.isocalendar().week)
        }

        lag1_new = history[-1]
        lag7_new = history[-7] if len(history) >= 7 else history[0]
        lag14_new = history[-14] if len(history) >= 14 else history[0]
        lag30_new = history[-30] if len(history) >= 30 else history[0]

        rm7_new = np.mean(history[-7:])
        rm14_new = np.mean(history[-14:]) if len(history) >= 14 else rm7_new
        rs7_new = np.std(history[-7:]) if len(history) >= 7 else std_7
        trend_new = curr_df['trend'].iloc[0] + i

        step_dict = {
            "day": float(d_feats["day"]), "month": float(d_feats["month"]),
            "weekday": float(d_feats["weekday"]), "is_weekend": float(d_feats["is_weekend"]),
            "weekofyear": float(d_feats["weekofyear"]), "lag_1": float(lag1_new),
            "lag_7": float(lag7_new), "lag_14": float(lag14_new), "lag_30": float(lag30_new),
            "rolling_mean_7": float(rm7_new), "rolling_mean_14": float(rm14_new),
            "rolling_std_7": float(rs7_new), "trend": float(trend_new),
        }
        step_df = pd.DataFrame([step_dict])[list(ENGINEERED_FEATURES)].astype(np.float64)
        val = max(0.0, float(xgb_model.predict(step_df)[0]))
        legacy_preds.append(val)
        history.append(val)

        unc = 1.96 * max(10.0, rs7_new) * np.sqrt(i / 2.0)
        legacy_lowers.append(max(0.0, val - unc))
        legacy_uppers.append(val + unc)

    # 4. Assert exact numerical parity
    np.testing.assert_allclose(
        service_result["Forecasted Sales ($)"].values,
        legacy_preds,
        rtol=1e-7,
        atol=1e-7,
        err_msg="ForecastingService predictions do not match legacy dashboard values!",
    )
    np.testing.assert_allclose(
        service_result["Lower Bound ($)"].values,
        legacy_lowers,
        rtol=1e-7,
        atol=1e-7,
        err_msg="ForecastingService lower bounds do not match legacy dashboard values!",
    )
    np.testing.assert_allclose(
        service_result["Upper Bound ($)"].values,
        legacy_uppers,
        rtol=1e-7,
        atol=1e-7,
        err_msg="ForecastingService upper bounds do not match legacy dashboard values!",
    )


def test_forecasting_service_ensemble_parity():
    """Verify Ensemble prediction is the arithmetic mean of XGBoost and RandomForest."""
    registry = ModelRegistry()
    service = ForecastingService(registry=registry)
    date_feats = extract_date_features("2024-06-15")
    df = build_feature_dataframe(date_feats, 150.0, 145.0, 140.0, 150.0, 148.0, 145.0, 12.0, 100.0)

    p_xgb = service.predict_single_step(df, model_choice="XGBoost")
    p_rf = service.predict_single_step(df, model_choice="RandomForest")
    p_ens = service.predict_single_step(df, model_choice="Ensemble")

    expected_ens = max(0.0, (p_xgb + p_rf) / 2.0)
    assert p_ens == pytest.approx(expected_ens, abs=1e-6)


def test_forecasting_service_interval_methods_distinction():
    """Verify heuristic_sqrt_step and calibrated_oof_residual use distinct uncertainty formulas."""
    registry = ModelRegistry()
    service = ForecastingService(registry=registry)
    date_feats = extract_date_features("2024-06-15")
    df = build_feature_dataframe(date_feats, 150.0, 145.0, 140.0, 150.0, 148.0, 145.0, 12.0, 100.0)

    res_heuristic = service.forecast_multi_horizon(df, horizon_days=7, interval_method="heuristic_sqrt_step")
    res_calibrated = service.forecast_multi_horizon(df, horizon_days=7, interval_method="calibrated_oof_residual", residual_std=50.0)

    # Point predictions are identical
    np.testing.assert_allclose(
        res_heuristic["Forecasted Sales ($)"].values,
        res_calibrated["Forecasted Sales ($)"].values,
        rtol=1e-7,
    )
    # Interval bounds differ due to different scaling formulas
    assert not np.allclose(
        res_heuristic["Upper Bound ($)"].values,
        res_calibrated["Upper Bound ($)"].values,
    )


def test_forecasting_service_persistence_integration(standalone_repo):
    """Verify forecast run and items are persisted when persist_run=True."""
    repo = standalone_repo["repo"]
    registry = ModelRegistry()
    service = ForecastingService(registry=registry, repository=repo)

    date_feats = extract_date_features("2024-06-15")
    df = build_feature_dataframe(date_feats, 150.0, 145.0, 140.0, 150.0, 148.0, 145.0, 12.0, 100.0)

    res = service.forecast_multi_horizon(
        initial_df=df,
        horizon_days=7,
        model_choice="XGBoost",
        origin_date="2024-06-15",
        persist_run=True,
        run_id="run-persist-service-01",
    )
    assert len(res) == 7

    logged_run = repo.get_forecast_run("run-persist-service-01")
    assert logged_run is not None
    assert logged_run.model_name == "XGBoost"
    assert len(logged_run.items) == 7


def test_validation_service_integration_and_audit_persistence(standalone_repo):
    """Verify ValidationService coordinates contract validation, SHA-256 computation, and audit logging."""
    repo = standalone_repo["repo"]
    service = ValidationService(repository=repo)

    valid_csv = b"Order Date,Sales,Row ID\n2024-01-01,100.0,1\n2024-01-02,150.0,2\n"
    res = service.validate_file(
        file_source=valid_csv,
        file_name="/tmp/uploads/sales_jan.csv",
        persist_audit=True,
    )

    assert res.report.status in ("PASSED", "WARNING")
    assert res.report.valid_rows_count == 2

    # Check persistence in audit DB
    import hashlib
    sha = hashlib.sha256(valid_csv).hexdigest()
    audit_entry = repo.get_validation_by_hash(sha)
    assert audit_entry is not None
    assert audit_entry.row_count == 2
    assert audit_entry.file_name == "sales_jan.csv"  # Sanitized


def test_validation_numeric_scalar_guards():
    """Verify numeric scalar validator rejects NaN, Inf, and below-minimum values."""
    assert validate_numeric_scalar(15.0, "test") == 15.0

    with pytest.raises(ValueError):
        validate_numeric_scalar(float("nan"), "test")

    with pytest.raises(ValueError):
        validate_numeric_scalar(float("inf"), "test")

    with pytest.raises(ValueError):
        validate_numeric_scalar(-5.0, "test", min_val=0.0)
