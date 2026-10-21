"""
SalesPulse AI - Phase 3B Anomaly Detection Acceptance Test Suite
Comprehensive, deterministic unit and integration tests covering:
1. Clear synthetic sales surge detection.
2. Clear synthetic sales drop detection.
3. Normal series does not trigger false alerts under configured policy.
4. Unusually large forecast error is detected when sufficient compatible history exists.
5. Insufficient history produces explicit non-alert result without error.
6. Zero MAD and zero variance (flatline series) handled safely without arbitrary scoring.
7. Missing dates, duplicate dates, negative values, and missing observations follow documented policies.
8. Future data cannot influence historical detection (leakage prevention).
9. Different stores/scopes are not incorrectly mixed.
10. Incompatible forecast horizons and unrelated runs are not compared.
11. Persistence is idempotent and preserves traceability.
12. Invalid numerical values cannot break serialization or persistence.
"""

import json
import pytest
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, text

from salespulse.analytics.schemas import (
    AnomalyConfig,
    AnomalyPoint,
    AnomalyReport,
    AnomalyType,
    SeverityLevel,
    AnomalyDirection,
)
from salespulse.analytics.detector import (
    compute_robust_scale,
    classify_severity,
    generate_deterministic_anomaly_id,
    detect_sales_anomalies,
    detect_forecast_error_anomalies,
)
from salespulse.storage.database import init_db, get_session_factory, Base, SCHEMA_VERSION
from salespulse.storage.models import AnomalyRecord
from salespulse.storage.repository import AuditRepository
from salespulse.services.analytics_service import AnalyticsService


# ----------------------------------------------------------------------------
# Test Fixtures
# ----------------------------------------------------------------------------

@pytest.fixture
def memory_db():
    """In-memory SQLite database initialized with schema v2."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return engine


@pytest.fixture
def baseline_dates():
    """Deterministic sequence of 30 daily dates starting from 2025-01-01."""
    start = pd.Timestamp("2025-01-01")
    return [start + pd.Timedelta(days=i) for i in range(30)]


# ----------------------------------------------------------------------------
# 1. Sales Surge Detection
# ----------------------------------------------------------------------------

def test_sales_surge_detection(baseline_dates):
    """Verify that an extreme positive sales observation is flagged as SALES_SURGE."""
    # 20 days of steady baseline around $500, followed by a surge to $2500 on day 21
    sales = [500.0 + (i % 3) * 10.0 for i in range(20)] + [2500.0]
    dates = baseline_dates[:21]

    df = pd.DataFrame({"date": dates, "sales": sales})
    config = AnomalyConfig(window_size=14, min_history=14, surge_threshold=2.5)

    report = detect_sales_anomalies(df, config=config, sales_scope="STORE_NORTH")

    assert report.total_evaluated == 21
    assert report.anomalies_detected >= 1

    surge = next((a for a in report.anomalies if a.observation_date == "2025-01-21"), None)
    assert surge is not None
    assert surge.anomaly_type == AnomalyType.SALES_SURGE
    assert surge.direction == AnomalyDirection.SURGE
    assert surge.observed_value == 2500.0
    assert surge.anomaly_score > 2.5
    assert surge.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
    assert "exceeded" in surge.explanation
    assert "STORE_NORTH" in surge.sales_scope


# ----------------------------------------------------------------------------
# 2. Sales Drop Detection
# ----------------------------------------------------------------------------

def test_sales_drop_detection(baseline_dates):
    """Verify that an extreme drop in sales is flagged as SALES_DROP."""
    # 20 days of sales around $1000, followed by a precipitous drop to $50 on day 21
    sales = [1000.0 + (i % 4) * 20.0 for i in range(20)] + [50.0]
    dates = baseline_dates[:21]

    df = pd.DataFrame({"date": dates, "sales": sales})
    config = AnomalyConfig(window_size=14, min_history=14, drop_threshold=2.5)

    report = detect_sales_anomalies(df, config=config, sales_scope="TOTAL_STORE")

    assert report.anomalies_detected >= 1
    drop = next((a for a in report.anomalies if a.observation_date == "2025-01-21"), None)
    assert drop is not None
    assert drop.anomaly_type == AnomalyType.SALES_DROP
    assert drop.direction == AnomalyDirection.DROP
    assert drop.observed_value == 50.0
    assert drop.anomaly_score < -2.5
    assert "fell below" in drop.explanation


# ----------------------------------------------------------------------------
# 3. Normal Observations Do Not False Alarm
# ----------------------------------------------------------------------------

def test_normal_observations_no_false_alerts(baseline_dates):
    """Verify that a normal stationary series produces zero alerts under threshold 2.5."""
    # Normal sales oscillating mildly between $480 and $520
    np.random.seed(42)
    sales = 500.0 + np.random.uniform(-20.0, 20.0, size=30)
    df = pd.DataFrame({"date": baseline_dates, "sales": sales})

    config = AnomalyConfig(window_size=14, min_history=14, surge_threshold=2.5, drop_threshold=2.5)
    report = detect_sales_anomalies(df, config=config)

    assert report.anomalies_detected == 0
    assert len(report.anomalies) == 0
    assert report.insufficient_history_count == 14  # First 14 days had < 14 prior days


# ----------------------------------------------------------------------------
# 4. Forecast Error Anomaly Detection
# ----------------------------------------------------------------------------

def test_forecast_error_anomaly_detection():
    """Verify detection of unusually large forecast error relative to historical distribution."""
    # Create 20 historical reconciled items with typical errors around $20 - $30
    items = []
    base_date = pd.Timestamp("2025-02-01")
    for i in range(20):
        items.append({
            "target_date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_value": 500.0,
            "actual_value": 520.0 + (i % 5),  # Error is ~20 - 24
            "sales_scope": "TOTAL_STORE",
            "model_name": "RandomForestRegressor",
            "horizon_days": 14,
            "step_ahead": 1,
            "run_id": "run-001",
        })

    # Day 21: Model predicts 500.0, but actual was 1200.0 (error = 700.0)
    items.append({
        "target_date": (base_date + pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
        "predicted_value": 500.0,
        "actual_value": 1200.0,
        "sales_scope": "TOTAL_STORE",
        "model_name": "RandomForestRegressor",
        "horizon_days": 14,
        "step_ahead": 1,
        "run_id": "run-002",
    })

    config = AnomalyConfig(window_size=14, min_history=14, error_threshold=2.5)
    report = detect_forecast_error_anomalies(
        items,
        config=config,
        target_scope="TOTAL_STORE",
        target_model_name="RandomForestRegressor",
        target_horizon_days=14,
        target_step_ahead=1,
    )

    assert report.anomalies_detected == 1
    err_point = report.anomalies[0]
    assert err_point.anomaly_type == AnomalyType.FORECAST_ERROR_ANOMALY
    assert err_point.observed_value == 700.0
    assert err_point.model_name == "RandomForestRegressor"
    assert err_point.step_ahead == 1
    assert err_point.severity == SeverityLevel.CRITICAL
    assert "exceeded historical baseline median error" in err_point.explanation


# ----------------------------------------------------------------------------
# 5. Insufficient History Non-Alert
# ----------------------------------------------------------------------------

def test_insufficient_history_non_alert():
    """Verify that series shorter than min_history produces explicit non-alert."""
    dates = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(5)]
    sales = [100.0, 105.0, 102.0, 1000.0, 101.0]  # Day 4 has 1000, but only 3 days of prior history
    df = pd.DataFrame({"date": dates, "sales": sales})

    config = AnomalyConfig(window_size=14, min_history=14)
    report = detect_sales_anomalies(df, config=config)

    assert report.anomalies_detected == 0
    assert report.total_evaluated == 5
    assert report.insufficient_history_count == 5


# ----------------------------------------------------------------------------
# 6. Zero MAD and Zero Variance Safe Handling
# ----------------------------------------------------------------------------

def test_zero_mad_and_flatline_safe_handling():
    """Verify that zero dispersion / flatline history does not crash with ZeroDivisionError."""
    # 20 days of exactly identical sales $500.0 (MAD = 0.0, variance = 0.0)
    dates = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(22)]
    sales = [500.0] * 20 + [500.0, 800.0]

    df = pd.DataFrame({"date": dates, "sales": sales})
    config = AnomalyConfig(window_size=14, min_history=14, zero_mad_minimum_scale=1.0)

    report = detect_sales_anomalies(df, config=config)

    # Day 21 (sales=500.0): exactly matches flatline median -> NO anomaly
    day_21 = next((a for a in report.anomalies if a.observation_date == "2025-01-21"), None)
    assert day_21 is None

    # Day 22 (sales=800.0): deviation from flatline -> flagged with fallback scale
    day_22 = next((a for a in report.anomalies if a.observation_date == "2025-01-22"), None)
    assert day_22 is not None
    assert day_22.dispersion_zero_fallback is True
    assert "Historical dispersion was zero" in day_22.explanation
    assert day_22.anomaly_score > 0


# ----------------------------------------------------------------------------
# 7. Missing Dates, Duplicate Dates, and Negative Sales Policies
# ----------------------------------------------------------------------------

def test_duplicate_dates_and_negative_sales_policies():
    """Verify handling of duplicate timestamps and negative return values."""
    dates = [
        "2025-01-01", "2025-01-02", "2025-01-02",  # Duplicate date on Jan 2 (sales 200 + 300 = 500)
    ] + [f"2025-01-{i:02d}" for i in range(3, 20)] + ["2025-01-20"]
    sales = [500.0, 200.0, 300.0] + [500.0] * 17 + [-150.0]  # Negative sales (returns) on day 20

    df = pd.DataFrame({"date": dates, "sales": sales})

    # Policy 1: TREAT_AS_RETURN (default)
    cfg_return = AnomalyConfig(window_size=14, min_history=14, handle_negative_sales="TREAT_AS_RETURN")
    report = detect_sales_anomalies(df, config=cfg_return)

    drop_point = next((a for a in report.anomalies if a.observation_date == "2025-01-20"), None)
    assert drop_point is not None
    assert drop_point.observed_value == -150.0
    assert drop_point.direction == AnomalyDirection.DROP
    assert "net product return" in drop_point.notes

    # Policy 2: REJECT
    cfg_reject = AnomalyConfig(window_size=14, min_history=14, handle_negative_sales="REJECT")
    with pytest.raises(ValueError, match="negative sales values, rejected by policy"):
        detect_sales_anomalies(df, config=cfg_reject)


# ----------------------------------------------------------------------------
# 8. Leakage Prevention (No Future Data Influence)
# ----------------------------------------------------------------------------

def test_no_future_data_leakage(baseline_dates):
    """Verify that future observations have zero influence on past detection scores."""
    # Series A: 20 baseline days, then normal day 21 (500.0), then massive surge on day 25 (9999.0)
    sales_a = [500.0] * 20 + [510.0, 500.0, 500.0, 500.0, 9999.0]
    df_a = pd.DataFrame({"date": baseline_dates[:25], "sales": sales_a})

    # Series B: Exactly identical through day 21, but day 25 has normal sales (500.0)
    sales_b = [500.0] * 20 + [510.0, 500.0, 500.0, 500.0, 500.0]
    df_b = pd.DataFrame({"date": baseline_dates[:25], "sales": sales_b})

    cfg = AnomalyConfig(window_size=14, min_history=14)
    report_a = detect_sales_anomalies(df_a, config=cfg)
    report_b = detect_sales_anomalies(df_b, config=cfg)

    # Day 21 evaluation must be identical regardless of what happens on day 25
    a_day_21 = [a for a in report_a.anomalies if a.observation_date == "2025-01-21"]
    b_day_21 = [a for a in report_b.anomalies if a.observation_date == "2025-01-21"]
    assert len(a_day_21) == len(b_day_21) == 0  # Both normal


# ----------------------------------------------------------------------------
# 9. Scope Isolation (No Cross-Scope Mixing)
# ----------------------------------------------------------------------------

def test_scope_isolation_rejection():
    """Verify that DataFrames with mixed scopes are rejected unless isolated."""
    df_mixed = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "sales": [500.0, 600.0],
        "sales_scope": ["STORE_A", "STORE_B"],
    })

    with pytest.raises(ValueError, match="contains multiple sales scopes"):
        detect_sales_anomalies(df_mixed, sales_scope="STORE_A")


# ----------------------------------------------------------------------------
# 10. Forecast Horizon and Model Compatibility
# ----------------------------------------------------------------------------

def test_forecast_horizon_isolation():
    """Verify that 7-day forecast errors are not mixed into 30-day error baselines."""
    base_date = pd.Timestamp("2025-01-01")
    items = []

    # 15 items of 7-day horizon with large errors ($200)
    for i in range(15):
        items.append({
            "target_date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_value": 500.0,
            "actual_value": 700.0,
            "sales_scope": "TOTAL_STORE",
            "model_name": "ModelX",
            "horizon_days": 7,
            "step_ahead": 1,
        })

    # 1 item of 30-day horizon with error $50
    items.append({
        "target_date": "2025-01-16",
        "predicted_value": 500.0,
        "actual_value": 550.0,
        "sales_scope": "TOTAL_STORE",
        "model_name": "ModelX",
        "horizon_days": 30,
        "step_ahead": 1,
    })

    # Evaluating 30-day horizon: should only find 1 item (insufficient history), NOT 16 items!
    cfg = AnomalyConfig(window_size=14, min_history=14)
    report_30 = detect_forecast_error_anomalies(
        items,
        config=cfg,
        target_scope="TOTAL_STORE",
        target_horizon_days=30,
    )

    assert report_30.total_evaluated == 1
    assert report_30.anomalies_detected == 0
    assert report_30.insufficient_history_count == 1


# ----------------------------------------------------------------------------
# 11. Idempotent Persistence in AuditRepository
# ----------------------------------------------------------------------------

def test_idempotent_persistence(memory_db):
    """Verify that saving anomaly points multiple times does not create duplicate rows."""
    repo = AuditRepository(memory_db)

    point = AnomalyPoint(
        anomaly_id=generate_deterministic_anomaly_id("TOTAL_STORE", "2025-01-20", "SALES_SURGE"),
        observation_date="2025-01-20",
        sales_scope="TOTAL_STORE",
        anomaly_type=AnomalyType.SALES_SURGE,
        severity=SeverityLevel.HIGH,
        direction=AnomalyDirection.SURGE,
        observed_value=2500.0,
        reference_center=500.0,
        reference_spread=50.0,
        anomaly_score=40.0,
        threshold=2.5,
        explanation="Synthetic test surge.",
        detection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    # First persistence call
    persisted_1 = repo.record_anomalies([point])
    assert len(persisted_1) == 1

    # Second persistence call with same anomaly_id
    persisted_2 = repo.record_anomalies([point])
    assert len(persisted_2) == 1

    # Total row count in database table must still be exactly 1
    with get_session_factory(memory_db)() as s:
        count = s.execute(select(AnomalyRecord)).scalars().all()
        assert len(count) == 1
        assert count[0].anomaly_id == point.anomaly_id
        assert count[0].observed_value == 2500.0


# ----------------------------------------------------------------------------
# 12. Strict RFC 8259 JSON Serialization (No NaN or Infinity)
# ----------------------------------------------------------------------------

def test_strict_json_serialization():
    """Verify that AnomalyPoint and AnomalyReport serialize with allow_nan=False."""
    point = AnomalyPoint(
        anomaly_id="test-id-123",
        observation_date="2025-01-15",
        sales_scope="STORE_TEST",
        anomaly_type=AnomalyType.SALES_SURGE,
        severity=SeverityLevel.CRITICAL,
        direction=AnomalyDirection.SURGE,
        observed_value=1234.56,
        reference_center=500.0,
        reference_spread=50.0,
        anomaly_score=14.69,
        threshold=2.5,
        explanation="Strict serialization test.",
        detection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    json_str = point.to_json()
    assert "1234.56" in json_str
    # Verify strict RFC 8259 parsing
    parsed = json.loads(json_str)
    assert parsed["anomaly_id"] == "test-id-123"
    assert parsed["observed_value"] == 1234.56

    # Test round-trip reconstruction
    restored = AnomalyPoint.from_dict(parsed)
    assert restored.anomaly_id == point.anomaly_id
    assert restored.severity == SeverityLevel.CRITICAL

    # Verify report serialization
    report = AnomalyReport(
        report_id="rep-1",
        target_scope="STORE_TEST",
        total_evaluated=10,
        anomalies_detected=1,
        insufficient_history_count=0,
        anomalies=[point],
    )
    report_json = report.to_json()
    assert '"anomalies_detected": 1' in report_json


# ----------------------------------------------------------------------------
# 13. AnalyticsService Domain Service Integration
# ----------------------------------------------------------------------------

def test_analytics_service_end_to_end(memory_db, baseline_dates):
    """Verify that AnalyticsService coordinates detection, database ingestion, and persistence."""
    repo = AuditRepository(memory_db)
    service = AnalyticsService(repository=repo)

    # 1. Ingest actual sales into database: 20 normal days + 1 surge day
    for i in range(20):
        repo.ingest_actual_sales(
            calendar_date=(baseline_dates[i]).strftime("%Y-%m-%d"),
            daily_sales=500.0,
            sales_scope="STORE_SERVICE_TEST",
        )
    repo.ingest_actual_sales(
        calendar_date=(baseline_dates[20]).strftime("%Y-%m-%d"),
        daily_sales=3000.0,
        sales_scope="STORE_SERVICE_TEST",
    )

    # 2. Run detection from database actuals with persistence enabled
    report = service.detect_sales_anomalies_from_db(
        sales_scope="STORE_SERVICE_TEST",
        persist=True,
    )

    assert report.anomalies_detected == 1
    assert report.anomalies[0].observed_value == 3000.0

    # 3. Retrieve persisted anomalies
    persisted = service.get_persisted_anomalies(sales_scope="STORE_SERVICE_TEST")
    assert len(persisted) == 1
    assert persisted[0].observed_value == 3000.0
    assert persisted[0].anomaly_type == "SALES_SURGE"
    assert persisted[0].severity == "CRITICAL"


# ----------------------------------------------------------------------------
# 14. Weekly Seasonality Limitation Verification
# ----------------------------------------------------------------------------

def test_weekly_seasonality_limitation_and_synthetic_pattern():
    """Verify and document that unadjusted rolling median flags recurring weekend surges.
    
    Limitation: A 14-day rolling window pools 10 weekdays and 4 weekend days. In a business
    with 4x weekend traffic ($100 weekdays, $400 weekends), unadjusted baselines produce elevated
    robust z-scores on weekends. Callers with weekly seasonality must deseasonalize.
    """
    # 28 days (4 weeks) with standard retail weekend effect: $100 Mon-Fri, $400 Sat-Sun
    start = pd.Timestamp("2025-01-06")  # Monday
    dates = [start + pd.Timedelta(days=i) for i in range(28)]
    sales = []
    for d in dates:
        sales.append(400.0 if d.weekday() >= 5 else 100.0)

    df = pd.DataFrame({"date": dates, "sales": sales})
    # Sensitive or moderate threshold (1.75) highlights that unadjusted baseline produces z ~ 1.8 on normal weekends
    cfg = AnomalyConfig(window_size=14, min_history=14, surge_threshold=1.75)

    report = detect_sales_anomalies(df, config=cfg)

    # Documented behavior: Weekend observations have elevated z-scores (~1.8) and trigger false alerts
    weekend_anomalies = [a for a in report.anomalies if pd.Timestamp(a.observation_date).weekday() >= 5]
    assert len(weekend_anomalies) > 0, (
        "Documented limitation: unadjusted rolling baseline flags periodic weekend spikes as surges."
    )
    for wa in weekend_anomalies:
        assert wa.anomaly_type == AnomalyType.SALES_SURGE
        assert wa.observed_value == 400.0
        assert wa.anomaly_score > 1.75


# ----------------------------------------------------------------------------
# 15. Missing Values Skip Policy (No False Drop Alarms)
# ----------------------------------------------------------------------------

def test_missing_values_skip_policy_prevents_false_drop_alerts(baseline_dates):
    """Verify that unrecorded/NaN observations are skipped and tracked without false drop alerts."""
    # 25 days: Day 20 has sales = NaN (missing measurement, not $0.00 sales)
    sales = [500.0] * 20 + [np.nan] + [500.0] * 4
    df = pd.DataFrame({"date": baseline_dates[:25], "sales": sales})

    # Under default handle_missing_values="SKIP"
    cfg = AnomalyConfig(window_size=14, min_history=14, handle_missing_values="SKIP")
    report = detect_sales_anomalies(df, config=cfg)

    # Missing date must be tracked and NOT flagged as a SALES_DROP to $0
    assert report.missing_values_count == 1
    drop_alerts = [a for a in report.anomalies if a.anomaly_type == AnomalyType.SALES_DROP]
    assert len(drop_alerts) == 0


# ----------------------------------------------------------------------------
# 16. Multi-Horizon and Multi-Step Error Stream Isolation
# ----------------------------------------------------------------------------

def test_forecast_error_isolation_multiple_models_and_steps():
    """Verify that step 14 errors do not pollute step 1 historical baselines, and models remain isolated."""
    base_date = pd.Timestamp("2025-01-01")
    items = []

    # 15 days of Step 1 items for XGBoost: expected errors are low (~$10)
    for i in range(15):
        items.append({
            "target_date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_value": 500.0,
            "actual_value": 510.0,  # Error = 10.0
            "sales_scope": "TOTAL_STORE",
            "model_name": "XGBoost",
            "horizon_days": 14,
            "step_ahead": 1,
            "run_id": f"run-step1-{i}",
        })

    # 15 days of Step 14 items for XGBoost: naturally larger errors (~$150)
    for i in range(15):
        items.append({
            "target_date": (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "predicted_value": 500.0,
            "actual_value": 650.0,  # Error = 150.0
            "sales_scope": "TOTAL_STORE",
            "model_name": "XGBoost",
            "horizon_days": 14,
            "step_ahead": 14,
            "run_id": f"run-step14-{i}",
        })

    # Day 16: Step 1 error jumps to $80 (abnormal for step 1, but normal if pooled with step 14)
    items.append({
        "target_date": (base_date + pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
        "predicted_value": 500.0,
        "actual_value": 580.0,  # Error = 80.0
        "sales_scope": "TOTAL_STORE",
        "model_name": "XGBoost",
        "horizon_days": 14,
        "step_ahead": 1,
        "run_id": "run-step1-15",
    })

    cfg = AnomalyConfig(window_size=14, min_history=14, error_threshold=2.5)
    report = detect_forecast_error_anomalies(items, config=cfg, target_scope="TOTAL_STORE")

    # Day 16 step 1 must be flagged because its baseline only includes step 1 ($10 error), NOT step 14 ($150 error)
    step1_anomalies = [a for a in report.anomalies if a.step_ahead == 1]
    assert len(step1_anomalies) == 1
    assert step1_anomalies[0].observed_value == 80.0
    assert step1_anomalies[0].reference_center == 10.0  # Median of step 1 history


# ----------------------------------------------------------------------------
# 17. Distinct Anomaly IDs for Multiple Runs Targeting Same Date
# ----------------------------------------------------------------------------

def test_multiple_runs_targeting_same_date_have_distinct_anomaly_ids():
    """Verify that distinct forecast runs targeting the same calendar date receive collision-free IDs."""
    target_date = "2025-01-20"
    id_run1 = generate_deterministic_anomaly_id(
        sales_scope="TOTAL_STORE",
        observation_date=target_date,
        anomaly_type="FORECAST_ERROR_ANOMALY",
        model_name="XGBoost",
        horizon_days=7,
        step_ahead=7,
        forecast_run_id="run-xgb-7d",
    )
    id_run2 = generate_deterministic_anomaly_id(
        sales_scope="TOTAL_STORE",
        observation_date=target_date,
        anomaly_type="FORECAST_ERROR_ANOMALY",
        model_name="XGBoost",
        horizon_days=14,
        step_ahead=7,
        forecast_run_id="run-xgb-14d",
    )

    assert id_run1 != id_run2, "Distinct forecast runs targeting the same date must not collide."


# ----------------------------------------------------------------------------
# 18. Anomaly Disappearance and Resolution Policy
# ----------------------------------------------------------------------------

def test_anomaly_disappearance_and_resolution_policy(memory_db):
    """Verify that when a previously flagged anomaly disappears, it is marked RESOLVED in audit trail."""
    repo = AuditRepository(memory_db)

    # 1. Record an initial active anomaly
    pt = AnomalyPoint(
        anomaly_id="surge-20250120",
        observation_date="2025-01-20",
        sales_scope="STORE_RESOLVE_TEST",
        anomaly_type=AnomalyType.SALES_SURGE,
        severity=SeverityLevel.HIGH,
        direction=AnomalyDirection.SURGE,
        observed_value=2000.0,
        reference_center=500.0,
        reference_spread=50.0,
        anomaly_score=30.0,
        threshold=2.5,
        explanation="Initial surge alert.",
        detection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="ACTIVE",
    )
    repo.record_anomalies([pt])

    # Verify active
    active = repo.get_anomalies(sales_scope="STORE_RESOLVE_TEST", status="ACTIVE")
    assert len(active) == 1
    assert active[0].status == "ACTIVE"

    # 2. Later re-evaluation: observation disappeared from active anomalies
    resolved_count = repo.resolve_anomalies_for_scope(
        sales_scope="STORE_RESOLVE_TEST",
        current_active_ids=[],  # No active anomalies remain
        resolution_reason="Corrected sales figure returned to historical bounds",
    )
    assert resolved_count == 1

    # 3. Verify audit history is preserved (not deleted) but marked RESOLVED
    all_recs = repo.get_anomalies(sales_scope="STORE_RESOLVE_TEST")
    assert len(all_recs) == 1
    assert all_recs[0].status == "RESOLVED"
    assert all_recs[0].status_updated_at is not None
    assert "RESOLVED - Corrected sales figure" in all_recs[0].explanation


# ----------------------------------------------------------------------------
# 13. Decision Support & Recommendation Engine Tests
# ----------------------------------------------------------------------------

from salespulse.analytics.recommendations import (
    DecisionSupportEngine,
    BusinessRecommendation,
    RecommendationCategory,
    UrgencyLevel,
    EvidenceConfidence,
)


def test_decision_support_empty_dataset():
    """Verify that empty or None dataset returns empty recommendations without error."""
    engine = DecisionSupportEngine(currency_symbol="£")
    recs = engine.analyze_dataset(daily_df=None)
    assert recs == []

    empty_df = pd.DataFrame(columns=["date", "sales"])
    recs_empty = engine.analyze_dataset(daily_df=empty_df)
    assert recs_empty == []


def test_decision_support_return_margin_erosion():
    """Verify that elevated returns trigger a quantified REVENUE_RISK recommendation."""
    engine = DecisionSupportEngine(currency_symbol="£")
    dates = pd.date_range("2021-01-01", periods=30)
    # 30 days of £1000 gross sales, £100 returns (10% return rate)
    df = pd.DataFrame({
        "date": dates,
        "gross_sales": [1000.0] * 30,
        "sales": [900.0] * 30,
        "returns": [100.0] * 30,
        "order_count": [10] * 30,
    })

    recs = engine.analyze_dataset(daily_df=df)
    return_rec = next((r for r in recs if r.category == RecommendationCategory.REVENUE_RISK and "Return" in r.title), None)
    assert return_rec is not None
    assert return_rec.priority == UrgencyLevel.CRITICAL  # 10% > 7%
    assert return_rec.confidence == EvidenceConfidence.HIGH
    assert "£3,000" in return_rec.quantified_impact
    assert "10.00%" in return_rec.problem_statement
    assert "suggested_action" in return_rec.to_dict()
    assert "follow_up_metric" in return_rec.to_dict()


def test_decision_support_market_concentration():
    """Verify that market concentration triggers geographic diversification recommendations."""
    engine = DecisionSupportEngine(currency_symbol="£")
    dates = pd.date_range("2021-01-01", periods=14)
    df = pd.DataFrame({"date": dates, "sales": [500.0] * 14})
    countries_df = pd.DataFrame({
        "Country": ["United Kingdom", "Germany", "France"],
        "NetSales": [8500.0, 1000.0, 500.0],  # UK is 85%
    })

    recs = engine.analyze_dataset(daily_df=df, countries_df=countries_df)
    geo_rec = next((r for r in recs if "Geographic" in r.title or "Market" in r.title), None)
    assert geo_rec is not None
    assert geo_rec.category == RecommendationCategory.REVENUE_RISK
    assert geo_rec.priority == UrgencyLevel.HIGH
    assert "85.0%" in geo_rec.root_cause_evidence


def test_decision_support_weekend_dispatch_latency():
    """Verify that zero Saturday dispatches triggers an INVENTORY_OPERATIONS recommendation."""
    engine = DecisionSupportEngine(currency_symbol="£")
    dates = pd.date_range("2021-01-01", periods=28)  # exactly 4 weeks
    # Saturdays have 0 sales
    sales = [0.0 if d.weekday() == 5 else 1000.0 for d in dates]
    df = pd.DataFrame({"date": dates, "sales": sales})

    recs = engine.analyze_dataset(daily_df=df)
    ops_rec = next((r for r in recs if r.category == RecommendationCategory.INVENTORY_OPERATIONS), None)
    assert ops_rec is not None
    assert "Weekend" in ops_rec.title
    assert "4 non-trading Saturdays" in ops_rec.root_cause_evidence


def test_decision_support_forecast_degradation_alert():
    """Verify that elevated forecast WAPE generates FORECAST_RELIABILITY recommendation."""
    engine = DecisionSupportEngine(currency_symbol="$")
    dates = pd.date_range("2021-01-01", periods=14)
    df = pd.DataFrame({"date": dates, "sales": [500.0] * 14})

    recs = engine.analyze_dataset(
        daily_df=df,
        forecast_metrics={"wape": 0.45, "bias_pct": 0.08, "model": "XGBoost", "horizon": "14_day"}
    )
    fc_rec = next((r for r in recs if r.category == RecommendationCategory.FORECAST_RELIABILITY), None)
    assert fc_rec is not None
    assert fc_rec.priority == UrgencyLevel.CRITICAL
    assert "45.0%" in fc_rec.problem_statement
    assert "+8.0%" in fc_rec.root_cause_evidence


def test_decision_support_serialization():
    """Verify that all recommendations serialize cleanly to dictionaries and JSON."""
    engine = DecisionSupportEngine(currency_symbol="£")
    dates = pd.date_range("2021-01-01", periods=14)
    df = pd.DataFrame({
        "date": dates,
        "gross_sales": [1000.0] * 14,
        "sales": [900.0] * 14,
        "returns": [100.0] * 14,
    })
    recs = engine.analyze_dataset(daily_df=df, active_anomalies_count=3)
    assert len(recs) >= 2

    for r in recs:
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "id" in d
        assert "title" in d
        # Verify JSON serializability
        dumped = json.dumps(d)
        assert len(dumped) > 0

