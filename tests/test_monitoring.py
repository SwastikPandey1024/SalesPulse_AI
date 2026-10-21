"""
SalesPulse AI - Phase 4A Data Freshness & Ingestion Monitoring Tests
Verifies:
- Detection of fresh, stale, and missing observation dates.
- Ingestion outcomes: PASSED, WARNING, FAILED, NO_ATTEMPTS.
- Status transition rules: HEALTHY, WARNING, CRITICAL, and UNKNOWN.
- Consecutive failure escalation.
- Threshold boundaries (exact boundary values).
- Database unavailable behavior (never falsely reports healthy).
- Idempotency of repeated checks.
- Historical monitoring check retrieval.
- Scheduling invocation boundary.
"""

import datetime
from unittest.mock import patch, MagicMock
import pytest
from sqlalchemy import create_engine

from salespulse.storage.database import init_db, get_db_session
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import DatasetAuditLog, ActualSalesRecord, DataMonitoringRecord
from salespulse.monitoring.schemas import (
    MonitoringStatus,
    FreshnessState,
    IngestionOutcome,
    CadenceType,
    MonitoringConfig,
    MonitoringVerdict,
)
from salespulse.monitoring.service import DataMonitoringService, run_scheduled_monitoring_check


@pytest.fixture
def mon_db(tmp_path):
    """Isolated SQLite test repository for monitoring checks."""
    db_file = tmp_path / "test_monitoring.db"
    engine = create_engine(f"sqlite:///{db_file}")
    init_db(engine)
    repo = AuditRepository(engine)
    service = DataMonitoringService(repository=repo, engine=engine)
    return {"engine": engine, "repo": repo, "service": service}


# ----------------------------------------------------------------------------
# 1. Fresh, Stale, and Missing Observation Dates
# ----------------------------------------------------------------------------
def test_monitoring_fresh_dataset(mon_db):
    """Verify that observations within threshold produce HEALTHY and FRESH state."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    # 1. Ingest clean validation log
    repo.record_validation(
        file_sha256="sha_clean_001",
        detected_grain="DAILY_AGGREGATED",
        row_count=100,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
        file_name="clean_daily_sales.csv",
    )

    # 2. Ingest recent actual observation (5 days before reference date)
    ref_date = datetime.date(2026, 6, 15)
    obs_date = datetime.date(2026, 6, 10)
    repo.ingest_actual_sales(calendar_date=obs_date, daily_sales=200.0, sales_scope="STORE_A")

    config = MonitoringConfig(
        staleness_threshold_days=30,
        reference_date=ref_date,
        sales_scope="STORE_A",
    )

    verdict = service.check_data_health(sales_scope="STORE_A", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.HEALTHY
    assert verdict.freshness.freshness_state == FreshnessState.FRESH
    assert verdict.freshness.staleness_days == 5
    assert verdict.freshness.is_stale is False
    assert verdict.ingestion.outcome == IngestionOutcome.PASSED
    assert verdict.ingestion.consecutive_failures == 0
    assert len(verdict.actionable_remediations) == 0


def test_monitoring_stale_dataset_triggers_warning(mon_db):
    """Verify observations older than staleness threshold trigger WARNING."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_stale_001",
        detected_grain="DAILY_AGGREGATED",
        row_count=100,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
    )

    # 40 days old relative to ref_date (threshold: 30 days, critical: 60 days)
    ref_date = datetime.date(2026, 6, 15)
    obs_date = datetime.date(2026, 5, 6)
    repo.ingest_actual_sales(calendar_date=obs_date, daily_sales=180.0, sales_scope="STORE_A")

    config = MonitoringConfig(
        staleness_threshold_days=30,
        critical_staleness_multiplier=2.0,
        reference_date=ref_date,
        sales_scope="STORE_A",
    )

    verdict = service.check_data_health(sales_scope="STORE_A", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.WARNING
    assert verdict.freshness.freshness_state == FreshnessState.STALE
    assert verdict.freshness.staleness_days == 40
    assert verdict.freshness.is_stale is True
    assert any("40 days old" in r for r in verdict.actionable_remediations)


def test_monitoring_severe_staleness_triggers_critical(mon_db):
    """Verify observations older than 2x threshold trigger CRITICAL status."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_severe_001",
        detected_grain="DAILY_AGGREGATED",
        row_count=50,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
    )

    # 70 days old (threshold: 30, critical: 60)
    ref_date = datetime.date(2026, 6, 15)
    obs_date = datetime.date(2026, 4, 6)
    repo.ingest_actual_sales(calendar_date=obs_date, daily_sales=180.0, sales_scope="STORE_A")

    config = MonitoringConfig(
        staleness_threshold_days=30,
        critical_staleness_multiplier=2.0,
        reference_date=ref_date,
        sales_scope="STORE_A",
    )

    verdict = service.check_data_health(sales_scope="STORE_A", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.CRITICAL
    assert verdict.freshness.staleness_days == 70
    assert "Critical Freshness Alert" in verdict.summary_message


def test_monitoring_missing_observations_uninitialized_baseline(mon_db):
    """Verify completely empty database returns UNKNOWN status (never falsely HEALTHY)."""
    service = mon_db["service"]
    config = MonitoringConfig(sales_scope="STORE_EMPTY")

    verdict = service.check_data_health(sales_scope="STORE_EMPTY", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.UNKNOWN
    assert verdict.freshness.freshness_state == FreshnessState.MISSING_DATA
    assert verdict.freshness.latest_observation_date is None
    assert verdict.ingestion.outcome == IngestionOutcome.NO_ATTEMPTS
    assert "No ingestion history" in verdict.summary_message


# ----------------------------------------------------------------------------
# 2. Ingestion Outcomes and Failure Runs
# ----------------------------------------------------------------------------
def test_monitoring_single_ingestion_failure_triggers_critical(mon_db):
    """Verify most recent failed validation triggers CRITICAL status."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_fail_001",
        detected_grain="TRANSACTION",
        row_count=100,
        validation_status="FAILED",
        is_usable=False,
        error_count=5,
        warning_count=2,
        report="""{"issues": [{"code": "DUPLICATE_ROWS", "severity": "ERROR", "message": "Duplicate daily dates found"}]}""",
        file_name="corrupt_batch.csv",
    )

    verdict = service.check_data_health(sales_scope="TOTAL_STORE", persist=True)

    assert verdict.overall_status == MonitoringStatus.CRITICAL
    assert verdict.ingestion.outcome == IngestionOutcome.FAILED
    assert verdict.ingestion.consecutive_failures == 1
    assert verdict.ingestion.recent_errors_count == 5
    assert len(verdict.ingestion.latest_issues_summary) > 0
    assert verdict.ingestion.latest_issues_summary[0]["code"] == "DUPLICATE_ROWS"


def test_monitoring_consecutive_failures_escalation(mon_db):
    """Verify multiple consecutive failures properly accumulate and trigger CRITICAL."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    # Ingest 3 failed validations in a row
    for i in range(1, 4):
        repo.record_validation(
            file_sha256=f"sha_fail_consec_{i}",
            detected_grain="DAILY_AGGREGATED",
            row_count=10,
            validation_status="FAILED",
            is_usable=False,
            error_count=1,
            warning_count=0,
            report="{}",
            file_name=f"bad_file_{i}.csv",
        )

    config = MonitoringConfig(consecutive_failure_threshold=2)
    verdict = service.check_data_health(sales_scope="TOTAL_STORE", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.CRITICAL
    assert verdict.ingestion.consecutive_failures == 3
    assert any("3 consecutive ingestion attempts have failed" in r for r in verdict.actionable_remediations)


def test_monitoring_validation_warnings_trigger_warning_status(mon_db):
    """Verify ingestion that passed with review flags triggers WARNING status."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_warn_001",
        detected_grain="TRANSACTION",
        row_count=200,
        validation_status="WARNING",
        is_usable=True,
        error_count=0,
        warning_count=3,
        report="{}",
        file_name="review_flags.csv",
    )

    # Clean recent observations
    ref_date = datetime.date(2026, 6, 15)
    repo.ingest_actual_sales(calendar_date="2026-06-14", daily_sales=300.0, sales_scope="STORE_WARN")

    config = MonitoringConfig(reference_date=ref_date, sales_scope="STORE_WARN")
    verdict = service.check_data_health(sales_scope="STORE_WARN", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.WARNING
    assert verdict.ingestion.outcome == IngestionOutcome.WARNING
    assert verdict.ingestion.recent_warnings_count == 3
    assert any("warning" in verdict.summary_message.lower() for _ in [1])


# ----------------------------------------------------------------------------
# 3. Threshold Boundaries & Edge Cases
# ----------------------------------------------------------------------------
def test_monitoring_exact_staleness_boundary(mon_db):
    """Verify boundary conditions for staleness_threshold_days."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_boundary",
        detected_grain="DAILY_AGGREGATED",
        row_count=50,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
    )

    ref_date = datetime.date(2026, 6, 30)
    # Case A: Exactly 30 days old (threshold: 30) -> FRESH (not > 30)
    repo.ingest_actual_sales(calendar_date="2026-05-31", daily_sales=100.0, sales_scope="STORE_BOUND_A")
    cfg_a = MonitoringConfig(staleness_threshold_days=30, reference_date=ref_date, sales_scope="STORE_BOUND_A")
    v_a = service.check_data_health(sales_scope="STORE_BOUND_A", config=cfg_a)
    assert v_a.freshness.staleness_days == 30
    assert v_a.freshness.is_stale is False
    assert v_a.overall_status == MonitoringStatus.HEALTHY

    # Case B: Exactly 31 days old (threshold: 30) -> STALE (31 > 30)
    repo.ingest_actual_sales(calendar_date="2026-05-30", daily_sales=100.0, sales_scope="STORE_BOUND_B")
    cfg_b = MonitoringConfig(staleness_threshold_days=30, reference_date=ref_date, sales_scope="STORE_BOUND_B")
    v_b = service.check_data_health(sales_scope="STORE_BOUND_B", config=cfg_b)
    assert v_b.freshness.staleness_days == 31
    assert v_b.freshness.is_stale is True
    assert v_b.overall_status == MonitoringStatus.WARNING


# ----------------------------------------------------------------------------
# 4. Database Unavailable Behavior
# ----------------------------------------------------------------------------
def test_monitoring_database_unavailable_returns_unknown():
    """Verify monitoring service handles disconnected database defensively without claiming health."""
    broken_service = DataMonitoringService(repository=None, engine=None, auto_init_engine=False)

    verdict = broken_service.check_data_health(sales_scope="ANY_STORE")

    assert verdict.overall_status == MonitoringStatus.UNKNOWN
    assert verdict.database_connected is False
    assert verdict.freshness.freshness_state == FreshnessState.UNKNOWN
    assert verdict.ingestion.outcome == IngestionOutcome.DATABASE_UNAVAILABLE
    assert "Database is unavailable" in verdict.summary_message
    assert len(verdict.actionable_remediations) > 0


# ----------------------------------------------------------------------------
# 5. Idempotency & Historical Check Retrieval
# ----------------------------------------------------------------------------
def test_monitoring_idempotency_and_history(mon_db):
    """Verify repeated monitoring checks with the same check_id update existing records idempotently."""
    service = mon_db["service"]
    repo = mon_db["repo"]

    fixed_check_id = "chk-fixed-001"
    config = MonitoringConfig(sales_scope="STORE_HIST")

    # Check 1
    v1 = service.check_data_health(sales_scope="STORE_HIST", config=config, persist=True, check_id=fixed_check_id)
    assert v1.check_id == fixed_check_id

    # Check 2 (Same check_id)
    v2 = service.check_data_health(sales_scope="STORE_HIST", config=config, persist=True, check_id=fixed_check_id)
    assert v2.check_id == fixed_check_id

    # Verify database has exactly 1 record for this check_id
    history = service.get_monitoring_history(sales_scope="STORE_HIST")
    matching = [h for h in history if h.check_id == fixed_check_id]
    assert len(matching) == 1

    # Verify get_latest_check returns it
    latest = service.get_latest_check(sales_scope="STORE_HIST")
    assert latest is not None
    assert latest.check_id == fixed_check_id


# ----------------------------------------------------------------------------
# 6. Scheduled Invocation Boundary
# ----------------------------------------------------------------------------
def test_scheduled_monitoring_check_execution(mon_db, monkeypatch):
    """Verify run_scheduled_monitoring_check entrypoint executes and logs appropriately."""
    monkeypatch.setattr("salespulse.monitoring.service.DataMonitoringService", lambda: mon_db["service"])

    verdict = run_scheduled_monitoring_check(sales_scope="STORE_CRON")
    assert isinstance(verdict, MonitoringVerdict)
    assert verdict.sales_scope == "STORE_CRON"
    assert verdict.overall_status in (
        MonitoringStatus.HEALTHY,
        MonitoringStatus.WARNING,
        MonitoringStatus.CRITICAL,
        MonitoringStatus.UNKNOWN,
    )


# ----------------------------------------------------------------------------
# 7. Dashboard Integration Tests
# ----------------------------------------------------------------------------
def test_dashboard_monitoring_integration():
    """Verify app.py integrates DataMonitoringService and displays all monitoring states."""
    import app

    # 1. Verify get_dashboard_services boots with mon_service
    services = app.get_dashboard_services()
    assert "mon_service" in services
    assert isinstance(services["mon_service"], DataMonitoringService)

    # 2. Verify fallback has mon_service in disconnected mode
    with patch("app.get_engine", side_effect=RuntimeError("Simulated DB Offline")):
        func = getattr(app.get_dashboard_services, "__wrapped__", getattr(app.get_dashboard_services, "func", app.get_dashboard_services))
        degraded = func()
        assert "mon_service" in degraded
        assert isinstance(degraded["mon_service"], DataMonitoringService)
        # Calling check on degraded service yields UNKNOWN
        v = degraded["mon_service"].check_data_health(sales_scope="TOTAL_STORE")
        assert v.overall_status == MonitoringStatus.UNKNOWN
        assert v.database_connected is False

    # 3. Verify app.py source contains all 4 status states and UI markers
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    assert "Data Freshness & Ingestion Health" in src
    assert "Monitoring Status: HEALTHY" in src
    assert "Monitoring Status: WARNING" in src
    assert "Monitoring Status: CRITICAL" in src
    assert "Monitoring Status: UNKNOWN" in src
    assert "DATA FRESHNESS SLA" in src
    assert "btn_run_monitoring_check" in src


# ----------------------------------------------------------------------------
# 8. Phase 4A.1 Reliability Hardening Tests
# ----------------------------------------------------------------------------
def test_ingestion_path_validation_service_success_and_failure(mon_db):
    """Verify ValidationService records both success and failure outcomes consistently in audit logs."""
    from salespulse.services.validation_service import ValidationService
    repo = mon_db["repo"]
    val_service = ValidationService(repository=repo)

    today = datetime.date.today()
    d1 = (today - datetime.timedelta(days=2)).isoformat()
    d2 = (today - datetime.timedelta(days=1)).isoformat()
    clean_csv = (
        f"Order Date,Sales,Region,Category,Segment\n"
        f"{d1},150.00,West,Furniture,Consumer\n"
        f"{d2},220.50,West,Technology,Corporate\n"
    ).encode("utf-8")
    res_clean = val_service.validate_file(file_source=clean_csv, file_name="batch_clean.csv", persist_audit=True)
    assert res_clean.report.status == "PASSED"

    # Verify audit log recorded for clean batch
    logs = repo.list_validations(limit=5)
    clean_log = next((l for l in logs if l.file_name == "batch_clean.csv"), None)
    assert clean_log is not None
    assert clean_log.validation_status == "PASSED"
    assert clean_log.error_count == 0

    # Corrupt CSV missing required Sales column
    bad_csv = b"Order Date,Region,Category\n2026-06-01,West,Furniture\n"
    res_bad = val_service.validate_file(file_source=bad_csv, file_name="batch_bad.csv", persist_audit=True)
    assert res_bad.report.status == "FAILED"

    # Verify audit log recorded for failed batch
    logs = repo.list_validations(limit=5)
    bad_log = next((l for l in logs if l.file_name == "batch_bad.csv"), None)
    assert bad_log is not None
    assert bad_log.validation_status == "FAILED"
    assert bad_log.error_count > 0


def test_ingest_and_aggregate_daily_lifecycle(mon_db):
    """Verify ingest_and_aggregate_daily coordinates validation and actuals persistence."""
    from salespulse.services.validation_service import ValidationService
    repo = mon_db["repo"]
    val_service = ValidationService(repository=repo)

    today = datetime.date.today()
    d1 = (today - datetime.timedelta(days=2)).isoformat()
    d2 = (today - datetime.timedelta(days=1)).isoformat()
    clean_csv = (
        f"Order Date,Sales\n"
        f"{d1},100.0\n"
        f"{d1},50.0\n"
        f"{d2},200.0\n"
    ).encode("utf-8")
    agg_df, result = val_service.ingest_and_aggregate_daily(
        file_source=clean_csv,
        file_name="clean_daily.csv",
        sales_scope="STORE_LIFECYCLE",
        persist_actuals=True,
        persist_audit=True,
    )
    assert result.report.status == "PASSED"
    assert agg_df is not None
    assert len(agg_df) == 2

    # Verify actuals were persisted into actual_sales_records
    with get_db_session(mon_db["engine"]) as s:
        actuals = s.query(ActualSalesRecord).filter(
            ActualSalesRecord.sales_scope == "STORE_LIFECYCLE",
            ActualSalesRecord.calendar_date.in_([datetime.date.fromisoformat(d1), datetime.date.fromisoformat(d2)]),
        ).all()
        assert len(actuals) == 2
        day1 = next(a for a in actuals if str(a.calendar_date) == d1)
        assert day1.daily_sales == 150.0  # aggregated 100 + 50


def test_ingest_actual_sales_batch_idempotency_and_reconciliation(mon_db):
    """Verify batch ingestion upserts idempotently and reconciles matching forecast items."""
    repo = mon_db["repo"]

    # 1. Create a forecast run with 2 items
    items = [
        {"target_date": "2026-07-01", "step_ahead": 1, "predicted_value": 140.0, "lower_bound": 120.0, "upper_bound": 160.0},
        {"target_date": "2026-07-02", "step_ahead": 2, "predicted_value": 150.0, "lower_bound": 130.0, "upper_bound": 170.0},
    ]
    repo.record_forecast_run(
        run_id="run-batch-recon-01",
        model_name="XGBoost",
        origin_date="2026-06-30",
        horizon_days=2,
        items=items,
        sales_scope="STORE_BATCH",
    )

    # 2. Ingest batch of actuals
    batch_records = [
        {"calendar_date": "2026-07-01", "daily_sales": 145.0},
        {"calendar_date": "2026-07-02", "daily_sales": 152.0},
    ]
    inserted, updated, reconciled = repo.ingest_actual_sales_batch(
        records=batch_records,
        sales_scope="STORE_BATCH",
    )
    assert inserted == 2
    assert updated == 0
    assert reconciled == 2

    # 3. Idempotent re-run: same batch should update existing, not duplicate
    ins2, upd2, recon2 = repo.ingest_actual_sales_batch(
        records=batch_records,
        sales_scope="STORE_BATCH",
    )
    assert ins2 == 0
    assert upd2 == 2
    assert recon2 == 2

    # Check forecast item errors
    run = repo.get_forecast_run("run-batch-recon-01")
    it1 = next(it for it in run.items if it.step_ahead == 1)
    assert it1.actual_value == 145.0
    assert it1.absolute_error == 5.0
    assert it1.interval_covered is True


def test_historical_reliability_preserved_after_subsequent_success(mon_db):
    """Verify that a later successful attempt does NOT erase historical failures from reliability rate."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    # Record 2 failed attempts
    for i in range(1, 3):
        repo.record_validation(
            file_sha256=f"sha_hist_fail_{i}",
            detected_grain="DAILY_AGGREGATED",
            row_count=10,
            validation_status="FAILED",
            is_usable=False,
            error_count=2,
            warning_count=0,
            report="{}",
            file_name=f"fail_{i}.csv",
        )

    # Record 1 recent observation and 1 subsequent successful attempt
    repo.ingest_actual_sales(calendar_date=datetime.date(2026, 6, 14), daily_sales=200.0, sales_scope="STORE_HIST")
    repo.record_validation(
        file_sha256="sha_hist_succ_1",
        detected_grain="DAILY_AGGREGATED",
        row_count=50,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
        file_name="success_1.csv",
    )

    config = MonitoringConfig(
        reference_date=datetime.date(2026, 6, 15),
        sales_scope="STORE_HIST",
        staleness_threshold_days=5,
    )
    verdict = service.check_data_health(sales_scope="STORE_HIST", config=config, persist=True)

    # Current health is HEALTHY because latest attempt passed and data is fresh
    assert verdict.overall_status == MonitoringStatus.HEALTHY
    assert verdict.ingestion.outcome == IngestionOutcome.PASSED
    assert verdict.ingestion.consecutive_failures == 0

    # Historical reliability accurately reflects past failures
    assert verdict.ingestion.historical_failure_count == 2
    assert verdict.ingestion.historical_success_count == 1
    assert verdict.ingestion.total_attempts_evaluated == 3
    assert verdict.ingestion.reliability_rate == pytest.approx(100 / 3, rel=1e-2)


def test_cadence_aware_freshness_boundaries(mon_db):
    """Verify configurable cadences (DAILY, WEEKLY, MONTHLY, CUSTOM) and exact SLA boundaries."""
    from salespulse.monitoring.schemas import CadenceType, CADENCE_DEFAULT_THRESHOLDS
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_cad_pass",
        detected_grain="DAILY_AGGREGATED",
        row_count=100,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
        file_name="cadence_pass.csv",
    )

    ref_date = datetime.date(2026, 6, 10)
    # Observation is 3 days old (2026-06-07)
    repo.ingest_actual_sales(calendar_date=datetime.date(2026, 6, 7), daily_sales=180.0, sales_scope="STORE_CAD")

    # 1. Under DAILY cadence (threshold 2 days, critical 4 days): 3 days > 2 and <= 4 -> STALE (WARNING)
    cfg_daily = MonitoringConfig(
        cadence=CadenceType.DAILY,
        reference_date=ref_date,
        sales_scope="STORE_CAD",
    )
    v_daily = service.check_data_health(sales_scope="STORE_CAD", config=cfg_daily)
    assert v_daily.freshness.is_stale is True
    assert v_daily.freshness.cadence.upper() == "DAILY"
    assert v_daily.overall_status == MonitoringStatus.WARNING

    # 2. Under WEEKLY cadence (threshold 8 days): 3 days <= 8 -> FRESH (HEALTHY)
    cfg_weekly = MonitoringConfig(
        cadence=CadenceType.WEEKLY,
        reference_date=ref_date,
        sales_scope="STORE_CAD",
    )
    v_weekly = service.check_data_health(sales_scope="STORE_CAD", config=cfg_weekly)
    assert v_weekly.freshness.is_stale is False
    assert v_weekly.freshness.cadence.upper() == "WEEKLY"
    assert v_weekly.overall_status == MonitoringStatus.HEALTHY

    # 3. Exact boundary test: threshold = 3 days (3 <= 3 -> HEALTHY)
    cfg_exact_border = MonitoringConfig(
        cadence=CadenceType.CUSTOM,
        staleness_threshold_days=3,
        reference_date=ref_date,
        sales_scope="STORE_CAD",
    )
    v_exact = service.check_data_health(sales_scope="STORE_CAD", config=cfg_exact_border)
    assert v_exact.freshness.is_stale is False
    assert v_exact.overall_status == MonitoringStatus.HEALTHY

    # Boundary + 1 day: threshold = 2 days (3 > 2 -> WARNING)
    cfg_past_border = MonitoringConfig(
        cadence=CadenceType.CUSTOM,
        staleness_threshold_days=2,
        reference_date=ref_date,
        sales_scope="STORE_CAD",
    )
    v_past = service.check_data_health(sales_scope="STORE_CAD", config=cfg_past_border)
    assert v_past.freshness.is_stale is True
    assert v_past.overall_status == MonitoringStatus.WARNING

    # 4. Severe staleness under DAILY cadence (critical = 2 * 2 = 4 days):
    # With 6 days old observation, staleness (6) > critical (4) -> CRITICAL
    ref_date_severe = datetime.date(2026, 6, 13)
    cfg_severe = MonitoringConfig(
        cadence=CadenceType.DAILY,
        reference_date=ref_date_severe,
        sales_scope="STORE_CAD",
    )
    v_severe = service.check_data_health(sales_scope="STORE_CAD", config=cfg_severe)
    assert v_severe.overall_status == MonitoringStatus.CRITICAL


def test_zero_sales_observation_distinguished_from_missing_data(mon_db):
    """Verify that a valid zero-sales observation is treated as a valid data point, not missing data."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_zero_valid",
        detected_grain="DAILY_AGGREGATED",
        row_count=10,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
        file_name="zero_sales.csv",
    )

    # Ingest a valid trading day with 0.0 sales
    obs_date = datetime.date(2026, 6, 14)
    ref_date = datetime.date(2026, 6, 15)
    repo.ingest_actual_sales(calendar_date=obs_date, daily_sales=0.0, sales_scope="STORE_ZERO")

    config = MonitoringConfig(
        reference_date=ref_date,
        sales_scope="STORE_ZERO",
        staleness_threshold_days=5,
    )
    verdict = service.check_data_health(sales_scope="STORE_ZERO", config=config, persist=True)

    assert verdict.overall_status == MonitoringStatus.HEALTHY
    assert verdict.freshness.freshness_state == FreshnessState.FRESH
    assert verdict.freshness.staleness_days == 1
    assert verdict.freshness.zero_sales_dates_count == 1
    assert verdict.freshness.latest_observation_date == "2026-06-14"
    assert "Verified zero-sales days: 1" in verdict.freshness.assessment


def test_privacy_sanitization_in_monitoring_evidence(mon_db):
    """Verify monitoring evidence never stores raw customer rows, credentials, or sensitive payloads."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    repo.record_validation(
        file_sha256="sha_priv_01",
        detected_grain="TRANSACTION",
        row_count=50,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report='{"issues": [], "diagnostics": {"date_coverage": {"min_date": "2026-06-01", "max_date": "2026-06-10"}}}',
        file_name="customer_orders_private.csv",
    )
    repo.ingest_actual_sales(calendar_date="2026-06-10", daily_sales=250.0, sales_scope="STORE_PRIV")

    config = MonitoringConfig(
        reference_date=datetime.date(2026, 6, 12),
        staleness_threshold_days=5,
        sales_scope="STORE_PRIV",
    )
    verdict = service.check_data_health(sales_scope="STORE_PRIV", config=config, persist=True)
    assert verdict.overall_status in (MonitoringStatus.HEALTHY, MonitoringStatus.WARNING)

    # Inspect persisted record in database
    rec = service.get_latest_monitoring_check(sales_scope="STORE_PRIV")
    assert rec is not None
    ev = rec.evidence_json

    # Must NOT contain raw customer strings, addresses, or credit cards
    for sensitive_keyword in ["customer_name", "credit_card", "ssn", "password", "token", "secret"]:
        assert sensitive_keyword not in ev.lower()


# ----------------------------------------------------------------------------
# 9. Phase 4B.1 Cadence Enum & Conversion Tests
# ----------------------------------------------------------------------------
def test_cadence_enum_parsing_and_defaults():
    """Verify all supported cadences convert canonically and supply accurate default SLAs."""
    from salespulse.monitoring.schemas import (
        CadenceType,
        CADENCE_DEFAULT_THRESHOLDS,
        get_cadence_default_threshold,
    )

    supported = ["HOURLY", "DAILY", "WEEKLY", "MONTHLY", "CUSTOM"]
    expected_thresholds = {
        CadenceType.HOURLY: 1,
        CadenceType.DAILY: 2,
        CadenceType.WEEKLY: 8,
        CadenceType.MONTHLY: 32,
        CadenceType.CUSTOM: 30,
    }

    for name in supported:
        enum_val = CadenceType[name]
        # Direct enum conversion
        assert CadenceType.from_str(enum_val) == enum_val
        # Uppercase string
        assert CadenceType.from_str(name) == enum_val
        # Lowercase string
        assert CadenceType.from_str(name.lower()) == enum_val
        # Mixed case string
        assert CadenceType.from_str(name.title()) == enum_val
        # Direct Enum construction via _missing_
        assert CadenceType(name.lower()) == enum_val

        # Default threshold helper
        expected_sla = expected_thresholds[enum_val]
        assert get_cadence_default_threshold(enum_val) == expected_sla
        assert get_cadence_default_threshold(name.lower()) == expected_sla

        # MonitoringConfig initialization with string or enum
        cfg1 = MonitoringConfig(cadence=name.lower())
        assert cfg1.cadence == enum_val
        assert cfg1.staleness_threshold_days == expected_sla

        cfg2 = MonitoringConfig(cadence=enum_val)
        assert cfg2.cadence == enum_val
        assert cfg2.staleness_threshold_days == expected_sla


def test_cadence_invalid_string_raises_clear_error():
    """Verify invalid cadence values raise ValueError with an informative message and do not swallow errors."""
    from salespulse.monitoring.schemas import CadenceType

    for bad_val in ["biweekly", "yearly", "quarterly", "invalid", "", "123"]:
        with pytest.raises(ValueError) as excinfo:
            CadenceType.from_str(bad_val)
        assert f"'{bad_val}' is not a valid CadenceType" in str(excinfo.value)
        assert "Supported cadences are: HOURLY, DAILY, WEEKLY, MONTHLY, CUSTOM" in str(excinfo.value)

        # MonitoringConfig must also raise ValueError and not default to CUSTOM or return HEALTHY
        with pytest.raises(ValueError):
            MonitoringConfig(cadence=bad_val)


def test_freshness_check_with_various_cadences(mon_db):
    """Verify freshness evaluation executes and persists accurately for multiple cadence types."""
    repo = mon_db["repo"]
    service = mon_db["service"]

    # Ingest actuals with observation 3 days ago relative to ref_date (2026-06-15)
    ref_date = datetime.date(2026, 6, 15)
    repo.ingest_actual_sales(calendar_date="2026-06-12", daily_sales=180.0, sales_scope="STORE_CADENCE")

    # Cadence DAILY: default threshold 2 days -> 3 days is STALE (WARNING)
    cfg_daily = MonitoringConfig(
        cadence=CadenceType.DAILY,
        reference_date=ref_date,
        sales_scope="STORE_CADENCE",
    )
    v_daily = service.check_data_health(sales_scope="STORE_CADENCE", config=cfg_daily, persist=True)
    assert v_daily.freshness.cadence == "DAILY"
    assert v_daily.freshness.staleness_days == 3
    assert v_daily.freshness.freshness_state == FreshnessState.STALE
    assert v_daily.overall_status == MonitoringStatus.WARNING

    # Cadence WEEKLY: default threshold 8 days -> 3 days is FRESH (HEALTHY)
    cfg_weekly = MonitoringConfig(
        cadence="weekly",  # lowercase string input
        reference_date=ref_date,
        sales_scope="STORE_CADENCE",
    )
    v_weekly = service.check_data_health(sales_scope="STORE_CADENCE", config=cfg_weekly, persist=True)
    assert v_weekly.freshness.cadence == "WEEKLY"
    assert v_weekly.freshness.staleness_days == 3
    assert v_weekly.freshness.freshness_state == FreshnessState.FRESH
    assert v_weekly.overall_status == MonitoringStatus.HEALTHY

    # Verify latest check recorded in database reflects the latest run
    latest = service.get_latest_monitoring_check(sales_scope="STORE_CADENCE")
    assert latest is not None
    assert latest.status == MonitoringStatus.HEALTHY.value


def test_parse_datetime_safe_and_format_datetime_safe():
    """Verify parse_datetime_safe and format_datetime_safe handle all supported types and edge cases."""
    from salespulse.monitoring.schemas import parse_datetime_safe, format_datetime_safe

    # 1. ISO strings with and without Z/offset
    iso_utc = "2026-10-21T10:45:00Z"
    dt1 = parse_datetime_safe(iso_utc)
    assert dt1 is not None
    assert dt1.tzinfo == datetime.timezone.utc
    assert dt1.year == 2026 and dt1.month == 10 and dt1.day == 21 and dt1.hour == 10
    assert format_datetime_safe(iso_utc, "%Y-%m-%d %H:%M") == "2026-10-21 10:45"

    iso_offset = "2026-10-21T12:45:00+02:00"
    dt2 = parse_datetime_safe(iso_offset)
    assert dt2 is not None
    assert dt2.hour == 10  # Normalized to UTC

    # 2. Python datetime (naive and aware)
    naive_dt = datetime.datetime(2026, 5, 1, 14, 30)
    dt3 = parse_datetime_safe(naive_dt)
    assert dt3.tzinfo == datetime.timezone.utc
    assert format_datetime_safe(naive_dt, "%Y-%m-%d %H:%M") == "2026-05-01 14:30"

    # 3. Python date object
    date_obj = datetime.date(2026, 7, 4)
    dt4 = parse_datetime_safe(date_obj)
    assert dt4.year == 2026 and dt4.month == 7 and dt4.day == 4
    assert format_datetime_safe(date_obj, "%Y-%m-%d") == "2026-07-04"

    # 4. Standard string formats
    space_str = "2026-10-21 08:30:00"
    assert format_datetime_safe(space_str, "%Y-%m-%d %H:%M") == "2026-10-21 08:30"

    date_only_str = "2026-10-21"
    assert format_datetime_safe(date_only_str, "%Y-%m-%d") == "2026-10-21"

    # 5. None, empty string, nan, and invalid garbage inputs -> never crash
    assert parse_datetime_safe(None) is None
    assert format_datetime_safe(None, fallback="Never Recorded") == "Never Recorded"
    assert format_datetime_safe("", fallback="Never Recorded") == "Never Recorded"
    assert format_datetime_safe("   ", fallback="None") == "None"
    assert format_datetime_safe("not-a-date", fallback="-") == "-"
    assert format_datetime_safe(float("nan"), fallback="Never Recorded") == "Never Recorded"


def test_monitoring_record_timestamp_persistence(mon_db):
    """Verify repository.record_monitoring_check handles ISO strings and datetimes gracefully."""
    from salespulse.monitoring.schemas import format_datetime_safe
    repo = mon_db["repo"]

    # Persist check with ISO string timestamps
    chk_str = "2026-10-21T10:00:00+00:00"
    succ_str = "2026-10-20T18:30:00+00:00"
    att_str = "2026-10-21T09:55:00+00:00"

    record = repo.record_monitoring_check(
        check_id="chk_ts_test_01",
        status="HEALTHY",
        sales_scope="STORE_TIMESTAMPS",
        checked_at=chk_str,
        last_successful_ingestion_at=succ_str,
        last_ingestion_attempt_at=att_str,
        last_ingestion_status="PASSED",
        latest_observation_date="2026-10-20",
        staleness_days=1,
        staleness_threshold_days=2,
    )
    assert record is not None
    assert isinstance(record.checked_at, datetime.datetime)
    assert isinstance(record.last_successful_ingestion_at, datetime.datetime)

    # Verify query and safe formatting
    retrieved = repo.get_latest_monitoring_check(sales_scope="STORE_TIMESTAMPS")
    assert retrieved is not None
    assert format_datetime_safe(retrieved.checked_at, "%Y-%m-%d %H:%M") == "2026-10-21 10:00"
    assert format_datetime_safe(retrieved.last_successful_ingestion_at, "%Y-%m-%d %H:%M") == "2026-10-20 18:30"



