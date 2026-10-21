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

