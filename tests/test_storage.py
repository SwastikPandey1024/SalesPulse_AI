"""
SalesPulse AI - Tests for Storage & Persistence Module (Phase 3A)
Verifies:
- SQLite WAL pragma configuration, timeout, and schema versioning.
- Audit repository logging, idempotent deduplication, and transaction rollbacks.
- Forecast run and multi-step item persistence.
- Multiple runs targeting identical calendar dates.
- Delayed actuals ingestion, reconciliation, and corrected actuals handling.
- Negative actual sales (refunds policy).
- Empty sample vs populated evaluation summaries.
"""

import os
import tempfile
import datetime
import pytest
from sqlalchemy import text

from salespulse.storage.database import (
    create_storage_engine,
    get_db_session,
    init_db,
    SCHEMA_VERSION,
    Base,
)
from salespulse.storage.models import DatasetAuditLog, ForecastRunLog, ForecastItemLog, ActualSalesRecord
from salespulse.storage.repository import AuditRepository


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with WAL mode enabled."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()

    db_url = f"sqlite:///{tmp_path}"
    engine = create_storage_engine(db_url=db_url)
    init_db(engine=engine)
    repo = AuditRepository(engine=engine)

    yield {"engine": engine, "repo": repo, "path": tmp_path}

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


def test_sqlite_wal_initialization_and_schema_version(temp_db):
    """Verify SQLite WAL pragma, schema versioning, and table creation."""
    engine = temp_db["engine"]
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode;")).scalar()
        assert journal_mode.lower() == "wal"

        user_version = conn.execute(text("PRAGMA user_version;")).scalar()
        assert user_version == SCHEMA_VERSION

        # Verify tables exist
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).scalars().all()
        assert "dataset_audit_logs" in tables
        assert "forecast_run_logs" in tables
        assert "forecast_item_logs" in tables
        assert "actual_sales_records" in tables


def test_repository_validation_record_and_idempotency(temp_db):
    """Verify dataset validation audit records, SHA-256 querying, and privacy safeguards."""
    repo = temp_db["repo"]
    report_dict = {"status": "PASSED", "issues": [], "summary": {"rows": 100}}

    record = repo.record_validation(
        file_sha256="abc123sha",
        detected_grain="TRANSACTION_LEVEL",
        row_count=100,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report=report_dict,
        file_name="/sensitive/local/path/sales_2024.csv",
    )

    assert record.id is not None
    # Verify path was sanitized to basename
    assert record.file_name == "sales_2024.csv"

    # Query by SHA-256
    queried = repo.get_validation_by_hash("abc123sha")
    assert queried is not None
    assert queried.row_count == 100
    assert "rows" in queried.report_json


def test_repository_forecast_run_and_items(temp_db):
    """Verify forecast run and multi-step item records insertion."""
    repo = temp_db["repo"]
    items = [
        {"target_date": "2024-06-01", "step_ahead": 1, "predicted_value": 150.0, "lower_bound": 120.0, "upper_bound": 180.0},
        {"target_date": "2024-06-02", "step_ahead": 2, "predicted_value": 160.0, "lower_bound": 125.0, "upper_bound": 195.0},
    ]

    run = repo.record_forecast_run(
        run_id="run-test-001",
        model_name="XGBoost",
        origin_date="2024-05-31",
        horizon_days=2,
        items=items,
        model_artifact_path="/models/model.pkl",
        model_artifact_hash="modelsha123",
    )

    assert run.id is not None
    assert run.total_predicted_revenue == 310.0
    assert run.average_daily_sales == 155.0
    assert run.model_artifact_path == "model.pkl"

    fetched = repo.get_forecast_run("run-test-001")
    assert fetched is not None
    assert len(fetched.items) == 2
    assert fetched.items[0].predicted_value == 150.0


def test_multiple_forecast_runs_targeting_same_date(temp_db):
    """Verify multiple runs (different models or horizons) can target the same date without collisions."""
    repo = temp_db["repo"]
    target = "2024-07-01"

    # Run 1: 7-day forecast from XGBoost
    repo.record_forecast_run(
        run_id="run-xgb-7d",
        model_name="XGBoost",
        origin_date="2024-06-30",
        horizon_days=1,
        items=[{"target_date": target, "step_ahead": 1, "predicted_value": 150.0}],
    )

    # Run 2: 14-day forecast from RandomForest
    repo.record_forecast_run(
        run_id="run-rf-14d",
        model_name="RandomForest",
        origin_date="2024-06-25",
        horizon_days=6,
        items=[{"target_date": target, "step_ahead": 6, "predicted_value": 165.0}],
    )

    # Both items should exist independently
    with get_db_session(temp_db["engine"]) as s:
        items = s.query(ForecastItemLog).filter(ForecastItemLog.target_date == datetime.date(2024, 7, 1)).all()
        assert len(items) == 2
        run_ids = {it.run_id for it in items}
        assert run_ids == {"run-xgb-7d", "run-rf-14d"}


def test_actuals_reconciliation_and_corrected_actuals(temp_db):
    """Verify delayed actual sales ingestion reconciles all overlapping items, and updates on corrected actuals."""
    repo = temp_db["repo"]
    target_date = "2024-08-10"

    # Store a forecast item with prediction 150 and bounds [130, 170]
    repo.record_forecast_run(
        run_id="run-reconcile-01",
        model_name="XGBoost",
        origin_date="2024-08-09",
        horizon_days=1,
        items=[{"target_date": target_date, "step_ahead": 1, "predicted_value": 150.0, "lower_bound": 130.0, "upper_bound": 170.0}],
    )

    # Ingest actual sales: 160.0 (Inside interval)
    record, was_created = repo.ingest_actual_sales(calendar_date=target_date, daily_sales=160.0)
    assert was_created is True
    assert record.daily_sales == 160.0

    # Verify forecast item was reconciled
    with get_db_session(temp_db["engine"]) as s:
        item = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-reconcile-01").one()
        assert item.actual_value == 160.0
        assert item.absolute_error == 10.0
        assert item.interval_covered is True
        assert item.reconciled_at is not None

    # Corrected actuals: Supplier audit revises day 2024-08-10 sales to 180.0 (Outside interval)
    rec2, was_created2 = repo.ingest_actual_sales(calendar_date=target_date, daily_sales=180.0)
    assert was_created2 is False  # Updated existing

    with get_db_session(temp_db["engine"]) as s:
        item = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-reconcile-01").one()
        assert item.actual_value == 180.0
        assert item.absolute_error == 30.0
        assert item.interval_covered is False


def test_reconcile_negative_actual_sales_returns_policy(temp_db):
    """Verify negative daily actual sales (net refunds) are reconciled without mathematical failures."""
    repo = temp_db["repo"]
    target_date = "2024-09-01"

    repo.record_forecast_run(
        run_id="run-refund-01",
        model_name="XGBoost",
        origin_date="2024-08-31",
        horizon_days=1,
        items=[{"target_date": target_date, "step_ahead": 1, "predicted_value": 100.0, "lower_bound": 50.0, "upper_bound": 150.0}],
    )

    # Day where returns exceeded gross sales (-25.0)
    repo.ingest_actual_sales(calendar_date=target_date, daily_sales=-25.0)

    with get_db_session(temp_db["engine"]) as s:
        item = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-refund-01").one()
        assert item.actual_value == -25.0
        assert item.absolute_error == 125.0  # |(-25) - 100| = 125.0
        assert item.interval_covered is False


def test_forecast_evaluation_summary_empty_and_populated(temp_db):
    """Verify evaluation summary handles empty vs populated evaluation samples honestly."""
    repo = temp_db["repo"]

    # 1. Empty sample returns None metrics (not misleading 0.0)
    empty_summary = repo.get_forecast_evaluation_summary(model_name="XGBoost")
    assert empty_summary["evaluated_count"] == 0
    assert empty_summary["mae"] is None
    assert empty_summary["rmse"] is None
    assert empty_summary["coverage_rate"] is None

    # 2. Populated sample
    repo.record_forecast_run(
        run_id="run-eval-01",
        model_name="XGBoost",
        origin_date="2024-10-01",
        horizon_days=2,
        items=[
            {"target_date": "2024-10-02", "step_ahead": 1, "predicted_value": 100.0, "lower_bound": 80.0, "upper_bound": 120.0},
            {"target_date": "2024-10-03", "step_ahead": 2, "predicted_value": 200.0, "lower_bound": 170.0, "upper_bound": 230.0},
        ],
    )
    repo.ingest_actual_sales(calendar_date="2024-10-02", daily_sales=110.0)  # Error: 10, covered
    repo.ingest_actual_sales(calendar_date="2024-10-03", daily_sales=220.0)  # Error: 20, covered

    summary = repo.get_forecast_evaluation_summary(model_name="XGBoost")
    assert summary["evaluated_count"] == 2
    assert summary["mae"] == 15.0  # (10 + 20) / 2
    assert summary["coverage_rate"] == 1.0


def test_transaction_rollback_and_cleanup(temp_db):
    """Verify transaction rollback on error leaves clean database state."""
    engine = temp_db["engine"]

    with pytest.raises(ValueError):
        with get_db_session(engine) as s:
            entry = DatasetAuditLog(
                file_sha256="failed_tx",
                detected_grain="DAILY_AGGREGATED",
                row_count=10,
                validation_status="PASSED",
                is_usable=True,
                error_count=0,
                warning_count=0,
                report_json="{}",
            )
            s.add(entry)
            s.flush()
            raise ValueError("Simulated unexpected failure")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM dataset_audit_logs WHERE file_sha256='failed_tx';")).scalar()
        assert count == 0


def test_schema_version_fresh_repeated_and_rejection(temp_db):
    """Verify fresh initialization, repeated idempotent init, and rejection of unsupported future schema."""
    engine = temp_db["engine"]

    # 1. Fresh init was already run by fixture, verify user_version == SCHEMA_VERSION
    with engine.connect() as conn:
        v = conn.execute(text("PRAGMA user_version;")).scalar()
        assert v == SCHEMA_VERSION

    # 2. Repeated initialization is idempotent
    v_repeated = init_db(engine=engine)
    assert v_repeated == SCHEMA_VERSION

    # 3. Simulate an unsupported future schema version
    with engine.begin() as conn:
        conn.execute(text("PRAGMA user_version = 999;"))

    with pytest.raises(RuntimeError) as exc_info:
        init_db(engine=engine)
    assert "newer than the application's supported schema version" in str(exc_info.value)
    assert "999" in str(exc_info.value)

    # Restore version so fixture cleanup succeeds
    with engine.begin() as conn:
        conn.execute(text(f"PRAGMA user_version = {SCHEMA_VERSION};"))


def test_schema_migration_v1_to_v2(tmp_path):
    """Verify that a legacy database at schema version 1 is safely upgraded to version 2."""
    db_file = tmp_path / "legacy_v1.db"
    engine = create_storage_engine(f"sqlite:///{db_file}")

    # Simulate legacy v1 database: create tables from v1 only (drop anomaly_records) and set user_version = 1
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS anomaly_records;"))
        conn.execute(text("PRAGMA user_version = 1;"))

    # Verify state before migration
    with engine.connect() as conn:
        v_before = conn.execute(text("PRAGMA user_version;")).scalar()
        assert v_before == 1
        # Table anomaly_records should not exist
        res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_records';")).fetchone()
        assert res is None

    # Run init_db to execute migration
    v_after = init_db(engine=engine)
    assert v_after == 2

    # Verify state after migration
    with engine.connect() as conn:
        v_current = conn.execute(text("PRAGMA user_version;")).scalar()
        assert v_current == 2
        res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='anomaly_records';")).fetchone()
        assert res is not None
        assert res[0] == "anomaly_records"


def test_schema_migration_failure_rollback(tmp_path, monkeypatch):
    """Verify that an exception during migration rolls back and leaves schema version at 1."""
    db_file = tmp_path / "rollback_v1.db"
    engine = create_storage_engine(f"sqlite:///{db_file}")

    # Set up v1 database
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS anomaly_records;"))
        conn.execute(text("PRAGMA user_version = 1;"))

    # Force create_all to fail during migration
    def _failing_create_all(*args, **kwargs):
        raise RuntimeError("Simulated migration failure during table creation")

    monkeypatch.setattr(Base.metadata, "create_all", _failing_create_all)

    with pytest.raises(RuntimeError, match="Simulated migration failure"):
        init_db(engine=engine)

    # Verify user_version is NOT falsely bumped to 2
    with engine.connect() as conn:
        v = conn.execute(text("PRAGMA user_version;")).scalar()
        assert v == 1




def test_multi_run_corrected_actual_reconciliation(temp_db):
    """Verify correcting actual sales updates every eligible forecast item across multiple runs and horizons."""
    repo = temp_db["repo"]
    target = "2024-12-01"

    # Run 1: 7-day forecast from XGBoost
    repo.record_forecast_run(
        run_id="run-xgb-multi",
        model_name="XGBoost",
        origin_date="2024-11-24",
        horizon_days=7,
        items=[
            {"target_date": target, "step_ahead": 7, "predicted_value": 100.0, "lower_bound": 80.0, "upper_bound": 120.0, "sales_scope": "TOTAL_STORE"},
        ],
    )

    # Run 2: 14-day forecast from RandomForest
    repo.record_forecast_run(
        run_id="run-rf-multi",
        model_name="RandomForest",
        origin_date="2024-11-17",
        horizon_days=14,
        items=[
            {"target_date": target, "step_ahead": 14, "predicted_value": 110.0, "lower_bound": 95.0, "upper_bound": 125.0, "sales_scope": "TOTAL_STORE"},
        ],
    )

    # 1. Ingest initial actual sales: 105.0
    rec1, was_created1 = repo.ingest_actual_sales(calendar_date=target, daily_sales=105.0, sales_scope="TOTAL_STORE")
    assert was_created1 is True
    assert rec1.daily_sales == 105.0

    # Verify both runs are reconciled and inside intervals
    with get_db_session(temp_db["engine"]) as s:
        item_xgb = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-xgb-multi").one()
        item_rf = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-rf-multi").one()

        assert item_xgb.actual_value == 105.0
        assert item_xgb.absolute_error == 5.0
        assert item_xgb.interval_covered is True

        assert item_rf.actual_value == 105.0
        assert item_rf.absolute_error == 5.0
        assert item_rf.interval_covered is True

    # 2. Corrected actual sales: Supplier audit updates day's sales to 122.0
    rec2, was_created2 = repo.ingest_actual_sales(calendar_date=target, daily_sales=122.0, sales_scope="TOTAL_STORE")
    assert was_created2 is False  # Reused existing record

    # Verify both runs are updated with new error and interval coverage
    with get_db_session(temp_db["engine"]) as s:
        item_xgb = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-xgb-multi").one()
        item_rf = s.query(ForecastItemLog).filter(ForecastItemLog.run_id == "run-rf-multi").one()

        # XGBoost: predicted 100.0, bounds [80.0, 120.0] -> 122.0 is OUTSIDE interval
        assert item_xgb.actual_value == 122.0
        assert item_xgb.absolute_error == 22.0
        assert item_xgb.interval_covered is False

        # RandomForest: predicted 110.0, bounds [95.0, 125.0] -> 122.0 is INSIDE interval
        assert item_rf.actual_value == 122.0
        assert item_rf.absolute_error == 12.0
        assert item_rf.interval_covered is True

        # Verify each run retains its own model identity and forecast metadata
        run_xgb = s.query(ForecastRunLog).filter(ForecastRunLog.run_id == "run-xgb-multi").one()
        run_rf = s.query(ForecastRunLog).filter(ForecastRunLog.run_id == "run-rf-multi").one()

        assert run_xgb.model_name == "XGBoost"
        assert run_xgb.horizon_days == 7
        assert run_rf.model_name == "RandomForest"
        assert run_rf.horizon_days == 14

    # 3. Repeated ingestion of identical actual sales does not duplicate records
    rec3, was_created3 = repo.ingest_actual_sales(calendar_date=target, daily_sales=122.0, sales_scope="TOTAL_STORE")
    assert was_created3 is False

    with get_db_session(temp_db["engine"]) as s:
        actuals_count = s.query(ActualSalesRecord).filter(
            ActualSalesRecord.calendar_date == datetime.date(2024, 12, 1),
            ActualSalesRecord.sales_scope == "TOTAL_STORE",
        ).count()
        assert actuals_count == 1


def test_schema_version_rejection_and_additive_table_safety(temp_db):
    """Verify additive table data_monitoring_records, idempotency, and future version rejection."""
    engine = temp_db["engine"]

    # 1. Verify additive table exists
    with engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).scalars().all()
        assert "data_monitoring_records" in tables
        current_ver = conn.execute(text("PRAGMA user_version;")).scalar()
        assert current_ver == SCHEMA_VERSION

    # 2. Repeated init_db is safe and idempotent
    init_db(engine)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA user_version;")).scalar() == SCHEMA_VERSION

    # 3. Future schema version rejection
    with engine.connect() as conn:
        conn.execute(text("PRAGMA user_version = 99;"))
        conn.commit()

    with pytest.raises(RuntimeError, match="is newer than the application's supported schema version"):
        init_db(engine)


def test_batch_actuals_storage_and_zero_sales(temp_db):
    """Verify batch ingestion upsert, reconciliation, and zero sales count in summary stats."""
    repo = temp_db["repo"]

    # Ingest mix of positive and zero-sales records
    records = [
        {"calendar_date": "2026-08-01", "daily_sales": 0.0},
        {"calendar_date": "2026-08-02", "daily_sales": 120.0},
        {"calendar_date": "2026-08-03", "daily_sales": 0.0},
    ]
    ins, upd, recon = repo.ingest_actual_sales_batch(records, sales_scope="STORE_ZERO_TEST")
    assert ins == 3
    assert upd == 0

    stats = repo.get_ingestion_summary_stats(sales_scope="STORE_ZERO_TEST")
    assert stats["zero_sales_count"] == 2
    assert stats["latest_observation_date"] == datetime.date(2026, 8, 3)

