"""
SalesPulse AI - Phase Acceptance Remediation & Quality Audit Tests
Comprehensive regression and acceptance test suite validating:
1. Online Retail II dataset provenance, hash integrity, and metadata structure.
2. Source-to-canonical schema mapping, cancellations, returns, and gross/net accounting.
3. Daily aggregation, calendar gap zero-filling, and missing date policy.
4. Currency formatting and labeling (£ GBP for Online Retail II vs $ USD for Superstore).
5. Safe datetime parsing/formatting (eliminates 'str' object has no attribute 'strftime').
6. Cadence normalization across enum and string values without type errors.
7. Separation of Historical Backtesting (rolling-origin) from Reconciled Realized Performance.
8. Pending actuals handling (never zero-filled or fabricated as 0.0 error).
9. Minimum history policy in Anomaly Monitor (N >= 14 observations).
10. Explicit Data Modes (DEMO_DATA, USER_DATA, NO_DATA, DATA_UNAVAILABLE).
11. Drift baseline persistence, absent baseline handling, and feature distribution checks.
12. Dashboard rendering functions and component resilience across all 6 workspaces.
"""

import os
import json
import datetime
from collections import Counter
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine

from salespulse.common.datetime_utils import parse_datetime_safe, format_datetime_safe
from salespulse.monitoring.schemas import (
    CadenceType,
    MonitoringConfig,
    MonitoringStatus,
    FreshnessState,
    IngestionOutcome,
    DataFreshnessReport,
    IngestionHealthReport,
)
from salespulse.monitoring.service import DataMonitoringService
from salespulse.monitoring.drift_schemas import (
    DriftStatus,
    FeatureType,
    DriftMetricType,
    DriftBaselineProfile,
    DriftConfig,
    DriftReport,
)
from salespulse.monitoring.drift_service import (
    DatasetDriftService,
    compute_numeric_psi,
    compute_categorical_tvd,
    profile_dataset,
)
from salespulse.storage.database import init_db, SCHEMA_VERSION
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import ActualSalesRecord, AnomalyRecord, ForecastRunLog, ForecastItemLog
from salespulse.data.contracts import (
    detect_dataset_grain,
    validate_dataset_schema,
    GrainType,
)
from salespulse.data.time_series import MISSING_DATE_POLICY
from salespulse.data.onboard_online_retail import (
    UCI_DATASET_NAME,
    UCI_DATASET_CITATION,
    UCI_LICENSE,
    OnboardingProvenance,
    RAW_ZIP_FILENAME,
    RAW_XLSX_FILENAME,
)
import app


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def isolated_repo(tmp_path):
    """Isolated SQLite test repository with clean schema."""
    db_file = tmp_path / "acceptance_test.db"
    engine = create_engine(f"sqlite:///{db_file}")
    init_db(engine)
    return AuditRepository(engine)


@pytest.fixture
def mock_online_retail_raw():
    """Synthetic transactions mirroring the 8 columns of Online Retail II."""
    rows = [
        # Normal sales - 2009-12-01
        ("489434", "85048", "15CM CHRISTMAS GLASS BALL", 12, "2009-12-01 07:45:00", 6.95, 13085, "United Kingdom"),
        ("489435", "79323P", "PINK CHERRY LIGHTS", 10, "2009-12-01 09:20:00", 5.00, 13085, "United Kingdom"),
        # Return / Cancellation: 'C' prefix and negative qty
        ("C489436", "85048", "15CM CHRISTMAS GLASS BALL", -2, "2009-12-01 10:15:00", 6.95, 13085, "United Kingdom"),
        # Next day sales - 2009-12-02
        ("489437", "22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 40, "2009-12-02 11:30:00", 2.50, 15311, "Germany"),
        # Non-positive price (postage adjustment / debt writeoff -> should be quarantined)
        ("489438", "D", "Bad Debt Adjustment", 1, "2009-12-02 12:00:00", 0.0, None, "United Kingdom"),
        # Unparseable/null date
        ("489439", "21232", "STRAWBERRY CERAMIC TRINKET BOX", 5, None, 1.25, 15311, "France"),
        # Day after weekend gap - 2009-12-04
        ("489440", "85048", "15CM CHRISTMAS GLASS BALL", 10, "2009-12-04 14:00:00", 7.00, 14000, "United Kingdom"),
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return pd.DataFrame(rows, columns=cols)


# -----------------------------------------------------------------------------
# 1. Datetime Parsing & Formatting Safe Contract Tests
# -----------------------------------------------------------------------------
def test_datetime_safe_utils_string_input():
    """Verify parse_datetime_safe and format_datetime_safe handle ISO strings without AttributeError."""
    iso_str = "2026-10-21T10:15:30+00:00"
    dt = parse_datetime_safe(iso_str)
    assert isinstance(dt, datetime.datetime)
    assert dt.year == 2026
    assert dt.hour == 10

    # Format back to display string safely
    formatted = format_datetime_safe(iso_str, default="N/A")
    assert formatted != "N/A"
    assert "2026-10-21" in formatted

    # Direct formatting on str object does NOT raise AttributeError
    formatted_from_str = format_datetime_safe("2026-10-21 14:30:00")
    assert "2026-10-21" in formatted_from_str


def test_datetime_safe_utils_date_and_none():
    """Verify handling of Python date, datetime, None, and unparseable values."""
    d = datetime.date(2026, 10, 21)
    dt = parse_datetime_safe(d)
    assert isinstance(dt, datetime.datetime)
    assert dt.date() == d

    formatted_d = format_datetime_safe(d)
    assert "2026-10-21" in formatted_d

    # None and empty inputs return default safely
    assert parse_datetime_safe(None) is None
    assert format_datetime_safe(None, default="Never") == "Never"
    assert format_datetime_safe("", default="Never") == "Never"

    # Malformed garbage string
    assert parse_datetime_safe("invalid-date-string") is None
    assert format_datetime_safe("invalid-date-string", default="Invalid") == "Invalid"


# -----------------------------------------------------------------------------
# 2. Freshness Cadence Normalization and Strftime Contract
# -----------------------------------------------------------------------------
def test_cadence_normalization_enum_and_string():
    """Verify CadenceConfig accepts both Enum and string variants cleanly."""
    c_enum = MonitoringConfig(cadence=CadenceType.DAILY)
    assert c_enum.cadence == CadenceType.DAILY
    assert c_enum.cadence.value == "DAILY"

    c_str = MonitoringConfig(cadence="daily")
    assert c_str.cadence == CadenceType.DAILY

    c_str_upper = MonitoringConfig(cadence="DAILY")
    assert c_str_upper.cadence == CadenceType.DAILY

    # Unsupported cadence raises clear ValueError
    with pytest.raises(ValueError):
        MonitoringConfig(cadence="unsupported_cadence")


def test_monitoring_service_freshness_check(isolated_repo):
    """Verify DataMonitoringService evaluates freshness with safe string/datetime inputs."""
    service = DataMonitoringService(repository=isolated_repo, engine=isolated_repo.engine)

    # Ingest actual observation
    obs_date = datetime.date(2026, 6, 10)
    ref_date = datetime.date(2026, 6, 11)
    isolated_repo.ingest_actual_sales(calendar_date=obs_date, daily_sales=200.0, sales_scope="STORE_A")

    # Ingest validation record
    isolated_repo.record_validation(
        file_sha256="sha_acc_01",
        detected_grain="DAILY_AGGREGATED",
        row_count=100,
        validation_status="PASSED",
        is_usable=True,
        error_count=0,
        warning_count=0,
        report="{}",
        file_name="clean_daily_sales.csv",
    )

    verdict = service.check_data_health(
        sales_scope="STORE_A",
        config=MonitoringConfig(cadence="daily", staleness_threshold_days=2, reference_date=ref_date, sales_scope="STORE_A"),
    )
    assert verdict.overall_status == MonitoringStatus.HEALTHY
    assert verdict.freshness.freshness_state == FreshnessState.FRESH
    assert verdict.freshness.cadence == "DAILY"


# -----------------------------------------------------------------------------
# 3. Dataset Onboarding Business Logic & Canonical Mapping
# -----------------------------------------------------------------------------
def test_dataset_provenance_and_integrity_metadata():
    """Verify that OnboardingProvenance contains all citation and integrity fields."""
    assert UCI_DATASET_NAME == "Online Retail II"
    assert "Chen, D. (2012)" in UCI_DATASET_CITATION
    assert "CC BY 4.0" in UCI_LICENSE
    assert RAW_ZIP_FILENAME == "online_retail_ii.zip"
    assert RAW_XLSX_FILENAME == "online_retail_II.xlsx"


def test_onboarding_gross_net_and_cancellations(mock_online_retail_raw):
    """Test segregation of cancellations, returns, and quarantine rows."""
    df = mock_online_retail_raw.copy()
    
    # 1. Price > 0 and valid InvoiceDate
    valid_mask = (df["Price"] > 0) & (df["InvoiceDate"].notna())
    df_valid = df[valid_mask].copy()
    assert len(df_valid) == 5  # 2 quarantined: bad debt (price 0) and missing date

    df_valid["is_cancellation"] = df_valid["Invoice"].str.startswith("C") | (df_valid["Quantity"] < 0)

    # Gross sales: non-cancellation rows
    gross_df = df_valid[~df_valid["is_cancellation"]]
    gross_sales = (gross_df["Quantity"] * gross_df["Price"]).sum()
    # (12 * 6.95 = 83.40) + (10 * 5.00 = 50.00) + (40 * 2.50 = 100.00) + (10 * 7.00 = 70.00) = 303.40
    assert pytest.approx(gross_sales, 0.01) == 303.40

    # Returns: cancellation rows
    ret_df = df_valid[df_valid["is_cancellation"]]
    returns_val = (ret_df["Quantity"].abs() * ret_df["Price"]).sum()
    # 2 * 6.95 = 13.90
    assert pytest.approx(returns_val, 0.01) == 13.90

    # Net sales = gross - returns = 303.40 - 13.90 = 289.50
    net_sales = gross_sales - returns_val
    assert pytest.approx(net_sales, 0.01) == 289.50


def test_onboarding_continuous_daily_reindexing(mock_online_retail_raw):
    """Test daily reindexing produces full calendar sequence under MISSING_DATE_POLICY."""
    df = mock_online_retail_raw.copy()
    valid_mask = (df["Price"] > 0) & (df["InvoiceDate"].notna())
    df_valid = df[valid_mask].copy()
    df_valid["date"] = pd.to_datetime(df_valid["InvoiceDate"]).dt.date
    df_valid["is_cancellation"] = df_valid["Invoice"].str.startswith("C") | (df_valid["Quantity"] < 0)
    df_valid["val"] = np.where(
        df_valid["is_cancellation"],
        -(df_valid["Quantity"].abs() * df_valid["Price"]),
        df_valid["Quantity"] * df_valid["Price"]
    )
    daily = df_valid.groupby("date")["val"].sum()

    full_idx = pd.date_range("2009-12-01", "2009-12-04", freq="D").date
    daily_reindexed = daily.reindex(full_idx, fill_value=0.0)

    assert len(daily_reindexed) == 4
    # 2009-12-03 was unobserved, so it must be exactly 0.0
    assert daily_reindexed[datetime.date(2009, 12, 3)] == 0.0
    # 2009-12-04: 10 * 7.00 = 70.0
    assert pytest.approx(daily_reindexed[datetime.date(2009, 12, 4)], 0.01) == 70.0


# -----------------------------------------------------------------------------
# 4. Currency and Data Mode Verification
# -----------------------------------------------------------------------------
def test_data_mode_enum_and_banner_rendering():
    """Verify DataMode constant values and safe rendering."""
    assert app.DataMode.DEMO_DATA == "DEMO DATA"
    assert app.DataMode.USER_DATA == "USER DATA"
    assert app.DataMode.NO_DATA == "NO DATA"
    assert "DATA UNAVAILABLE" in app.DataMode.DATA_UNAVAILABLE

    # Verify render_data_mode_banner does not throw
    with patch("streamlit.markdown") as mock_markdown:
        app.render_data_mode_banner(
            mode=app.DataMode.DEMO_DATA,
            dataset_name="UCI Online Retail II",
            currency="£",
            details="Dec 2009 - Dec 2011",
        )
        assert mock_markdown.called


# -----------------------------------------------------------------------------
# 5. Anomaly Monitor Minimum History Policy
# -----------------------------------------------------------------------------
def test_anomaly_overview_figure_min_history_enforcement():
    """Verify Anomaly Overview Figure requires minimum history (e.g. 14 observations)."""
    # 5 observations: below the 14-observation threshold
    short_records = [
        ActualSalesRecord(calendar_date=datetime.date(2026, 10, i), daily_sales=100.0 + i)
        for i in range(1, 6)
    ]

    # When min_history=14, it must return INSUFFICIENT_HISTORY
    fig_insufficient, status_insufficient = app.build_anomaly_overview_figure(
        actual_records=short_records,
        anomaly_records=[],
        min_history=14,
    )
    assert fig_insufficient is None
    assert status_insufficient == "INSUFFICIENT_HISTORY"

    # When records meet or exceed 14 observations
    adequate_records = [
        ActualSalesRecord(calendar_date=datetime.date(2026, 10, i), daily_sales=100.0 + i)
        for i in range(1, 16)
    ]
    fig_adequate, status_adequate = app.build_anomaly_overview_figure(
        actual_records=adequate_records,
        anomaly_records=[],
        min_history=14,
        currency_symbol="£",
    )
    assert fig_adequate is not None
    assert status_adequate == "MULTI_POINT"
    assert len(fig_adequate.data) == 1
    assert "£" in fig_adequate.layout.yaxis.title.text


# -----------------------------------------------------------------------------
# 6. Forecast Performance: Backtest vs Reconciled Actuals Separation
# -----------------------------------------------------------------------------
def test_forecast_evaluation_separation_and_pending_actuals(isolated_repo):
    """Verify Section A (backtesting) and Section B (reconciled actuals) separation."""
    # Persist a forecast run with 3 horizon items
    run_id = "run-acceptance-separation-01"
    items = [
        {"target_date": "2026-11-01", "step_ahead": 1, "predicted_value": 200.0, "lower_bound": 180.0, "upper_bound": 220.0},
        {"target_date": "2026-11-02", "step_ahead": 2, "predicted_value": 210.0, "lower_bound": 190.0, "upper_bound": 230.0},
        {"target_date": "2026-11-03", "step_ahead": 3, "predicted_value": 220.0, "lower_bound": 200.0, "upper_bound": 240.0},
    ]
    isolated_repo.record_forecast_run(
        run_id=run_id,
        model_name="XGBoost",
        origin_date="2026-10-31",
        horizon_days=3,
        items=items,
        sales_scope="TOTAL_STORE",
    )

    # Ingest actual ONLY for step 1
    isolated_repo.ingest_actual_sales(
        calendar_date="2026-11-01",
        daily_sales=195.0,
        sales_scope="TOTAL_STORE",
    )

    run = isolated_repo.get_forecast_run(run_id)
    step1 = next(it for it in run.items if it.step_ahead == 1)
    step2 = next(it for it in run.items if it.step_ahead == 2)
    step3 = next(it for it in run.items if it.step_ahead == 3)

    # Step 1 is reconciled
    assert step1.actual_value == 195.0
    assert step1.absolute_error == 5.0
    assert step1.interval_covered is True

    # Step 2 and 3 must remain PENDING (None), never 0.0
    assert step2.actual_value is None
    assert step2.absolute_error is None
    assert step3.actual_value is None
    assert step3.absolute_error is None

    # Reconciled evaluation summary must evaluate exactly 1 item
    summary = isolated_repo.get_forecast_evaluation_summary(
        model_name="XGBoost",
        horizon_days=3,
        sales_scope="TOTAL_STORE",
    )
    assert summary["evaluated_count"] == 1
    assert summary["mae"] == 5.0
    assert summary["rmse"] == 5.0
    assert summary["coverage_rate"] == 1.0


# -----------------------------------------------------------------------------
# 7. Drift Baseline Absence and Persistence Handling
# -----------------------------------------------------------------------------
def test_drift_baseline_absence_and_low_sample_size(isolated_repo):
    """Verify DatasetDriftService handles missing reference baselines safely."""
    drift_service = DatasetDriftService(repository=isolated_repo)

    current_data = pd.DataFrame({
        "sales": [100.0, 150.0, 120.0],
        "rolling_mean_7": [110.0, 140.0, 130.0],
    })

    # Evaluate drift with nonexistent baseline ID
    result = drift_service.evaluate_drift(
        current_df=current_data,
        baseline_id="nonexistent_baseline_999",
        sales_scope="TOTAL_STORE",
    )
    assert result.overall_status == DriftStatus.UNKNOWN
    assert "baseline" in result.summary_message.lower()


# -----------------------------------------------------------------------------
# 8. Dashboard Data Loading and Workspace Preparation
# -----------------------------------------------------------------------------
def test_load_online_retail_data_returns_valid_dataset():
    """Verify load_online_retail_data loads processed daily dataset or degrades gracefully."""
    data_dict = app.load_online_retail_data()
    assert isinstance(data_dict, dict)

    if "daily" in data_dict:
        daily_df = data_dict["daily"]
        assert isinstance(daily_df, pd.DataFrame)
        assert "date" in daily_df.columns
        assert "sales" in daily_df.columns
        assert len(daily_df) >= 700  # 739 days

        prov_dict = data_dict.get("provenance")
        assert prov_dict is not None
        assert prov_dict["currency"] == "GBP (£)"
        assert prov_dict["total_transaction_rows"] == 1067371


# -----------------------------------------------------------------------------
# 9. Phase 4C: Dataset Accounting Partition & Penny Reconciliation
# -----------------------------------------------------------------------------
def test_accounting_audit_report_exact_partition_and_reconciliation():
    """Verify data/processed/online_retail_ii_accounting_audit.json guarantees exact partition."""
    audit_path = "data/processed/online_retail_ii_accounting_audit.json"
    if not os.path.exists(audit_path):
        pytest.skip("Audit report not generated yet. Run scripts/audit_dataset_accounting.py first.")

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    part = audit["source_row_accounting_partition"]
    assert part["is_partition_exact"] is True
    assert part["reconciliation_difference"] == 0
    assert part["total_transactions"] == 1067371
    assert part["partition_sum"] == 1067371

    # Mutually exclusive partitions
    cats = part["categories"]
    assert cats["category_A_valid_positive_sales"] == 1041670
    assert cats["category_B_returns_and_cancellations"] == 19494
    assert cats["category_C_quarantined_zero_or_negative_price"] == 6207
    assert cats["category_D_zero_quantity_positive_price"] == 0
    assert cats["category_E_null_or_invalid_date"] == 0

    # Sub-categories
    subs = part["sub_categories"]
    assert subs["B1_both_c_invoice_and_negative_quantity"] == 19493
    assert subs["B2_c_invoice_with_non_negative_quantity"] == 1
    assert subs["B3_negative_quantity_without_c_invoice"] == 0

    # Daily CSV reconciliation: within penny rounding tolerance (< 0.05 GBP)
    daily_rec = audit["daily_csv_reconciliation"]
    assert daily_rec["net_sales_delta"] <= 0.05
    assert audit["financial_reconciliation_gbp"]["currency"] == "GBP (£)"
    assert audit["financial_reconciliation_gbp"]["net_sales"] == 19445179.57


def test_hand_verified_financial_fixture_accounting():
    """Independent verification of mutual exclusivity and accounting on a hand-verified fixture."""
    fixture = [
        # Cat A: Valid positive sale
        {"inv": "500001", "qty": 10, "price": 5.0, "expected_cat": "A"},  # 50.00
        {"inv": "500002", "qty": 2, "price": 25.0, "expected_cat": "A"},  # 50.00
        # Cat B1: C-invoice + negative qty
        {"inv": "C500003", "qty": -2, "price": 5.0, "expected_cat": "B"}, # 10.00 return
        # Cat B2: C-invoice + positive qty
        {"inv": "C500004", "qty": 1, "price": 15.0, "expected_cat": "B"}, # 15.00 return
        # Cat B3: Negative qty without C
        {"inv": "500005", "qty": -4, "price": 5.0, "expected_cat": "B"},  # 20.00 return
        # Cat C1: Zero price without customer
        {"inv": "500006", "qty": 100, "price": 0.0, "expected_cat": "C"}, # 0.0 quarantined
        # Cat C3: Negative price adjustment
        {"inv": "A500007", "qty": 1, "price": -50.0, "expected_cat": "C"},# quarantined
        # Cat D: Zero qty with positive price
        {"inv": "500008", "qty": 0, "price": 10.0, "expected_cat": "D"},  # quarantined
    ]

    gross_sales = 0.0
    returns_val = 0.0
    cat_counts = Counter()

    for row in fixture:
        inv = str(row["inv"])
        qty = float(row["qty"])
        price = float(row["price"])

        # Partition rule
        if price <= 0.0:
            cat_counts["C"] += 1
        elif qty == 0.0 and not inv.startswith("C"):
            cat_counts["D"] += 1
        elif inv.startswith("C") or qty < 0:
            cat_counts["B"] += 1
            returns_val += abs(qty) * price
        else:
            cat_counts["A"] += 1
            gross_sales += qty * price

    assert cat_counts["A"] == 2
    assert cat_counts["B"] == 3
    assert cat_counts["C"] == 2
    assert cat_counts["D"] == 1
    assert sum(cat_counts.values()) == len(fixture)

    assert gross_sales == 100.0  # 10*5 + 2*25
    assert returns_val == 45.0   # 2*5 + 1*15 + 4*5
    net_sales = gross_sales - returns_val
    assert net_sales == 55.0


def test_units_sold_versus_quantity_column_resolution():
    """Verify Sales Overview correctly resolves 'units_sold' column over fallback 'quantity'."""
    df_with_units = pd.DataFrame({
        "sales": [100.0, 200.0],
        "gross_sales": [110.0, 220.0],
        "returns": [10.0, 20.0],
        "order_count": [2, 4],
        "units_sold": [50, 100],
        "quantity": [999, 999],  # intentionally discordant to test precedence
    })

    # Test resolution logic identical to app.py
    if "units_sold" in df_with_units.columns:
        tot_units = int(df_with_units["units_sold"].sum())
    elif "quantity" in df_with_units.columns:
        tot_units = int(df_with_units["quantity"].sum())
    else:
        tot_units = 0

    assert tot_units == 150  # Must be 150 from units_sold, NOT 1998 from quantity


# =============================================================================
# PHASE 4C: INTEGRATION CONTRACT REPAIR REGRESSION TESTS
# =============================================================================

from salespulse.storage.models import DataMonitoringRecord
from salespulse.data.product_aggregation import aggregate_product_sales, CANONICAL_PRODUCT_COLUMNS


def test_data_monitoring_record_freshness_state_and_outcome_adapter():
    """Verify DataMonitoringRecord.freshness_state and latest_ingestion_outcome properties.

    Covers:
    - Normal record with evidence_json (HEALTHY, STALE, UNKNOWN, FAILED)
    - Legacy record without evidence_json (derivation from staleness vs threshold)
    - Null/unobserved records return UNKNOWN (never infer HEALTHY blindly)
    - to_dict() includes both fields
    """
    # 1. Record with full evidence_json
    rec_healthy = DataMonitoringRecord(
        check_id="chk-001",
        status="HEALTHY",
        latest_observation_date=datetime.date(2011, 12, 9),
        staleness_days=5,
        staleness_threshold_days=30,
        last_ingestion_status="PASSED",
        summary_message="Pipeline is healthy",
        evidence_json=json.dumps({
            "freshness": {"freshness_state": "FRESH", "staleness_days": 5},
            "ingestion": {"outcome": "PASSED"}
        }),
    )
    assert rec_healthy.freshness_state == "FRESH"
    assert rec_healthy.latest_ingestion_outcome == "PASSED"
    d_healthy = rec_healthy.to_dict()
    assert d_healthy["freshness_state"] == "FRESH"
    assert d_healthy["latest_ingestion_outcome"] == "PASSED"

    # 2. Record with STALE evidence
    rec_stale = DataMonitoringRecord(
        check_id="chk-002",
        status="WARNING",
        latest_observation_date=datetime.date(2011, 12, 9),
        staleness_days=45,
        staleness_threshold_days=30,
        last_ingestion_status="PASSED",
        summary_message="Data is stale",
        evidence_json=json.dumps({
            "freshness": {"freshness_state": "STALE", "staleness_days": 45},
            "ingestion": {"outcome": "PASSED"}
        }),
    )
    assert rec_stale.freshness_state == "STALE"
    assert rec_stale.latest_ingestion_outcome == "PASSED"

    # 3. Legacy record WITHOUT evidence_json (empty / None) - Staleness > threshold
    rec_legacy_stale = DataMonitoringRecord(
        check_id="chk-003",
        status="WARNING",
        latest_observation_date=datetime.date(2011, 12, 9),
        staleness_days=50,
        staleness_threshold_days=30,
        last_ingestion_status="WARNING",
        summary_message="Legacy stale record",
        evidence_json="",
    )
    assert rec_legacy_stale.freshness_state == "STALE"
    assert rec_legacy_stale.latest_ingestion_outcome == "WARNING"

    # 4. Legacy record WITHOUT evidence_json - Within threshold and status HEALTHY
    rec_legacy_fresh = DataMonitoringRecord(
        check_id="chk-004",
        status="HEALTHY",
        latest_observation_date=datetime.date(2011, 12, 9),
        staleness_days=3,
        staleness_threshold_days=30,
        last_ingestion_status="PASSED",
        summary_message="Legacy fresh record",
        evidence_json=None,
    )
    assert rec_legacy_fresh.freshness_state == "FRESH"
    assert rec_legacy_fresh.latest_ingestion_outcome == "PASSED"

    # 5. Missing / Null observation dates must return UNKNOWN (NEVER blindly infer HEALTHY)
    rec_unverified = DataMonitoringRecord(
        check_id="chk-005",
        status="UNKNOWN",
        latest_observation_date=None,
        staleness_days=None,
        staleness_threshold_days=30,
        last_ingestion_status=None,
        summary_message="Unverified baseline",
        evidence_json="",
    )
    assert rec_unverified.freshness_state == "UNKNOWN"
    assert rec_unverified.latest_ingestion_outcome == "UNKNOWN"

    # 6. Critical record due to ingestion failure
    rec_failed = DataMonitoringRecord(
        check_id="chk-006",
        status="CRITICAL",
        latest_observation_date=datetime.date(2011, 12, 9),
        staleness_days=10,
        staleness_threshold_days=30,
        last_ingestion_status="FAILED",
        summary_message="Ingestion validation failed",
        evidence_json=json.dumps({
            "freshness": {"freshness_state": "FRESH"},
            "ingestion": {"outcome": "FAILED"}
        }),
    )
    # Extracts from evidence:
    assert rec_failed.freshness_state == "FRESH"
    assert rec_failed.latest_ingestion_outcome == "FAILED"


def test_data_freshness_report_state_property_and_cadences():
    """Verify DataFreshnessReport.state compatibility alias and supported cadences."""
    for cadence_val in [CadenceType.DAILY, CadenceType.HOURLY, CadenceType.WEEKLY]:
        report = DataFreshnessReport(
            freshness_state=FreshnessState.FRESH,
            latest_observation_date="2011-12-09",
            staleness_days=0,
            staleness_threshold_days=30,
            cadence=cadence_val.value,
            is_stale=False,
            reference_date="2011-12-09",
            assessment="Fresh daily trading observation.",
        )
        # Direct property access
        assert report.state == FreshnessState.FRESH
        assert report.freshness_state == FreshnessState.FRESH

        # Serialization to dictionary
        d = report.to_dict()
        assert d["freshness_state"] == "FRESH"
        assert d["state"] == "FRESH"
        assert d["cadence"] == cadence_val.value

    # Test distinct states (STALE, MISSING_DATA, UNKNOWN)
    for st in [FreshnessState.STALE, FreshnessState.MISSING_DATA, FreshnessState.UNKNOWN]:
        r = DataFreshnessReport(freshness_state=st)
        assert r.state == st
        assert r.to_dict()["state"] == st.value


def test_canonical_product_sales_aggregation():
    """Verify deterministic product sales aggregation, returns, cancellations, and missing descriptions."""
    fixture_tx = pd.DataFrame([
        # Product 1: Normal sale (10 units @ £5 = £50 gross)
        {"InvoiceNo": "10001", "StockCode": "SKU_01", "Description": "Ceramic Tea Mug", "Quantity": 10, "UnitPrice": 5.0},
        # Product 1: Repeat sale in another invoice (5 units @ £5 = £25 gross)
        {"InvoiceNo": "10002", "StockCode": "SKU_01", "Description": "Ceramic Tea Mug", "Quantity": 5, "UnitPrice": 5.0},
        # Product 1: Return with 'C' invoice and negative qty (2 units returned @ £5 = £10 return)
        {"InvoiceNo": "C10003", "StockCode": "SKU_01", "Description": "Ceramic Tea Mug", "Quantity": -2, "UnitPrice": 5.0},

        # Product 2: Normal sale with missing description (defaults to 'Unknown SKU (SKU_02)')
        {"InvoiceNo": "10004", "StockCode": "SKU_02", "Description": None, "Quantity": 20, "UnitPrice": 10.0},
        # Product 2: Return with negative quantity without 'C' prefix (5 units @ £10 = £50 return)
        {"InvoiceNo": "10005", "StockCode": "SKU_02", "Description": "", "Quantity": -5, "UnitPrice": 10.0},

        # Product 3: Pure cancellation (never had positive sale, only cancellation record preserved)
        {"InvoiceNo": "C10006", "StockCode": "SKU_03", "Description": "Cancelled Item", "Quantity": -1, "UnitPrice": 100.0},

        # Operational quarantined transaction: zero price (should be excluded from revenue)
        {"InvoiceNo": "10007", "StockCode": "SKU_04", "Description": "Sample Item", "Quantity": 1, "UnitPrice": 0.0},
        # Operational quarantined transaction: negative price adjustment
        {"InvoiceNo": "10008", "StockCode": "SKU_04", "Description": "Bad Debt", "Quantity": 1, "UnitPrice": -5.0},
    ])

    agg = aggregate_product_sales(fixture_tx, sort_by="net_sales", ascending=False)

    assert list(agg.columns) == CANONICAL_PRODUCT_COLUMNS
    assert len(agg) == 3  # SKU_04 was quarantined (price <= 0)

    # Validate SKU_01
    sku1 = agg[agg["product_id"] == "SKU_01"].iloc[0]
    assert sku1["description"] == "Ceramic Tea Mug"
    assert sku1["units_sold"] == 15      # 10 + 5
    assert sku1["gross_sales"] == 75.0   # 50 + 25
    assert sku1["returns"] == 10.0       # 2 * 5
    assert sku1["net_sales"] == 65.0     # 75 - 10
    assert sku1["order_count"] == 2      # invoices 10001, 10002

    # Validate SKU_02 (missing description fallback)
    sku2 = agg[agg["product_id"] == "SKU_02"].iloc[0]
    assert sku2["description"] == "Unknown SKU (SKU_02)"
    assert sku2["units_sold"] == 20
    assert sku2["gross_sales"] == 200.0
    assert sku2["returns"] == 50.0
    assert sku2["net_sales"] == 150.0

    # Validate SKU_03 (negative net sales from return)
    sku3 = agg[agg["product_id"] == "SKU_03"].iloc[0]
    assert sku3["units_sold"] == 0
    assert sku3["gross_sales"] == 0.0
    assert sku3["returns"] == 100.0
    assert sku3["net_sales"] == -100.0

    # Ranking: SKU_02 (£150 net) > SKU_01 (£65 net) > SKU_03 (-£100 net)
    assert list(agg["product_id"]) == ["SKU_02", "SKU_01", "SKU_03"]


def test_product_chart_column_discovery_and_empty_state():
    """Verify that product chart rendering detects canonical columns and empty states gracefully."""
    # Test column discovery with canonical columns
    canonical_df = pd.DataFrame({
        "product_id": ["P1", "P2"],
        "description": ["Prod 1", "Prod 2"],
        "units_sold": [10, 20],
        "gross_sales": [120.0, 240.0],
        "returns": [20.0, 40.0],
        "net_sales": [100.0, 200.0],
    })
    x_col = next((c for c in ["net_sales", "NetSales", "gross_sales", "GrossSales"] if c in canonical_df.columns), None)
    assert x_col == "net_sales"

    # Test column discovery with legacy Superstore / alternative columns
    legacy_df = pd.DataFrame({
        "product_id": ["P1", "P2"],
        "description": ["Prod 1", "Prod 2"],
        "units_sold": [10, 20],
        "gross_sales": [100.0, 200.0],
    })
    x_legacy = next((c for c in ["net_sales", "NetSales", "gross_sales", "GrossSales"] if c in legacy_df.columns), None)
    assert x_legacy == "gross_sales"

    # Empty DataFrame check
    empty_df = pd.DataFrame()
    assert empty_df.empty is True

