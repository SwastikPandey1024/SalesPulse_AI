"""
Unit tests for SalesPulse AI Data Contracts & Schema Validation
"""

import pytest
import pandas as pd
import numpy as np
import json

from salespulse.data.contracts import (
    GrainType,
    ValidationSeverity,
    ValidationReport,
    ValidationIssue,
    detect_dataset_grain,
    validate_dataset_schema,
    safe_parse_calendar_dates,
    CODE_EMPTY_DATASET,
    CODE_MISSING_COLUMNS,
    CODE_MALFORMED_DATES,
    CODE_NON_NUMERIC_SALES,
    CODE_INFINITE_VALUES,
    CODE_ALL_ROWS_INVALID,
    CODE_UNKNOWN_GRAIN,
    CODE_DUPLICATE_DATES,
    CODE_QUARANTINED_ROWS,
    CODE_DUPLICATES,
    CODE_DUPLICATE_IDENTIFIERS,
    CODE_DATA_STALENESS,
    CODE_NEGATIVE_DAILY_SALES,
    CODE_LEGITIMATE_RETURNS,
    CODE_EXTRA_COLUMNS,
    CODE_MISSING_CALENDAR_DATES,
)
from salespulse.data.time_series import aggregate_to_daily_sales
from salespulse.features.temporal import ENGINEERED_FEATURES


def test_detect_dataset_grain_transaction():
    """Verify transaction grain is detected when multiple rows exist on same date or Order ID is present."""
    # With duplicate dates
    df1 = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "Sales": [10.0, 20.0, 30.0],
    })
    grain1, col_map1 = detect_dataset_grain(df1)
    assert grain1 == GrainType.TRANSACTION
    assert col_map1["date"] == "Order Date"
    assert col_map1["sales"] == "Sales"

    # With order ID
    df2 = pd.DataFrame({
        "Order ID": ["CA-101", "CA-102"],
        "Order Date": ["2023-01-01", "2023-01-02"],
        "Sales": [10.0, 20.0],
    })
    grain2, _ = detect_dataset_grain(df2)
    assert grain2 == GrainType.TRANSACTION


def test_detect_dataset_grain_daily():
    """Verify daily grain is detected for single observation per calendar date without transaction IDs."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "sales": [150.0, 200.0, 175.0],
    })
    grain, col_map = detect_dataset_grain(df)
    assert grain == GrainType.DAILY_AGGREGATED
    assert col_map["date"] == "date"
    assert col_map["sales"] == "sales"


def test_detect_dataset_grain_feature_batch():
    """Verify feature batch grain is detected when 13 engineered features are present."""
    data = {feat: [1.0] for feat in ENGINEERED_FEATURES}
    df = pd.DataFrame(data)
    grain, _ = detect_dataset_grain(df)
    assert grain == GrainType.FEATURE_BATCH


def test_detect_dataset_grain_unknown():
    """Verify unknown grain when required date or sales columns are absent."""
    df = pd.DataFrame({"random_col_1": [1, 2], "random_col_2": ["a", "b"]})
    grain, _ = detect_dataset_grain(df)
    assert grain == GrainType.UNKNOWN


def test_validate_empty_dataset():
    """Verify that an empty dataset generates an ERROR report with CODE_EMPTY_DATASET."""
    df = pd.DataFrame()
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert report.status == "FAILED"
    assert len(report.issues) == 1
    assert report.issues[0].code == CODE_EMPTY_DATASET
    assert report.issues[0].severity == ValidationSeverity.ERROR
    assert len(valid_df) == 0


def test_validate_missing_columns():
    """Verify missing required columns produces an ERROR and status FAILED."""
    df = pd.DataFrame({"Customer": ["Alice", "Bob"], "City": ["Austin", "Denver"]})
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert report.status == "FAILED"
    issue_codes = [i.code for i in report.issues]
    assert CODE_UNKNOWN_GRAIN in issue_codes or CODE_MISSING_COLUMNS in issue_codes


def test_validate_unexpected_extra_columns_preserved():
    """Verify that extra/unexpected columns trigger an INFO issue and are preserved."""
    df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-02"],
        "Sales": [100.0, 200.0],
        "Salesperson": ["Jane", "John"],
        "Store_Region": ["North", "South"],
    })
    report, valid_df, _ = validate_dataset_schema(df)
    assert report.status in ["PASSED", "WARNING"]
    info_issues = [i for i in report.issues if i.code == CODE_EXTRA_COLUMNS]
    assert len(info_issues) == 1
    assert info_issues[0].severity == ValidationSeverity.INFO
    assert "Salesperson" in valid_df.columns
    assert "Store_Region" in valid_df.columns


def test_validate_malformed_dates():
    """Verify unparseable and malformed date strings are quarantined."""
    df = pd.DataFrame({
        "Order Date": ["2023-01-01", "malformed_date_str", "2023-01-03", "9999-99-99"],
        "Sales": [100.0, 150.0, 200.0, 250.0],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert len(valid_df) == 2
    assert len(quarantined_df) == 2
    issue_codes = [i.code for i in report.issues]
    assert CODE_MALFORMED_DATES in issue_codes
    assert CODE_QUARANTINED_ROWS in issue_codes


def test_validate_non_numeric_sales_and_nans():
    """Verify non-numeric text strings and NaNs in sales are quarantined."""
    df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "Sales": [100.0, "INVALID_SALES_STR", np.nan, 200.0],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert len(valid_df) == 2
    assert len(quarantined_df) == 2
    issue_codes = [i.code for i in report.issues]
    assert CODE_NON_NUMERIC_SALES in issue_codes


def test_validate_infinite_sales():
    """Verify infinite numeric values (inf/-inf) are caught and quarantined."""
    df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Sales": [100.0, np.inf, -np.inf],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert len(valid_df) == 1
    assert len(quarantined_df) == 2
    issue_codes = [i.code for i in report.issues]
    assert CODE_INFINITE_VALUES in issue_codes


def test_validate_legitimate_negative_sales_transaction():
    """Verify negative transaction sales are treated as legitimate returns (INFO, not error)."""
    df = pd.DataFrame({
        "Order ID": ["ORD-1", "ORD-2", "ORD-3"],
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "Sales": [100.0, -35.0, 200.0],  # -35.0 is a merchandise return
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert len(valid_df) == 3
    assert len(quarantined_df) == 0
    issue_codes = [i.code for i in report.issues]
    assert CODE_LEGITIMATE_RETURNS in issue_codes
    return_issue = next(i for i in report.issues if i.code == CODE_LEGITIMATE_RETURNS)
    assert return_issue.severity == ValidationSeverity.INFO
    assert return_issue.affected_count == 1


def test_validate_negative_sales_daily_aggregated():
    """Verify negative net daily sales trigger a WARNING review flag."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "sales": [100.0, -25.0, 150.0],  # Net returns exceeded sales on Jan 2
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df, expected_grain=GrainType.DAILY_AGGREGATED)
    assert len(valid_df) == 3
    issue_codes = [i.code for i in report.issues]
    assert CODE_NEGATIVE_DAILY_SALES in issue_codes
    warn_issue = next(i for i in report.issues if i.code == CODE_NEGATIVE_DAILY_SALES)
    assert warn_issue.severity == ValidationSeverity.WARNING


def test_validate_duplicate_records_handling():
    """Verify exact duplicate records are detected and flagged as WARNING."""
    df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "Sales": [100.0, 100.0, 200.0],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    issue_codes = [i.code for i in report.issues]
    assert CODE_DUPLICATES in issue_codes
    dup_issue = next(i for i in report.issues if i.code == CODE_DUPLICATES)
    assert dup_issue.severity == ValidationSeverity.WARNING
    assert dup_issue.affected_count == 1


def test_validate_freshness_threshold():
    """Verify configurable freshness threshold detects stale datasets."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-10"],
        "sales": [100.0, 150.0],
    })
    # Reference date is 2023-03-01 (50 days after Jan 10)
    report, valid_df, _ = validate_dataset_schema(
        df,
        max_staleness_days=30,
        reference_date="2023-03-01",
    )
    issue_codes = [i.code for i in report.issues]
    assert CODE_DATA_STALENESS in issue_codes
    stale_issue = next(i for i in report.issues if i.code == CODE_DATA_STALENESS)
    assert stale_issue.severity == ValidationSeverity.WARNING
    assert stale_issue.affected_count == 50


def test_validate_missing_calendar_dates_info():
    """Verify missing calendar dates are highlighted with zero-fill policy reference."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-05"],  # Missing Jan 2, 3, 4
        "sales": [100.0, 200.0],
    })
    report, _, _ = validate_dataset_schema(df)
    issue_codes = [i.code for i in report.issues]
    assert CODE_MISSING_CALENDAR_DATES in issue_codes
    missing_issue = next(i for i in report.issues if i.code == CODE_MISSING_CALENDAR_DATES)
    assert missing_issue.affected_count == 3
    assert "zero-filled" in missing_issue.message.lower()


def test_safe_rejection_when_all_rows_invalid():
    """Verify dataset is rejected when 100% of rows have fatal schema errors."""
    df = pd.DataFrame({
        "Order Date": ["bad_date_1", "bad_date_2"],
        "Sales": ["not_a_number", "also_text"],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert report.status == "FAILED"
    assert len(valid_df) == 0
    assert len(quarantined_df) == 2
    issue_codes = [i.code for i in report.issues]
    assert CODE_ALL_ROWS_INVALID in issue_codes


def test_report_json_serialization_privacy():
    """Verify report serialization avoids leaking confidential payload data and caps sample rows."""
    report = ValidationReport(grain=GrainType.TRANSACTION, total_rows_read=100)
    report.add_issue(
        code=CODE_MALFORMED_DATES,
        severity=ValidationSeverity.ERROR,
        message="Malformed date format",
        affected_count=20,
        sample_rows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10 indices
        remediation="Format as YYYY-MM-DD",
    )
    json_str = report.to_json()
    data = json.loads(json_str)

    assert data["verdict"] == "FAILED"
    # Verify sample rows is capped to 5 elements
    assert len(data["issues"][0]["sample_rows"]) == 5
    assert data["issues"][0]["sample_rows"] == [1, 2, 3, 4, 5]
    # Verify raw confidential data is absent
    assert "Alice" not in json_str


# ============================================================================
# AUDIT REMEDIATION FOCUSED TESTS (F-01, F-02, F-03, F-04, F-05)
# ============================================================================

def test_f01_feature_batch_preserves_calendar_columns():
    """Fix F-01: Verify feature batch validation strictly requires and preserves all 13 features."""
    row_data = {
        "day": [15.0],
        "month": [6.0],
        "weekday": [3.0],
        "is_weekend": [0.0],
        "weekofyear": [24.0],
        "lag_1": [110.0],
        "lag_7": [105.0],
        "lag_14": [100.0],
        "lag_30": [95.0],
        "rolling_mean_7": [108.0],
        "rolling_mean_14": [102.0],
        "rolling_std_7": [12.0],
        "trend": [45.0],
    }
    df = pd.DataFrame(row_data)
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert report.status == "PASSED"
    assert report.grain == GrainType.FEATURE_BATCH
    assert len(valid_df) == 1
    # Verify day, month, weekday are not modified
    assert valid_df.loc[0, "day"] == 15.0
    assert valid_df.loc[0, "month"] == 6.0
    assert valid_df.loc[0, "weekday"] == 3.0


def test_f01_feature_batch_rejects_missing_features():
    """Fix F-01: Incomplete feature batches cannot silently fall back to defaults."""
    # Omit 'rolling_mean_14'
    incomplete = {f: [1.0] for f in ENGINEERED_FEATURES if f != "rolling_mean_14"}
    df = pd.DataFrame(incomplete)
    report, valid_df, _ = validate_dataset_schema(df, expected_grain=GrainType.FEATURE_BATCH)
    assert report.status == "FAILED"
    assert len(valid_df) == 0
    issue_codes = [i.code for i in report.issues]
    assert CODE_MISSING_COLUMNS in issue_codes


def test_f02_duplicate_daily_dates_block_ingestion():
    """Fix F-02: Duplicate calendar dates in DAILY_AGGREGATED grain must produce ERROR and block ingestion."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "sales": [100.0, 100.0, 200.0],
    })
    report, valid_df, _ = validate_dataset_schema(df, expected_grain=GrainType.DAILY_AGGREGATED)
    assert report.status == "FAILED"
    issue_codes = [i.code for i in report.issues]
    assert CODE_DUPLICATE_DATES in issue_codes
    err_issue = next(i for i in report.issues if i.code == CODE_DUPLICATE_DATES)
    assert err_issue.severity == ValidationSeverity.ERROR
    assert "revenue double-counting" in err_issue.message.lower()


def test_f02_unique_daily_dates_and_transaction_aggregation_preserved():
    """Fix F-02: Unique daily dates pass, and multiple transaction rows on the same day still aggregate correctly."""
    # 1. Unique daily series passes
    daily_df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "sales": [100.0, 200.0],
    })
    report_daily, _, _ = validate_dataset_schema(daily_df, expected_grain=GrainType.DAILY_AGGREGATED)
    assert report_daily.status in ["PASSED", "WARNING"]

    # 2. Transaction grain with same day multiple transactions aggregates correctly
    tx_df = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "Sales": [100.0, 50.0, 200.0],
    })
    report_tx, valid_tx, _ = validate_dataset_schema(tx_df)
    assert report_tx.grain == GrainType.TRANSACTION
    daily_res = aggregate_to_daily_sales(valid_tx, date_col="Order Date", sales_col="Sales")
    assert list(daily_res["sales"]) == [150.0, 200.0]


def test_f03_robust_date_parsing_mixed_timezones_and_boundary():
    """Fix F-03: Robust date parser handles mixed timezones without crashing or shifting business dates."""
    dates_input = pd.Series([
        "2023-01-01T23:30:00-05:00",  # Aware with negative offset
        "2023-01-01T10:00:00Z",       # UTC aware
        "2023-01-02 14:15:00",        # Naive datetime
        "2023-01-03",                 # Date only
        "bad_date_format",            # Malformed string
        None                          # Null
    ])
    parsed = safe_parse_calendar_dates(dates_input)
    assert pd.api.types.is_datetime64_any_dtype(parsed)
    # Check that intended business calendar date is preserved without UTC shifting across midnight
    assert parsed.iloc[0] == pd.Timestamp("2023-01-01")
    assert parsed.iloc[1] == pd.Timestamp("2023-01-01")
    assert parsed.iloc[2] == pd.Timestamp("2023-01-02")
    assert parsed.iloc[3] == pd.Timestamp("2023-01-03")
    assert pd.isna(parsed.iloc[4])
    assert pd.isna(parsed.iloc[5])


def test_f03_mixed_timezone_dataset_validation():
    """Fix F-03: validate_dataset_schema gracefully handles mixed timezone strings."""
    df = pd.DataFrame({
        "date": ["2023-01-01T10:00:00Z", "2023-01-02 10:00:00"],
        "sales": [100.0, 200.0],
    })
    report, valid_df, quarantined_df = validate_dataset_schema(df)
    assert report.status in ["PASSED", "WARNING"]
    assert len(valid_df) == 2
    assert len(quarantined_df) == 0


def test_f04_identifier_aware_duplicate_diagnostics():
    """Fix F-04: Flag duplicate primary keys (Row ID) and legitimate multi-line Order IDs."""
    # 1. Repeated primary key (Row ID) with differing sales amount
    df_row_id = pd.DataFrame({
        "Row ID": [101, 101, 102],
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "Sales": [100.0, 150.0, 200.0],  # Differing amounts
    })
    report, valid_df, _ = validate_dataset_schema(df_row_id)
    issue_codes = [i.code for i in report.issues]
    assert CODE_DUPLICATE_IDENTIFIERS in issue_codes
    # Valid rows must NOT be automatically dropped
    assert len(valid_df) == 3

    # 2. Legitimate multi-line order items (same Order ID, distinct Product ID)
    df_order_prod = pd.DataFrame({
        "Order ID": ["CA-1", "CA-1"],
        "Product ID": ["PROD-A", "PROD-B"],
        "Order Date": ["2023-01-01", "2023-01-01"],
        "Sales": [50.0, 75.0],
    })
    report_op, valid_op, _ = validate_dataset_schema(df_order_prod)
    # Distinct products under one order is legitimate, so no duplicate identifier warning
    assert CODE_DUPLICATE_IDENTIFIERS not in [i.code for i in report_op.issues]
    assert len(valid_op) == 2


def test_f05_strict_json_rfc_compliance():
    """Fix F-05: Serialization strictly outputs valid RFC 8259 JSON and sanitizes non-finite floats."""
    report = ValidationReport(grain=GrainType.TRANSACTION, total_rows_read=10)
    report.diagnostics = {
        "nan_metric": np.nan,
        "inf_metric": np.inf,
        "neg_inf_metric": -np.inf,
        "valid_float": 123.45,
        "int_scalar": np.int64(99),
        "timestamp": pd.Timestamp("2023-01-01"),
        "nested": {"array": [np.nan, 42.0, np.inf]},
    }
    json_str = report.to_json()
    # Must be valid RFC 8259 JSON (no unquoted NaN or Infinity)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str
    # Must parse cleanly with strict standard JSON loader
    parsed = json.loads(json_str)
    assert parsed["diagnostics"]["nan_metric"] is None
    assert parsed["diagnostics"]["inf_metric"] is None
    assert parsed["diagnostics"]["neg_inf_metric"] is None
    assert parsed["diagnostics"]["nested"]["array"] == [None, 42.0, None]
