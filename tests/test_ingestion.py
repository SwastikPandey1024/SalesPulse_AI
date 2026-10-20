"""
Unit tests for SalesPulse AI Safe Ingestion & Data Quality Diagnostics
"""

import os
import io
import pytest
import pandas as pd
import numpy as np

from salespulse.data.ingestion import (
    read_tabular_file,
    safe_ingest_file,
    safe_ingest_and_aggregate_to_daily,
)
from salespulse.data.quality import compute_data_quality_diagnostics
from salespulse.data.contracts import GrainType


def test_read_tabular_csv_utf8_and_latin1():
    """Verify CSV reading handles both UTF-8 and latin-1 encoded content."""
    csv_content = "Order Date,Sales\n2023-01-01,100.50\n2023-01-02,200.75\n"

    # UTF-8 buffer
    utf8_buf = io.BytesIO(csv_content.encode("utf-8"))
    df_utf8, _ = read_tabular_file(utf8_buf, file_name="sales_utf8.csv")
    assert len(df_utf8) == 2
    assert list(df_utf8.columns) == ["Order Date", "Sales"]

    # Latin1 buffer with special character
    latin1_content = "Order Date,Sales,Region\n2023-01-01,100.50,Montréal\n"
    latin1_buf = io.BytesIO(latin1_content.encode("latin1"))
    df_lat, _ = read_tabular_file(latin1_buf, file_name="sales_latin1.csv")
    assert len(df_lat) == 1
    assert "Montréal" in df_lat["Region"].iloc[0]


def test_read_tabular_xlsx(tmp_path):
    """Verify XLSX Excel spreadsheet reading."""
    df_orig = pd.DataFrame({
        "Order Date": ["2023-01-01", "2023-01-02"],
        "Sales": [150.0, 300.0],
    })
    xlsx_path = os.path.join(tmp_path, "test_sales.xlsx")
    df_orig.to_excel(xlsx_path, index=False)

    df_loaded, fname = read_tabular_file(xlsx_path)
    assert len(df_loaded) == 2
    assert "test_sales.xlsx" in fname
    assert list(df_loaded["Sales"]) == [150.0, 300.0]


def test_read_unsupported_file_extension():
    """Verify ValueError is raised when encountering unsupported formats like .json or .zip."""
    buf = io.BytesIO(b'{"key": "value"}')
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_file(buf, file_name="data.json")

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_file(buf, file_name="archive.zip")


def test_safe_ingest_file_empty_bytes():
    """Verify empty file is safely rejected with FAILED verdict."""
    empty_buf = io.BytesIO(b"")
    result = safe_ingest_file(empty_buf, file_name="empty.csv")
    assert result.is_usable is False
    assert result.report.status == "FAILED"
    assert len(result.valid_df) == 0


def test_safe_ingest_file_end_to_end_valid():
    """Verify valid transaction dataset produces an IngestionResult with PASSED status and diagnostics."""
    csv_data = (
        "Order ID,Order Date,Sales\n"
        "CA-001,2023-01-01,150.00\n"
        "CA-002,2023-01-01,75.00\n"
        "CA-003,2023-01-02,220.00\n"
    )
    buf = io.BytesIO(csv_data.encode("utf-8"))
    result = safe_ingest_file(buf, file_name="orders.csv", reference_date="2023-01-10")

    assert result.is_usable is True
    assert result.report.status == "PASSED"
    assert result.report.grain == GrainType.TRANSACTION
    assert len(result.valid_df) == 3
    assert len(result.quarantined_df) == 0

    # Diagnostics verification
    diag = result.report.diagnostics
    assert diag["valid"] is True
    assert diag["date_coverage"]["calendar_span_days"] == 2
    assert diag["sales_distribution"]["mean"] == 148.33
    assert diag["freshness"]["is_stale"] is False


def test_safe_ingest_and_aggregate_to_daily():
    """Verify safe ingestion aggregates transactions into a contiguous daily sales series."""
    csv_data = (
        "Order Date,Sales\n"
        "2023-01-01,100.00\n"
        "2023-01-01,50.00\n"
        "2023-01-03,200.00\n"  # Jan 2 missing
    )
    buf = io.BytesIO(csv_data.encode("utf-8"))
    daily_df, result = safe_ingest_and_aggregate_to_daily(buf, file_name="pos_transactions.csv")

    assert result.is_usable is True
    assert daily_df is not None
    # Jan 1, Jan 2 (zero-filled), Jan 3
    assert len(daily_df) == 3
    assert list(daily_df["sales"]) == [150.0, 0.0, 200.0]
    assert pd.to_datetime(daily_df.loc[1, "date"]) == pd.Timestamp("2023-01-02")


def test_compute_data_quality_diagnostics_outliers_and_zeroes():
    """Verify diagnostic anomaly detection flags extreme outliers and high zero-sales ratios."""
    # Data with high zero ratio and an extreme outlier
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    sales = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 15.0, 20.0, 10000.0]
    df = pd.DataFrame({"date": dates, "sales": sales})

    diag = compute_data_quality_diagnostics(
        df=df,
        grain=GrainType.DAILY_AGGREGATED,
        date_col="date",
        sales_col="sales",
        max_staleness_days=30,
        reference_date="2023-01-15",
    )

    assert diag["valid"] is True
    flags = [a["flag"] for a in diag["anomalies"]]
    assert "HIGH_ZERO_SALES_RATIO" in flags
    assert "STATISTICAL_DISTRIBUTION_OUTLIER" in flags
    assert diag["sales_distribution"]["max"] == 10000.0
    assert diag["freshness"]["is_stale"] is False
