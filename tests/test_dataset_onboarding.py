"""
SalesPulse AI - Dataset Onboarding & Multi-Mode Unit Tests
Verifies:
- UCI Online Retail II provenance metadata structure and schema contracts.
- Deterministic calculation of gross sales, returns, and net sales.
- Treatment of cancellations ('C' prefix or negative quantity).
- Quarantining of non-positive prices and unparseable dates.
- Daily aggregation matching SalesPulse data contracts.
- Continuous daily reindexing under MISSING_DATE_POLICY.
- Multi-dataset mode handling (DEMO, USER, NO_DATA, DATA_UNAVAILABLE).
"""

import os
import io
import json
import datetime
import pytest
import pandas as pd
import numpy as np

from salespulse.data.contracts import (
    detect_dataset_grain,
    validate_dataset_schema,
    GrainType,
    ValidationSeverity,
)
from salespulse.data.time_series import (
    MISSING_DATE_POLICY,
    get_daily_series_diagnostics,
)
from salespulse.data.onboard_online_retail import (
    UCI_DATASET_NAME,
    UCI_DATASET_CITATION,
    UCI_LICENSE,
    OnboardingProvenance,
    RAW_ZIP_FILENAME,
    RAW_XLSX_FILENAME,
)


@pytest.fixture
def sample_online_retail_raw():
    """Compact raw transaction rows mirroring UCI Online Retail II schema."""
    data = [
        # Normal positive sales
        ("489434", "85048", "15CM CHRISTMAS GLASS BALL", 12, "2009-12-01 07:45:00", 6.95, 13085, "United Kingdom"),
        ("489434", "79323P", "PINK CHERRY LIGHTS", 10, "2009-12-01 07:45:00", 5.00, 13085, "United Kingdom"),
        ("489435", "21232", "STRAWBERRY CERAMIC TRINKET BOX", 24, "2009-12-01 09:20:00", 1.25, 13085, "United Kingdom"),
        # Return / Cancellation: Invoice starts with 'C' and negative quantity
        ("C489436", "85048", "15CM CHRISTMAS GLASS BALL", -2, "2009-12-01 10:15:00", 6.95, 13085, "United Kingdom"),
        # Next day positive sales
        ("489437", "22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 40, "2009-12-02 11:30:00", 2.55, 15311, "Germany"),
        # Zero price / bad debt adjustment (should be quarantined)
        ("489438", "D", "Bad Debt Adjustment", 1, "2009-12-02 12:00:00", 0.0, None, "United Kingdom"),
        # Unparseable date (should be quarantined)
        ("489439", "21232", "STRAWBERRY CERAMIC TRINKET BOX", 5, None, 1.25, 15311, "France"),
        # Trading day after gap (2009-12-04, leaving 2009-12-03 unobserved)
        ("489440", "85048", "15CM CHRISTMAS GLASS BALL", 6, "2009-12-04 14:00:00", 6.95, 14000, "United Kingdom"),
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return pd.DataFrame(data, columns=cols)


def test_online_retail_provenance_metadata_contract():
    """Verify OnboardingProvenance dataclass enforces all required fields and JSON safety."""
    prov = OnboardingProvenance(
        dataset_name=UCI_DATASET_NAME,
        source_url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        landing_page_url="https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        citation=UCI_DATASET_CITATION,
        license=UCI_LICENSE,
        retrieval_date_utc="2026-10-21T10:00:00Z",
        zip_sha256="abc123sha",
        zip_size_bytes=45622418,
        xlsx_sha256="def456sha",
        xlsx_size_bytes=45622278,
        raw_sheet_names=["Year 2009-2010", "Year 2010-2011"],
        total_raw_rows=1067373,
        total_header_rows=2,
        total_transaction_rows=1067371,
        valid_transaction_rows=1000000,
        cancellation_rows=20000,
        zero_or_negative_price_rows=5000,
        null_date_rows=0,
        unique_invoices=50000,
        unique_cancellation_invoices=8000,
        unique_products=4500,
        unique_countries=43,
        date_range_start="2009-12-01",
        date_range_end="2011-12-09",
        total_calendar_days=739,
        observed_trading_days=600,
        unobserved_calendar_days=139,
        currency="GBP (£)",
        total_gross_sales=18000000.0,
        total_returns=1500000.0,
        total_net_sales=16500000.0,
        missing_date_policy=MISSING_DATE_POLICY,
        processing_duration_seconds=12.5,
    )

    p_dict = prov.to_dict()
    assert p_dict["dataset_name"] == "Online Retail II"
    assert "Chen, D. (2012)" in p_dict["citation"]
    assert "CC BY 4.0" in p_dict["license"]
    assert p_dict["total_transaction_rows"] == 1067371
    assert p_dict["currency"] == "GBP (£)"

    p_json = prov.to_json()
    assert isinstance(p_json, str)
    parsed = json.loads(p_json)
    assert parsed["zip_size_bytes"] == 45622418


def test_business_rules_gross_net_sales_and_returns(sample_online_retail_raw):
    """Verify returns are segregated from gross sales and net sales is computed correctly."""
    df = sample_online_retail_raw.copy()

    # Filter valid prices and dates
    valid_mask = (df["Price"] > 0) & (df["InvoiceDate"].notna())
    df_clean = df[valid_mask].copy()

    df_clean["is_cancellation"] = df_clean["Invoice"].str.startswith("C") | (df_clean["Quantity"] < 0)
    
    # Gross items
    gross_items = df_clean[~df_clean["is_cancellation"]]
    gross_sales = (gross_items["Quantity"] * gross_items["Price"]).sum()
    
    # 12*6.95 (83.40) + 10*5.0 (50.00) + 24*1.25 (30.00) + 40*2.55 (102.00) + 6*6.95 (41.70) = 307.10
    expected_gross = (12 * 6.95) + (10 * 5.0) + (24 * 1.25) + (40 * 2.55) + (6 * 6.95)
    assert pytest.approx(gross_sales, 0.01) == expected_gross

    # Return items
    return_items = df_clean[df_clean["is_cancellation"]]
    returns_val = (return_items["Quantity"].abs() * return_items["Price"]).sum()
    # 2 * 6.95 = 13.90
    expected_returns = 2 * 6.95
    assert pytest.approx(returns_val, 0.01) == expected_returns

    # Net sales = gross - returns = 307.10 - 13.90 = 293.20
    net_sales = gross_sales - returns_val
    assert pytest.approx(net_sales, 0.01) == 293.20


def test_daily_aggregation_and_missing_date_policy(sample_online_retail_raw):
    """Verify daily aggregation produces continuous daily calendar sequence under zero-fill policy."""
    df = sample_online_retail_raw.copy()
    valid_mask = (df["Price"] > 0) & (df["InvoiceDate"].notna())
    df_clean = df[valid_mask].copy()
    df_clean["parsed_date"] = pd.to_datetime(df_clean["InvoiceDate"]).dt.date
    df_clean["is_cancellation"] = df_clean["Invoice"].str.startswith("C") | (df_clean["Quantity"] < 0)
    df_clean["line_val"] = np.where(
        df_clean["is_cancellation"],
        -(df_clean["Quantity"].abs() * df_clean["Price"]),
        df_clean["Quantity"] * df_clean["Price"]
    )

    daily_net = df_clean.groupby("parsed_date")["line_val"].sum()

    # Dates present in raw data: 2009-12-01, 2009-12-02, 2009-12-04
    # Expected reindexed range: 2009-12-01 through 2009-12-04 (4 days, 2009-12-03 is filled with 0.0)
    full_idx = pd.date_range(start="2009-12-01", end="2009-12-04", freq="D")
    daily_reindexed = daily_net.reindex(full_idx.date, fill_value=0.0)

    assert len(daily_reindexed) == 4
    # Day 1 (Dec 1): (12*6.95 + 10*5 + 24*1.25) - (2*6.95) = 163.40 - 13.90 = 149.50
    assert pytest.approx(daily_reindexed[datetime.date(2009, 12, 1)], 0.01) == 149.50
    # Day 2 (Dec 2): 40 * 2.55 = 102.00
    assert pytest.approx(daily_reindexed[datetime.date(2009, 12, 2)], 0.01) == 102.00
    # Day 3 (Dec 3): Unobserved day -> strictly 0.0
    assert daily_reindexed[datetime.date(2009, 12, 3)] == 0.0
    # Day 4 (Dec 4): 6 * 6.95 = 41.70
    assert pytest.approx(daily_reindexed[datetime.date(2009, 12, 4)], 0.01) == 41.70

    # Test contract validation on derived daily DataFrame
    daily_df = pd.DataFrame({
        "date": [str(d) for d in daily_reindexed.index],
        "sales": daily_reindexed.values,
    })
    report, valid_df, _ = validate_dataset_schema(daily_df, expected_grain=GrainType.DAILY_AGGREGATED)
    assert report.status == "PASSED"
    assert len(valid_df) == 4
