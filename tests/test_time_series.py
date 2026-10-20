"""
Unit tests for Daily Sales Time Series Aggregation & Missing Date Semantics
"""

import pytest
import pandas as pd
import numpy as np

from salespulse.data.time_series import (
    aggregate_to_daily_sales,
    get_daily_series_diagnostics,
    MISSING_DATE_POLICY,
)


def test_aggregate_to_daily_sales_basic():
    """Verify that multiple transactions on the same day are summed correctly."""
    data = {
        "Order Date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03"],
        "Sales": [100.0, 50.0, 200.0, 75.0],
    }
    df = pd.DataFrame(data)
    daily = aggregate_to_daily_sales(df)

    assert len(daily) == 3
    assert list(daily["sales"]) == [150.0, 200.0, 75.0]
    assert pd.to_datetime(daily.loc[0, "date"]) == pd.Timestamp("2023-01-01")


def test_aggregate_to_daily_sales_missing_dates_zero_fill():
    """Verify missing dates are inserted with 0.0 sales according to zero-fill policy."""
    data = {
        "Order Date": ["2023-01-01", "2023-01-04"],  # Missing Jan 2 and Jan 3
        "Sales": [100.0, 400.0],
    }
    df = pd.DataFrame(data)
    daily = aggregate_to_daily_sales(df, fill_missing_dates=True)

    assert len(daily) == 4
    expected_sales = [100.0, 0.0, 0.0, 400.0]
    assert list(daily["sales"]) == expected_sales

    # Dates must be continuous daily
    date_diffs = daily["date"].diff().dropna()
    assert (date_diffs == pd.Timedelta(days=1)).all()


def test_missing_date_policy_documented():
    """Verify that zero-fill policy is explicitly documented and accessible."""
    assert "ZERO_FILL_POLICY" in MISSING_DATE_POLICY
    assert "$0.00" in MISSING_DATE_POLICY


def test_aggregate_missing_columns():
    """Ensure KeyError is raised if required columns are absent."""
    df = pd.DataFrame({"Wrong Date": ["2023-01-01"], "Sales": [10.0]})
    with pytest.raises(KeyError, match="Order Date"):
        aggregate_to_daily_sales(df)

    df2 = pd.DataFrame({"Order Date": ["2023-01-01"], "Wrong Sales": [10.0]})
    with pytest.raises(KeyError, match="Sales"):
        aggregate_to_daily_sales(df2)


def test_aggregate_empty_df():
    """Ensure ValueError is raised on empty input DataFrame."""
    with pytest.raises(ValueError):
        aggregate_to_daily_sales(pd.DataFrame())


def test_daily_series_diagnostics():
    """Verify diagnostics reports correct summary stats and integrity."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    sales = [10.0, 20.0, 0.0, 40.0, 50.0, 0.0, 70.0, 80.0, 90.0, 100.0]
    df = pd.DataFrame({"date": dates, "sales": sales})

    diag = get_daily_series_diagnostics(df)
    assert diag["valid"] is True
    assert diag["total_days"] == 10
    assert diag["is_strictly_daily"] is True
    assert diag["zero_sales_days"] == 2
    assert diag["min_date"] == "2023-01-01"
    assert diag["max_date"] == "2023-01-10"
