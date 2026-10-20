"""
SalesPulse AI - Data Ingestion & Daily Time Series Aggregation Module
"""

import os
import warnings
import datetime
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


MISSING_DATE_POLICY = (
    "ZERO_FILL_POLICY: In retail point-of-sale systems, calendar days absent "
    "from the order log represent zero sales transactions ($0.00), not missing "
    "or unobserved data. Reindexing to a continuous daily frequency ('D') with "
    "fill_value=0.0 preserves true calendar spacing for 1-, 7-, 14-, and 30-day "
    "lags and rolling windows."
)


def parse_business_calendar_date(val: Any) -> pd.Timestamp:
    """Extract local calendar date (year, month, day) without shifting across midnight boundaries.

    In retail point-of-sale datasets, dates indicate the local business trading day.
    This helper extracts normalized calendar midnight dates while safely ignoring time offsets.
    """
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return pd.Timestamp(year=val.year, month=val.month, day=val.day)
    val_str = str(val).strip()
    if not val_str:
        return pd.NaT
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(val_str, errors="coerce", format="mixed")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)
    except Exception:
        return pd.NaT


def safe_parse_calendar_dates(series: pd.Series) -> pd.Series:
    """Robust date parsing preserving intended business calendar dates.

    Safely handles:
    - Naive date/datetime strings
    - UTC ISO-8601 strings
    - Timezone offset-aware strings (without shifting dates across midnight)
    - Mixed timezone-aware and timezone-naive series
    - Unparseable or malformed date values (returned as NaT without unhandled exceptions)
    """
    if series is None or series.empty:
        return pd.Series([], dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dtype, "tz", None) is not None:
            return series.dt.tz_localize(None).dt.normalize()
        return series.dt.normalize()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if hasattr(parsed, "dt") and not (hasattr(parsed, "dtype") and parsed.dtype == object):
                if getattr(parsed.dtype, "tz", None) is not None:
                    return parsed.dt.tz_localize(None).dt.normalize()
                return parsed.dt.normalize()
        except Exception:
            pass

    # Element-wise fallback for mixed timezone offsets or object-wrapped series
    res = series.apply(parse_business_calendar_date)
    return pd.to_datetime(res, errors="coerce")


def load_raw_transactions(csv_path: str, encoding: str = "latin1") -> pd.DataFrame:
    """Load raw transactions from CSV with date parsing and encoding safety.

    Args:
        csv_path: Path to the transactions CSV file.
        encoding: File encoding (defaults to 'latin1' for Superstore dataset).

    Returns:
        pd.DataFrame containing raw transaction rows.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    if df.empty:
        raise ValueError(f"Transaction dataset at {csv_path} is empty.")

    return df


def aggregate_to_daily_sales(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    sales_col: str = "Sales",
    fill_missing_dates: bool = True,
    missing_fill_value: float = 0.0,
) -> pd.DataFrame:
    """Aggregate transaction line-items by date to create a unified daily sales time series.

    Applies the documented ZERO_FILL_POLICY to guarantee contiguous daily calendar indexing.

    Args:
        df: Input transactions DataFrame.
        date_col: Name of the timestamp/date column.
        sales_col: Name of the monetary sales amount column.
        fill_missing_dates: Whether to reindex to continuous daily calendar frequency.
        missing_fill_value: Value to assign to calendar days without transactions (default 0.0).

    Returns:
        pd.DataFrame with columns ['date', 'sales'], sorted chronologically.

    Raises:
        KeyError: If date_col or sales_col are missing.
        ValueError: If input DataFrame is empty or dates cannot be parsed.
    """
    if df.empty:
        raise ValueError("Cannot aggregate an empty DataFrame.")

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    if sales_col not in df.columns:
        raise KeyError(f"Sales column '{sales_col}' not found in DataFrame.")

    # Convert date column safely without modifying original DataFrame
    working_df = df[[date_col, sales_col]].copy()
    working_df[date_col] = safe_parse_calendar_dates(working_df[date_col])

    # Reject null dates
    invalid_dates = working_df[date_col].isna().sum()
    if invalid_dates > 0:
        working_df = working_df.dropna(subset=[date_col])

    if working_df.empty:
        raise ValueError("No valid date records found in dataset.")

    # Group by normalized date (safe_parse_calendar_dates already returns normalized date)
    working_df["date_only"] = working_df[date_col]
    daily_series = working_df.groupby("date_only")[sales_col].sum()

    if fill_missing_dates:
        full_date_range = pd.date_range(
            start=daily_series.index.min(),
            end=daily_series.index.max(),
            freq="D",
            name="date",
        )
        daily_series = daily_series.reindex(full_date_range, fill_value=missing_fill_value)
    else:
        daily_series.index.name = "date"

    result_df = daily_series.reset_index()
    result_df.columns = ["date", "sales"]
    result_df = result_df.sort_values("date").reset_index(drop=True)

    # Ensure sales is float64
    result_df["sales"] = result_df["sales"].astype(np.float64)

    return result_df


def get_daily_series_diagnostics(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key integrity and completeness metrics for a daily sales series."""
    if daily_df.empty or "date" not in daily_df.columns or "sales" not in daily_df.columns:
        return {"valid": False, "error": "Invalid daily sales DataFrame structure."}

    dates = pd.to_datetime(daily_df["date"])
    total_days = len(daily_df)
    min_date = dates.min()
    max_date = dates.max()
    expected_calendar_days = (max_date - min_date).days + 1
    zero_sales_days = int((daily_df["sales"] == 0.0).sum())
    negative_sales_days = int((daily_df["sales"] < 0.0).sum())

    diffs = dates.diff().dropna()
    is_strictly_daily = bool((diffs == pd.Timedelta(days=1)).all()) if len(diffs) > 0 else True

    return {
        "valid": True,
        "total_days": total_days,
        "expected_calendar_days": expected_calendar_days,
        "is_strictly_daily": is_strictly_daily,
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "zero_sales_days": zero_sales_days,
        "negative_sales_days": negative_sales_days,
        "mean_daily_sales": float(daily_df["sales"].mean()),
        "std_daily_sales": float(daily_df["sales"].std()),
        "median_daily_sales": float(daily_df["sales"].median()),
        "policy": MISSING_DATE_POLICY,
    }
