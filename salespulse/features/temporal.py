"""
SalesPulse AI - Leakage-Free Temporal & Rolling Feature Engineering
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


ENGINEERED_FEATURES: List[str] = [
    "day",
    "month",
    "weekday",
    "is_weekend",
    "weekofyear",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "trend",
]


def extract_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Extract temporal calendar features from a date series.

    Guarantees no future leakage as calendar features depend strictly on the timestamp.
    """
    date_dt = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "day": date_dt.dt.day.astype(np.float64),
            "month": date_dt.dt.month.astype(np.float64),
            "weekday": date_dt.dt.weekday.astype(np.float64),
            "is_weekend": (date_dt.dt.weekday >= 5).astype(np.float64),
            "weekofyear": date_dt.dt.isocalendar().week.astype(np.float64),
        },
        index=dates.index,
    )


def build_supervised_daily_features(
    daily_df: pd.DataFrame,
    sales_col: str = "sales",
    date_col: str = "date",
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate the 13 engineered features from a daily sales DataFrame without target leakage.

    All lag and rolling features are shifted such that row t only observes
    information available strictly prior to t (i.e. t-1, t-7, etc.).

    Args:
        daily_df: DataFrame with contiguous daily dates and sales.
        sales_col: Column containing target sales.
        date_col: Column containing timestamp dates.
        drop_na: Whether to drop rows where lag/rolling history is unavailable (initial 30 days).

    Returns:
        Tuple of (X: pd.DataFrame with 13 features, y: pd.Series of target sales)
    """
    df = daily_df.sort_values(date_col).copy()
    sales = df[sales_col].astype(np.float64)

    # 1. Calendar features
    cal_df = extract_calendar_features(df[date_col])

    # 2. Strict lagged features (prior observations)
    # lag_1 is y_{t-1}, lag_7 is y_{t-7}, etc.
    lag_1 = sales.shift(1)
    lag_7 = sales.shift(7)
    lag_14 = sales.shift(14)
    lag_30 = sales.shift(30)

    # 3. Rolling window features (must roll on lag_1 to strictly exclude current target y_t)
    # rolling_mean_7 on lag_1 averages [y_{t-7}, ..., y_{t-1}]
    rolling_mean_7 = lag_1.rolling(window=7, min_periods=7).mean()
    rolling_mean_14 = lag_1.rolling(window=14, min_periods=14).mean()
    # Fill rolling_std with 0.0 if zero variance in window, min_periods=7
    rolling_std_7 = lag_1.rolling(window=7, min_periods=7).std().fillna(0.0)

    # 4. Trend momentum index
    trend = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)

    features_df = pd.DataFrame(
        {
            "day": cal_df["day"],
            "month": cal_df["month"],
            "weekday": cal_df["weekday"],
            "is_weekend": cal_df["is_weekend"],
            "weekofyear": cal_df["weekofyear"],
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_30": lag_30,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": rolling_mean_14,
            "rolling_std_7": rolling_std_7,
            "trend": trend,
        },
        index=df.index,
    )[ENGINEERED_FEATURES]

    target = sales.copy()

    if drop_na:
        valid_idx = features_df.dropna().index
        features_df = features_df.loc[valid_idx].reset_index(drop=True)
        target = target.loc[valid_idx].reset_index(drop=True)

    return features_df, target


def compute_step_feature_vector(
    target_date: pd.Timestamp,
    history_sales: List[float],
    trend_idx: float,
) -> pd.DataFrame:
    """Compute the 13-feature input vector for a single future prediction step.

    Used during iterative recursive forecasting.

    Args:
        target_date: Future date to forecast.
        history_sales: Contiguous past sales series ending at target_date - 1 day.
                       Must have at least 30 observations for complete feature support.
        trend_idx: Trend index value for the forecast date.

    Returns:
        1-row pd.DataFrame matching ENGINEERED_FEATURES schema.

    Raises:
        ValueError: If history_sales contains fewer than 30 observations.
    """
    if len(history_sales) < 30:
        raise ValueError(
            f"Insufficient history: {len(history_sales)} observations provided, "
            f"minimum required for lag_30 is 30."
        )

    date_obj = pd.Timestamp(target_date)
    day = float(date_obj.day)
    month = float(date_obj.month)
    weekday = float(date_obj.weekday())
    is_weekend = 1.0 if weekday >= 5 else 0.0
    weekofyear = float(date_obj.isocalendar().week)

    lag_1 = float(history_sales[-1])
    lag_7 = float(history_sales[-7])
    lag_14 = float(history_sales[-14])
    lag_30 = float(history_sales[-30])

    rm_7_window = history_sales[-7:]
    rm_14_window = history_sales[-14:]

    rolling_mean_7 = float(np.mean(rm_7_window))
    rolling_mean_14 = float(np.mean(rm_14_window))
    rolling_std_7 = float(np.std(rm_7_window, ddof=1)) if len(rm_7_window) > 1 else 0.0
    if np.isnan(rolling_std_7):
        rolling_std_7 = 0.0

    data = {
        "day": [day],
        "month": [month],
        "weekday": [weekday],
        "is_weekend": [is_weekend],
        "weekofyear": [weekofyear],
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "lag_14": [lag_14],
        "lag_30": [lag_30],
        "rolling_mean_7": [rolling_mean_7],
        "rolling_mean_14": [rolling_mean_14],
        "rolling_std_7": [rolling_std_7],
        "trend": [float(trend_idx)],
    }

    return pd.DataFrame(data)[ENGINEERED_FEATURES].astype(np.float64)
