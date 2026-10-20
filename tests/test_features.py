"""
Unit tests for Temporal Feature Engineering & Leakage Prevention
"""

import pytest
import pandas as pd
import numpy as np

from salespulse.features.temporal import (
    ENGINEERED_FEATURES,
    build_supervised_daily_features,
    compute_step_feature_vector,
)


def test_engineered_features_contract():
    """Verify the 13-feature contract matches the required schema."""
    assert len(ENGINEERED_FEATURES) == 13
    expected = [
        "day", "month", "weekday", "is_weekend", "weekofyear",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "trend"
    ]
    assert ENGINEERED_FEATURES == expected


def test_no_target_leakage():
    """CRITICAL TEST: Verify feature calculation at time t does NOT leak data from time t or future.

    Modifying future sales or current sales y_t must not alter the feature vector at time t.
    """
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    base_sales = np.arange(100.0, 160.0)

    df1 = pd.DataFrame({"date": dates, "sales": base_sales})
    X1, y1 = build_supervised_daily_features(df1, drop_na=True)

    # In df2, we drastically alter the target values of the last 5 days
    df2 = pd.DataFrame({"date": dates, "sales": base_sales.copy()})
    df2.loc[55:, "sales"] = 999999.0  # Massive spike at t=55..59

    X2, y2 = build_supervised_daily_features(df2, drop_na=True)

    # For any index t < 55, features must be 100% IDENTICAL between X1 and X2
    # Because drop_na drops the first 30 days, row index 20 corresponds to original day 50
    t_inspect = 20  # original day 50 (well before day 55)
    pd.testing.assert_series_equal(X1.iloc[t_inspect], X2.iloc[t_inspect])

    # Check lag_1 at row 0 (original day 30):
    # lag_1 should equal day 29 sales
    assert X1.loc[0, "lag_1"] == df1.loc[29, "sales"]
    assert X1.loc[0, "lag_7"] == df1.loc[23, "sales"]
    assert X1.loc[0, "lag_14"] == df1.loc[16, "sales"]
    assert X1.loc[0, "lag_30"] == df1.loc[0, "sales"]

    # Rolling mean 7 at day 30 should be the mean of days [23..29]
    expected_rm7 = df1.loc[23:29, "sales"].mean()
    assert np.isclose(X1.loc[0, "rolling_mean_7"], expected_rm7)


def test_compute_step_feature_vector_insufficient_history():
    """Verify that fewer than 30 history points raises ValueError."""
    history = [100.0] * 29
    with pytest.raises(ValueError, match="Insufficient history"):
        compute_step_feature_vector(
            target_date=pd.Timestamp("2023-02-01"),
            history_sales=history,
            trend_idx=30,
        )


def test_compute_step_feature_vector_valid():
    """Verify single-step feature vector returns valid 1-row DataFrame with all 13 features."""
    history = list(range(1, 35))  # 34 points
    feat_df = compute_step_feature_vector(
        target_date=pd.Timestamp("2023-05-15"),
        history_sales=history,
        trend_idx=100.0,
    )

    assert feat_df.shape == (1, 13)
    assert list(feat_df.columns) == ENGINEERED_FEATURES
    assert feat_df.loc[0, "day"] == 15.0
    assert feat_df.loc[0, "month"] == 5.0
    assert feat_df.loc[0, "lag_1"] == 34.0
    assert feat_df.loc[0, "lag_7"] == 28.0
    assert feat_df.loc[0, "lag_14"] == 21.0
    assert feat_df.loc[0, "lag_30"] == 5.0
    assert feat_df.loc[0, "trend"] == 100.0
