"""
Unit tests for Forecasting Metrics & Edge Cases
"""

import pytest
import numpy as np

from salespulse.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    calculate_bias,
    calculate_mean_error,
    calculate_coverage,
    compute_all_metrics,
)


def test_metric_formulas_exact():
    """Verify metrics with exact hand-calculated expected values.

    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 330.0]
    errors = [+10.0, -10.0, +30.0]
    abs_errors = [10.0, 10.0, 30.0]
    MAE = 50 / 3 = 16.6666...
    RMSE = sqrt((100 + 100 + 900) / 3) = sqrt(1100 / 3) = sqrt(366.666...) = 19.1485...
    WAPE = 50 / 600 = 0.08333... (8.33%)
    Bias = (+10 - 10 + 30) / 600 = 30 / 600 = +0.05 (+5%)
    Mean Error = 30 / 3 = +10.0
    """
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 330.0]

    assert np.isclose(calculate_mae(y_true, y_pred), 50.0 / 3.0)
    assert np.isclose(calculate_rmse(y_true, y_pred), np.sqrt(1100.0 / 3.0))
    assert np.isclose(calculate_wape(y_true, y_pred), 50.0 / 600.0)
    assert np.isclose(calculate_bias(y_true, y_pred), 30.0 / 600.0)
    assert np.isclose(calculate_mean_error(y_true, y_pred), 10.0)


def test_zero_denominator_handling():
    """Verify explicit zero denominator handling in WAPE and Bias."""
    # When both actuals and predictions are all zeros:
    y_zero = [0.0, 0.0, 0.0]
    y_pred_zero = [0.0, 0.0, 0.0]

    assert calculate_wape(y_zero, y_pred_zero) == 0.0
    assert calculate_bias(y_zero, y_pred_zero) == 0.0

    # When actuals are all zeros but predictions are non-zero:
    y_pred_nonzero = [10.0, 20.0, 30.0]
    assert np.isnan(calculate_wape(y_zero, y_pred_nonzero))
    assert np.isnan(calculate_bias(y_zero, y_pred_nonzero))


def test_coverage_metric():
    """Verify empirical interval coverage calculation."""
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_lower = [5.0, 15.0, 35.0, 30.0]  # item 3 (30.0) is below lower bound 35.0 -> missed
    y_upper = [15.0, 25.0, 45.0, 50.0]

    # Covered items: 0, 1, 3 (3 out of 4) -> 0.75
    cov = calculate_coverage(y_true, y_lower, y_upper)
    assert cov == 0.75

    # If lower > upper, must raise ValueError
    with pytest.raises(ValueError, match="Lower bound exceeds upper bound"):
        calculate_coverage([10.0], [20.0], [15.0])


def test_invalid_metric_inputs():
    """Verify mismatched lengths, NaNs, and empty inputs raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_mae([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="empty input"):
        calculate_mae([], [])

    with pytest.raises(ValueError, match="NaN"):
        calculate_mae([1.0, np.nan], [1.0, 2.0])
