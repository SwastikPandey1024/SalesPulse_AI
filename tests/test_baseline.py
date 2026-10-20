"""
Unit tests for Seasonal-Naive Baseline Model
"""

import pytest
import numpy as np

from salespulse.models.baseline import SeasonalNaiveModel


def test_seasonal_naive_weekly_cycle():
    """Verify seasonal naive forecasts repeat the corresponding day from the previous week."""
    # Days Mon(10), Tue(20), Wed(30), Thu(40), Fri(50), Sat(60), Sun(70)
    history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    model = SeasonalNaiveModel(seasonality_period=7)
    model.fit(history)

    # 7-day forecast must equal history exactly
    preds_7 = model.predict(horizon_days=7)
    assert np.array_equal(preds_7, np.array(history))

    # 14-day forecast must tile the 7-day pattern twice
    preds_14 = model.predict(horizon_days=14)
    expected_14 = np.array(history + history)
    assert np.array_equal(preds_14, expected_14)


def test_seasonal_naive_insufficient_history():
    """Verify ValueError when history has fewer than seasonality_period observations."""
    model = SeasonalNaiveModel(seasonality_period=7)
    with pytest.raises(ValueError, match="requires at least 7 observations"):
        model.fit([1.0, 2.0, 3.0])


def test_seasonal_naive_unfitted():
    """Verify ValueError when predict is called before fit."""
    model = SeasonalNaiveModel(seasonality_period=7)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(7)


def test_seasonal_naive_residuals_and_intervals():
    """Verify residual-based uncertainty intervals have valid shapes and bounds."""
    # 21 days of history (3 full cycles)
    history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 3
    model = SeasonalNaiveModel(seasonality_period=7)
    model.fit(history)

    preds, lowers, uppers = model.predict_with_residuals(horizon_days=14)

    assert len(preds) == 14
    assert len(lowers) == 14
    assert len(uppers) == 14
    assert (lowers <= preds).all()
    assert (preds <= uppers).all()
    assert (lowers >= 0.0).all()  # Non-negative lower bound
