"""
SalesPulse AI - Seasonal-Naive Benchmark Model
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd


class SeasonalNaiveModel:
    """Seasonal-Naive time-series baseline.

    For daily sales with weekly seasonality (seasonality_period=7),
    the forecast for horizon step h from origin t is the actual sales
    observed on the corresponding day-of-week in the most recent cycle:
        y_hat_{t+h} = y_{t - 7 + ((h - 1) % 7) + 1}
    """

    def __init__(self, seasonality_period: int = 7):
        if seasonality_period < 1:
            raise ValueError("seasonality_period must be >= 1.")
        self.seasonality_period = seasonality_period
        self.history_: Optional[np.ndarray] = None

    def fit(self, y: Union[np.ndarray, pd.Series, List[float]]) -> "SeasonalNaiveModel":
        """Fit baseline on historical target series.

        Requires at least seasonality_period observations.
        """
        arr = np.asarray(y, dtype=np.float64).ravel()
        if len(arr) < self.seasonality_period:
            raise ValueError(
                f"SeasonalNaiveModel requires at least {self.seasonality_period} "
                f"observations, got {len(arr)}."
            )
        self.history_ = arr.copy()
        return self

    def predict(self, horizon_days: int = 7) -> np.ndarray:
        """Forecast next horizon_days steps."""
        if self.history_ is None or len(self.history_) < self.seasonality_period:
            raise ValueError("Model is not fitted. Call fit() first with sufficient history.")

        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1.")

        m = self.seasonality_period
        recent_cycle = self.history_[-m:]  # Last m observations: [-m, ..., -1]

        predictions = []
        for h in range(1, horizon_days + 1):
            # Index into recent_cycle:
            # For h=1 -> (1-1)%m = 0 -> recent_cycle[0] (which is history[-m])
            # For h=m -> (m-1)%m = m-1 -> recent_cycle[m-1] (history[-1])
            idx = (h - 1) % m
            predictions.append(recent_cycle[idx])

        return np.asarray(predictions, dtype=np.float64)

    def predict_with_residuals(
        self, horizon_days: int = 7
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate point forecasts along with empirical residual standard deviation intervals.

        Computes residuals e_t = y_t - y_{t-m} on historical series to derive uncertainty bounds.
        """
        preds = self.predict(horizon_days=horizon_days)

        # Residuals on historical fitted series
        m = self.seasonality_period
        residuals = self.history_[m:] - self.history_[:-m]
        res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 10.0

        # Multi-step variance scaling: sqrt(1 + (h-1)//m)
        lowers = []
        uppers = []
        for h, p in enumerate(preds, 1):
            scaling = np.sqrt(1.0 + (h - 1) // m)
            half_width = 1.96 * res_std * scaling
            lowers.append(max(0.0, p - half_width))
            uppers.append(p + half_width)

        return preds, np.asarray(lowers, dtype=np.float64), np.asarray(uppers, dtype=np.float64)
