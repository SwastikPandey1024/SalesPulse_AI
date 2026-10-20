"""
SalesPulse AI - Daily Sales Multi-Horizon Forecaster
Supports XGBoost, RandomForest, and Weighted Ensemble with recursive multi-step simulation.
"""

from typing import Union, List, Optional, Dict, Any, Tuple
import datetime
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from salespulse.features.temporal import ENGINEERED_FEATURES, compute_step_feature_vector


class DailySalesForecaster:
    """Multi-horizon iterative forecaster using tree-based regressors on daily sales."""

    def __init__(
        self,
        model_type: str = "XGBoost",
        model_instance: Optional[Any] = None,
        residual_std: float = 100.0,
        training_end_date: Optional[Union[str, pd.Timestamp]] = None,
        training_cutoff_idx: Optional[int] = None,
        is_experimental_interval: bool = True,
    ):
        self.model_type = model_type
        self.model = model_instance
        self.residual_std = max(1.0, float(residual_std))
        self.training_end_date = pd.Timestamp(training_end_date) if training_end_date is not None else None
        self.training_cutoff_idx = int(training_cutoff_idx) if training_cutoff_idx is not None else None
        self.is_experimental_interval = is_experimental_interval

    def predict_step(self, feature_df: pd.DataFrame) -> float:
        """Predict a single time step."""
        if self.model is None:
            raise ValueError(f"Model {self.model_type} is not initialized.")
        X = feature_df[ENGINEERED_FEATURES].astype(np.float64)
        pred = float(self.model.predict(X)[0])
        return max(0.0, pred)

    def forecast_horizon(
        self,
        origin_date: pd.Timestamp,
        history_sales: List[float],
        trend_start_idx: float,
        horizon_days: int = 7,
    ) -> pd.DataFrame:
        """Generate multi-step recursive daily forecast.

        Args:
            origin_date: Last date of known history (forecast starts at origin_date + 1 day).
            history_sales: List of historical daily sales up to origin_date (at least 30 values).
            trend_start_idx: Integer trend index corresponding to origin_date + 1.
            horizon_days: Forecast horizon (e.g. 7, 14, 30).

        Returns:
            pd.DataFrame with ['date', 'forecast', 'lower', 'upper'].
        """
        if len(history_sales) < 30:
            raise ValueError(f"Need at least 30 historical days, got {len(history_sales)}")

        dates = []
        forecasts = []
        lowers = []
        uppers = []

        working_history = list(history_sales)
        origin_dt = pd.Timestamp(origin_date)

        for step in range(1, horizon_days + 1):
            target_date = origin_dt + pd.Timedelta(days=step)
            step_trend = trend_start_idx + (step - 1)

            step_features = compute_step_feature_vector(
                target_date=target_date,
                history_sales=working_history,
                trend_idx=step_trend,
            )

            pred_val = self.predict_step(step_features)
            dates.append(target_date)
            forecasts.append(pred_val)
            working_history.append(pred_val)

            # Uncertainty expands with horizon step
            interval_half_width = 1.96 * self.residual_std * np.sqrt(1.0 + (step - 1) / 7.0)
            lowers.append(max(0.0, pred_val - interval_half_width))
            uppers.append(pred_val + interval_half_width)

        return pd.DataFrame(
            {
                "date": dates,
                "forecast": forecasts,
                "lower": lowers,
                "upper": uppers,
            }
        )


class DailyEnsembleForecaster:
    """Weighted ensemble combining XGBoost and RandomForest daily forecasters."""

    def __init__(
        self,
        xgb_forecaster: DailySalesForecaster,
        rf_forecaster: DailySalesForecaster,
        xgb_weight: float = 0.5,
    ):
        self.xgb_forecaster = xgb_forecaster
        self.rf_forecaster = rf_forecaster
        self.xgb_weight = xgb_weight
        self.rf_weight = 1.0 - xgb_weight
        self.residual_std = (
            self.xgb_weight * xgb_forecaster.residual_std
            + self.rf_weight * rf_forecaster.residual_std
        )

    def forecast_horizon(
        self,
        origin_date: pd.Timestamp,
        history_sales: List[float],
        trend_start_idx: float,
        horizon_days: int = 7,
    ) -> pd.DataFrame:
        """Generate blended multi-step recursive forecast."""
        dates = []
        forecasts = []
        lowers = []
        uppers = []

        working_history = list(history_sales)
        origin_dt = pd.Timestamp(origin_date)

        for step in range(1, horizon_days + 1):
            target_date = origin_dt + pd.Timedelta(days=step)
            step_trend = trend_start_idx + (step - 1)

            step_features = compute_step_feature_vector(
                target_date=target_date,
                history_sales=working_history,
                trend_idx=step_trend,
            )

            p_xgb = self.xgb_forecaster.predict_step(step_features)
            p_rf = self.rf_forecaster.predict_step(step_features)
            pred_val = max(0.0, (self.xgb_weight * p_xgb) + (self.rf_weight * p_rf))

            dates.append(target_date)
            forecasts.append(pred_val)
            working_history.append(pred_val)

            interval_half_width = 1.96 * self.residual_std * np.sqrt(1.0 + (step - 1) / 7.0)
            lowers.append(max(0.0, pred_val - interval_half_width))
            uppers.append(pred_val + interval_half_width)

        return pd.DataFrame(
            {
                "date": dates,
                "forecast": forecasts,
                "lower": lowers,
                "upper": uppers,
            }
        )
