"""
SalesPulse AI - Shared Forecasting Domain Service
Provides:
- Framework-agnostic single-step inference and dynamic business insights.
- Multi-horizon recursive forecasting with exact numerical parity to the legacy dashboard.
- Explicit interval methodology: 'heuristic_sqrt_step' (dashboard default) vs 'calibrated_oof_residual'.
- Optional audit persistence via AuditRepository with zero PII retention.
- Strict non-fallback error handling: fails explicitly if requested model is unavailable.
"""

import uuid
import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np

from salespulse.features.temporal import ENGINEERED_FEATURES
from salespulse.services.model_registry import ModelRegistry, get_model_registry
from salespulse.storage.repository import AuditRepository


def extract_date_features(date_input: Union[datetime.date, datetime.datetime, str, pd.Timestamp]) -> Dict[str, int]:
    """Extract required calendar components from date input."""
    dt = pd.Timestamp(date_input)
    return {
        "day": int(dt.day),
        "month": int(dt.month),
        "weekday": int(dt.weekday()),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "weekofyear": int(dt.isocalendar().week),
    }


def validate_numeric_scalar(value: Any, name: str, min_val: Optional[float] = 0.0) -> float:
    """Validate a numeric scalar against NaN, Infinity, and minimum bounds."""
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Feature '{name}' must be numeric, got {value}")
    if np.isnan(val) or np.isinf(val):
        raise ValueError(f"Feature '{name}' contains non-finite NaN or Infinity: {val}")
    if min_val is not None and val < min_val:
        raise ValueError(f"Feature '{name}' must be >= {min_val}, got {val}")
    return val


def build_feature_dataframe(
    date_features: Optional[Dict[str, int]] = None,
    lag_1: float = 0.0,
    lag_7: float = 0.0,
    lag_14: float = 0.0,
    lag_30: float = 0.0,
    rolling_mean_7: float = 0.0,
    rolling_mean_14: float = 0.0,
    rolling_std_7: float = 0.0,
    trend: float = 100.0,
    **kwargs,
) -> pd.DataFrame:
    """Construct and strictly validate 13-feature DataFrame conforming to ENGINEERED_FEATURES contract."""
    d_feats = date_features if date_features is not None else kwargs.get("date_feats", {})
    data = {
        "day": validate_numeric_scalar(d_feats["day"], "day", min_val=1.0),
        "month": validate_numeric_scalar(d_feats["month"], "month", min_val=1.0),
        "weekday": validate_numeric_scalar(d_feats["weekday"], "weekday", min_val=0.0),
        "is_weekend": validate_numeric_scalar(d_feats["is_weekend"], "is_weekend", min_val=0.0),
        "weekofyear": validate_numeric_scalar(d_feats["weekofyear"], "weekofyear", min_val=1.0),
        "lag_1": validate_numeric_scalar(lag_1, "lag_1", min_val=0.0),
        "lag_7": validate_numeric_scalar(lag_7, "lag_7", min_val=0.0),
        "lag_14": validate_numeric_scalar(lag_14, "lag_14", min_val=0.0),
        "lag_30": validate_numeric_scalar(lag_30, "lag_30", min_val=0.0),
        "rolling_mean_7": validate_numeric_scalar(rolling_mean_7, "rolling_mean_7", min_val=0.0),
        "rolling_mean_14": validate_numeric_scalar(rolling_mean_14, "rolling_mean_14", min_val=0.0),
        "rolling_std_7": validate_numeric_scalar(rolling_std_7, "rolling_std_7", min_val=0.0),
        "trend": validate_numeric_scalar(trend, "trend", min_val=1.0),
    }
    df = pd.DataFrame([data])[list(ENGINEERED_FEATURES)].astype(np.float64)
    return df


class ForecastingService:
    """Enterprise domain service for sales prediction, multi-horizon rollouts, and audit persistence."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        repository: Optional[AuditRepository] = None,
    ):
        self.registry = registry or get_model_registry()
        self.repository = repository or AuditRepository()

    # ------------------------------------------------------------------------
    # 1. Single-Step Prediction
    # ------------------------------------------------------------------------
    def predict_single_step(
        self,
        input_df: pd.DataFrame,
        model_choice: str = "XGBoost",
    ) -> float:
        """Execute single-step inference with model verification and non-negative clipping.
        
        Strict error handling: Fails if requested model is unavailable; NO silent fallback.
        """
        # Validate exact 13-feature vector
        X = input_df[list(ENGINEERED_FEATURES)].astype(np.float64)

        if model_choice == "Ensemble":
            xgb_m = self.registry.get_model("XGBoost")
            rf_m = self.registry.get_model("RandomForest")
            p_xgb = float(xgb_m.predict(X)[0])
            p_rf = float(rf_m.predict(X)[0])
            pred = (p_xgb + p_rf) / 2.0
            return max(0.0, pred)

        target_model = self.registry.get_model(model_choice)
        pred = float(target_model.predict(X)[0])
        return max(0.0, pred)

    # ------------------------------------------------------------------------
    # 2. Dynamic Business Insights Generator
    # ------------------------------------------------------------------------
    def generate_business_insights(
        self,
        predicted_val: float,
        lag_1: float,
        rm_7: float,
        rm_14: float,
    ) -> List[Dict[str, str]]:
        """Generate structured business recommendations matching legacy dashboard logic."""
        insights = []
        diff_lag1 = ((predicted_val - lag_1) / lag_1) * 100 if lag_1 > 0 else 0.0
        diff_rm7 = predicted_val - rm_7

        if diff_lag1 >= 5.0:
            insights.append({
                "icon": "📈",
                "title": "Strong Growth Trajectory Detected",
                "desc": f"Sales predicted to increase by +{diff_lag1:.1f}% compared to yesterday (${lag_1:,.2f}).",
                "action": "Recommendation: Ensure sufficient inventory & schedule additional store staff for demand surges.",
            })
        elif diff_lag1 <= -5.0:
            insights.append({
                "icon": "📉",
                "title": "Sales Dip Anticipated",
                "desc": f"Sales projected to contract by {diff_lag1:.1f}% below yesterday's benchmark.",
                "action": "Recommendation: Trigger target flash promotions or offer bundle discounts to stimulate volume.",
            })
        else:
            insights.append({
                "icon": "➡️",
                "title": "Steady & Stable Demand",
                "desc": "Predicted sales remain within a stable ±5% corridor of recent daily totals.",
                "action": "Recommendation: Maintain current marketing and operational allocation.",
            })

        if diff_rm7 > 0:
            insights.append({
                "icon": "⬆️",
                "title": "Outperforming 7-Day Moving Average",
                "desc": f"Predicted value is +${diff_rm7:,.2f} above the 7-day rolling mean (${rm_7:,.2f}).",
                "action": "Momentum Indicator: Strong baseline performance.",
            })
        else:
            insights.append({
                "icon": "⬇️",
                "title": "Below 7-Day Baseline",
                "desc": f"Predicted value is -${abs(diff_rm7):,.2f} lower than the 7-day rolling mean (${rm_7:,.2f}).",
                "action": "Focus Indicator: Investigate potential localized bottlenecks or seasonality.",
            })

        return insights

    # ------------------------------------------------------------------------
    # 3. Multi-Horizon Recursive Forecasting Engine
    # ------------------------------------------------------------------------
    def forecast_multi_horizon(
        self,
        initial_df: pd.DataFrame,
        horizon_days: int = 7,
        model_choice: str = "XGBoost",
        origin_date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
        interval_method: str = "heuristic_sqrt_step",
        residual_std: Optional[float] = None,
        persist_run: bool = False,
        run_id: Optional[str] = None,
        execution_source: str = "standalone_service",
        sales_scope: str = "TOTAL_STORE",
    ) -> pd.DataFrame:
        """Generate multi-day recursive forecast with empirical or calibrated prediction intervals.
        
        Guarantees exact numerical parity with legacy dashboard under default interval_method='heuristic_sqrt_step'.
        """
        if horizon_days not in (7, 14, 30) and horizon_days < 1:
            raise ValueError(f"horizon_days must be positive integer, got {horizon_days}")

        if origin_date is not None:
            if isinstance(origin_date, str):
                base_dt = datetime.datetime.fromisoformat(origin_date)
            elif isinstance(origin_date, datetime.date) and not isinstance(origin_date, datetime.datetime):
                base_dt = datetime.datetime.combine(origin_date, datetime.time.min)
            else:
                base_dt = origin_date
        else:
            base_dt = datetime.datetime.now()

        dates = []
        preds = []
        lowers = []
        uppers = []

        curr_df = initial_df.copy()
        history = [float(curr_df["lag_1"].iloc[0])]
        std_7 = float(curr_df["rolling_std_7"].iloc[0])
        trend_base = float(curr_df["trend"].iloc[0])

        for i in range(1, horizon_days + 1):
            target_date = base_dt + datetime.timedelta(days=i)
            dates.append(target_date)

            date_feats = extract_date_features(target_date)
            lag1_new = history[-1]
            lag7_new = history[-7] if len(history) >= 7 else history[0]
            lag14_new = history[-14] if len(history) >= 14 else history[0]
            lag30_new = history[-30] if len(history) >= 30 else history[0]

            rm7_new = float(np.mean(history[-7:]))
            rm14_new = float(np.mean(history[-14:])) if len(history) >= 14 else rm7_new
            rs7_new = float(np.std(history[-7:])) if len(history) >= 7 else std_7
            trend_new = trend_base + i

            step_df = build_feature_dataframe(
                date_features=date_feats,
                lag_1=lag1_new,
                lag_7=lag7_new,
                lag_14=lag14_new,
                lag_30=lag30_new,
                rolling_mean_7=rm7_new,
                rolling_mean_14=rm14_new,
                rolling_std_7=rs7_new,
                trend=trend_new,
            )

            val = self.predict_single_step(step_df, model_choice=model_choice)
            preds.append(val)
            history.append(val)

            # Uncertainty Interval calculation
            if interval_method == "heuristic_sqrt_step":
                # Matches legacy dashboard formula exactly: 1.96 * max(10.0, rs7_new) * sqrt(i / 2.0)
                uncertainty = 1.96 * max(10.0, rs7_new) * np.sqrt(i / 2.0)
            elif interval_method == "calibrated_oof_residual":
                # Matches Phase 1 out-of-fold calibrated scaling: 1.96 * residual_std * sqrt(1 + (i-1)/7.0)
                sig = float(residual_std) if residual_std is not None else max(10.0, rs7_new)
                uncertainty = 1.96 * sig * np.sqrt(1.0 + (i - 1) / 7.0)
            else:
                raise ValueError(
                    f"Unknown interval_method '{interval_method}'. Choose 'heuristic_sqrt_step' or 'calibrated_oof_residual'."
                )

            lowers.append(max(0.0, val - uncertainty))
            uppers.append(val + uncertainty)

        result_df = pd.DataFrame({
            "Date": dates,
            "Forecasted Sales ($)": preds,
            "Lower Bound ($)": lowers,
            "Upper Bound ($)": uppers,
        })

        # Optional persistence to audit storage
        if persist_run:
            active_run_id = run_id or str(uuid.uuid4())
            try:
                meta = self.registry.get_metadata(model_choice)
                art_path = meta.get("artifact_filename")
                art_sha = meta.get("artifact_sha256")
            except Exception:
                art_path = None
                art_sha = None

            items_to_log = []
            for idx, row in enumerate(result_df.itertuples(), start=1):
                item_date = row.Date.date() if isinstance(row.Date, (pd.Timestamp, datetime.datetime)) else row.Date
                items_to_log.append({
                    "target_date": item_date,
                    "step_ahead": idx,
                    "predicted_value": float(row._2),
                    "lower_bound": float(row._3),
                    "upper_bound": float(row._4),
                    "sales_scope": sales_scope,
                })

            self.repository.record_forecast_run(
                run_id=active_run_id,
                model_name=model_choice,
                origin_date=base_dt,
                horizon_days=horizon_days,
                items=items_to_log,
                model_artifact_path=art_path,
                model_artifact_hash=art_sha,
                feature_contract_version="v1_13features",
                interval_method=interval_method,
                execution_source=execution_source,
                status="SUCCESS",
                sales_scope=sales_scope,
            )

        return result_df
