"""
SalesPulse AI - Chronological Rolling-Origin Backtesting Engine
Supports true dynamic walk-forward refitting, static model boundary validation,
and dual micro/macro aggregation metrics with transparent overlap tracking.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import datetime
import pandas as pd
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
from salespulse.models.baseline import SeasonalNaiveModel
from salespulse.models.daily_forecaster import DailySalesForecaster, DailyEnsembleForecaster


class RollingOriginBacktester:
    """Chronological rolling-origin evaluation engine."""

    def __init__(
        self,
        daily_df: pd.DataFrame,
        horizons: List[int] = [7, 14, 30],
        min_train_days: int = 730,  # 2 years of history for initial training
        step_days: int = 14,  # Move origin forward by 14 days
        eval_window_days: Optional[int] = 180,  # Evaluate over last 180 days
    ):
        if "date" not in daily_df.columns or "sales" not in daily_df.columns:
            raise KeyError("daily_df must contain 'date' and 'sales' columns.")

        self.df = daily_df.sort_values("date").reset_index(drop=True)
        self.horizons = sorted(horizons)
        self.max_horizon = max(self.horizons)
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.eval_window_days = eval_window_days

        total_days = len(self.df)
        if total_days < self.min_train_days + self.max_horizon:
            raise ValueError(
                f"Insufficient data length: {total_days} days. "
                f"Need at least {self.min_train_days + self.max_horizon} days."
            )

    def generate_origins(self) -> List[int]:
        """Generate integer indices for forecast origins."""
        total_days = len(self.df)
        max_origin = total_days - self.max_horizon - 1

        if self.eval_window_days is not None:
            earliest_origin = max(self.min_train_days, total_days - self.eval_window_days - self.max_horizon)
        else:
            earliest_origin = self.min_train_days

        origins = list(range(earliest_origin, max_origin + 1, self.step_days))
        return origins

    def validate_static_model_boundary(
        self,
        model_name: str,
        model: Any,
        earliest_origin_idx: int,
    ) -> None:
        """Verify that a static pre-fitted model was not trained on data overlapping the backtest origins."""
        cutoff_date = getattr(model, "training_end_date", None)
        cutoff_idx = getattr(model, "training_cutoff_idx", None)

        if cutoff_idx is not None and cutoff_idx >= earliest_origin_idx:
            earliest_date = self.df.loc[earliest_origin_idx, "date"]
            raise ValueError(
                f"LEAKAGE ERROR for static model '{model_name}': Model training cutoff index ({cutoff_idx}) "
                f"is at or after earliest backtest origin index ({earliest_origin_idx}, date {earliest_date.date()}). "
                f"Static models must only be evaluated strictly after their training window."
            )

        if cutoff_date is not None:
            cutoff_dt = pd.Timestamp(cutoff_date)
            earliest_dt = pd.Timestamp(self.df.loc[earliest_origin_idx, "date"])
            if cutoff_dt >= earliest_dt:
                raise ValueError(
                    f"LEAKAGE ERROR for static model '{model_name}': Model training cutoff date ({cutoff_dt.date()}) "
                    f"is at or after earliest backtest origin date ({earliest_dt.date()})."
                )

    def run_backtest(
        self,
        models_or_factories: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute rolling-origin backtest with support for dynamic refitting and static boundary checks.

        Args:
            models_or_factories: Dict mapping model name to either:
                - A callable factory: `fn(history_df: pd.DataFrame) -> model` (true dynamic walk-forward refitting)
                - A model instance: `SeasonalNaiveModel`, `DailySalesForecaster`, etc. (validated for boundaries)

        Returns:
            Dict containing detailed fold records, micro metrics, macro metrics, and overlap statistics.
        """
        origins = self.generate_origins()
        if not origins:
            raise ValueError("No valid forecast origins could be generated with current configuration.")

        earliest_origin = origins[0]

        # Validate static model boundaries before running
        for m_name, item in models_or_factories.items():
            if not callable(item):
                self.validate_static_model_boundary(m_name, item, earliest_origin)

        # Structure to collect predictions and fold-level metrics
        fold_results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
            m: {h: [] for h in self.horizons} for m in models_or_factories.keys()
        }
        pooled_results: Dict[str, Dict[int, Dict[str, List[float]]]] = {
            m: {h: {"actuals": [], "preds": [], "lowers": [], "uppers": []} for h in self.horizons}
            for m in models_or_factories.keys()
        }

        # Track unique target dates per horizon
        unique_dates_per_horizon: Dict[int, set] = {h: set() for h in self.horizons}
        fold_records = []

        for fold_idx, origin_idx in enumerate(origins):
            origin_date = pd.Timestamp(self.df.loc[origin_idx, "date"])
            history_slice = self.df.iloc[: origin_idx + 1].copy()
            history_sales = history_slice["sales"].tolist()

            target_slice = self.df.iloc[origin_idx + 1 : origin_idx + 1 + self.max_horizon]
            target_dates = target_slice["date"].tolist()
            target_sales = target_slice["sales"].tolist()

            for h in self.horizons:
                unique_dates_per_horizon[h].update(target_dates[:h])

            fold_info = {
                "fold": fold_idx + 1,
                "origin_date": str(origin_date.date()),
                "origin_idx": origin_idx,
                "history_length": len(history_slice),
                "forecast_start": str(target_dates[0].date()),
                "forecast_end": str(target_dates[-1].date()),
            }
            fold_records.append(fold_info)

            # Evaluate each model/factory on this exact fold
            for model_name, item in models_or_factories.items():
                if callable(item):
                    # TRUE DYNAMIC WALK-FORWARD REFITTING:
                    # Model is instantiated and trained strictly on history up to origin_idx
                    model_instance = item(history_slice)
                else:
                    model_instance = item

                if isinstance(model_instance, SeasonalNaiveModel):
                    model_instance.fit(history_sales)
                    preds, lowers, uppers = model_instance.predict_with_residuals(self.max_horizon)
                elif isinstance(model_instance, (DailySalesForecaster, DailyEnsembleForecaster)):
                    fc_df = model_instance.forecast_horizon(
                        origin_date=origin_date,
                        history_sales=history_sales,
                        trend_start_idx=float(origin_idx + 1),
                        horizon_days=self.max_horizon,
                    )
                    preds = fc_df["forecast"].to_numpy()
                    lowers = fc_df["lower"].to_numpy()
                    uppers = fc_df["upper"].to_numpy()
                else:
                    raise TypeError(f"Unsupported model type for backtesting: {type(model_instance)}")

                for h in self.horizons:
                    acts_h = target_sales[:h]
                    preds_h = preds[:h]
                    lows_h = lowers[:h]
                    ups_h = uppers[:h]

                    # Record pooled
                    pooled_results[model_name][h]["actuals"].extend(acts_h)
                    pooled_results[model_name][h]["preds"].extend(preds_h)
                    pooled_results[model_name][h]["lowers"].extend(lows_h)
                    pooled_results[model_name][h]["uppers"].extend(ups_h)

                    # Compute fold-level metrics
                    fold_metrics = compute_all_metrics(
                        y_true=acts_h,
                        y_pred=preds_h,
                        y_lower=lows_h,
                        y_upper=ups_h,
                    )
                    fold_metrics["fold"] = fold_idx + 1
                    fold_results[model_name][h].append(fold_metrics)

        # Compute dual Micro and Macro Summary Metrics
        summary_metrics: Dict[str, Dict[str, Any]] = {}
        for model_name in models_or_factories.keys():
            summary_metrics[model_name] = {}
            for h in self.horizons:
                h_key = f"{h}_day"
                # Micro (Pooled)
                acts = pooled_results[model_name][h]["actuals"]
                preds = pooled_results[model_name][h]["preds"]
                lows = pooled_results[model_name][h]["lowers"]
                ups = pooled_results[model_name][h]["uppers"]

                micro_m = compute_all_metrics(
                    y_true=acts,
                    y_pred=preds,
                    y_lower=lows,
                    y_upper=ups,
                )

                # Macro (Mean across folds)
                folds_m = fold_results[model_name][h]
                macro_mae = float(np.mean([f["mae"] for f in folds_m]))
                macro_mae_std = float(np.std([f["mae"] for f in folds_m], ddof=1)) if len(folds_m) > 1 else 0.0
                macro_rmse = float(np.mean([f["rmse"] for f in folds_m]))
                macro_rmse_std = float(np.std([f["rmse"] for f in folds_m], ddof=1)) if len(folds_m) > 1 else 0.0

                valid_wapes = [f["wape"] for f in folds_m if not np.isnan(f["wape"])]
                macro_wape = float(np.mean(valid_wapes)) if valid_wapes else float("nan")
                macro_wape_std = float(np.std(valid_wapes, ddof=1)) if len(valid_wapes) > 1 else 0.0

                valid_biases = [f["bias_pct"] for f in folds_m if not np.isnan(f["bias_pct"])]
                macro_bias = float(np.mean(valid_biases)) if valid_biases else float("nan")
                macro_bias_std = float(np.std(valid_biases, ddof=1)) if len(valid_biases) > 1 else 0.0

                coverages = [f.get("coverage", 0.0) for f in folds_m if "coverage" in f]
                macro_cov = float(np.mean(coverages)) if coverages else None

                # Calculate mean interval width
                interval_widths = [u - l for l, u in zip(lows, ups)]
                mean_interval_width = float(np.mean(interval_widths)) if interval_widths else 0.0

                unique_dates_count = len(unique_dates_per_horizon[h])
                total_evals = len(acts)
                overlap_ratio = float(total_evals / unique_dates_count) if unique_dates_count > 0 else 1.0

                summary_metrics[model_name][h_key] = {
                    "total_evaluations": total_evals,
                    "unique_target_dates": unique_dates_count,
                    "overlap_ratio": overlap_ratio,
                    "mean_interval_width": mean_interval_width,
                    "micro": micro_m,
                    "macro": {
                        "mae_mean": macro_mae,
                        "mae_std": macro_mae_std,
                        "rmse_mean": macro_rmse,
                        "rmse_std": macro_rmse_std,
                        "wape_mean": macro_wape,
                        "wape_std": macro_wape_std,
                        "bias_mean": macro_bias,
                        "bias_std": macro_bias_std,
                        "coverage_mean": macro_cov,
                    },
                }

        return {
            "folds_evaluated": len(origins),
            "origins_count": len(origins),
            "step_days": self.step_days,
            "eval_window_days": self.eval_window_days,
            "fold_details": fold_records,
            "horizons": self.horizons,
            "summary_metrics": summary_metrics,
        }
