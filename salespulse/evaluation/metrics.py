"""
SalesPulse AI - Rigorous Forecasting Evaluation Metrics

Provides documented, reproducible implementations of:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- Signed Net Forecast Bias (Percentage and Absolute Dollar ME)
- Empirical Prediction Interval Coverage
"""

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd


def _to_numpy_arrays(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D float64 numpy arrays."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} elements, "
            f"y_pred has {len(y_p)} elements."
        )

    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on empty input arrays.")

    if np.isnan(y_t).any() or np.isnan(y_p).any():
        raise ValueError("Inputs contain NaN values.")

    if np.isinf(y_t).any() or np.isinf(y_p).any():
        raise ValueError("Inputs contain infinite values.")

    return y_t, y_p


def calculate_mae(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Mean Absolute Error (MAE):
    Formula: (1 / N) * sum(|y_i - y_hat_i|)
    """
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_rmse(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Root Mean Squared Error (RMSE):
    Formula: sqrt((1 / N) * sum((y_i - y_hat_i)^2))
    """
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def calculate_wape(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Weighted Absolute Percentage Error (WAPE):
    Formula: sum(|y_i - y_hat_i|) / sum(|y_i|)

    Zero-denominator handling:
    If sum(|y_i|) == 0:
        returns 0.0 if sum(|y_i - y_hat_i|) == 0 (perfect zero forecast),
        otherwise returns float('nan').
    """
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    sum_abs_err = np.sum(np.abs(y_t - y_p))
    sum_abs_actual = np.sum(np.abs(y_t))

    if sum_abs_actual == 0.0:
        return 0.0 if sum_abs_err == 0.0 else float("nan")

    return float(sum_abs_err / sum_abs_actual)


def calculate_bias(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Signed Net Forecast Percentage Bias:
    Formula: sum(y_hat_i - y_i) / sum(y_i)
    Positive value indicates systematic over-forecasting.
    Negative value indicates systematic under-forecasting.

    Zero-denominator handling:
    If sum(y_i) == 0:
        returns 0.0 if sum(y_hat_i - y_i) == 0,
        otherwise returns float('nan').
    """
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    net_error = np.sum(y_p - y_t)
    sum_actual = np.sum(y_t)

    if sum_actual == 0.0:
        return 0.0 if net_error == 0.0 else float("nan")

    return float(net_error / sum_actual)


def calculate_mean_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Signed Mean Error (Dollar Bias):
    Formula: (1 / N) * sum(y_hat_i - y_i)
    Positive indicates average dollar over-forecast.
    Negative indicates average dollar under-forecast.
    """
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    return float(np.mean(y_p - y_t))


def calculate_coverage(
    y_true: Union[np.ndarray, pd.Series, list],
    y_lower: Union[np.ndarray, pd.Series, list],
    y_upper: Union[np.ndarray, pd.Series, list],
) -> float:
    """Empirical Prediction Interval Coverage:
    Formula: (1 / N) * sum(1 if y_lower_i <= y_i <= y_upper_i else 0)
    """
    y_t, y_l = _to_numpy_arrays(y_true, y_lower)
    _, y_u = _to_numpy_arrays(y_true, y_upper)

    if (y_l > y_u).any():
        raise ValueError("Lower bound exceeds upper bound in interval predictions.")

    covered = (y_t >= y_l) & (y_t <= y_u)
    return float(np.mean(covered))


def compute_all_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_lower: Optional[Union[np.ndarray, pd.Series, list]] = None,
    y_upper: Optional[Union[np.ndarray, pd.Series, list]] = None,
) -> Dict[str, Any]:
    """Calculate complete forecasting metric suite."""
    y_t, y_p = _to_numpy_arrays(y_true, y_pred)
    mae = calculate_mae(y_t, y_p)
    rmse = calculate_rmse(y_t, y_p)
    wape = calculate_wape(y_t, y_p)
    bias = calculate_bias(y_t, y_p)
    mean_error = calculate_mean_error(y_t, y_p)

    metrics = {
        "count": len(y_t),
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "bias_pct": bias,
        "mean_error": mean_error,
    }

    if y_lower is not None and y_upper is not None:
        metrics["coverage"] = calculate_coverage(y_t, y_lower, y_upper)

    return metrics
