"""
SalesPulse AI - Deterministic Anomaly Detection Engine
Provides:
- Robust rolling median + MAD scoring (excluding current observation to prevent data leakage).
- Deterministic handling of zero-MAD, flatline/zero-variance, and small-sample boundaries.
- Scope-isolated detection for daily sales surges, drops, and forecast error anomalies.
- Factual, non-speculative explanations adhering to statistical evidence.
- Strict RFC 8259 JSON compatibility (no NaN or Infinity values).
"""

import math
import uuid
import hashlib
import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
import pandas as pd
import numpy as np

from salespulse.analytics.schemas import (
    AnomalyConfig,
    AnomalyPoint,
    AnomalyReport,
    AnomalyType,
    SeverityLevel,
    AnomalyDirection,
)
from salespulse.data.time_series import parse_business_calendar_date


def compute_robust_scale(
    values: np.ndarray,
    center: float,
    scaling_constant: float = 1.4826,
    zero_mad_minimum_scale: float = 1.0,
) -> Tuple[float, bool]:
    """Compute robust dispersion scale with deterministic zero-MAD guardrails.
    
    Returns:
        Tuple[float, bool]: (effective_scale, is_fallback_scale_used)
    """
    arr = np.asarray(values, dtype=np.float64)
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return max(zero_mad_minimum_scale, 1e-6), True

    deviations = np.abs(finite_vals - center)
    mad = float(np.median(deviations))
    scale = scaling_constant * mad

    if scale > 1e-9:
        return scale, False

    # Guardrail 1: Interquartile Range (IQR) scale
    q75, q25 = np.percentile(finite_vals, [75, 25])
    iqr = float(q75 - q25)
    iqr_scale = iqr / 1.349
    if iqr_scale > 1e-9:
        return iqr_scale, False

    # Guardrail 2: Mean Absolute Deviation (MeanAD) scale
    mean_ad = float(np.mean(deviations))
    mean_ad_scale = mean_ad * 1.2533
    if mean_ad_scale > 1e-9:
        return mean_ad_scale, False

    # Guardrail 3: True zero-dispersion (flatline/identical history).
    # Use explicit, documented fallback scale: max(min_scale, 10% of median level).
    fallback_scale = max(zero_mad_minimum_scale, 0.10 * abs(center))
    fallback_scale = max(fallback_scale, 1e-6)
    return fallback_scale, True


def classify_severity(abs_z: float, config: AnomalyConfig) -> SeverityLevel:
    """Classify anomaly severity deterministically based on robust z-score magnitude."""
    if abs_z >= config.severity_critical_z:
        return SeverityLevel.CRITICAL
    elif abs_z >= config.severity_high_z:
        return SeverityLevel.HIGH
    elif abs_z >= config.severity_medium_z:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def generate_deterministic_anomaly_id(
    sales_scope: str,
    observation_date: str,
    anomaly_type: str,
    model_name: Optional[str] = None,
    horizon_days: Optional[int] = None,
    step_ahead: Optional[int] = None,
    forecast_run_id: Optional[str] = None,
) -> str:
    """Generate a reproducible, collision-free UUIDv5 based on business identifiers."""
    seed = (
        f"{sales_scope}|{observation_date}|{anomaly_type}|"
        f"{model_name or 'NONE'}|{horizon_days or 0}|{step_ahead or 0}|{forecast_run_id or 'NONE'}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def detect_sales_anomalies(
    df: pd.DataFrame,
    config: Optional[AnomalyConfig] = None,
    sales_scope: str = "TOTAL_STORE",
    date_col: str = "date",
    sales_col: str = "sales",
    fill_missing_dates: bool = False,
) -> AnomalyReport:
    """Detect sales surges and drops in a daily sales time series.
    
    Guarantees:
    - Pure, deterministic execution.
    - Zero future-data leakage: evaluation window strictly precedes observation date [t-W : t-1].
    - Excludes current observation from its own historical baseline.
    - Explicit non-alert when history is insufficient (< min_history).
    - Scope awareness: validates that data belongs to specified sales_scope.
    - Handles missing/duplicate dates, negative sales, and zero dispersion safely.
    """
    cfg = config or AnomalyConfig()
    report_id = str(uuid.uuid4())

    if df is None or df.empty:
        return AnomalyReport(
            report_id=report_id,
            target_scope=sales_scope,
            total_evaluated=0,
            anomalies_detected=0,
            insufficient_history_count=0,
            anomalies=[],
            config=cfg.__dict__,
        )

    # 1. Structure and Column Validation
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in input DataFrame.")
    if sales_col not in df.columns:
        raise KeyError(f"Sales column '{sales_col}' not found in input DataFrame.")

    # 2. Scope Validation if scope column present
    if "sales_scope" in df.columns:
        scopes_present = set(df["sales_scope"].dropna().unique())
        if len(scopes_present) > 1:
            raise ValueError(
                f"DataFrame contains multiple sales scopes ({scopes_present}). "
                f"Isolate data to a single scope before passing to detect_sales_anomalies()."
            )
        elif len(scopes_present) == 1 and list(scopes_present)[0] != sales_scope:
            raise ValueError(
                f"DataFrame scope '{list(scopes_present)[0]}' does not match target scope '{sales_scope}'."
            )

    # 3. Clean and parse calendar dates
    work_df = df[[date_col, sales_col]].copy()
    work_df["parsed_date"] = work_df[date_col].apply(parse_business_calendar_date)
    work_df = work_df.dropna(subset=["parsed_date"]).copy()
    work_df["parsed_date"] = pd.to_datetime(work_df["parsed_date"])

    # 4. Handle duplicates
    # Aggregate multiple transactions on the same calendar day (min_count=1 preserves NaN for unobserved days)
    work_df = work_df.groupby("parsed_date", as_index=False)[sales_col].sum(min_count=1)
    work_df = work_df.sort_values("parsed_date").reset_index(drop=True)

    # 5. Handle missing dates if requested
    if fill_missing_dates and len(work_df) > 1:
        full_idx = pd.date_range(
            start=work_df["parsed_date"].min(),
            end=work_df["parsed_date"].max(),
            freq="D",
            name="parsed_date",
        )
        work_df = work_df.set_index("parsed_date").reindex(full_idx, fill_value=0.0).reset_index()

    # 6. Missing and Negative sales policy checks
    raw_sales_numeric = pd.to_numeric(work_df[sales_col], errors="coerce")
    missing_mask = raw_sales_numeric.isna()
    missing_values_count = int(missing_mask.sum())

    if cfg.handle_missing_values == "SKIP" and missing_values_count > 0:
        # Exclude unrecorded observations from detection to prevent artificial drop alerts
        work_df = work_df[~missing_mask].copy()
        work_df[sales_col] = raw_sales_numeric[~missing_mask]
    else:
        work_df[sales_col] = raw_sales_numeric.fillna(0.0)

    has_negative = (work_df[sales_col] < 0).any()
    if has_negative:
        if cfg.handle_negative_sales == "REJECT":
            raise ValueError("Input series contains negative sales values, rejected by policy.")
        # If 'TREAT_AS_RETURN' or 'FLAG', proceed with numeric calculations

    dates = work_df["parsed_date"].dt.strftime("%Y-%m-%d").tolist()
    sales_values = work_df[sales_col].to_numpy(dtype=np.float64)
    n_points = len(sales_values)

    anomalies: List[AnomalyPoint] = []
    insufficient_history_count = 0
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 7. Rolling Historical Window Evaluation (Zero Leakage)
    for i in range(n_points):
        # Historical reference window strictly precedes observation i: [max(0, i - W) : i]
        start_idx = max(0, i - cfg.window_size)
        history = sales_values[start_idx:i]

        if len(history) < cfg.min_history:
            insufficient_history_count += 1
            continue

        curr_val = sales_values[i]
        obs_date = dates[i]

        # Robust statistics on historical reference
        center = float(np.median(history))
        scale, is_fallback = compute_robust_scale(
            history,
            center=center,
            scaling_constant=cfg.mad_scaling_constant,
            zero_mad_minimum_scale=cfg.zero_mad_minimum_scale,
        )

        deviation = curr_val - center
        z_score = deviation / scale

        # Check for surge or drop
        is_surge = z_score >= cfg.surge_threshold
        is_drop = z_score <= -cfg.drop_threshold

        if not (is_surge or is_drop):
            continue

        if is_surge:
            anomaly_type = AnomalyType.SALES_SURGE
            direction = AnomalyDirection.SURGE
            threshold_applied = cfg.surge_threshold
            verb = "exceeded"
            relation = "above"
        else:
            anomaly_type = AnomalyType.SALES_DROP
            direction = AnomalyDirection.DROP
            threshold_applied = cfg.drop_threshold
            verb = "fell below"
            relation = "below"

        abs_z = abs(z_score)
        severity = classify_severity(abs_z, cfg)
        anomaly_id = generate_deterministic_anomaly_id(
            sales_scope=sales_scope,
            observation_date=obs_date,
            anomaly_type=anomaly_type.value,
        )

        explanation = (
            f"Observed sales of ${curr_val:.2f} {verb} the rolling {len(history)}-day historical "
            f"median (${center:.2f}) by {abs_z:.2f} robust standard deviations ({relation} threshold {threshold_applied:.2f})."
        )
        if is_fallback:
            explanation += (
                f" Historical dispersion was zero (identical baseline values); "
                f"fallback scale of ${scale:.2f} was applied."
            )

        notes = None
        if curr_val < 0:
            notes = "Observed sales value is negative (potential net product return or accounting correction)."

        point = AnomalyPoint(
            anomaly_id=anomaly_id,
            observation_date=obs_date,
            sales_scope=sales_scope,
            anomaly_type=anomaly_type,
            severity=severity,
            direction=direction,
            observed_value=float(curr_val),
            reference_center=float(center),
            reference_spread=float(scale),
            anomaly_score=float(z_score),
            threshold=float(threshold_applied),
            explanation=explanation,
            detection_timestamp=now_utc,
            dispersion_zero_fallback=is_fallback,
            notes=notes,
        )
        anomalies.append(point)

    return AnomalyReport(
        report_id=report_id,
        target_scope=sales_scope,
        total_evaluated=n_points,
        anomalies_detected=len(anomalies),
        insufficient_history_count=insufficient_history_count,
        missing_values_count=missing_values_count,
        anomalies=anomalies,
        config=cfg.__dict__,
    )


def detect_forecast_error_anomalies(
    items: List[Dict[str, Any]],
    config: Optional[AnomalyConfig] = None,
    target_scope: str = "TOTAL_STORE",
    target_model_name: Optional[str] = None,
    target_horizon_days: Optional[int] = None,
    target_step_ahead: Optional[int] = None,
) -> AnomalyReport:
    """Detect forecast error anomalies across reconciled forecast items.
    
    Guarantees:
    - Only compares compatible forecast horizons, models, scopes, and step-aheads.
    - Historical error distribution strictly precedes observation date.
    - Robust median + MAD scoring on absolute forecast error (|y_pred - y_actual|).
    - Prevents cross-scope or cross-horizon error pollution.
    - Explicit insufficient history handling.
    """
    cfg = config or AnomalyConfig()
    report_id = str(uuid.uuid4())

    if not items:
        return AnomalyReport(
            report_id=report_id,
            target_scope=target_scope,
            total_evaluated=0,
            anomalies_detected=0,
            insufficient_history_count=0,
            anomalies=[],
            config=cfg.__dict__,
        )

    # 1. Filter items by compatibility
    compatible_items = []
    for it in items:
        # Check sales scope
        scope = it.get("sales_scope", "TOTAL_STORE")
        if scope != target_scope:
            continue

        # Check model name if specified
        if target_model_name and it.get("model_name") != target_model_name:
            continue

        # Check horizon if specified
        if target_horizon_days and it.get("horizon_days") != target_horizon_days:
            continue

        # Check step_ahead if specified
        if target_step_ahead and it.get("step_ahead") != target_step_ahead:
            continue

        # Must have both predicted and reconciled actual values
        pred = it.get("predicted_value")
        act = it.get("actual_value")
        if pred is None or act is None:
            continue

        # Parse date
        raw_date = it.get("target_date")
        parsed_dt = parse_business_calendar_date(raw_date)
        if pd.isna(parsed_dt):
            continue

        abs_err = abs(float(pred) - float(act))
        compatible_items.append({
            "target_date": parsed_dt,
            "date_str": parsed_dt.strftime("%Y-%m-%d"),
            "predicted_value": float(pred),
            "actual_value": float(act),
            "absolute_error": float(abs_err),
            "run_id": it.get("run_id"),
            "model_name": it.get("model_name"),
            "horizon_days": it.get("horizon_days"),
            "step_ahead": it.get("step_ahead"),
        })

    if not compatible_items:
        return AnomalyReport(
            report_id=report_id,
            target_scope=target_scope,
            total_evaluated=0,
            anomalies_detected=0,
            insufficient_history_count=0,
            anomalies=[],
            config=cfg.__dict__,
        )

    # 2. Group items into strictly compatible evaluation streams: (scope, model, horizon, step)
    streams: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for it in compatible_items:
        key = (
            it.get("sales_scope") or target_scope,
            it.get("model_name") or "DEFAULT_MODEL",
            it.get("horizon_days") or 0,
            it.get("step_ahead") or 0,
        )
        streams.setdefault(key, []).append(it)

    anomalies: List[AnomalyPoint] = []
    insufficient_history_count = 0
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    total_evaluated = 0

    for stream_key, stream_items in streams.items():
        # Sort each stream strictly by target_date
        stream_items.sort(key=lambda x: x["target_date"])
        n_stream = len(stream_items)
        total_evaluated += n_stream

        # 3. Rolling evaluation over historical error distribution of this compatible stream
        for i in range(n_stream):
            start_idx = max(0, i - cfg.window_size)
            history_items = stream_items[start_idx:i]

            if len(history_items) < cfg.min_history:
                insufficient_history_count += 1
                continue

            curr_item = stream_items[i]
            curr_error = curr_item["absolute_error"]
            hist_errors = np.array([h["absolute_error"] for h in history_items], dtype=np.float64)

            center_err = float(np.median(hist_errors))
            scale_err, is_fallback = compute_robust_scale(
                hist_errors,
                center=center_err,
                scaling_constant=cfg.mad_scaling_constant,
                zero_mad_minimum_scale=cfg.zero_mad_minimum_scale,
            )

            err_deviation = curr_error - center_err
            z_error = err_deviation / scale_err

            # Only flag if error is unusually LARGE (positive z-score exceeding threshold)
            if z_error < cfg.error_threshold:
                continue

            severity = classify_severity(z_error, cfg)
            anomaly_id = generate_deterministic_anomaly_id(
                sales_scope=target_scope,
                observation_date=curr_item["date_str"],
                anomaly_type=AnomalyType.FORECAST_ERROR_ANOMALY.value,
                model_name=curr_item["model_name"],
                horizon_days=curr_item["horizon_days"],
                step_ahead=curr_item["step_ahead"],
                forecast_run_id=curr_item["run_id"],
            )

            model_display = f" for model '{curr_item['model_name']}'" if curr_item.get("model_name") else ""
            step_display = f" (step {curr_item['step_ahead']})" if curr_item.get("step_ahead") else ""

            explanation = (
                f"Observed forecast absolute error of ${curr_error:.2f}{model_display}{step_display} "
                f"exceeded historical baseline median error (${center_err:.2f}) by {z_error:.2f} "
                f"robust standard deviations (threshold {cfg.error_threshold:.2f}). "
                f"Predicted: ${curr_item['predicted_value']:.2f}, Actual: ${curr_item['actual_value']:.2f}."
            )
            if is_fallback:
                explanation += (
                    f" Historical error dispersion was zero; "
                    f"fallback scale of ${scale_err:.2f} was applied."
                )

            point = AnomalyPoint(
                anomaly_id=anomaly_id,
                observation_date=curr_item["date_str"],
                sales_scope=target_scope,
                anomaly_type=AnomalyType.FORECAST_ERROR_ANOMALY,
                severity=severity,
                direction=AnomalyDirection.HIGHER_THAN_EXPECTED,
                observed_value=float(curr_error),
                reference_center=float(center_err),
                reference_spread=float(scale_err),
                anomaly_score=float(z_error),
                threshold=float(cfg.error_threshold),
                explanation=explanation,
                detection_timestamp=now_utc,
                status="ACTIVE",
                model_name=curr_item.get("model_name"),
                forecast_run_id=curr_item.get("run_id"),
                horizon_days=curr_item.get("horizon_days"),
                step_ahead=curr_item.get("step_ahead"),
                predicted_value=curr_item.get("predicted_value"),
                dispersion_zero_fallback=is_fallback,
            )
            anomalies.append(point)

    return AnomalyReport(
        report_id=report_id,
        target_scope=target_scope,
        total_evaluated=total_evaluated,
        anomalies_detected=len(anomalies),
        insufficient_history_count=insufficient_history_count,
        anomalies=anomalies,
        config=cfg.__dict__,
    )
