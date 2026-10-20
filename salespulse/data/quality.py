"""
SalesPulse AI - Data Quality Diagnostics & Anomaly Assessment Module
Computes coverage, freshness, duplicate counts, numeric validity, distribution metrics,
and explainable anomaly review flags.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

from salespulse.data.contracts import GrainType, safe_parse_calendar_dates
from salespulse.data.time_series import MISSING_DATE_POLICY


def compute_data_quality_diagnostics(
    df: pd.DataFrame,
    grain: GrainType = GrainType.UNKNOWN,
    date_col: Optional[str] = None,
    sales_col: Optional[str] = None,
    max_staleness_days: Optional[int] = 30,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive data-quality diagnostics and explainable review flags.

    Args:
        df: Input DataFrame (typically the valid rows subset).
        grain: GrainType of the data.
        date_col: Column name containing dates.
        sales_col: Column name containing sales.
        max_staleness_days: Configurable threshold for data freshness assessment.
        reference_date: Reference date for freshness (defaults to now).

    Returns:
        Structured dictionary containing diagnostics, distribution summaries, and anomaly flags.
    """
    if df is None or df.empty:
        return {
            "valid": False,
            "grain": grain.value if hasattr(grain, "value") else str(grain),
            "total_rows": 0,
            "message": "Empty dataset provided; cannot compute diagnostics.",
        }

    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())

    # Initialize results container
    diagnostics: Dict[str, Any] = {
        "valid": True,
        "grain": grain.value if hasattr(grain, "value") else str(grain),
        "row_counts": {
            "total_records": total_rows,
            "exact_duplicate_records": exact_duplicates,
        },
        "anomalies": [],
        "missing_date_policy": MISSING_DATE_POLICY,
    }

    # ------------------------------------------------------------------------
    # 1. Date Coverage & Freshness
    # ------------------------------------------------------------------------
    if date_col and date_col in df.columns:
        parsed_dates = safe_parse_calendar_dates(df[date_col]).dropna()
        if not parsed_dates.empty:
            min_date = parsed_dates.min()
            max_date = parsed_dates.max()
            calendar_span_days = (max_date - min_date).days + 1
            unique_dates_count = parsed_dates.nunique()

            # Missing calendar periods
            expected_range = pd.date_range(min_date, max_date, freq="D")
            missing_dates = expected_range.difference(parsed_dates.unique())
            missing_calendar_count = len(missing_dates)

            # Freshness threshold evaluation
            ref_ts = pd.Timestamp(reference_date).normalize() if reference_date else pd.Timestamp.now().normalize()
            staleness_days = max(0, (ref_ts - max_date).days)
            threshold = max_staleness_days if max_staleness_days is not None else 30
            is_stale = staleness_days > threshold

            diagnostics["date_coverage"] = {
                "date_column": date_col,
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "calendar_span_days": calendar_span_days,
                "unique_active_dates": unique_dates_count,
                "missing_calendar_days": missing_calendar_count,
                "missing_dates_sample": [str(d.date()) for d in missing_dates[:5]],
            }

            diagnostics["freshness"] = {
                "latest_observation_date": str(max_date.date()),
                "reference_date": str(ref_ts.date()),
                "staleness_days": staleness_days,
                "threshold_days": threshold,
                "is_stale": is_stale,
                "freshness_status": "STALE" if is_stale else "FRESH",
                "assessment_note": (
                    "Configurable threshold-based staleness check; does not represent automated "
                    "continuous live monitoring."
                ),
            }

            # Anomaly check: Calendar coverage gap ratio
            if calendar_span_days > 7 and (missing_calendar_count / calendar_span_days) > 0.40:
                diagnostics["anomalies"].append({
                    "flag": "HIGH_CALENDAR_GAP_RATIO",
                    "severity": "REVIEW",
                    "description": (
                        f"{missing_calendar_count} of {calendar_span_days} calendar days ({missing_calendar_count/calendar_span_days:.1%}) "
                        f"have no transactions. Verify store operating schedule."
                    ),
                })

    # ------------------------------------------------------------------------
    # 2. Numeric Sales Validity & Distribution
    # ------------------------------------------------------------------------
    if sales_col and sales_col in df.columns:
        numeric_sales = pd.to_numeric(df[sales_col], errors="coerce")
        valid_sales = numeric_sales.dropna()
        finite_sales = valid_sales[np.isfinite(valid_sales)]

        null_count = int(numeric_sales.isna().sum())
        inf_count = int(np.isinf(numeric_sales.fillna(0)).sum())
        zero_count = int((finite_sales == 0.0).sum())
        negative_count = int((finite_sales < 0.0).sum())
        positive_count = int((finite_sales > 0.0).sum())

        diagnostics["numeric_validity"] = {
            "sales_column": sales_col,
            "null_or_unparseable_count": null_count,
            "infinite_count": inf_count,
            "zero_sales_count": zero_count,
            "negative_sales_count": negative_count,
            "positive_sales_count": positive_count,
        }

        if len(finite_sales) > 0:
            mean_val = float(finite_sales.mean())
            std_val = float(finite_sales.std()) if len(finite_sales) > 1 else 0.0
            p25 = float(finite_sales.quantile(0.25))
            median_val = float(finite_sales.median())
            p75 = float(finite_sales.quantile(0.75))
            iqr_val = float(p75 - p25)
            min_val = float(finite_sales.min())
            max_val = float(finite_sales.max())

            diagnostics["sales_distribution"] = {
                "count": len(finite_sales),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": round(min_val, 2),
                "p25": round(p25, 2),
                "median": round(median_val, 2),
                "p75": round(p75, 2),
                "max": round(max_val, 2),
                "iqr": round(iqr_val, 2),
            }

            # ----------------------------------------------------------------
            # 3. Explainable Distribution Anomaly Rules (Review Flags Only)
            # ----------------------------------------------------------------
            # Rule A: High Zero Sales Ratio
            zero_ratio = zero_count / len(finite_sales)
            if zero_ratio > 0.50:
                diagnostics["anomalies"].append({
                    "flag": "HIGH_ZERO_SALES_RATIO",
                    "severity": "REVIEW",
                    "description": (
                        f"Zero-sales records account for {zero_ratio:.1%} of all observations. "
                        f"Review if inactive periods or store shutdowns occurred."
                    ),
                })

            # Rule B: High Return/Negative Sales Ratio
            neg_ratio = negative_count / len(finite_sales)
            if neg_ratio > 0.15:
                diagnostics["anomalies"].append({
                    "flag": "ELEVATED_RETURN_VOLUME",
                    "severity": "REVIEW",
                    "description": (
                        f"Negative sales (returns/refunds) represent {neg_ratio:.1%} of records. "
                        f"Confirm return transaction legitimacy."
                    ),
                })

            # Rule C: Extreme Outliers (Z-score > 5 or > 3*IQR above Q3)
            if std_val > 0:
                z_scores = np.abs((finite_sales - mean_val) / std_val)
                extreme_z_count = int((z_scores > 5.0).sum())
            else:
                extreme_z_count = 0

            iqr_cutoff = p75 + 3.0 * iqr_val if iqr_val > 0 else max_val
            iqr_outliers_count = int((finite_sales > iqr_cutoff).sum())

            if extreme_z_count > 0 or iqr_outliers_count > 0:
                outlier_count = max(extreme_z_count, iqr_outliers_count)
                diagnostics["anomalies"].append({
                    "flag": "STATISTICAL_DISTRIBUTION_OUTLIER",
                    "severity": "REVIEW",
                    "description": (
                        f"Detected {outlier_count} transaction/daily values exceeding 5 standard deviations "
                        f"or 3x IQR above the 75th percentile (max observed: ${max_val:,.2f}). "
                        f"Review for bulk wholesale orders or data-entry errors."
                    ),
                })

    return diagnostics
