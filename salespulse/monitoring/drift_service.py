"""
SalesPulse AI - Dataset Drift Monitoring Service
Provides:
- Versioned reference baseline profiling for numeric and categorical features.
- Population Stability Index (PSI) with epsilon smoothing and zero-bin safeguards.
- Total Variation Distance (TVD) and unseen category detection for categoricals.
- Null-rate shift, schema deviation, and day-of-week seasonality temporal context.
- Four-state classification: NO_DRIFT, WARNING, DRIFT, UNKNOWN.
- Strict isolation of sensitive records (only summary statistics persisted).
- Safe handling of database errors and offline telemetry.
"""

import math
import uuid
import logging
import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from salespulse.data.contracts import sanitize_for_json
from salespulse.storage.repository import AuditRepository
from salespulse.monitoring.drift_schemas import (
    DriftStatus,
    FeatureType,
    DriftMetricType,
    NumericFeatureProfile,
    CategoricalFeatureProfile,
    DriftBaselineProfile,
    DriftConfig,
    FeatureDriftResult,
    DriftReport,
)

logger = logging.getLogger(__name__)


def compute_numeric_psi(
    baseline_profile: NumericFeatureProfile,
    current_series: pd.Series,
    config: Optional[DriftConfig] = None,
) -> Tuple[Optional[float], DriftStatus, Dict[str, Any], str]:
    """Compute Population Stability Index (PSI) between baseline bins and current data.
    
    Safeguards:
    - Epsilon smoothing (config.epsilon, default 1e-4) applied to avoid log(0) or division by zero.
    - Constant feature handling: if baseline std is zero, test if current values diverge.
    - Low sample sizes (< config.min_sample_size) return UNKNOWN.
    """
    cfg = config or DriftConfig()
    non_null_curr = current_series.dropna()
    curr_n = len(non_null_curr)

    if curr_n < cfg.min_sample_size or baseline_profile.count < cfg.min_sample_size:
        return (
            None,
            DriftStatus.UNKNOWN,
            {"current_mean": float(non_null_curr.mean()) if curr_n > 0 else None},
            f"Insufficient samples for PSI (current non-null: {curr_n}, baseline: {baseline_profile.count}, minimum: {cfg.min_sample_size})",
        )

    # Constant feature in baseline
    if baseline_profile.std == 0.0 or len(baseline_profile.bin_edges) <= 1:
        curr_std = float(non_null_curr.std(ddof=0)) if curr_n > 1 else 0.0
        curr_mean = float(non_null_curr.mean())
        if curr_std == 0.0 and math.isclose(curr_mean, baseline_profile.mean, rel_tol=1e-5, abs_tol=1e-5):
            return (
                0.0,
                DriftStatus.NO_DRIFT,
                {"current_mean": curr_mean, "current_std": 0.0},
                "Feature is constant in both baseline and current batch; no drift detected.",
            )
        else:
            return (
                1.0,
                DriftStatus.DRIFT,
                {"current_mean": curr_mean, "current_std": curr_std},
                f"Feature was constant ({baseline_profile.mean}) in baseline but varies in current batch (mean={curr_mean:.2f}, std={curr_std:.2f}).",
            )

    # Bin current observations using baseline bin edges
    # Outer boundaries extended to (-inf, +inf) to capture out-of-range observations
    edges = list(baseline_profile.bin_edges)
    extended_edges = [-np.inf] + edges[1:-1] + [np.inf]

    curr_counts, _ = np.histogram(non_null_curr.to_numpy(), bins=extended_edges)
    curr_proportions = curr_counts / curr_n

    base_proportions = np.array(baseline_profile.bin_proportions)
    if len(base_proportions) != len(curr_proportions):
        # Fallback if bin counts differ
        return (
            None,
            DriftStatus.UNKNOWN,
            {},
            f"Bin dimension mismatch between baseline ({len(base_proportions)}) and current ({len(curr_proportions)})",
        )

    # Apply epsilon smoothing
    eps = cfg.epsilon
    adj_base = (base_proportions + eps) / (1.0 + eps * len(base_proportions))
    adj_curr = (curr_proportions + eps) / (1.0 + eps * len(curr_proportions))

    # PSI formula: sum((O - E) * ln(O / E))
    psi_terms = (adj_curr - adj_base) * np.log(adj_curr / adj_base)
    psi_val = float(np.sum(psi_terms))
    psi_val = max(0.0, psi_val)  # Numerical precision safeguard

    curr_mean = float(non_null_curr.mean())
    curr_std = float(non_null_curr.std(ddof=1)) if curr_n > 1 else 0.0
    summary = {
        "current_mean": round(curr_mean, 4),
        "current_std": round(curr_std, 4),
        "current_min": float(non_null_curr.min()),
        "current_max": float(non_null_curr.max()),
        "current_p50": float(non_null_curr.median()),
    }

    if psi_val >= cfg.numeric_drift_threshold:
        status = DriftStatus.DRIFT
        explanation = f"Significant numeric distribution shift (PSI={psi_val:.4f} >= {cfg.numeric_drift_threshold})."
    elif psi_val >= cfg.numeric_warning_threshold:
        status = DriftStatus.WARNING
        explanation = f"Moderate numeric distribution shift (PSI={psi_val:.4f} >= {cfg.numeric_warning_threshold}). Monitoring recommended."
    else:
        status = DriftStatus.NO_DRIFT
        explanation = f"Stable numeric distribution (PSI={psi_val:.4f} < {cfg.numeric_warning_threshold})."

    return psi_val, status, summary, explanation


def compute_categorical_tvd(
    baseline_profile: CategoricalFeatureProfile,
    current_series: pd.Series,
    config: Optional[DriftConfig] = None,
) -> Tuple[Optional[float], DriftStatus, Dict[str, Any], List[str], float, str]:
    """Compute Total Variation Distance (TVD) and identify unseen categories.
    
    TVD = 0.5 * sum(|P(c) - Q(c)|) over all observed categories.
    Returns (tvd, status, current_summary, unseen_categories, unseen_category_rate, explanation).
    """
    cfg = config or DriftConfig()
    non_null_curr = current_series.dropna().astype(str)
    curr_n = len(non_null_curr)

    if curr_n < cfg.min_sample_size or baseline_profile.count < cfg.min_sample_size:
        return (
            None,
            DriftStatus.UNKNOWN,
            {"current_distinct_count": non_null_curr.nunique() if curr_n > 0 else 0},
            [],
            0.0,
            f"Insufficient samples for categorical check (current: {curr_n}, baseline: {baseline_profile.count}, min: {cfg.min_sample_size})",
        )

    curr_counts = non_null_curr.value_counts()
    curr_props = (curr_counts / curr_n).to_dict()
    base_props = baseline_profile.category_proportions

    all_cats = set(base_props.keys()).union(set(curr_props.keys()))
    tvd_sum = 0.0
    unseen_cats = []
    unseen_count = 0

    for cat in all_cats:
        p_base = base_props.get(cat, 0.0)
        p_curr = curr_props.get(cat, 0.0)
        tvd_sum += abs(p_curr - p_base)
        if cat not in base_props:
            unseen_cats.append(cat)
            unseen_count += curr_counts.get(cat, 0)

    tvd = float(0.5 * tvd_sum)
    unseen_rate = float(unseen_count / curr_n) if curr_n > 0 else 0.0

    summary = {
        "current_distinct_count": len(curr_props),
        "top_current_proportions": {k: round(v, 4) for k, v in list(curr_props.items())[:10]},
    }

    # Evaluate drift
    if unseen_rate >= cfg.unseen_category_rate_threshold or tvd >= cfg.categorical_drift_threshold:
        status = DriftStatus.DRIFT
        reasons = []
        if unseen_rate >= cfg.unseen_category_rate_threshold:
            reasons.append(f"{len(unseen_cats)} unseen categories representing {unseen_rate*100:.1f}% of current volume")
        if tvd >= cfg.categorical_drift_threshold:
            reasons.append(f"TVD={tvd:.4f} >= {cfg.categorical_drift_threshold}")
        explanation = f"Significant categorical divergence: {'; '.join(reasons)}."
    elif tvd >= cfg.categorical_warning_threshold:
        status = DriftStatus.WARNING
        explanation = f"Moderate categorical shift (TVD={tvd:.4f} >= {cfg.categorical_warning_threshold})."
    else:
        status = DriftStatus.NO_DRIFT
        explanation = f"Stable categorical distribution (TVD={tvd:.4f} < {cfg.categorical_warning_threshold})."

    return tvd, status, summary, unseen_cats, round(unseen_rate, 4), explanation


def profile_dataset(
    df: pd.DataFrame,
    baseline_id: Optional[str] = None,
    sales_scope: str = "TOTAL_STORE",
    source_file_name: Optional[str] = None,
    source_file_sha256: Optional[str] = None,
    notes: Optional[str] = None,
    num_bins: int = 10,
) -> DriftBaselineProfile:
    """Construct a versioned, reproducible reference baseline profile from a dataframe.
    
    Extracts summary statistics, histogram bins, and categorical frequency tables.
    Never retains raw customer rows or PII.
    """
    b_id = baseline_id or f"base_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{sales_scope.lower()}"
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    row_count = len(df)

    # Determine observation date window if date column present
    data_window_start = None
    data_window_end = None
    date_col = next((c for c in df.columns if c.lower() in ("date", "calendar_date", "timestamp")), None)
    if date_col is not None and not df[date_col].dropna().empty:
        try:
            parsed_dates = pd.to_datetime(df[date_col].dropna(), errors="coerce").dropna()
            if not parsed_dates.empty:
                data_window_start = parsed_dates.min().strftime("%Y-%m-%d")
                data_window_end = parsed_dates.max().strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Could not parse date coverage window for baseline: {e}")

    feature_names = []
    feature_types = {}
    numeric_profiles = {}
    categorical_profiles = {}

    for col in df.columns:
        if col.lower() in ("date", "calendar_date", "timestamp"):
            continue

        series = df[col]
        null_count = int(series.isna().sum())
        null_rate = float(null_count / row_count) if row_count > 0 else 0.0
        feature_names.append(col)

        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            feature_types[col] = FeatureType.NUMERIC.value
            valid_series = series.dropna()
            n_valid = len(valid_series)

            if n_valid == 0:
                mean_val, std_val, min_val, max_val = 0.0, 0.0, 0.0, 0.0
                quantiles = {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
                bin_edges = [0.0, 1.0]
                bin_proportions = [1.0]
            else:
                arr = valid_series.to_numpy(dtype=float)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                quantiles = {
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p95": float(np.percentile(arr, 95)),
                }

                # Construct reproducible bin edges
                if min_val == max_val:
                    bin_edges = [min_val - 1.0, min_val + 1.0]
                    bin_proportions = [1.0]
                else:
                    # Attempt quantile binning; fallback to uniform if duplicates occur
                    probs = np.linspace(0, 100, num_bins + 1)
                    q_edges = np.percentile(arr, probs)
                    if len(np.unique(q_edges)) == len(q_edges):
                        bin_edges = [float(x) for x in q_edges]
                    else:
                        bin_edges = [float(x) for x in np.linspace(min_val, max_val, num_bins + 1)]

                    counts, _ = np.histogram(arr, bins=bin_edges)
                    bin_proportions = [float(c / n_valid) for c in counts]

            numeric_profiles[col] = NumericFeatureProfile(
                feature_name=col,
                count=n_valid,
                null_count=null_count,
                null_rate=round(null_rate, 4),
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
                quantiles=quantiles,
                bin_edges=bin_edges,
                bin_proportions=bin_proportions,
            )
        else:
            # Categorical feature
            feature_types[col] = FeatureType.CATEGORICAL.value
            valid_series = series.dropna().astype(str)
            n_valid = len(valid_series)

            if n_valid == 0:
                distinct_count = 0
                cat_props = {}
            else:
                counts = valid_series.value_counts()
                distinct_count = len(counts)
                cat_props = {str(k): round(float(v / n_valid), 4) for k, v in counts.items()}

            categorical_profiles[col] = CategoricalFeatureProfile(
                feature_name=col,
                count=n_valid,
                null_count=null_count,
                null_rate=round(null_rate, 4),
                distinct_count=distinct_count,
                category_proportions=cat_props,
            )

    return DriftBaselineProfile(
        baseline_id=b_id,
        created_at=created_at,
        sales_scope=sales_scope,
        data_window_start=data_window_start,
        data_window_end=data_window_end,
        row_count=row_count,
        feature_names=feature_names,
        feature_types=feature_types,
        numeric_profiles=numeric_profiles,
        categorical_profiles=categorical_profiles,
        source_file_name=source_file_name,
        source_file_sha256=source_file_sha256,
        notes=notes,
    )


class DatasetDriftService:
    """Production service for baseline lifecycle management and explainable dataset drift detection."""

    def __init__(
        self,
        repository: Optional[AuditRepository] = None,
        default_config: Optional[DriftConfig] = None,
    ):
        self.repo = repository or AuditRepository()
        self.config = default_config or DriftConfig()

    def create_baseline(
        self,
        df: pd.DataFrame,
        sales_scope: str = "TOTAL_STORE",
        baseline_id: Optional[str] = None,
        source_file_name: Optional[str] = None,
        source_file_sha256: Optional[str] = None,
        notes: Optional[str] = None,
        activate: bool = False,
    ) -> DriftBaselineProfile:
        """Create and persist a reference baseline from a dataset without raw row leakage."""
        if df is None or len(df) == 0:
            raise ValueError("Cannot profile an empty dataset for baseline.")

        profile = profile_dataset(
            df=df,
            baseline_id=baseline_id,
            sales_scope=sales_scope,
            source_file_name=source_file_name,
            source_file_sha256=source_file_sha256,
            notes=notes,
            num_bins=self.config.num_bins,
        )

        try:
            self.repo.create_drift_baseline(
                baseline_id=profile.baseline_id,
                profile_json=profile.to_json(),
                sales_scope=sales_scope,
                data_window_start=profile.data_window_start,
                data_window_end=profile.data_window_end,
                row_count=profile.row_count,
                feature_count=len(profile.feature_names),
                is_active=activate,
                source_file_name=source_file_name,
                source_file_sha256=source_file_sha256,
                notes=notes,
            )
            logger.info(f"Persisted drift baseline {profile.baseline_id} (active={activate})")
        except Exception as e:
            logger.error(f"Failed to persist baseline to database: {e}")

        return profile

    def promote_baseline(
        self,
        baseline_id: str,
        sales_scope: str = "TOTAL_STORE",
    ) -> bool:
        """Explicitly promote a baseline to active status for a scope."""
        try:
            res = self.repo.promote_drift_baseline(baseline_id=baseline_id, sales_scope=sales_scope)
            return res is not None
        except Exception as e:
            logger.error(f"Error promoting baseline {baseline_id}: {e}")
            return False

    def get_active_baseline(
        self,
        sales_scope: str = "TOTAL_STORE",
    ) -> Optional[DriftBaselineProfile]:
        """Fetch the current active baseline profile for a scope."""
        try:
            rec = self.repo.get_active_drift_baseline(sales_scope=sales_scope)
            if rec and rec.profile_json:
                import json
                data = json.loads(rec.profile_json)
                return DriftBaselineProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Error reading active baseline: {e}")
        return None

    def list_baselines(
        self,
        sales_scope: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List baseline metadata for operator management."""
        try:
            records = self.repo.list_drift_baselines(sales_scope=sales_scope, limit=limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"Error listing baselines: {e}")
            return []

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        sales_scope: str = "TOTAL_STORE",
        baseline_id: Optional[str] = None,
        baseline_profile: Optional[DriftBaselineProfile] = None,
        config: Optional[DriftConfig] = None,
        persist: bool = True,
    ) -> DriftReport:
        """Execute a comprehensive, explainable dataset drift check.
        
        Guarantees:
        - If database offline or telemetry missing, returns UNKNOWN.
        - If no baseline registered or specified, returns UNKNOWN.
        - If sample size insufficient (< min_sample_size), returns UNKNOWN.
        - Never reports UNKNOWN as NO_DRIFT.
        """
        cfg = config or self.config
        chk_id = f"chk_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        chk_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Check DB connectivity
        db_connected = True
        try:
            # Probe engine connection
            with self.repo.engine.connect() as conn:
                pass
        except Exception as e:
            db_connected = False
            logger.warning(f"Database connection unavailable during drift check: {e}")

        # Resolve reference baseline
        target_baseline = baseline_profile
        if target_baseline is None:
            if not db_connected:
                return DriftReport(
                    check_id=chk_id,
                    checked_at=chk_time,
                    sales_scope=sales_scope,
                    baseline_id="UNKNOWN",
                    overall_status=DriftStatus.UNKNOWN,
                    baseline_row_count=0,
                    current_row_count=len(current_df) if current_df is not None else 0,
                    total_features_evaluated=0,
                    drifted_features_count=0,
                    warning_features_count=0,
                    summary_message="Database telemetry offline; cannot retrieve reference baseline.",
                    database_connected=False,
                )

            try:
                if baseline_id:
                    rec = self.repo.get_drift_baseline(baseline_id)
                else:
                    rec = self.repo.get_active_drift_baseline(sales_scope=sales_scope)

                if rec and rec.profile_json:
                    import json
                    target_baseline = DriftBaselineProfile.from_dict(json.loads(rec.profile_json))
            except Exception as e:
                logger.error(f"Failed to load baseline from storage: {e}")
                return DriftReport(
                    check_id=chk_id,
                    checked_at=chk_time,
                    sales_scope=sales_scope,
                    baseline_id=baseline_id or "UNKNOWN",
                    overall_status=DriftStatus.UNKNOWN,
                    baseline_row_count=0,
                    current_row_count=len(current_df) if current_df is not None else 0,
                    total_features_evaluated=0,
                    drifted_features_count=0,
                    warning_features_count=0,
                    summary_message=f"Storage error reading reference baseline: {str(e)}",
                    database_connected=False,
                )

        if target_baseline is None:
            return DriftReport(
                check_id=chk_id,
                checked_at=chk_time,
                sales_scope=sales_scope,
                baseline_id="MISSING",
                overall_status=DriftStatus.UNKNOWN,
                baseline_row_count=0,
                current_row_count=len(current_df) if current_df is not None else 0,
                total_features_evaluated=0,
                drifted_features_count=0,
                warning_features_count=0,
                summary_message=f"No active reference baseline found for scope '{sales_scope}'. Create and activate a baseline first.",
                database_connected=db_connected,
            )

        curr_rows = len(current_df) if current_df is not None else 0
        if current_df is None or curr_rows < cfg.min_sample_size:
            return DriftReport(
                check_id=chk_id,
                checked_at=chk_time,
                sales_scope=sales_scope,
                baseline_id=target_baseline.baseline_id,
                overall_status=DriftStatus.UNKNOWN,
                baseline_row_count=target_baseline.row_count,
                current_row_count=curr_rows,
                total_features_evaluated=0,
                drifted_features_count=0,
                warning_features_count=0,
                summary_message=f"Current batch has insufficient observations ({curr_rows} < {cfg.min_sample_size} minimum).",
                remediation_advice="Provide at least 30 observations for statistically reliable drift evaluation.",
                database_connected=db_connected,
            )

        # Evaluate feature drift
        feature_results: List[FeatureDriftResult] = []
        schema_differences: List[str] = []
        drifted_count = 0
        warning_count = 0

        # Check for missing baseline features in current dataframe
        for b_feat in target_baseline.feature_names:
            if b_feat not in current_df.columns:
                schema_differences.append(f"Baseline feature '{b_feat}' is missing from current batch.")
                feature_results.append(
                    FeatureDriftResult(
                        feature_name=b_feat,
                        feature_type=FeatureType.NUMERIC if b_feat in target_baseline.numeric_profiles else FeatureType.CATEGORICAL,
                        metric_type=DriftMetricType.SCHEMA_CHANGE,
                        metric_value=None,
                        warning_threshold=0.0,
                        drift_threshold=0.0,
                        status=DriftStatus.DRIFT,
                        baseline_sample_size=target_baseline.row_count,
                        current_sample_size=curr_rows,
                        baseline_summary={},
                        current_summary={},
                        explanation=f"Critical schema deviation: required baseline feature '{b_feat}' is absent.",
                    )
                )
                drifted_count += 1

        # Check for unexpected new columns in current dataframe
        for c_col in current_df.columns:
            if c_col.lower() in ("date", "calendar_date", "timestamp"):
                continue
            if c_col not in target_baseline.feature_names:
                schema_differences.append(f"New feature '{c_col}' in current batch is unobserved in baseline.")

        # Numeric features
        for feat_name, num_profile in target_baseline.numeric_profiles.items():
            if feat_name not in current_df.columns:
                continue

            curr_series = current_df[feat_name]
            curr_null_count = int(curr_series.isna().sum())
            curr_null_rate = float(curr_null_count / curr_rows) if curr_rows > 0 else 0.0
            null_delta = abs(curr_null_rate - num_profile.null_rate)

            psi_val, stat, summary, expl = compute_numeric_psi(num_profile, curr_series, cfg)
            summary["current_null_rate"] = round(curr_null_rate, 4)
            summary["null_rate_delta"] = round(null_delta, 4)

            # Check null rate shift
            if null_delta >= cfg.null_rate_shift_threshold:
                if stat != DriftStatus.DRIFT:
                    stat = DriftStatus.WARNING
                    expl += f" Warning: Null rate changed by {null_delta*100:.1f}% (baseline {num_profile.null_rate*100:.1f}%, current {curr_null_rate*100:.1f}%)."

            if stat == DriftStatus.DRIFT:
                drifted_count += 1
            elif stat == DriftStatus.WARNING:
                warning_count += 1

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feat_name,
                    feature_type=FeatureType.NUMERIC,
                    metric_type=DriftMetricType.PSI,
                    metric_value=round(psi_val, 4) if psi_val is not None else None,
                    warning_threshold=cfg.numeric_warning_threshold,
                    drift_threshold=cfg.numeric_drift_threshold,
                    status=stat,
                    baseline_sample_size=num_profile.count,
                    current_sample_size=curr_rows - curr_null_count,
                    baseline_summary={
                        "mean": num_profile.mean,
                        "std": num_profile.std,
                        "min": num_profile.min_val,
                        "max": num_profile.max_val,
                        "null_rate": num_profile.null_rate,
                    },
                    current_summary=summary,
                    explanation=expl,
                )
            )

        # Categorical features
        for feat_name, cat_profile in target_baseline.categorical_profiles.items():
            if feat_name not in current_df.columns:
                continue

            curr_series = current_df[feat_name]
            curr_null_count = int(curr_series.isna().sum())
            curr_null_rate = float(curr_null_count / curr_rows) if curr_rows > 0 else 0.0
            null_delta = abs(curr_null_rate - cat_profile.null_rate)

            tvd_val, stat, summary, unseen_cats, unseen_rate, expl = compute_categorical_tvd(
                cat_profile, curr_series, cfg
            )
            summary["current_null_rate"] = round(curr_null_rate, 4)
            summary["null_rate_delta"] = round(null_delta, 4)

            if null_delta >= cfg.null_rate_shift_threshold:
                if stat != DriftStatus.DRIFT:
                    stat = DriftStatus.WARNING
                    expl += f" Warning: Null rate changed by {null_delta*100:.1f}%."

            if stat == DriftStatus.DRIFT:
                drifted_count += 1
            elif stat == DriftStatus.WARNING:
                warning_count += 1

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feat_name,
                    feature_type=FeatureType.CATEGORICAL,
                    metric_type=DriftMetricType.TVD,
                    metric_value=round(tvd_val, 4) if tvd_val is not None else None,
                    warning_threshold=cfg.categorical_warning_threshold,
                    drift_threshold=cfg.categorical_drift_threshold,
                    status=stat,
                    baseline_sample_size=cat_profile.count,
                    current_sample_size=curr_rows - curr_null_count,
                    baseline_summary={
                        "distinct_count": cat_profile.distinct_count,
                        "null_rate": cat_profile.null_rate,
                        "top_categories": list(cat_profile.category_proportions.keys())[:5],
                    },
                    current_summary=summary,
                    unseen_categories=unseen_cats,
                    unseen_category_rate=unseen_rate,
                    explanation=expl,
                )
            )

        # Evaluate Temporal / Seasonality Context (e.g., weekend proportions)
        temporal_context = None
        date_col = next((c for c in current_df.columns if c.lower() in ("date", "calendar_date", "timestamp")), None)
        if date_col is not None:
            try:
                curr_dates = pd.to_datetime(current_df[date_col].dropna(), errors="coerce").dropna()
                if not curr_dates.empty:
                    curr_weekends = (curr_dates.dt.dayofweek >= 5).mean()
                    curr_start = curr_dates.min().strftime("%Y-%m-%d")
                    curr_end = curr_dates.max().strftime("%Y-%m-%d")
                    temporal_context = (
                        f"Current window: {curr_start} to {curr_end} (Weekend share: {curr_weekends*100:.1f}%). "
                        f"Baseline window: {target_baseline.data_window_start or 'N/A'} to {target_baseline.data_window_end or 'N/A'}."
                    )
            except Exception as e:
                logger.warning(f"Could not compute temporal context: {e}")

        # Compute overall status
        if any(f.status == DriftStatus.UNKNOWN for f in feature_results) and not any(f.status == DriftStatus.DRIFT for f in feature_results):
            overall_status = DriftStatus.UNKNOWN
            summary_message = "Drift status is UNKNOWN due to indeterminate feature evaluations or sample size constraints."
            remediation = "Collect more sample observations before triggering retraining or operational alerts."
        elif drifted_count > 0:
            overall_status = DriftStatus.DRIFT
            summary_message = f"DRIFT DETECTED: {drifted_count} feature(s) exceed drift thresholds. Distributional divergence confirmed."
            remediation = "Investigate recent business changes or consider promoting a new reference baseline."
        elif warning_count > 0:
            overall_status = DriftStatus.WARNING
            summary_message = f"WARNING: {warning_count} feature(s) exhibit moderate distributional movement."
            remediation = "Monitor ongoing ingestion cycles; verify whether calendar or holiday seasonality explains the shift."
        else:
            overall_status = DriftStatus.NO_DRIFT
            summary_message = "NO DRIFT: Feature distributions remain stable and conform to the active reference baseline."
            remediation = None

        report = DriftReport(
            check_id=chk_id,
            checked_at=chk_time,
            sales_scope=sales_scope,
            baseline_id=target_baseline.baseline_id,
            overall_status=overall_status,
            baseline_row_count=target_baseline.row_count,
            current_row_count=curr_rows,
            total_features_evaluated=len(feature_results),
            drifted_features_count=drifted_count,
            warning_features_count=warning_count,
            feature_results=feature_results,
            schema_differences=schema_differences,
            temporal_context=temporal_context,
            summary_message=summary_message,
            remediation_advice=remediation,
            database_connected=db_connected,
        )

        # Persist report if requested and DB connected
        if persist and db_connected:
            try:
                self.repo.record_drift_check(
                    check_id=report.check_id,
                    baseline_id=report.baseline_id,
                    overall_status=report.overall_status.value,
                    summary_message=report.summary_message,
                    report_json=report.to_json(),
                    sales_scope=sales_scope,
                    current_row_count=report.current_row_count,
                    baseline_row_count=report.baseline_row_count,
                    total_features_evaluated=report.total_features_evaluated,
                    drifted_features_count=report.drifted_features_count,
                    warning_features_count=report.warning_features_count,
                )
            except Exception as e:
                logger.error(f"Failed to persist drift check report: {e}")

        return report

    def get_latest_drift_check(
        self,
        sales_scope: str = "TOTAL_STORE",
    ) -> Optional[DriftReport]:
        """Fetch the most recent drift check from storage."""
        try:
            rec = self.repo.get_latest_drift_check(sales_scope=sales_scope)
            if rec and rec.report_json:
                import json
                data = json.loads(rec.report_json)
                feat_res = [
                    FeatureDriftResult(
                        feature_name=f["feature_name"],
                        feature_type=FeatureType(f["feature_type"]),
                        metric_type=DriftMetricType(f["metric_type"]),
                        metric_value=f["metric_value"],
                        warning_threshold=f["warning_threshold"],
                        drift_threshold=f["drift_threshold"],
                        status=DriftStatus(f["status"]),
                        baseline_sample_size=f["baseline_sample_size"],
                        current_sample_size=f["current_sample_size"],
                        baseline_summary=f.get("baseline_summary", {}),
                        current_summary=f.get("current_summary", {}),
                        unseen_categories=f.get("unseen_categories", []),
                        unseen_category_rate=f.get("unseen_category_rate", 0.0),
                        explanation=f.get("explanation", ""),
                    )
                    for f in data.get("feature_results", [])
                ]
                return DriftReport(
                    check_id=data["check_id"],
                    checked_at=data["checked_at"],
                    sales_scope=data.get("sales_scope", sales_scope),
                    baseline_id=data.get("baseline_id", "UNKNOWN"),
                    overall_status=DriftStatus(data["overall_status"]),
                    baseline_row_count=data.get("baseline_row_count", 0),
                    current_row_count=data.get("current_row_count", 0),
                    total_features_evaluated=data.get("total_features_evaluated", 0),
                    drifted_features_count=data.get("drifted_features_count", 0),
                    warning_features_count=data.get("warning_features_count", 0),
                    feature_results=feat_res,
                    schema_differences=data.get("schema_differences", []),
                    temporal_context=data.get("temporal_context"),
                    summary_message=data.get("summary_message", ""),
                    remediation_advice=data.get("remediation_advice"),
                    database_connected=data.get("database_connected", True),
                )
        except Exception as e:
            logger.error(f"Error fetching latest drift check: {e}")
        return None

    def get_drift_history(
        self,
        sales_scope: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch historical drift check evaluations."""
        try:
            records = self.repo.list_drift_checks(sales_scope=sales_scope, limit=limit)
            return [r.to_dict() for r in records]
        except Exception as e:
            logger.error(f"Error fetching drift history: {e}")
            return []
