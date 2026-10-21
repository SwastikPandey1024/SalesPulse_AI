"""
SalesPulse AI - Dataset Drift Monitoring Schemas
Defines strictly typed, explainable contracts for reference baselines and dataset drift signals:
- Drift statuses: NO_DRIFT, WARNING, DRIFT, UNKNOWN.
- Feature profiling schemas (numeric quantiles, histogram bins, categorical frequency tables).
- Versioned reference baseline definitions with source provenance.
- Feature-level drift evaluation results (PSI, TVD, missingness shift, unseen categories).
- High-level comprehensive DriftReport with RFC 8259 JSON sanitization.
"""

import json
import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict

from salespulse.data.contracts import sanitize_for_json


class DriftStatus(str, Enum):
    """Categorical verdict for feature-level and dataset-level drift."""
    NO_DRIFT = "NO_DRIFT"  # Stable distribution within expected boundaries
    WARNING = "WARNING"    # Moderate statistical shift; monitoring warranted
    DRIFT = "DRIFT"        # Significant distribution divergence requiring intervention
    UNKNOWN = "UNKNOWN"    # Insufficient samples, missing baseline, or telemetry offline


class FeatureType(str, Enum):
    """Supported feature modalities for distribution profiling."""
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"


class DriftMetricType(str, Enum):
    """Statistical metric utilized for drift evaluation."""
    PSI = "PSI"                          # Population Stability Index (Numeric continuous)
    TVD = "TVD"                          # Total Variation Distance (Categorical frequencies)
    NULL_RATE_SHIFT = "NULL_RATE_SHIFT"  # Data quality missingness rate delta
    SCHEMA_CHANGE = "SCHEMA_CHANGE"      # Missing or incompatible feature column


@dataclass
class NumericFeatureProfile:
    """Statistical profile of a numeric feature in a reference dataset."""
    feature_name: str
    count: int
    null_count: int
    null_rate: float
    mean: float
    std: float
    min_val: float
    max_val: float
    quantiles: Dict[str, float]       # e.g. {"p25": ..., "p50": ..., "p75": ..., "p95": ...}
    bin_edges: List[float]            # Monotonically increasing bin boundaries
    bin_proportions: List[float]      # Reference relative frequency per bin (sums to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "count": self.count,
            "null_count": self.null_count,
            "null_rate": self.null_rate,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "quantiles": self.quantiles,
            "bin_edges": self.bin_edges,
            "bin_proportions": self.bin_proportions,
        }


@dataclass
class CategoricalFeatureProfile:
    """Statistical profile of a categorical feature in a reference dataset."""
    feature_name: str
    count: int
    null_count: int
    null_rate: float
    distinct_count: int
    category_proportions: Dict[str, float]  # Top categories and their baseline proportions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "count": self.count,
            "null_count": self.null_count,
            "null_rate": self.null_rate,
            "distinct_count": self.distinct_count,
            "category_proportions": self.category_proportions,
        }


@dataclass
class DriftBaselineProfile:
    """Immutable, versioned statistical profile of a reference dataset."""
    baseline_id: str
    created_at: str                                    # ISO-8601 UTC
    sales_scope: str = "TOTAL_STORE"
    data_window_start: Optional[str] = None           # YYYY-MM-DD
    data_window_end: Optional[str] = None             # YYYY-MM-DD
    row_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    numeric_profiles: Dict[str, NumericFeatureProfile] = field(default_factory=dict)
    categorical_profiles: Dict[str, CategoricalFeatureProfile] = field(default_factory=dict)
    source_file_name: Optional[str] = None
    source_file_sha256: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "created_at": self.created_at,
            "sales_scope": self.sales_scope,
            "data_window_start": self.data_window_start,
            "data_window_end": self.data_window_end,
            "row_count": self.row_count,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "numeric_profiles": {k: v.to_dict() for k, v in self.numeric_profiles.items()},
            "categorical_profiles": {k: v.to_dict() for k, v in self.categorical_profiles.items()},
            "source_file_name": self.source_file_name,
            "source_file_sha256": self.source_file_sha256,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        sanitized = sanitize_for_json(self.to_dict())
        return json.dumps(sanitized, indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftBaselineProfile":
        num_profs = {}
        for k, v in data.get("numeric_profiles", {}).items():
            num_profs[k] = NumericFeatureProfile(**v)
        cat_profs = {}
        for k, v in data.get("categorical_profiles", {}).items():
            cat_profs[k] = CategoricalFeatureProfile(**v)
        return cls(
            baseline_id=data["baseline_id"],
            created_at=data["created_at"],
            sales_scope=data.get("sales_scope", "TOTAL_STORE"),
            data_window_start=data.get("data_window_start"),
            data_window_end=data.get("data_window_end"),
            row_count=data.get("row_count", 0),
            feature_names=data.get("feature_names", []),
            feature_types=data.get("feature_types", {}),
            numeric_profiles=num_profs,
            categorical_profiles=cat_profs,
            source_file_name=data.get("source_file_name"),
            source_file_sha256=data.get("source_file_sha256"),
            notes=data.get("notes"),
        )


@dataclass
class DriftConfig:
    """Configurable thresholds and parameters for dataset drift detection.
    
    Standard Industry Thresholds (Yurdakul, 2018):
    - PSI < 0.10: No significant change (NO_DRIFT).
    - 0.10 <= PSI < 0.25: Moderate shift; monitoring review advised (WARNING).
    - PSI >= 0.25: Significant distributional shift (DRIFT).
    
    Categorical Total Variation Distance (TVD):
    - TVD < 0.10: Minimal frequency movement (NO_DRIFT).
    - 0.10 <= TVD < 0.25: Moderate category proportion change (WARNING).
    - TVD >= 0.25: Pronounced category shift or unseen category emergence (DRIFT).
    """
    numeric_warning_threshold: float = 0.10
    numeric_drift_threshold: float = 0.25
    categorical_warning_threshold: float = 0.10
    categorical_drift_threshold: float = 0.25
    null_rate_shift_threshold: float = 0.15
    unseen_category_rate_threshold: float = 0.05
    min_sample_size: int = 30
    num_bins: int = 10
    epsilon: float = 1e-4

    def __post_init__(self):
        if self.numeric_warning_threshold <= 0:
            raise ValueError(f"numeric_warning_threshold must be > 0, got {self.numeric_warning_threshold}")
        if self.numeric_drift_threshold <= self.numeric_warning_threshold:
            raise ValueError("numeric_drift_threshold must exceed numeric_warning_threshold")
        if self.min_sample_size < 5:
            raise ValueError("min_sample_size must be >= 5")


@dataclass
class FeatureDriftResult:
    """Granular statistical evaluation of an individual feature."""
    feature_name: str
    feature_type: FeatureType
    metric_type: DriftMetricType
    metric_value: Optional[float]
    warning_threshold: float
    drift_threshold: float
    status: DriftStatus
    baseline_sample_size: int
    current_sample_size: int
    baseline_summary: Dict[str, Any]
    current_summary: Dict[str, Any]
    unseen_categories: List[str] = field(default_factory=list)
    unseen_category_rate: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type.value if hasattr(self.feature_type, "value") else str(self.feature_type),
            "metric_type": self.metric_type.value if hasattr(self.metric_type, "value") else str(self.metric_type),
            "metric_value": self.metric_value,
            "warning_threshold": self.warning_threshold,
            "drift_threshold": self.drift_threshold,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "baseline_sample_size": self.baseline_sample_size,
            "current_sample_size": self.current_sample_size,
            "baseline_summary": self.baseline_summary,
            "current_summary": self.current_summary,
            "unseen_categories": self.unseen_categories,
            "unseen_category_rate": self.unseen_category_rate,
            "explanation": self.explanation,
        }


@dataclass
class DriftReport:
    """Comprehensive, typed verdict of a dataset drift check."""
    check_id: str
    checked_at: str                                    # ISO-8601 UTC
    sales_scope: str
    baseline_id: str
    overall_status: DriftStatus
    baseline_row_count: int
    current_row_count: int
    total_features_evaluated: int
    drifted_features_count: int
    warning_features_count: int
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    schema_differences: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None
    summary_message: str = ""
    remediation_advice: Optional[str] = None
    database_connected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "checked_at": self.checked_at,
            "sales_scope": self.sales_scope,
            "baseline_id": self.baseline_id,
            "overall_status": self.overall_status.value if hasattr(self.overall_status, "value") else str(self.overall_status),
            "baseline_row_count": self.baseline_row_count,
            "current_row_count": self.current_row_count,
            "total_features_evaluated": self.total_features_evaluated,
            "drifted_features_count": self.drifted_features_count,
            "warning_features_count": self.warning_features_count,
            "feature_results": [f.to_dict() for f in self.feature_results],
            "schema_differences": self.schema_differences,
            "temporal_context": self.temporal_context,
            "summary_message": self.summary_message,
            "remediation_advice": self.remediation_advice,
            "database_connected": self.database_connected,
        }

    def to_json(self, indent: int = 2) -> str:
        sanitized = sanitize_for_json(self.to_dict())
        return json.dumps(sanitized, indent=indent, allow_nan=False)
