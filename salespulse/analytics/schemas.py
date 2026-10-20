"""
SalesPulse AI - Anomaly Detection Schemas and Contracts
Defines strictly typed, RFC 8259-compliant data structures for:
- Anomaly configurations (thresholds, windows, min history, policies).
- Anomaly point records (score, severity, observed, reference, explanation).
- Anomaly reports (summary, list of anomalies, evaluation diagnostics).
"""

import json
import uuid
import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict

from salespulse.data.contracts import sanitize_for_json


class AnomalyType(str, Enum):
    """Categorization of detected anomalies."""
    SALES_SURGE = "SALES_SURGE"
    SALES_DROP = "SALES_DROP"
    FORECAST_ERROR_ANOMALY = "FORECAST_ERROR_ANOMALY"


class SeverityLevel(str, Enum):
    """Deterministic severity classification based on statistical extremity."""
    LOW = "LOW"            # 2.0 <= |z| < 2.5 (Watchlist / Warning)
    MEDIUM = "MEDIUM"      # 2.5 <= |z| < 3.5 (Significant deviation)
    HIGH = "HIGH"          # 3.5 <= |z| < 5.0 (Severe deviation)
    CRITICAL = "CRITICAL"  # |z| >= 5.0 or extreme structural shift


class AnomalyDirection(str, Enum):
    """Direction of observed deviation relative to baseline center."""
    SURGE = "SURGE"                       # Value significantly above center
    DROP = "DROP"                         # Value significantly below center
    HIGHER_THAN_EXPECTED = "HIGHER_THAN_EXPECTED"  # Error significantly above expected error
    NORMAL = "NORMAL"                     # Within normal range


@dataclass
class AnomalyConfig:
    """Explicit configuration parameters for statistical anomaly detection.
    
    Attributes:
        window_size: Number of prior days in the rolling historical reference window (W).
        min_history: Minimum required historical observations before evaluating detection.
        surge_threshold: Robust z-score threshold for flagging surges (k_surge).
        drop_threshold: Robust z-score threshold for flagging drops (k_drop, positive number).
        error_threshold: Robust z-score threshold for forecast error anomalies.
        mad_scaling_constant: Normal consistency factor (1.4826 for standard normal).
        zero_mad_minimum_scale: Fallback minimum dispersion when MAD and IQR are zero.
        severity_low_z: Lower z-score boundary for LOW severity.
        severity_medium_z: Lower z-score boundary for MEDIUM severity.
        severity_high_z: Lower z-score boundary for HIGH severity.
        severity_critical_z: Lower z-score boundary for CRITICAL severity.
        handle_negative_sales: Policy for negative sales: 'FLAG', 'REJECT', or 'TREAT_AS_RETURN'.
    """
    window_size: int = 14
    min_history: int = 14
    surge_threshold: float = 2.5
    drop_threshold: float = 2.5
    error_threshold: float = 2.5
    mad_scaling_constant: float = 1.4826
    zero_mad_minimum_scale: float = 1.0
    severity_low_z: float = 2.0
    severity_medium_z: float = 2.5
    severity_high_z: float = 3.5
    severity_critical_z: float = 5.0
    handle_negative_sales: str = "TREAT_AS_RETURN"

    def __post_init__(self):
        if self.window_size < 3:
            raise ValueError(f"window_size must be >= 3, got {self.window_size}")
        if self.min_history < 3:
            raise ValueError(f"min_history must be >= 3, got {self.min_history}")
        if self.min_history > self.window_size:
            raise ValueError(
                f"min_history ({self.min_history}) cannot exceed window_size ({self.window_size})"
            )
        if self.surge_threshold <= 0:
            raise ValueError(f"surge_threshold must be positive, got {self.surge_threshold}")
        if self.drop_threshold <= 0:
            raise ValueError(f"drop_threshold must be positive, got {self.drop_threshold}")
        if self.error_threshold <= 0:
            raise ValueError(f"error_threshold must be positive, got {self.error_threshold}")


@dataclass
class AnomalyPoint:
    """Individual explainable anomaly detection record.
    
    Guarantees:
    - Never makes unproven causal claims in explanation.
    - Fully serializable to strict JSON without NaN or Infinity.
    """
    anomaly_id: str
    observation_date: str                   # ISO-8601 'YYYY-MM-DD'
    sales_scope: str                        # e.g. 'TOTAL_STORE', 'STORE_01', 'ITEM_XYZ'
    anomaly_type: AnomalyType
    severity: SeverityLevel
    direction: AnomalyDirection
    observed_value: float                   # Actual sales or observed forecast error
    reference_center: float                 # Rolling median
    reference_spread: float                 # Scale (1.4826 * MAD or IQR fallback)
    anomaly_score: float                    # Robust z-score
    threshold: float                        # Configured threshold applied
    explanation: str                        # Factual statistical explanation
    detection_timestamp: str                # ISO-8601 UTC
    
    # Optional forecast metadata (populated when anomaly_type == FORECAST_ERROR_ANOMALY)
    model_name: Optional[str] = None
    forecast_run_id: Optional[str] = None
    horizon_days: Optional[int] = None
    step_ahead: Optional[int] = None
    predicted_value: Optional[float] = None

    # Edge-case diagnostics
    dispersion_zero_fallback: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sanitized primitives safe for strict JSON."""
        d = asdict(self)
        d["anomaly_type"] = self.anomaly_type.value if isinstance(self.anomaly_type, AnomalyType) else str(self.anomaly_type)
        d["severity"] = self.severity.value if isinstance(self.severity, SeverityLevel) else str(self.severity)
        d["direction"] = self.direction.value if isinstance(self.direction, AnomalyDirection) else str(self.direction)
        return sanitize_for_json(d)

    def to_json(self) -> str:
        """Serialize to strict RFC 8259 JSON without NaN or Infinity."""
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyPoint":
        """Reconstruct AnomalyPoint from dictionary."""
        d = dict(data)
        if isinstance(d.get("anomaly_type"), str):
            d["anomaly_type"] = AnomalyType(d["anomaly_type"])
        if isinstance(d.get("severity"), str):
            d["severity"] = SeverityLevel(d["severity"])
        if isinstance(d.get("direction"), str):
            d["direction"] = AnomalyDirection(d["direction"])
        return cls(**d)


@dataclass
class AnomalyReport:
    """Comprehensive anomaly detection result report."""
    report_id: str
    target_scope: str
    total_evaluated: int
    anomalies_detected: int
    insufficient_history_count: int
    anomalies: List[AnomalyPoint] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for strict JSON."""
        return {
            "report_id": self.report_id,
            "target_scope": self.target_scope,
            "total_evaluated": self.total_evaluated,
            "anomalies_detected": self.anomalies_detected,
            "insufficient_history_count": self.insufficient_history_count,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "config": sanitize_for_json(self.config),
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        """Serialize to strict RFC 8259 JSON without NaN or Infinity."""
        return json.dumps(self.to_dict(), allow_nan=False)
