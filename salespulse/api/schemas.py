"""
SalesPulse AI - FastAPI Request and Response Schemas
Defines strictly typed Pydantic models for:
- Health and readiness endpoints.
- Dataset validation (file upload and JSON).
- Forecast generation and run history retrieval.
- Anomaly detection requests, reports, and persisted record queries.
- Uniform, security-conscious API error responses.
"""

from typing import List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel, Field, ConfigDict


# ----------------------------------------------------------------------------
# 1. Error Response Schemas
# ----------------------------------------------------------------------------

class APIErrorResponse(BaseModel):
    """Uniform, security-conscious error response. Never leaks file paths or stack traces."""
    model_config = ConfigDict(extra="forbid")

    error_code: str = Field(..., description="Stable programmatic error identifier")
    detail: str = Field(..., description="Human-readable safe explanation")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the error event")
    path: Optional[str] = Field(None, description="API route path where error occurred")


# ----------------------------------------------------------------------------
# 2. Health & Readiness Schemas
# ----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field("ok", description="Liveness status indicator")
    timestamp: str = Field(..., description="Current UTC timestamp")
    service: str = Field("salespulse-api", description="Service identifier")
    version: str = Field("1.0.0", description="API semantic version")


class ReadinessResponse(BaseModel):
    """Readiness probe response indicating infrastructure and model availability."""
    status: str = Field(..., description="'ready' or 'not_ready'")
    database: str = Field(..., description="Database connection health")
    schema_version: int = Field(..., description="Active database schema version")
    supported_models: List[str] = Field(..., description="Candidate models available in registry")
    timestamp: str = Field(..., description="Current UTC timestamp")


# ----------------------------------------------------------------------------
# 3. Data Validation Schemas
# ----------------------------------------------------------------------------

class ValidationIssueSchema(BaseModel):
    """Structured report item for data contract issues."""
    code: str
    severity: str
    message: str
    affected_count: int = 0
    sample_rows: Optional[List[int]] = None
    remediation: str = ""


class DataValidationResponse(BaseModel):
    """Sanitized validation verdict conforming to existing data contracts."""
    verdict: str = Field(..., description="'PASSED', 'WARNING', or 'FAILED'")
    grain: str = Field(..., description="Detected dataset grain")
    total_rows_read: int
    valid_rows_count: int
    quarantined_rows_count: int
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    issues: List[ValidationIssueSchema] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# 4. Forecast Schemas
# ----------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    """Multi-step time-series recursive forecast request payload."""
    model_config = ConfigDict(extra="forbid")

    base_date: str = Field(
        ...,
        description="Forecast origin date in YYYY-MM-DD format",
        examples=["2025-01-20"],
    )
    recent_history_sales: List[float] = Field(
        ...,
        min_length=30,
        description="At least 30 consecutive daily historical sales observations (oldest to newest)",
    )
    horizon_days: int = Field(
        14,
        ge=1,
        le=90,
        description="Forecast horizon in calendar days (1 to 90)",
    )
    model_choice: str = Field(
        "RandomForest",
        description="Candidate model name (e.g. 'RandomForest', 'LinearRegression', 'XGBoost')",
    )
    interval_method: str = Field(
        "heuristic_sqrt_step",
        description="Prediction interval method ('heuristic_sqrt_step' or 'calibrated_oof_residual')",
    )
    sales_scope: str = Field(
        "TOTAL_STORE",
        max_length=50,
        description="Business sales scope identifier",
    )
    persist_run: bool = Field(
        False,
        description="Whether to record this execution to the persistent audit database",
    )


class ForecastItemSchema(BaseModel):
    """Individual step-ahead prediction with uncertainty bounds."""
    target_date: str
    step_ahead: int
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    actual_value: Optional[float] = None
    absolute_error: Optional[float] = None
    interval_covered: Optional[bool] = None


class ForecastResponse(BaseModel):
    """Complete multi-step forecast response."""
    run_id: Optional[str] = None
    model_name: str
    forecast_origin: str
    horizon_days: int
    sales_scope: str
    interval_method: str
    total_predicted_revenue: float
    average_daily_sales: float
    items: List[ForecastItemSchema]


# ----------------------------------------------------------------------------
# 5. Anomaly Detection Schemas
# ----------------------------------------------------------------------------

class AnomalyPointInput(BaseModel):
    """Single observation point for in-memory series anomaly detection."""
    date: str = Field(..., description="Observation calendar date (YYYY-MM-DD)")
    sales: float = Field(..., description="Observed daily sales revenue")


class AnomalyDetectionRequest(BaseModel):
    """Trigger payload for time-series anomaly detection."""
    model_config = ConfigDict(extra="forbid")

    sales_scope: str = Field("TOTAL_STORE", max_length=50, description="Target sales scope")
    series: Optional[List[AnomalyPointInput]] = Field(
        None,
        max_length=10000,
        description="Optional in-memory daily sales points (up to 10,000 points). If omitted, evaluates database actuals.",
    )
    window_size: int = Field(14, ge=3, le=90, description="Rolling historical window size in days (W)")
    min_history: int = Field(14, ge=3, le=90, description="Minimum prior history required for alert")
    surge_threshold: float = Field(2.5, gt=0, description="Robust z-score surge cutoff")
    drop_threshold: float = Field(2.5, gt=0, description="Robust z-score drop cutoff")
    persist: bool = Field(False, description="Whether to persist detected anomalies into audit storage")
    fill_missing_dates: bool = Field(False, description="Whether to reindex calendar gaps with 0.0")
    handle_negative_sales: str = Field("TREAT_AS_RETURN", description="'TREAT_AS_RETURN' or 'REJECT'")
    handle_missing_values: str = Field("SKIP", description="'SKIP' (exclude NaN) or 'ZERO_FILL'")


class AnomalyPointSchema(BaseModel):
    """Structured, explainable anomaly event record."""
    anomaly_id: str
    observation_date: str
    sales_scope: str
    anomaly_type: str
    severity: str
    direction: str
    observed_value: float
    reference_center: float
    reference_spread: float
    anomaly_score: float
    threshold: float
    explanation: str
    detection_timestamp: str
    status: str
    status_updated_at: Optional[str] = None
    model_name: Optional[str] = None
    forecast_run_id: Optional[str] = None
    horizon_days: Optional[int] = None
    step_ahead: Optional[int] = None
    predicted_value: Optional[float] = None
    dispersion_zero_fallback: bool = False
    notes: Optional[str] = None


class AnomalyDetectionResponse(BaseModel):
    """Response payload for anomaly detection runs."""
    report_id: str
    target_scope: str
    total_evaluated: int
    anomalies_detected: int
    insufficient_history_count: int
    missing_values_count: int
    anomalies: List[AnomalyPointSchema]
    config: Dict[str, Any]
    seasonality_limitation_notice: str = (
        "Notice: The anomaly engine employs an unadjusted 14-day rolling median baseline. "
        "Intra-week recurring surges (such as weekend retail traffic) may exhibit elevated "
        "z-scores (~1.8). For high-amplitude weekly seasonality, caller should deseasonalize "
        "or apply day-of-week stratification."
    )
