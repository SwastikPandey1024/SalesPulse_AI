"""
SalesPulse AI - Declarative Persistent Audit Models
Defines schemas for:
- Dataset validation audit logs (sanitized strict JSON reports, file SHA-256).
- Forecast run records (model identity, origin, horizon, summary).
- Forecast item records (target date, step, bounds, delayed actuals).
- Actual sales records (calendar date, sales scope, ground truth values).
"""

import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    Date,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from salespulse.storage.database import Base


class DatasetAuditLog(Base):
    """Audit record for tabular dataset ingestion and validation."""
    __tablename__ = "dataset_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_sha256 = Column(String(64), nullable=False, index=True)
    file_name = Column(String(255), nullable=True)  # Basename only, no paths
    detected_grain = Column(String(50), nullable=False, index=True)
    row_count = Column(Integer, nullable=False)
    validation_status = Column(String(20), nullable=False, index=True)  # PASSED, WARNING, FAILED
    is_usable = Column(Boolean, nullable=False, default=True)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    report_json = Column(Text, nullable=False)  # Strict RFC 8259 JSON
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index("idx_audit_sha_created", "file_sha256", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "file_sha256": self.file_sha256,
            "file_name": self.file_name,
            "detected_grain": self.detected_grain,
            "row_count": self.row_count,
            "validation_status": self.validation_status,
            "is_usable": self.is_usable,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "report_json": self.report_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ForecastRunLog(Base):
    """Audit record for a multi-step or single-step forecast execution."""
    __tablename__ = "forecast_run_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_artifact_path = Column(String(255), nullable=True)
    model_artifact_hash = Column(String(64), nullable=True)
    feature_contract_version = Column(String(50), nullable=False, default="v1_13features")
    forecast_origin = Column(DateTime, nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False)
    total_predicted_revenue = Column(Float, nullable=True)
    average_daily_sales = Column(Float, nullable=True)
    interval_method = Column(String(50), nullable=False, default="heuristic_sqrt_step")
    execution_source = Column(String(50), nullable=False, default="standalone_service")
    status = Column(String(20), nullable=False, default="SUCCESS")
    summary_json = Column(Text, nullable=True)  # Strict RFC 8259 JSON
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)

    items = relationship("ForecastItemLog", back_populates="run", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_artifact_path": self.model_artifact_path,
            "model_artifact_hash": self.model_artifact_hash,
            "feature_contract_version": self.feature_contract_version,
            "forecast_origin": self.forecast_origin.isoformat() if self.forecast_origin else None,
            "horizon_days": self.horizon_days,
            "total_predicted_revenue": self.total_predicted_revenue,
            "average_daily_sales": self.average_daily_sales,
            "interval_method": self.interval_method,
            "execution_source": self.execution_source,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ForecastItemLog(Base):
    """Granular prediction record per forecast time-step."""
    __tablename__ = "forecast_item_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("forecast_run_logs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    target_date = Column(Date, nullable=False, index=True)
    step_ahead = Column(Integer, nullable=False)
    predicted_value = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    
    # Delayed realization fields (initially NULL until actuals are reconciled)
    actual_value = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)
    interval_covered = Column(Boolean, nullable=True)
    reconciled_at = Column(DateTime, nullable=True)

    run = relationship("ForecastRunLog", back_populates="items")

    __table_args__ = (
        UniqueConstraint("run_id", "target_date", "sales_scope", name="uq_run_target_scope"),
        Index("idx_item_target_scope", "target_date", "sales_scope"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "sales_scope": self.sales_scope,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "step_ahead": self.step_ahead,
            "predicted_value": self.predicted_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "actual_value": self.actual_value,
            "absolute_error": self.absolute_error,
            "interval_covered": self.interval_covered,
            "reconciled_at": self.reconciled_at.isoformat() if self.reconciled_at else None,
        }


class ActualSalesRecord(Base):
    """Actual observed daily sales ground truth keyed by calendar date and scope."""
    __tablename__ = "actual_sales_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calendar_date = Column(Date, nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    daily_sales = Column(Float, nullable=False)
    source_file_sha256 = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)

    __table_args__ = (
        UniqueConstraint("calendar_date", "sales_scope", name="uq_actual_date_scope"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "calendar_date": self.calendar_date.isoformat() if self.calendar_date else None,
            "sales_scope": self.sales_scope,
            "daily_sales": self.daily_sales,
            "source_file_sha256": self.source_file_sha256,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AnomalyRecord(Base):
    """Persistent audit record for an explainable anomaly event."""
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    anomaly_id = Column(String(64), unique=True, nullable=False, index=True)
    observation_date = Column(Date, nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    anomaly_type = Column(String(50), nullable=False, index=True)  # SALES_SURGE, SALES_DROP, FORECAST_ERROR_ANOMALY
    severity = Column(String(20), nullable=False, index=True)      # LOW, MEDIUM, HIGH, CRITICAL
    direction = Column(String(30), nullable=False)
    observed_value = Column(Float, nullable=False)
    reference_center = Column(Float, nullable=False)
    reference_spread = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    explanation = Column(Text, nullable=False)
    detection_timestamp = Column(DateTime, nullable=False, index=True)

    # Optional forecast metadata
    model_name = Column(String(100), nullable=True, index=True)
    forecast_run_id = Column(String(64), nullable=True, index=True)
    horizon_days = Column(Integer, nullable=True)
    step_ahead = Column(Integer, nullable=True)
    predicted_value = Column(Float, nullable=True)
    dispersion_zero_fallback = Column(Boolean, nullable=False, default=False)
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)
    status_updated_at = Column(DateTime, nullable=True)
    raw_json = Column(Text, nullable=False)  # Strict RFC 8259 JSON payload
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_anomaly_date_scope", "observation_date", "sales_scope"),
        Index("idx_anomaly_type_sev", "anomaly_type", "severity"),
        Index("idx_anomaly_status", "status"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "anomaly_id": self.anomaly_id,
            "observation_date": self.observation_date.isoformat() if self.observation_date else None,
            "sales_scope": self.sales_scope,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "direction": self.direction,
            "observed_value": self.observed_value,
            "reference_center": self.reference_center,
            "reference_spread": self.reference_spread,
            "anomaly_score": self.anomaly_score,
            "threshold": self.threshold,
            "explanation": self.explanation,
            "detection_timestamp": self.detection_timestamp.isoformat() if self.detection_timestamp else None,
            "status": self.status,
            "status_updated_at": self.status_updated_at.isoformat() if self.status_updated_at else None,
            "model_name": self.model_name,
            "forecast_run_id": self.forecast_run_id,
            "horizon_days": self.horizon_days,
            "step_ahead": self.step_ahead,
            "predicted_value": self.predicted_value,
            "dispersion_zero_fallback": self.dispersion_zero_fallback,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DataMonitoringRecord(Base):
    """Persistent audit record for an ingestion and data freshness monitoring check."""
    __tablename__ = "data_monitoring_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    check_id = Column(String(64), unique=True, nullable=False, index=True)
    checked_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    status = Column(String(20), nullable=False, index=True)  # HEALTHY, WARNING, CRITICAL, UNKNOWN
    
    # Freshness signals
    last_successful_ingestion_at = Column(DateTime, nullable=True)
    last_ingestion_attempt_at = Column(DateTime, nullable=True)
    last_ingestion_status = Column(String(20), nullable=True)  # PASSED, WARNING, FAILED, NO_ATTEMPTS
    latest_observation_date = Column(Date, nullable=True)
    staleness_days = Column(Integer, nullable=True)
    staleness_threshold_days = Column(Integer, nullable=False, default=30)
    
    # Ingestion failure signals
    consecutive_failures = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    
    # Actionable diagnosis & evidence
    summary_message = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=False)  # Strict RFC 8259 JSON
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_mon_scope_checked", "sales_scope", "checked_at"),
        Index("idx_mon_status", "status"),
    )

    @property
    def freshness_state(self) -> str:
        """Canonical freshness state adapter for UI and evaluation consumers.

        Evaluates evidence_json first, then falls back safely to recorded signals.
        Preserves UNKNOWN, STALE, CRITICAL, and FRESH states.
        Never infers HEALTHY/FRESH merely because the field or evidence is missing.
        """
        # 1. Check evidence_json payload if present
        if self.evidence_json:
            try:
                import json
                ev = json.loads(self.evidence_json)
                if isinstance(ev, dict):
                    fresh = ev.get("freshness")
                    if isinstance(fresh, dict):
                        st_val = fresh.get("freshness_state") or fresh.get("state")
                        if st_val:
                            return str(st_val).upper()
            except Exception:
                pass

        # 2. Strict evaluation based on recorded observation date and staleness
        if self.latest_observation_date is None or self.staleness_days is None:
            return "UNKNOWN"

        thresh = self.staleness_threshold_days or 30
        if self.staleness_days > thresh:
            return "STALE"

        if self.status == "HEALTHY":
            return "FRESH"
        elif self.status == "WARNING":
            return "FRESH"
        elif self.status == "CRITICAL":
            # If critical status, check if staleness caused it
            if self.staleness_days > thresh:
                return "STALE"
            return "FAILED"

        return "UNKNOWN"

    @property
    def latest_ingestion_outcome(self) -> str:
        """Compatibility property providing the latest ingestion outcome string."""
        if self.last_ingestion_status:
            return str(self.last_ingestion_status)
        if self.evidence_json:
            try:
                import json
                ev = json.loads(self.evidence_json)
                if isinstance(ev, dict):
                    ing = ev.get("ingestion")
                    if isinstance(ing, dict):
                        out = ing.get("outcome")
                        if out:
                            return str(out)
            except Exception:
                pass
        return "UNKNOWN"

    def to_dict(self):
        return {
            "id": self.id,
            "check_id": self.check_id,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "sales_scope": self.sales_scope,
            "status": self.status,
            "freshness_state": self.freshness_state,
            "latest_ingestion_outcome": self.latest_ingestion_outcome,
            "last_successful_ingestion_at": self.last_successful_ingestion_at.isoformat() if self.last_successful_ingestion_at else None,
            "last_ingestion_attempt_at": self.last_ingestion_attempt_at.isoformat() if self.last_ingestion_attempt_at else None,
            "last_ingestion_status": self.last_ingestion_status,
            "latest_observation_date": self.latest_observation_date.isoformat() if self.latest_observation_date else None,
            "staleness_days": self.staleness_days,
            "staleness_threshold_days": self.staleness_threshold_days,
            "consecutive_failures": self.consecutive_failures,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "summary_message": self.summary_message,
            "evidence_json": self.evidence_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DriftBaselineRecord(Base):
    """Persistent audit record for a versioned reference baseline used in dataset drift monitoring."""
    __tablename__ = "drift_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    baseline_id = Column(String(64), unique=True, nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    data_window_start = Column(Date, nullable=True)
    data_window_end = Column(Date, nullable=True)
    row_count = Column(Integer, nullable=False, default=0)
    feature_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    source_file_name = Column(String(255), nullable=True)
    source_file_sha256 = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    profile_json = Column(Text, nullable=False)  # Strict RFC 8259 JSON of DriftBaselineProfile

    __table_args__ = (
        Index("idx_drift_base_scope_active", "sales_scope", "is_active"),
        Index("idx_drift_base_created", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "baseline_id": self.baseline_id,
            "sales_scope": self.sales_scope,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "data_window_start": self.data_window_start.isoformat() if self.data_window_start else None,
            "data_window_end": self.data_window_end.isoformat() if self.data_window_end else None,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "is_active": self.is_active,
            "source_file_name": self.source_file_name,
            "source_file_sha256": self.source_file_sha256,
            "notes": self.notes,
            "profile_json": self.profile_json,
        }


class DriftCheckRecord(Base):
    """Persistent audit record for a dataset drift evaluation execution."""
    __tablename__ = "drift_check_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    check_id = Column(String(64), unique=True, nullable=False, index=True)
    checked_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)
    sales_scope = Column(String(50), nullable=False, default="TOTAL_STORE", index=True)
    baseline_id = Column(String(64), nullable=False, index=True)
    overall_status = Column(String(20), nullable=False, index=True)  # NO_DRIFT, WARNING, DRIFT, UNKNOWN
    current_row_count = Column(Integer, nullable=False, default=0)
    baseline_row_count = Column(Integer, nullable=False, default=0)
    total_features_evaluated = Column(Integer, nullable=False, default=0)
    drifted_features_count = Column(Integer, nullable=False, default=0)
    warning_features_count = Column(Integer, nullable=False, default=0)
    summary_message = Column(Text, nullable=False)
    report_json = Column(Text, nullable=False)  # Strict RFC 8259 JSON of DriftReport
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)

    __table_args__ = (
        Index("idx_drift_check_scope_time", "sales_scope", "checked_at"),
        Index("idx_drift_check_status", "overall_status"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "check_id": self.check_id,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "sales_scope": self.sales_scope,
            "baseline_id": self.baseline_id,
            "overall_status": self.overall_status,
            "current_row_count": self.current_row_count,
            "baseline_row_count": self.baseline_row_count,
            "total_features_evaluated": self.total_features_evaluated,
            "drifted_features_count": self.drifted_features_count,
            "warning_features_count": self.warning_features_count,
            "summary_message": self.summary_message,
            "report_json": self.report_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }



