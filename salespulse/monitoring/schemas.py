"""
SalesPulse AI - Data Freshness & Ingestion Monitoring Schemas
Defines strictly typed, deterministic contracts for data observability signals:
- Freshness states, ingestion outcomes, and overall monitoring statuses.
- Monitoring configuration with configurable thresholds.
- Granular reports and overall monitoring verdicts.
- Strict RFC 8259 JSON serialization without raw customer data exposure.
"""

import json
import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict

from salespulse.data.contracts import sanitize_for_json


class MonitoringStatus(str, Enum):
    """Overall data health verdict based on deterministic evaluation rules."""
    HEALTHY = "HEALTHY"      # Pipeline fresh, recent ingestion passed, no critical issues
    WARNING = "WARNING"      # Minor staleness or non-blocking validation warnings
    CRITICAL = "CRITICAL"    # Severe staleness or recent/repeated ingestion failures
    UNKNOWN = "UNKNOWN"      # Database unavailable or no historical observations recorded


class FreshnessState(str, Enum):
    """Categorization of dataset calendar freshness."""
    FRESH = "FRESH"                # Latest observation within configured staleness threshold
    STALE = "STALE"                # Latest observation older than threshold
    MISSING_DATA = "MISSING_DATA"  # No observations found for the evaluated scope
    UNKNOWN = "UNKNOWN"            # Freshness cannot be evaluated (e.g. database offline)


class IngestionOutcome(str, Enum):
    """Verdict of the most recent ingestion attempt."""
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    NO_ATTEMPTS = "NO_ATTEMPTS"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"


@dataclass
class MonitoringConfig:
    """Configurable thresholds and reference settings for monitoring evaluations.
    
    Attributes:
        staleness_threshold_days: Calendar days after which data is considered STALE (default: 30).
        critical_staleness_multiplier: Multiple of staleness threshold triggering CRITICAL (default: 2.0).
        consecutive_failure_threshold: Consecutive failed attempts triggering CRITICAL (default: 2).
        reference_date: Reference calendar date for staleness evaluation (default: UTC today).
        sales_scope: Scope identifier being monitored (default: 'TOTAL_STORE').
        lookback_attempts: Number of recent audit logs inspected for failure runs (default: 20).
    """
    staleness_threshold_days: int = 30
    critical_staleness_multiplier: float = 2.0
    consecutive_failure_threshold: int = 2
    reference_date: Optional[datetime.date] = None
    sales_scope: str = "TOTAL_STORE"
    lookback_attempts: int = 20

    def __post_init__(self):
        if self.staleness_threshold_days < 1:
            raise ValueError(f"staleness_threshold_days must be >= 1, got {self.staleness_threshold_days}")
        if self.critical_staleness_multiplier < 1.0:
            raise ValueError(f"critical_staleness_multiplier must be >= 1.0, got {self.critical_staleness_multiplier}")
        if self.consecutive_failure_threshold < 1:
            raise ValueError(f"consecutive_failure_threshold must be >= 1, got {self.consecutive_failure_threshold}")
        if self.lookback_attempts < 1:
            raise ValueError(f"lookback_attempts must be >= 1, got {self.lookback_attempts}")


@dataclass
class DataFreshnessReport:
    """Detailed evaluation of calendar data freshness."""
    freshness_state: FreshnessState
    latest_observation_date: Optional[str] = None     # YYYY-MM-DD
    staleness_days: Optional[int] = None
    staleness_threshold_days: int = 30
    is_stale: bool = False
    reference_date: str = ""                         # YYYY-MM-DD
    assessment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freshness_state": self.freshness_state.value if hasattr(self.freshness_state, "value") else str(self.freshness_state),
            "latest_observation_date": self.latest_observation_date,
            "staleness_days": self.staleness_days,
            "staleness_threshold_days": self.staleness_threshold_days,
            "is_stale": self.is_stale,
            "reference_date": self.reference_date,
            "assessment": self.assessment,
        }


@dataclass
class IngestionHealthReport:
    """Detailed evaluation of ingestion and validation pipeline reliability."""
    outcome: IngestionOutcome
    latest_attempt_at: Optional[str] = None          # ISO-8601 UTC
    last_successful_ingestion_at: Optional[str] = None # ISO-8601 UTC
    consecutive_failures: int = 0
    recent_errors_count: int = 0
    recent_warnings_count: int = 0
    latest_file_name: Optional[str] = None
    latest_file_sha256: Optional[str] = None
    latest_issues_summary: List[Dict[str, Any]] = field(default_factory=list)
    assessment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value if hasattr(self.outcome, "value") else str(self.outcome),
            "latest_attempt_at": self.latest_attempt_at,
            "last_successful_ingestion_at": self.last_successful_ingestion_at,
            "consecutive_failures": self.consecutive_failures,
            "recent_errors_count": self.recent_errors_count,
            "recent_warnings_count": self.recent_warnings_count,
            "latest_file_name": self.latest_file_name,
            "latest_file_sha256": self.latest_file_sha256,
            "latest_issues_summary": self.latest_issues_summary,
            "assessment": self.assessment,
        }


@dataclass
class MonitoringVerdict:
    """Comprehensive, deterministic data observability check verdict."""
    check_id: str
    checked_at: str                                  # ISO-8601 UTC
    sales_scope: str
    overall_status: MonitoringStatus
    freshness: DataFreshnessReport
    ingestion: IngestionHealthReport
    actionable_remediations: List[str] = field(default_factory=list)
    summary_message: str = ""
    database_connected: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "checked_at": self.checked_at,
            "sales_scope": self.sales_scope,
            "overall_status": self.overall_status.value if hasattr(self.overall_status, "value") else str(self.overall_status),
            "freshness": self.freshness.to_dict(),
            "ingestion": self.ingestion.to_dict(),
            "actionable_remediations": self.actionable_remediations,
            "summary_message": self.summary_message,
            "database_connected": self.database_connected,
        }

    def to_json(self, indent: int = 2) -> str:
        sanitized = sanitize_for_json(self.to_dict())
        return json.dumps(sanitized, indent=indent, allow_nan=False)
