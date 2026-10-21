"""
SalesPulse AI - Data Freshness & Ingestion Monitoring Service
Provides:
- Modular, deterministic monitoring of dataset freshness and ingestion health.
- Evaluation of staleness thresholds, consecutive validation failures, and data quality review flags.
- Strict transition rules distinguishing HEALTHY, WARNING, CRITICAL, and UNKNOWN.
- Defensive error handling: database offline states are never reported as fresh or healthy.
- Idempotent persistence to AuditRepository without customer PII retention.
- Standalone execution boundary ready for future scheduler/worker integration.
"""

import sys
import uuid
import datetime
import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.engine import Engine

from salespulse.monitoring.schemas import (
    MonitoringStatus,
    FreshnessState,
    IngestionOutcome,
    MonitoringConfig,
    DataFreshnessReport,
    IngestionHealthReport,
    MonitoringVerdict,
)
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import DataMonitoringRecord
from salespulse.storage.database import get_engine

logger = logging.getLogger("SalesPulse_Monitoring")


class DataMonitoringService:
    """Domain service for automated data freshness, contract compliance, and ingestion monitoring."""

    def __init__(
        self,
        repository: Optional[AuditRepository] = None,
        engine: Optional[Engine] = None,
        auto_init_engine: bool = True,
    ):
        if engine is not None:
            self.engine = engine
        elif auto_init_engine:
            try:
                self.engine = get_engine()
            except Exception as e:
                logger.warning(f"Could not initialize database engine for monitoring: {e}")
                self.engine = None
        else:
            self.engine = None

        if repository is not None:
            self.repository = repository
        elif self.engine is not None:
            self.repository = AuditRepository(self.engine)
        else:
            self.repository = None


    # ------------------------------------------------------------------------
    # 1. Core Health Evaluation Engine
    # ------------------------------------------------------------------------
    def check_data_health(
        self,
        sales_scope: str = "TOTAL_STORE",
        config: Optional[MonitoringConfig] = None,
        persist: bool = True,
        check_id: Optional[str] = None,
    ) -> MonitoringVerdict:
        """Execute a comprehensive, deterministic data freshness and ingestion health check.
        
        Args:
            sales_scope: Identifier for the sales domain or store being evaluated.
            config: Configurable thresholds (defaults to MonitoringConfig()).
            persist: Whether to store the monitoring verdict into persistent audit storage.
            check_id: Optional unique check ID (generated if not provided).
            
        Returns:
            MonitoringVerdict with typed freshness and reliability sub-reports.
        """
        cfg = config or MonitoringConfig(sales_scope=sales_scope)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        ref_date = cfg.reference_date or datetime.date.today()
        cid = check_id or str(uuid.uuid4())

        # Defensive Guard: Database Unavailable State
        if self.repository is None or self.engine is None:
            return self._build_database_unavailable_verdict(
                check_id=cid,
                checked_at=now_utc,
                sales_scope=sales_scope,
                config=cfg,
                ref_date=ref_date,
            )

        try:
            stats = self.repository.get_ingestion_summary_stats(
                sales_scope=sales_scope,
                lookback_attempts=cfg.lookback_attempts,
            )
        except Exception as db_err:
            logger.error(f"Database error while querying ingestion stats: {db_err}")
            return self._build_database_unavailable_verdict(
                check_id=cid,
                checked_at=now_utc,
                sales_scope=sales_scope,
                config=cfg,
                ref_date=ref_date,
                error_detail=str(db_err),
            )

        # 1. Freshness Evaluation
        freshness = self._evaluate_freshness(stats, cfg, ref_date)

        # 2. Ingestion Health Evaluation
        ingestion = self._evaluate_ingestion_health(stats, cfg)

        # 3. Apply Deterministic Status Transition Rules
        overall_status, remediations, summary_msg = self._apply_transition_rules(
            freshness=freshness,
            ingestion=ingestion,
            config=cfg,
        )

        verdict = MonitoringVerdict(
            check_id=cid,
            checked_at=now_utc.isoformat(),
            sales_scope=sales_scope,
            overall_status=overall_status,
            freshness=freshness,
            ingestion=ingestion,
            actionable_remediations=remediations,
            summary_message=summary_msg,
            database_connected=True,
        )

        # 4. Optional Idempotent Persistence
        if persist:
            try:
                last_succ_dt = (
                    stats["last_successful_attempt"].created_at
                    if stats.get("last_successful_attempt")
                    else None
                )
                last_att_dt = (
                    stats["latest_attempt"].created_at
                    if stats.get("latest_attempt")
                    else None
                )
                last_att_status = (
                    stats["latest_attempt"].validation_status
                    if stats.get("latest_attempt")
                    else None
                )

                self.repository.record_monitoring_check(
                    check_id=cid,
                    status=overall_status.value,
                    sales_scope=sales_scope,
                    checked_at=now_utc,
                    last_successful_ingestion_at=last_succ_dt,
                    last_ingestion_attempt_at=last_att_dt,
                    last_ingestion_status=last_att_status,
                    latest_observation_date=freshness.latest_observation_date,
                    staleness_days=freshness.staleness_days,
                    staleness_threshold_days=cfg.staleness_threshold_days,
                    consecutive_failures=ingestion.consecutive_failures,
                    error_count=ingestion.recent_errors_count,
                    warning_count=ingestion.recent_warnings_count,
                    summary_message=summary_msg,
                    evidence=verdict.to_dict(),
                )
            except Exception as p_err:
                logger.error(f"Failed to persist monitoring check log: {p_err}")

        return verdict

    # ------------------------------------------------------------------------
    # 2. Sub-component Evaluators
    # ------------------------------------------------------------------------
    def _evaluate_freshness(
        self,
        stats: Dict[str, Any],
        config: MonitoringConfig,
        ref_date: datetime.date,
    ) -> DataFreshnessReport:
        """Compute calendar freshness from latest observation date."""
        latest_obs = stats.get("latest_observation_date")
        if latest_obs is None:
            return DataFreshnessReport(
                freshness_state=FreshnessState.MISSING_DATA,
                latest_observation_date=None,
                staleness_days=None,
                staleness_threshold_days=config.staleness_threshold_days,
                is_stale=False,
                reference_date=ref_date.isoformat(),
                assessment="No historical observation records exist for this scope.",
            )

        if isinstance(latest_obs, str):
            obs_dt = datetime.date.fromisoformat(latest_obs)
        elif isinstance(latest_obs, datetime.datetime):
            obs_dt = latest_obs.date()
        else:
            obs_dt = latest_obs

        # Calculate calendar staleness
        delta_days = max(0, (ref_date - obs_dt).days)
        is_stale = delta_days > config.staleness_threshold_days
        state = FreshnessState.STALE if is_stale else FreshnessState.FRESH

        assessment = (
            f"Observations are {delta_days} days old relative to {ref_date}. "
            f"{'EXCEEDS' if is_stale else 'Within'} threshold of {config.staleness_threshold_days} days."
        )

        return DataFreshnessReport(
            freshness_state=state,
            latest_observation_date=obs_dt.isoformat(),
            staleness_days=delta_days,
            staleness_threshold_days=config.staleness_threshold_days,
            is_stale=is_stale,
            reference_date=ref_date.isoformat(),
            assessment=assessment,
        )

    def _evaluate_ingestion_health(
        self,
        stats: Dict[str, Any],
        config: MonitoringConfig,
    ) -> IngestionHealthReport:
        """Evaluate validation outcomes and failure runs from audit logs."""
        latest_attempt = stats.get("latest_attempt")
        last_success = stats.get("last_successful_attempt")
        consecutive_failures = stats.get("consecutive_failures", 0)

        if latest_attempt is None:
            return IngestionHealthReport(
                outcome=IngestionOutcome.NO_ATTEMPTS,
                latest_attempt_at=None,
                last_successful_ingestion_at=None,
                consecutive_failures=0,
                recent_errors_count=0,
                recent_warnings_count=0,
                assessment="No dataset ingestion attempts have been recorded yet.",
            )

        # Parse outcome
        v_status = latest_attempt.validation_status
        if v_status == "PASSED":
            outcome = IngestionOutcome.PASSED
        elif v_status == "WARNING":
            outcome = IngestionOutcome.WARNING
        else:
            outcome = IngestionOutcome.FAILED

        # Extract issue codes from report_json safely
        issues_summary = []
        if latest_attempt.report_json:
            try:
                import json
                r_dict = json.loads(latest_attempt.report_json)
                for iss in r_dict.get("issues", [])[:5]:
                    issues_summary.append({
                        "code": iss.get("code"),
                        "severity": iss.get("severity"),
                        "message": iss.get("message"),
                    })
            except Exception:
                pass

        assessment = (
            f"Most recent ingestion outcome: {outcome.value}. "
            f"Consecutive failures: {consecutive_failures}. "
            f"Audit log ID: {latest_attempt.id} ({latest_attempt.file_name or 'unnamed'})."
        )

        return IngestionHealthReport(
            outcome=outcome,
            latest_attempt_at=latest_attempt.created_at.isoformat() if latest_attempt.created_at else None,
            last_successful_ingestion_at=last_success.created_at.isoformat() if last_success and last_success.created_at else None,
            consecutive_failures=consecutive_failures,
            recent_errors_count=latest_attempt.error_count,
            recent_warnings_count=latest_attempt.warning_count,
            latest_file_name=latest_attempt.file_name,
            latest_file_sha256=latest_attempt.file_sha256,
            latest_issues_summary=issues_summary,
            assessment=assessment,
        )

    # ------------------------------------------------------------------------
    # 3. Transition Rules Engine
    # ------------------------------------------------------------------------
    def _apply_transition_rules(
        self,
        freshness: DataFreshnessReport,
        ingestion: IngestionHealthReport,
        config: MonitoringConfig,
    ) -> Tuple[MonitoringStatus, List[str], str]:
        """Apply deterministic, trustworthy rules to decide overall health status."""
        remediations: List[str] = []

        # State 0: Missing or uninitialized baseline
        if (
            freshness.freshness_state == FreshnessState.MISSING_DATA
            and ingestion.outcome == IngestionOutcome.NO_ATTEMPTS
        ):
            remediations.append("Upload and ingest an initial sales dataset to establish operational baseline.")
            return (
                MonitoringStatus.UNKNOWN,
                remediations,
                "No ingestion history or observation records exist in storage.",
            )

        # State 1: CRITICAL triggers
        # - Consecutive failures >= threshold (e.g. 2+ failed attempts in a row)
        # - Most recent attempt failed with critical contract violations
        # - Severe data staleness (> critical_staleness_multiplier * threshold, e.g. > 60 days)
        crit_staleness = int(config.staleness_threshold_days * config.critical_staleness_multiplier)

        if ingestion.consecutive_failures >= config.consecutive_failure_threshold:
            remediations.append(
                f"Triage recurring pipeline errors: {ingestion.consecutive_failures} consecutive ingestion attempts have failed."
            )
            remediations.append("Verify CSV/Excel schema against required column definitions in Data Quality.")
            return (
                MonitoringStatus.CRITICAL,
                remediations,
                f"Critical Ingestion Alert: {ingestion.consecutive_failures} consecutive ingestion attempts failed.",
            )

        if ingestion.outcome == IngestionOutcome.FAILED:
            remediations.append("Review Data Quality findings and remediate blocking contract violations.")
            remediations.append("Ensure numeric sales columns contain valid values and dates are properly formatted.")
            return (
                MonitoringStatus.CRITICAL,
                remediations,
                "Critical Ingestion Alert: Most recent dataset upload failed validation and was rejected.",
            )

        if freshness.staleness_days is not None and freshness.staleness_days > crit_staleness:
            remediations.append(
                f"Ingest fresh data immediately. Dataset is {freshness.staleness_days} days stale (critical threshold: {crit_staleness} days)."
            )
            return (
                MonitoringStatus.CRITICAL,
                remediations,
                f"Critical Freshness Alert: Data is {freshness.staleness_days} days behind schedule.",
            )

        # State 2: WARNING triggers
        # - Stale data (staleness_threshold_days < staleness <= critical_staleness)
        # - Most recent ingestion passed with review flags/warnings
        # - Validation warnings present
        if freshness.is_stale:
            remediations.append(
                f"Dataset observations are {freshness.staleness_days} days old. Ingest recent sales data to maintain forecast calibration."
            )
            return (
                MonitoringStatus.WARNING,
                remediations,
                f"Data Freshness Warning: Observations are {freshness.staleness_days} days old (threshold: {config.staleness_threshold_days} days).",
            )

        if ingestion.outcome == IngestionOutcome.WARNING:
            remediations.append("Inspect non-blocking data quality warnings in the Data Quality workspace.")
            return (
                MonitoringStatus.WARNING,
                remediations,
                f"Data Quality Warning: Ingestion passed with {ingestion.recent_warnings_count} review flag(s).",
            )

        if ingestion.recent_warnings_count > 0:
            remediations.append("Review non-critical validation findings in historical ingestion audits.")
            return (
                MonitoringStatus.WARNING,
                remediations,
                f"Data Quality Warning: {ingestion.recent_warnings_count} quality review flag(s) active.",
            )

        # State 3: HEALTHY
        return (
            MonitoringStatus.HEALTHY,
            [],
            "Data pipeline is healthy: observations are fresh and the latest ingestion passed all contracts.",
        )

    def _build_database_unavailable_verdict(
        self,
        check_id: str,
        checked_at: datetime.datetime,
        sales_scope: str,
        config: MonitoringConfig,
        ref_date: datetime.date,
        error_detail: Optional[str] = None,
    ) -> MonitoringVerdict:
        """Produce an explicit, honest UNKNOWN verdict when database storage is unreachable."""
        freshness = DataFreshnessReport(
            freshness_state=FreshnessState.UNKNOWN,
            latest_observation_date=None,
            staleness_days=None,
            staleness_threshold_days=config.staleness_threshold_days,
            is_stale=False,
            reference_date=ref_date.isoformat(),
            assessment="Database connection is unavailable; observation dates cannot be determined.",
        )

        ingestion = IngestionHealthReport(
            outcome=IngestionOutcome.DATABASE_UNAVAILABLE,
            latest_attempt_at=None,
            last_successful_ingestion_at=None,
            consecutive_failures=0,
            recent_errors_count=0,
            recent_warnings_count=0,
            assessment=f"Database connection error: {error_detail or 'Engine uninitialized'}",
        )

        remed = [
            "Check SQLite storage readiness and filesystem permissions in System Health.",
            "Verify database connection parameters and run init_db().",
        ]

        return MonitoringVerdict(
            check_id=check_id,
            checked_at=checked_at.isoformat(),
            sales_scope=sales_scope,
            overall_status=MonitoringStatus.UNKNOWN,
            freshness=freshness,
            ingestion=ingestion,
            actionable_remediations=remed,
            summary_message="Database is unavailable; data freshness and ingestion health cannot be verified.",
            database_connected=False,
        )

    # ------------------------------------------------------------------------
    # 4. Historical Retrieval & Query
    # ------------------------------------------------------------------------
    def get_monitoring_history(
        self,
        sales_scope: Optional[str] = None,
        limit: int = 50,
    ) -> List[DataMonitoringRecord]:
        """Fetch historical monitoring check logs from persistent storage."""
        if not self.repository:
            return []
        try:
            return self.repository.get_monitoring_history(sales_scope=sales_scope, limit=limit)
        except Exception as e:
            logger.error(f"Failed to query monitoring history: {e}")
            return []

    def get_latest_check(
        self,
        sales_scope: str = "TOTAL_STORE",
    ) -> Optional[DataMonitoringRecord]:
        """Fetch the most recent persisted monitoring check for a scope."""
        if not self.repository:
            return None
        try:
            return self.repository.get_latest_monitoring_check(sales_scope=sales_scope)
        except Exception as e:
            logger.error(f"Failed to query latest monitoring check: {e}")
            return None


# ----------------------------------------------------------------------------
# 5. Scheduled Invocation Boundary
# ----------------------------------------------------------------------------
def run_scheduled_monitoring_check(
    sales_scope: str = "TOTAL_STORE",
    config: Optional[MonitoringConfig] = None,
) -> MonitoringVerdict:
    """Entry point for external schedulers (cron, Airflow, Windows Task Scheduler).
    
    Example Scheduled Invocation:
        # In a cron job or scheduled worker:
        from salespulse.monitoring.service import run_scheduled_monitoring_check
        verdict = run_scheduled_monitoring_check(sales_scope="TOTAL_STORE")
        if verdict.overall_status == MonitoringStatus.CRITICAL:
            alert_ops_team(verdict)
    """
    service = DataMonitoringService()
    verdict = service.check_data_health(sales_scope=sales_scope, config=config, persist=True)
    logger.info(
        f"[Scheduled Monitoring Check] Scope: {sales_scope} | "
        f"Status: {verdict.overall_status.value} | "
        f"Summary: {verdict.summary_message}"
    )
    return verdict


if __name__ == "__main__":
    # Command-line invocation for testing or cron jobs
    import argparse
    parser = argparse.ArgumentParser(description="SalesPulse AI Data Freshness & Ingestion Health Check")
    parser.add_argument("--scope", default="TOTAL_STORE", help="Sales scope identifier")
    parser.add_argument("--threshold", type=int, default=30, help="Staleness threshold in days")
    args = parser.parse_args()

    cfg = MonitoringConfig(sales_scope=args.scope, staleness_threshold_days=args.threshold)
    v = run_scheduled_monitoring_check(sales_scope=args.scope, config=cfg)
    print(v.to_json())
    sys.exit(0 if v.overall_status in (MonitoringStatus.HEALTHY, MonitoringStatus.WARNING) else 1)
