"""
SalesPulse AI - Audit & Persistence Repository
Provides:
- Idempotent dataset validation logging with SHA-256 deduplication and strict JSON.
- Forecast run and multi-step item persistence.
- Ground truth actuals ingestion and delayed realization reconciliation.
- Isolated, transactional execution with reliable resource cleanup.
"""

import os
import datetime
import math
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
from sqlalchemy import select, update, and_, func
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.engine import Engine

from salespulse.storage.database import get_engine, get_db_session
from salespulse.storage.models import DatasetAuditLog, ForecastRunLog, ForecastItemLog, ActualSalesRecord, AnomalyRecord
from salespulse.data.contracts import ValidationReport, sanitize_for_json


class AuditRepository:
    """Thread-safe persistence repository for audit trails and forecast realization."""

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or get_engine()

    # ------------------------------------------------------------------------
    # 1. Dataset Validation Audit Logging
    # ------------------------------------------------------------------------
    def record_validation(
        self,
        file_sha256: str,
        detected_grain: str,
        row_count: int,
        validation_status: str,
        is_usable: bool,
        error_count: int,
        warning_count: int,
        report: Union[ValidationReport, Dict[str, Any], str],
        file_name: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> DatasetAuditLog:
        """Record dataset validation outcome. Privacy-safe: never stores raw tabular rows."""
        # Sanitize file name to avoid leaking local system directory paths
        sanitized_filename = os.path.basename(file_name) if file_name else None

        # Format strict RFC 8259 JSON report
        if isinstance(report, ValidationReport):
            report_json = report.to_json()
        elif isinstance(report, dict):
            import json
            sanitized = sanitize_for_json(report)
            report_json = json.dumps(sanitized, allow_nan=False)
        elif isinstance(report, str):
            report_json = report
        else:
            report_json = "{}"

        def _execute(s: Session) -> DatasetAuditLog:
            entry = DatasetAuditLog(
                file_sha256=file_sha256,
                file_name=sanitized_filename,
                detected_grain=detected_grain,
                row_count=row_count,
                validation_status=validation_status,
                is_usable=is_usable,
                error_count=error_count,
                warning_count=warning_count,
                report_json=report_json,
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )
            s.add(entry)
            s.flush()
            return entry

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    def get_validation_by_hash(self, file_sha256: str, session: Optional[Session] = None) -> Optional[DatasetAuditLog]:
        """Lookup latest validation record by file SHA-256 fingerprint."""
        stmt = (
            select(DatasetAuditLog)
            .where(DatasetAuditLog.file_sha256 == file_sha256)
            .order_by(DatasetAuditLog.created_at.desc())
            .limit(1)
        )
        if session is not None:
            return session.execute(stmt).scalar_one_or_none()
        with get_db_session(self.engine) as s:
            return s.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------------
    # 2. Forecast Run & Item Persistence
    # ------------------------------------------------------------------------
    def record_forecast_run(
        self,
        run_id: str,
        model_name: str,
        origin_date: Union[datetime.date, datetime.datetime, str],
        horizon_days: int,
        items: List[Dict[str, Any]],
        model_artifact_path: Optional[str] = None,
        model_artifact_hash: Optional[str] = None,
        feature_contract_version: str = "v1_13features",
        interval_method: str = "heuristic_sqrt_step",
        execution_source: str = "standalone_service",
        status: str = "SUCCESS",
        sales_scope: str = "TOTAL_STORE",
        session: Optional[Session] = None,
    ) -> ForecastRunLog:
        """Persist a multi-step forecast run with constituent item steps."""
        if isinstance(origin_date, str):
            origin_dt = datetime.datetime.fromisoformat(origin_date)
        elif isinstance(origin_date, datetime.date) and not isinstance(origin_date, datetime.datetime):
            origin_dt = datetime.datetime.combine(origin_date, datetime.time.min)
        else:
            origin_dt = origin_date

        sanitized_path = os.path.basename(model_artifact_path) if model_artifact_path else None

        preds = [float(it["predicted_value"]) for it in items if "predicted_value" in it]
        total_rev = float(sum(preds)) if preds else 0.0
        avg_sales = float(np.mean(preds)) if preds else 0.0

        def _execute(s: Session) -> ForecastRunLog:
            run_log = ForecastRunLog(
                run_id=run_id,
                model_name=model_name,
                model_artifact_path=sanitized_path,
                model_artifact_hash=model_artifact_hash,
                feature_contract_version=feature_contract_version,
                forecast_origin=origin_dt,
                horizon_days=horizon_days,
                total_predicted_revenue=total_rev,
                average_daily_sales=avg_sales,
                interval_method=interval_method,
                execution_source=execution_source,
                status=status,
                created_at=datetime.datetime.now(datetime.timezone.utc),
            )
            s.add(run_log)
            s.flush()

            # Insert items
            for idx, item in enumerate(items, start=1):
                raw_target = item["target_date"]
                if isinstance(raw_target, str):
                    t_date = datetime.date.fromisoformat(raw_target)
                elif isinstance(raw_target, datetime.datetime):
                    t_date = raw_target.date()
                else:
                    t_date = raw_target

                item_scope = item.get("sales_scope", sales_scope)
                p_val = float(item["predicted_value"])
                l_bound = float(item["lower_bound"]) if item.get("lower_bound") is not None else None
                u_bound = float(item["upper_bound"]) if item.get("upper_bound") is not None else None

                item_log = ForecastItemLog(
                    run_id=run_id,
                    sales_scope=item_scope,
                    target_date=t_date,
                    step_ahead=item.get("step_ahead", idx),
                    predicted_value=p_val,
                    lower_bound=l_bound,
                    upper_bound=u_bound,
                )

                # Check if ground truth actual already exists in database
                actual_stmt = select(ActualSalesRecord).where(
                    and_(
                        ActualSalesRecord.calendar_date == t_date,
                        ActualSalesRecord.sales_scope == item_scope,
                    )
                )
                actual_rec = s.execute(actual_stmt).scalar_one_or_none()
                if actual_rec is not None:
                    act_val = float(actual_rec.daily_sales)
                    item_log.actual_value = act_val
                    item_log.absolute_error = abs(act_val - p_val)
                    if l_bound is not None and u_bound is not None:
                        item_log.interval_covered = (l_bound <= act_val <= u_bound)
                    item_log.reconciled_at = datetime.datetime.now(datetime.timezone.utc)

                s.add(item_log)

            s.flush()
            return run_log

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    def get_forecast_run(self, run_id: str, session: Optional[Session] = None) -> Optional[ForecastRunLog]:
        """Fetch forecast run by run_id with items eagerly loaded."""
        stmt = select(ForecastRunLog).options(selectinload(ForecastRunLog.items)).where(ForecastRunLog.run_id == run_id)
        if session is not None:
            return session.execute(stmt).scalar_one_or_none()
        with get_db_session(self.engine) as s:
            return s.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------------
    # 3. Delayed Actuals Ingestion & Reconciliation
    # ------------------------------------------------------------------------
    def ingest_actual_sales(
        self,
        calendar_date: Union[datetime.date, str],
        daily_sales: float,
        sales_scope: str = "TOTAL_STORE",
        source_file_sha256: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Tuple[ActualSalesRecord, bool]:
        """Idempotently ingest ground truth actual daily sales and reconcile matching items.
        
        Returns:
            Tuple[ActualSalesRecord, bool]: (record, was_created)
        """
        c_date = datetime.date.fromisoformat(calendar_date) if isinstance(calendar_date, str) else calendar_date
        val = float(daily_sales)

        def _execute(s: Session) -> Tuple[ActualSalesRecord, bool]:
            stmt = select(ActualSalesRecord).where(
                and_(
                    ActualSalesRecord.calendar_date == c_date,
                    ActualSalesRecord.sales_scope == sales_scope,
                )
            )
            existing = s.execute(stmt).scalar_one_or_none()
            was_created = False

            if existing is None:
                record = ActualSalesRecord(
                    calendar_date=c_date,
                    sales_scope=sales_scope,
                    daily_sales=val,
                    source_file_sha256=source_file_sha256,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    updated_at=datetime.datetime.now(datetime.timezone.utc),
                )
                s.add(record)
                was_created = True
            else:
                existing.daily_sales = val
                existing.source_file_sha256 = source_file_sha256
                existing.updated_at = datetime.datetime.now(datetime.timezone.utc)
                record = existing

            s.flush()

            # Reconcile all forecast items matching this date and scope
            self._reconcile_items_for_date_in_session(s, c_date, sales_scope, val)
            return record, was_created

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    def _reconcile_items_for_date_in_session(
        self, s: Session, calendar_date: datetime.date, sales_scope: str, actual_sales: float
    ) -> int:
        """Helper to reconcile forecast items matching calendar_date and sales_scope."""
        stmt = select(ForecastItemLog).where(
            and_(
                ForecastItemLog.target_date == calendar_date,
                ForecastItemLog.sales_scope == sales_scope,
            )
        )
        items = s.execute(stmt).scalars().all()
        now = datetime.datetime.now(datetime.timezone.utc)
        count = 0

        for item in items:
            item.actual_value = actual_sales
            item.absolute_error = abs(actual_sales - item.predicted_value)
            if item.lower_bound is not None and item.upper_bound is not None:
                item.interval_covered = (item.lower_bound <= actual_sales <= item.upper_bound)
            item.reconciled_at = now
            count += 1

        s.flush()
        return count

    def reconcile_actuals_for_date(
        self,
        calendar_date: Union[datetime.date, str],
        actual_sales: float,
        sales_scope: str = "TOTAL_STORE",
        session: Optional[Session] = None,
    ) -> int:
        """Reconcile forecast items targeting calendar_date."""
        c_date = datetime.date.fromisoformat(calendar_date) if isinstance(calendar_date, str) else calendar_date
        val = float(actual_sales)

        if session is not None:
            return self._reconcile_items_for_date_in_session(session, c_date, sales_scope, val)
        with get_db_session(self.engine) as s:
            return self._reconcile_items_for_date_in_session(s, c_date, sales_scope, val)

    def reconcile_all_pending(self, sales_scope: str = "TOTAL_STORE", session: Optional[Session] = None) -> int:
        """Scan all unreconciled items and match against existing ActualSalesRecords."""
        def _execute(s: Session) -> int:
            stmt = (
                select(ForecastItemLog, ActualSalesRecord.daily_sales)
                .join(
                    ActualSalesRecord,
                    and_(
                        ForecastItemLog.target_date == ActualSalesRecord.calendar_date,
                        ForecastItemLog.sales_scope == ActualSalesRecord.sales_scope,
                    ),
                )
                .where(
                    and_(
                        ForecastItemLog.sales_scope == sales_scope,
                        ForecastItemLog.actual_value.is_(None),
                    )
                )
            )
            matches = s.execute(stmt).all()
            now = datetime.datetime.now(datetime.timezone.utc)
            count = 0
            for item, actual_val in matches:
                act = float(actual_val)
                item.actual_value = act
                item.absolute_error = abs(act - item.predicted_value)
                if item.lower_bound is not None and item.upper_bound is not None:
                    item.interval_covered = (item.lower_bound <= act <= item.upper_bound)
                item.reconciled_at = now
                count += 1
            s.flush()
            return count

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    # ------------------------------------------------------------------------
    # 4. Evaluation Metrics on Reconciled Forecasts
    # ------------------------------------------------------------------------
    def get_forecast_evaluation_summary(
        self,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
        sales_scope: str = "TOTAL_STORE",
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Compute realized out-of-sample evaluation metrics on reconciled forecast items.
        
        Guarantees:
        - Never returns false 0.0 values on empty sample.
        - Preserves model name and horizon filters.
        """
        def _execute(s: Session) -> Dict[str, Any]:
            query = (
                select(
                    ForecastItemLog.predicted_value,
                    ForecastItemLog.actual_value,
                    ForecastItemLog.absolute_error,
                    ForecastItemLog.interval_covered,
                )
                .join(ForecastRunLog, ForecastItemLog.run_id == ForecastRunLog.run_id)
                .where(
                    and_(
                        ForecastItemLog.sales_scope == sales_scope,
                        ForecastItemLog.actual_value.is_not(None),
                    )
                )
            )
            if model_name:
                query = query.where(ForecastRunLog.model_name == model_name)
            if horizon_days:
                query = query.where(ForecastRunLog.horizon_days == horizon_days)

            records = s.execute(query).all()
            if not records:
                return {
                    "evaluated_count": 0,
                    "model_name": model_name,
                    "horizon_days": horizon_days,
                    "sales_scope": sales_scope,
                    "mae": None,
                    "rmse": None,
                    "coverage_rate": None,
                }

            abs_errors = [r[2] for r in records if r[2] is not None]
            sq_errors = [(r[1] - r[0]) ** 2 for r in records if r[0] is not None and r[1] is not None]
            covered = [r[3] for r in records if r[3] is not None]

            mae = float(np.mean(abs_errors)) if abs_errors else None
            rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else None
            coverage_rate = float(np.mean(covered)) if covered else None

            return {
                "evaluated_count": len(records),
                "model_name": model_name,
                "horizon_days": horizon_days,
                "sales_scope": sales_scope,
                "mae": mae,
                "rmse": rmse,
                "coverage_rate": coverage_rate,
            }

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    # ------------------------------------------------------------------------
    # 5. Anomaly Detection Event Persistence & Query
    # ------------------------------------------------------------------------
    def record_anomalies(
        self,
        anomalies: Any,
        session: Optional[Session] = None,
    ) -> List[AnomalyRecord]:
        """Persist anomaly points idempotently.
        
        Guarantees:
        - Traceable timestamps and scope.
        - Idempotent upsert by anomaly_id: rerunning detection updates existing records
          without creating duplicate rows.
        - Strictly validates numerical stability and JSON serialization.
        """
        # Accept AnomalyReport or raw list of AnomalyPoint / dicts
        from salespulse.analytics.schemas import AnomalyReport, AnomalyPoint

        point_list: List[AnomalyPoint] = []
        if isinstance(anomalies, AnomalyReport):
            point_list = anomalies.anomalies
        elif isinstance(anomalies, list):
            for item in anomalies:
                if isinstance(item, AnomalyPoint):
                    point_list.append(item)
                elif isinstance(item, dict):
                    point_list.append(AnomalyPoint.from_dict(item))
        elif isinstance(anomalies, AnomalyPoint):
            point_list = [anomalies]

        if not point_list:
            return []

        def _execute(s: Session) -> List[AnomalyRecord]:
            persisted: List[AnomalyRecord] = []
            for p in point_list:
                # Parse dates
                if isinstance(p.observation_date, datetime.date):
                    obs_date = p.observation_date
                else:
                    obs_date = datetime.date.fromisoformat(str(p.observation_date))

                try:
                    det_time = datetime.datetime.fromisoformat(str(p.detection_timestamp))
                except Exception:
                    det_time = datetime.datetime.now(datetime.timezone.utc)

                raw_json = p.to_json()

                # Idempotent lookup
                existing = s.execute(
                    select(AnomalyRecord).where(AnomalyRecord.anomaly_id == p.anomaly_id)
                ).scalar_one_or_none()

                if existing:
                    # Update mutable attributes
                    existing.observation_date = obs_date
                    existing.sales_scope = p.sales_scope
                    existing.anomaly_type = p.anomaly_type.value if hasattr(p.anomaly_type, "value") else str(p.anomaly_type)
                    existing.severity = p.severity.value if hasattr(p.severity, "value") else str(p.severity)
                    existing.direction = p.direction.value if hasattr(p.direction, "value") else str(p.direction)
                    existing.observed_value = float(p.observed_value)
                    existing.reference_center = float(p.reference_center)
                    existing.reference_spread = float(p.reference_spread)
                    existing.anomaly_score = float(p.anomaly_score)
                    existing.threshold = float(p.threshold)
                    existing.explanation = str(p.explanation)
                    existing.detection_timestamp = det_time
                    existing.model_name = p.model_name
                    existing.forecast_run_id = p.forecast_run_id
                    existing.horizon_days = p.horizon_days
                    existing.step_ahead = p.step_ahead
                    existing.predicted_value = float(p.predicted_value) if p.predicted_value is not None else None
                    existing.dispersion_zero_fallback = bool(p.dispersion_zero_fallback)
                    existing.raw_json = raw_json
                    persisted.append(existing)
                else:
                    new_rec = AnomalyRecord(
                        anomaly_id=p.anomaly_id,
                        observation_date=obs_date,
                        sales_scope=p.sales_scope,
                        anomaly_type=p.anomaly_type.value if hasattr(p.anomaly_type, "value") else str(p.anomaly_type),
                        severity=p.severity.value if hasattr(p.severity, "value") else str(p.severity),
                        direction=p.direction.value if hasattr(p.direction, "value") else str(p.direction),
                        observed_value=float(p.observed_value),
                        reference_center=float(p.reference_center),
                        reference_spread=float(p.reference_spread),
                        anomaly_score=float(p.anomaly_score),
                        threshold=float(p.threshold),
                        explanation=str(p.explanation),
                        detection_timestamp=det_time,
                        model_name=p.model_name,
                        forecast_run_id=p.forecast_run_id,
                        horizon_days=p.horizon_days,
                        step_ahead=p.step_ahead,
                        predicted_value=float(p.predicted_value) if p.predicted_value is not None else None,
                        dispersion_zero_fallback=bool(p.dispersion_zero_fallback),
                        raw_json=raw_json,
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                    )
                    s.add(new_rec)
                    persisted.append(new_rec)

            s.flush()
            return persisted

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    def get_anomalies(
        self,
        sales_scope: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[Union[datetime.date, str]] = None,
        end_date: Optional[Union[datetime.date, str]] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
        session: Optional[Session] = None,
    ) -> List[AnomalyRecord]:
        """Query persisted anomaly records with optional scope, type, and temporal filters."""
        def _execute(s: Session) -> List[AnomalyRecord]:
            query = select(AnomalyRecord)

            if sales_scope:
                query = query.where(AnomalyRecord.sales_scope == sales_scope)
            if anomaly_type:
                query = query.where(AnomalyRecord.anomaly_type == anomaly_type)
            if severity:
                query = query.where(AnomalyRecord.severity == severity)
            if model_name:
                query = query.where(AnomalyRecord.model_name == model_name)

            if start_date:
                s_dt = datetime.date.fromisoformat(str(start_date)) if isinstance(start_date, str) else start_date
                query = query.where(AnomalyRecord.observation_date >= s_dt)
            if end_date:
                e_dt = datetime.date.fromisoformat(str(end_date)) if isinstance(end_date, str) else end_date
                query = query.where(AnomalyRecord.observation_date <= e_dt)

            query = query.order_by(
                AnomalyRecord.observation_date.desc(),
                AnomalyRecord.detection_timestamp.desc(),
            ).limit(limit)

            return list(s.execute(query).scalars().all())

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

