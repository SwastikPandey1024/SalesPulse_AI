"""
SalesPulse AI - Shared Analytics Domain Service
Provides:
- High-level orchestration for explainable time-series anomaly detection.
- Detection across in-memory DataFrames or persisted database actuals and forecast items.
- Configurable thresholds, window sizes, and severity policies.
- Optional persistence to AuditRepository with idempotent upsert.
- Clear integration boundary for Phase 3C (FastAPI) and Phase 3D (monitoring UI).
"""

import datetime
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from salespulse.analytics.schemas import (
    AnomalyConfig,
    AnomalyReport,
    AnomalyPoint,
    AnomalyType,
    SeverityLevel,
)
from salespulse.analytics.detector import (
    detect_sales_anomalies as core_detect_sales,
    detect_forecast_error_anomalies as core_detect_forecast_errors,
)
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import ActualSalesRecord, ForecastItemLog, ForecastRunLog, AnomalyRecord
from salespulse.storage.database import get_db_session, get_engine


class AnalyticsService:
    """Domain service for sales and forecast performance anomaly detection."""

    def __init__(self, repository: Optional[AuditRepository] = None, engine: Optional[Engine] = None):
        self.engine = engine or (repository.engine if repository else get_engine())
        self.repository = repository or AuditRepository(self.engine)

    def detect_sales_anomalies_df(
        self,
        df: pd.DataFrame,
        config: Optional[AnomalyConfig] = None,
        sales_scope: str = "TOTAL_STORE",
        date_col: str = "date",
        sales_col: str = "sales",
        fill_missing_dates: bool = False,
        persist: bool = False,
    ) -> AnomalyReport:
        """Run anomaly detection on an in-memory DataFrame.
        
        Args:
            df: DataFrame containing at least date_col and sales_col.
            config: AnomalyConfig parameters (or defaults).
            sales_scope: Scope identifier.
            date_col: Column name for dates.
            sales_col: Column name for sales values.
            fill_missing_dates: Whether to reindex missing calendar dates with 0.0.
            persist: Whether to store detected anomalies into AuditRepository.
            
        Returns:
            AnomalyReport containing evaluated count, detected anomalies, and diagnostics.
        """
        report = core_detect_sales(
            df=df,
            config=config,
            sales_scope=sales_scope,
            date_col=date_col,
            sales_col=sales_col,
            fill_missing_dates=fill_missing_dates,
        )

        if persist and report.anomalies:
            self.repository.record_anomalies(report)

        return report

    def detect_sales_anomalies_from_db(
        self,
        sales_scope: str = "TOTAL_STORE",
        config: Optional[AnomalyConfig] = None,
        fill_missing_dates: bool = False,
        persist: bool = False,
        session: Optional[Session] = None,
    ) -> AnomalyReport:
        """Run anomaly detection on ground truth actual sales stored in the database."""
        def _execute(s: Session) -> AnomalyReport:
            stmt = (
                select(ActualSalesRecord.calendar_date, ActualSalesRecord.daily_sales)
                .where(ActualSalesRecord.sales_scope == sales_scope)
                .order_by(ActualSalesRecord.calendar_date.asc())
            )
            records = s.execute(stmt).all()
            if not records:
                cfg = config or AnomalyConfig()
                return AnomalyReport(
                    report_id=str(datetime.datetime.now(datetime.timezone.utc).timestamp()),
                    target_scope=sales_scope,
                    total_evaluated=0,
                    anomalies_detected=0,
                    insufficient_history_count=0,
                    anomalies=[],
                    config=cfg.__dict__,
                )

            data = [{"date": r[0], "sales": float(r[1])} for r in records]
            df = pd.DataFrame(data)
            return self.detect_sales_anomalies_df(
                df=df,
                config=config,
                sales_scope=sales_scope,
                fill_missing_dates=fill_missing_dates,
                persist=persist,
            )

        if session is not None:
            return _execute(session)
        with get_db_session(self.engine) as s:
            return _execute(s)

    def detect_forecast_error_anomalies(
        self,
        items: Optional[List[Dict[str, Any]]] = None,
        config: Optional[AnomalyConfig] = None,
        sales_scope: str = "TOTAL_STORE",
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
        step_ahead: Optional[int] = None,
        persist: bool = False,
        session: Optional[Session] = None,
    ) -> AnomalyReport:
        """Run forecast error anomaly detection.
        
        If `items` is None, queries all reconciled forecast items from the database.
        """
        if items is None:
            def _fetch(s: Session) -> List[Dict[str, Any]]:
                query = (
                    select(
                        ForecastItemLog.target_date,
                        ForecastItemLog.predicted_value,
                        ForecastItemLog.actual_value,
                        ForecastItemLog.absolute_error,
                        ForecastItemLog.sales_scope,
                        ForecastItemLog.step_ahead,
                        ForecastItemLog.run_id,
                        ForecastRunLog.model_name,
                        ForecastRunLog.horizon_days,
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
                if step_ahead:
                    query = query.where(ForecastItemLog.step_ahead == step_ahead)

                records = s.execute(query).all()
                out = []
                for r in records:
                    out.append({
                        "target_date": r[0],
                        "predicted_value": r[1],
                        "actual_value": r[2],
                        "absolute_error": r[3],
                        "sales_scope": r[4],
                        "step_ahead": r[5],
                        "run_id": r[6],
                        "model_name": r[7],
                        "horizon_days": r[8],
                    })
                return out

            if session is not None:
                fetched_items = _fetch(session)
            else:
                with get_db_session(self.engine) as s:
                    fetched_items = _fetch(s)
        else:
            fetched_items = items

        report = core_detect_forecast_errors(
            items=fetched_items,
            config=config,
            target_scope=sales_scope,
            target_model_name=model_name,
            target_horizon_days=horizon_days,
            target_step_ahead=step_ahead,
        )

        if persist and report.anomalies:
            self.repository.record_anomalies(report)

        return report

    def get_persisted_anomalies(
        self,
        sales_scope: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[Union[datetime.date, str]] = None,
        end_date: Optional[Union[datetime.date, str]] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[AnomalyRecord]:
        """Retrieve persisted anomaly records from the audit repository."""
        return self.repository.get_anomalies(
            sales_scope=sales_scope,
            anomaly_type=anomaly_type,
            severity=severity,
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
            limit=limit,
        )
