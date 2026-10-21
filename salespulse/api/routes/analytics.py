"""
SalesPulse AI - Analytics & Anomaly Detection Routes
Provides:
- POST /api/v1/analytics/anomalies/detect: Triggers explainable time-series anomaly detection.
- GET /api/v1/analytics/anomalies: Queries persisted anomaly records with validated filters.
- Documents the known weekly seasonality limitation in response payloads.
"""

import datetime
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from salespulse.api.schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    AnomalyPointSchema,
)
from salespulse.api.dependencies import get_analytics_service, get_audit_repository
from salespulse.analytics.schemas import AnomalyConfig, AnomalyPoint
from salespulse.services.analytics_service import AnalyticsService
from salespulse.storage.repository import AuditRepository

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics & Anomalies"])


@router.post(
    "/anomalies/detect",
    response_model=AnomalyDetectionResponse,
    summary="Trigger Time-Series Anomaly Detection",
    description=(
        "Executes explainable anomaly detection on submitted daily sales points or persisted database actuals. "
        "Applies rolling median and MAD robust z-scores with explicit zero-MAD guardrails."
    ),
)
def detect_anomalies(
    request: AnomalyDetectionRequest,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnomalyDetectionResponse:
    # 1. Construct explicit statistical configuration
    try:
        config = AnomalyConfig(
            window_size=request.window_size,
            min_history=request.min_history,
            surge_threshold=request.surge_threshold,
            drop_threshold=request.drop_threshold,
            handle_negative_sales=request.handle_negative_sales,
            handle_missing_values=request.handle_missing_values,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid anomaly configuration: {str(e)}",
        )

    # 2. Execute detection across in-memory payload or database actuals
    try:
        if request.series is not None and len(request.series) > 0:
            df = pd.DataFrame([{"date": p.date, "sales": p.sales} for p in request.series])
            report = analytics_service.detect_sales_anomalies_df(
                df=df,
                config=config,
                sales_scope=request.sales_scope,
                fill_missing_dates=request.fill_missing_dates,
                persist=request.persist,
            )
        else:
            # Evaluate ground truth actual sales stored in database
            report = analytics_service.detect_sales_anomalies_from_db(
                sales_scope=request.sales_scope,
                config=config,
                fill_missing_dates=request.fill_missing_dates,
                persist=request.persist,
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Anomaly detection failed: {str(e)}",
        )

    # 3. Format response items
    anomalies_out = [
        AnomalyPointSchema(
            anomaly_id=p.anomaly_id,
            observation_date=p.observation_date,
            sales_scope=p.sales_scope,
            anomaly_type=p.anomaly_type.value if hasattr(p.anomaly_type, "value") else str(p.anomaly_type),
            severity=p.severity.value if hasattr(p.severity, "value") else str(p.severity),
            direction=p.direction.value if hasattr(p.direction, "value") else str(p.direction),
            observed_value=p.observed_value,
            reference_center=p.reference_center,
            reference_spread=p.reference_spread,
            anomaly_score=p.anomaly_score,
            threshold=p.threshold,
            explanation=p.explanation,
            detection_timestamp=p.detection_timestamp,
            status=getattr(p, "status", "ACTIVE"),
            status_updated_at=getattr(p, "status_updated_at", None),
            model_name=p.model_name,
            forecast_run_id=p.forecast_run_id,
            horizon_days=p.horizon_days,
            step_ahead=p.step_ahead,
            predicted_value=p.predicted_value,
            dispersion_zero_fallback=p.dispersion_zero_fallback,
            notes=p.notes,
        )
        for p in report.anomalies
    ]

    return AnomalyDetectionResponse(
        report_id=report.report_id,
        target_scope=report.target_scope,
        total_evaluated=report.total_evaluated,
        anomalies_detected=report.anomalies_detected,
        insufficient_history_count=report.insufficient_history_count,
        missing_values_count=getattr(report, "missing_values_count", 0),
        anomalies=anomalies_out,
        config=report.config,
    )


@router.get(
    "/anomalies",
    response_model=List[AnomalyPointSchema],
    summary="Query Persisted Anomaly Records",
    description="Fetches persisted anomaly event records with scope, type, severity, status, and date filters.",
)
def get_anomalies(
    sales_scope: Optional[str] = Query(None, max_length=50, description="Filter by sales scope"),
    anomaly_type: Optional[str] = Query(None, description="SALES_SURGE, SALES_DROP, FORECAST_ERROR_ANOMALY"),
    severity: Optional[str] = Query(None, description="LOW, MEDIUM, HIGH, CRITICAL"),
    status: Optional[str] = Query(None, description="ACTIVE or RESOLVED"),
    start_date: Optional[str] = Query(None, description="Start date (inclusive, YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (inclusive, YYYY-MM-DD)"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    repository: AuditRepository = Depends(get_audit_repository),
) -> List[AnomalyPointSchema]:
    # Validate date formats if passed
    if start_date:
        try:
            datetime.date.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid start_date '{start_date}'. Expected format YYYY-MM-DD.",
            )
    if end_date:
        try:
            datetime.date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid end_date '{end_date}'. Expected format YYYY-MM-DD.",
            )

    records = repository.get_anomalies(
        sales_scope=sales_scope,
        anomaly_type=anomaly_type,
        severity=severity,
        status=status,
        start_date=start_date,
        end_date=end_date,
        model_name=model_name,
        limit=limit,
    )

    out = []
    for r in records:
        out.append(
            AnomalyPointSchema(
                anomaly_id=r.anomaly_id,
                observation_date=r.observation_date.isoformat() if r.observation_date else "",
                sales_scope=r.sales_scope,
                anomaly_type=r.anomaly_type,
                severity=r.severity,
                direction=r.direction,
                observed_value=float(r.observed_value),
                reference_center=float(r.reference_center),
                reference_spread=float(r.reference_spread),
                anomaly_score=float(r.anomaly_score),
                threshold=float(r.threshold),
                explanation=r.explanation,
                detection_timestamp=r.detection_timestamp.isoformat() if r.detection_timestamp else "",
                status=r.status,
                status_updated_at=r.status_updated_at.isoformat() if r.status_updated_at else None,
                model_name=r.model_name,
                forecast_run_id=r.forecast_run_id,
                horizon_days=r.horizon_days,
                step_ahead=r.step_ahead,
                predicted_value=float(r.predicted_value) if r.predicted_value is not None else None,
                dispersion_zero_fallback=bool(r.dispersion_zero_fallback),
            )
        )

    return out
