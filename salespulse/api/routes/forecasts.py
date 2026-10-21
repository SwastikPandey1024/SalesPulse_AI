"""
SalesPulse AI - Forecasting Service Routes
Provides:
- POST /api/v1/forecasts: Generates multi-step recursive forecasts with interval bounds.
- GET /api/v1/forecasts/{run_id}: Retrieves a persisted forecast run and item realization status.
- Strict non-fallback model selection: fails explicitly if requested model is unavailable.
"""

import uuid
import datetime
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from salespulse.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastItemSchema,
)
from salespulse.api.dependencies import get_forecasting_service, get_audit_repository, get_registry
import numpy as np
from salespulse.services.forecasting_service import (
    ForecastingService,
    build_feature_dataframe,
    extract_date_features,
)
from salespulse.services.model_registry import ModelRegistry
from salespulse.storage.repository import AuditRepository

router = APIRouter(prefix="/api/v1/forecasts", tags=["Forecasting"])


@router.post(
    "",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Multi-Step Sales Forecast",
    description="Generates recursive multi-step daily sales forecasts with uncertainty intervals using candidate models.",
)
def generate_forecast(
    request: ForecastRequest,
    forecasting_service: ForecastingService = Depends(get_forecasting_service),
    registry: ModelRegistry = Depends(get_registry),
) -> ForecastResponse:
    # 1. Validate candidate model availability explicitly (no silent fallback)
    available_models = registry.list_models()
    if request.model_choice not in available_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{request.model_choice}' is not available. "
                f"Available models: {available_models}"
            ),
        )

    # 2. Validate interval methodology
    if request.interval_method not in ("heuristic_sqrt_step", "calibrated_oof_residual"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid interval_method '{request.interval_method}'. "
                f"Must be 'heuristic_sqrt_step' or 'calibrated_oof_residual'."
            ),
        )

    # 3. Parse and validate base date
    try:
        base_dt = datetime.date.fromisoformat(request.base_date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base_date '{request.base_date}'. Expected format YYYY-MM-DD.",
        )

    # 4. Generate forecast
    assigned_run_id = str(uuid.uuid4()) if request.persist_run else None

    try:
        hist = [float(x) for x in request.recent_history_sales]
        d_feats = extract_date_features(base_dt)
        initial_df = build_feature_dataframe(
            date_features=d_feats,
            lag_1=hist[-1],
            lag_7=hist[-7],
            lag_14=hist[-14],
            lag_30=hist[-30],
            rolling_mean_7=float(np.mean(hist[-7:])),
            rolling_mean_14=float(np.mean(hist[-14:])),
            rolling_std_7=float(np.std(hist[-7:])),
            trend=float(len(hist)),
        )

        result_df = forecasting_service.forecast_multi_horizon(
            initial_df=initial_df,
            horizon_days=request.horizon_days,
            model_choice=request.model_choice,
            origin_date=base_dt,
            interval_method=request.interval_method,
            sales_scope=request.sales_scope,
            persist_run=request.persist_run,
            run_id=assigned_run_id,
            execution_source="fastapi_service",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    # 5. Format response items
    items: List[ForecastItemSchema] = []
    for idx, row in result_df.reset_index(drop=True).iterrows():
        target_val = row["Date"]
        target_str = target_val.strftime("%Y-%m-%d") if hasattr(target_val, "strftime") else str(target_val)
        items.append(
            ForecastItemSchema(
                target_date=target_str,
                step_ahead=int(idx) + 1,
                predicted_value=float(row["Forecasted Sales ($)"]),
                lower_bound=float(row["Lower Bound ($)"]) if pd.notnull(row["Lower Bound ($)"]) else None,
                upper_bound=float(row["Upper Bound ($)"]) if pd.notnull(row["Upper Bound ($)"]) else None,
            )
        )

    total_rev = float(result_df["Forecasted Sales ($)"].sum())
    avg_sales = float(result_df["Forecasted Sales ($)"].mean())

    return ForecastResponse(
        run_id=assigned_run_id,
        model_name=request.model_choice,
        forecast_origin=request.base_date,
        horizon_days=request.horizon_days,
        sales_scope=request.sales_scope,
        interval_method=request.interval_method,
        total_predicted_revenue=total_rev,
        average_daily_sales=avg_sales,
        items=items,
    )


@router.get(
    "/{run_id}",
    response_model=ForecastResponse,
    summary="Retrieve Persisted Forecast Run",
    description="Fetches a previously persisted forecast run and its item realization status by run_id.",
)
def get_forecast_run(
    run_id: str,
    repository: AuditRepository = Depends(get_audit_repository),
) -> ForecastResponse:
    run_log = repository.get_forecast_run(run_id=run_id)
    if run_log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast run '{run_id}' not found.",
        )

    items: List[ForecastItemSchema] = []
    for item in sorted(run_log.items, key=lambda x: x.step_ahead):
        items.append(
            ForecastItemSchema(
                target_date=item.target_date.isoformat() if item.target_date else "",
                step_ahead=item.step_ahead,
                predicted_value=float(item.predicted_value),
                lower_bound=float(item.lower_bound) if item.lower_bound is not None else None,
                upper_bound=float(item.upper_bound) if item.upper_bound is not None else None,
                actual_value=float(item.actual_value) if item.actual_value is not None else None,
                absolute_error=float(item.absolute_error) if item.absolute_error is not None else None,
                interval_covered=item.interval_covered,
            )
        )

    return ForecastResponse(
        run_id=run_log.run_id,
        model_name=run_log.model_name,
        forecast_origin=run_log.forecast_origin.strftime("%Y-%m-%d") if run_log.forecast_origin else "",
        horizon_days=run_log.horizon_days,
        sales_scope=run_log.items[0].sales_scope if run_log.items else "TOTAL_STORE",
        interval_method=run_log.interval_method,
        total_predicted_revenue=float(run_log.total_predicted_revenue or 0.0),
        average_daily_sales=float(run_log.average_daily_sales or 0.0),
        items=items,
    )
