"""
SalesPulse AI - Health and Readiness Endpoints
Provides:
- Liveness probe at GET /health (quick sanity check).
- Readiness probe at GET /ready (checks database connectivity, schema version, and model registry).
- Security safeguard: never exposes database credentials or local filesystem paths.
"""

import datetime
from fastapi import APIRouter, Depends, status, Response
from sqlalchemy import text
from sqlalchemy.engine import Engine

from salespulse.api.schemas import HealthResponse, ReadinessResponse
from salespulse.api.dependencies import get_database_engine, get_registry
from salespulse.services.model_registry import ModelRegistry
from salespulse.storage.database import SCHEMA_VERSION

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns HTTP 200 indicating the API service process is alive.",
)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        service="salespulse-api",
        version="1.0.0",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Probe",
    description="Verifies database connectivity, active schema version, and model registry availability.",
)
def get_ready(
    response: Response,
    engine: Engine = Depends(get_database_engine),
    registry: ModelRegistry = Depends(get_registry),
) -> ReadinessResponse:
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    db_status = "ok"
    active_version = 0

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
            if engine.url.drivername.startswith("sqlite"):
                v = conn.execute(text("PRAGMA user_version;")).scalar()
                active_version = int(v) if v is not None else 0
            else:
                active_version = SCHEMA_VERSION

            if active_version > SCHEMA_VERSION:
                db_status = "unsupported_future_schema"
            elif active_version < SCHEMA_VERSION:
                db_status = "unmigrated_schema"
            else:
                db_status = "ok"
    except Exception:
        db_status = "unavailable"
        active_version = 0

    try:
        supported_models = registry.list_models()
    except Exception:
        supported_models = []

    is_ready = (
        (db_status == "ok")
        and (active_version == SCHEMA_VERSION)
        and (len(supported_models) > 0)
    )

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        database=db_status,
        schema_version=active_version,
        supported_models=supported_models,
        timestamp=now_utc,
    )
