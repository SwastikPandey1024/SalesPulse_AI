"""
SalesPulse AI - FastAPI Application Factory
Provides:
- Production-minded FastAPI app initialization via create_app().
- Modular route registration under /api/v1.
- Strict security boundaries: sanitized error handlers preventing filesystem or stack-trace leaks.
- Conservative CORS configuration (never wildcard with credentials).
- Dependency injection overrides for isolated testing.
"""

import os
import logging
import datetime
import re
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.engine import Engine

from salespulse.api.schemas import APIErrorResponse
from salespulse.api.routes import (
    health_router,
    data_router,
    forecasts_router,
    analytics_router,
)
from salespulse.api.dependencies import get_database_engine, get_audit_repository
from salespulse.storage.repository import AuditRepository

logger = logging.getLogger("salespulse.api")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

_PATH_OR_CRED_PATTERN = re.compile(
    r"(?:[a-zA-Z]:[/\\]|/(?:home|users|tmp|var|etc|app|usr|bin|lib)/|sqlite:|postgres|mysql:|password\s*=|secret\s*=|bearer\s+)",
    re.IGNORECASE,
)


def sanitize_error_detail(detail: str) -> str:
    """Ensure error message never leaks local filesystem paths, database URLs, or credentials."""
    sanitized = str(detail)
    if _PATH_OR_CRED_PATTERN.search(sanitized):
        return "Internal system error occurred. Diagnostic path and credential details redacted for security."
    return sanitized


def create_app(
    engine: Optional[Engine] = None,
    repository: Optional[AuditRepository] = None,
    cors_origins: Optional[List[str]] = None,
    enable_docs: bool = True,
) -> FastAPI:
    """Application factory for SalesPulse AI API service.
    
    Security Boundary Note:
    - This API layer is configured for development-local or trusted internal network environments.
    - Write endpoints (/api/v1/data/validate, /api/v1/forecasts, /api/v1/analytics/anomalies/detect)
      are unauthenticated by default. In production environments, an API Gateway or reverse proxy
      with OAuth2/OIDC, JWT, or mTLS authentication is strictly required.
    - CORS restricts browser origins but does NOT provide authentication or authorization.
    """
    app = FastAPI(
        title="SalesPulse AI Engine API",
        description=(
            "Production-minded REST API for data quality contracts, time-series forecasting, "
            "and explainable anomaly detection.\n\n"
            "**Security Note:** This service is designated for development-local or protected VPC access. "
            "Public internet deployments require an external authentication proxy or API Gateway."
        ),
        version="1.0.0",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )

    # 1. Dependency overrides for testing/isolation
    if engine is not None:
        app.dependency_overrides[get_database_engine] = lambda: engine
    if repository is not None:
        app.dependency_overrides[get_audit_repository] = lambda: repository

    # 2. Conservative CORS configuration
    raw_origins = cors_origins or os.environ.get("SALESPULSE_CORS_ORIGINS", "").split(",")
    valid_origins = [o.strip() for o in raw_origins if o.strip()] or DEFAULT_CORS_ORIGINS

    allow_creds = "*" not in valid_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=valid_origins,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )

    # 3. Security Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        detail = sanitize_error_detail(exc.detail)
        error_code = f"ERR_HTTP_{exc.status_code}"
        if exc.status_code == 404:
            error_code = "RESOURCE_NOT_FOUND"
        elif exc.status_code == 400:
            error_code = "BAD_REQUEST"
        elif exc.status_code == 413:
            error_code = "PAYLOAD_TOO_LARGE"

        payload = APIErrorResponse(
            error_code=error_code,
            detail=detail,
            timestamp=now_utc,
            path=request.url.path,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Summarize field validation errors cleanly without raw internal exception dumps
        error_details = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            error_details.append(f"{loc}: {msg}")

        summary = "; ".join(error_details)
        payload = APIErrorResponse(
            error_code="VALIDATION_ERROR",
            detail=summary,
            timestamp=now_utc,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
        payload = APIErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            detail="An internal server error occurred while processing the request.",
            timestamp=now_utc,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )

    # 4. Route registration
    app.include_router(health_router)
    app.include_router(data_router)
    app.include_router(forecasts_router)
    app.include_router(analytics_router)

    return app
