"""
SalesPulse AI - FastAPI Dependency Injection Providers
Allows injecting shared domain services, repositories, and transactional database sessions.
Enables clean test-time mocking and isolated temporary database injection via app.dependency_overrides.
"""

from typing import Generator, Optional
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from salespulse.storage.database import get_engine, get_db_session, get_session_factory
from salespulse.storage.repository import AuditRepository
from salespulse.services.model_registry import ModelRegistry, get_model_registry
from salespulse.services.validation_service import ValidationService
from salespulse.services.forecasting_service import ForecastingService
from salespulse.services.analytics_service import AnalyticsService


def get_database_engine() -> Engine:
    """Dependency provider for global or configured SQLAlchemy Engine."""
    return get_engine()


def get_db(engine: Engine = Depends(get_database_engine)) -> Generator[Session, None, None]:
    """Dependency provider yielding a scoped transactional database session with cleanup."""
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_audit_repository(engine: Engine = Depends(get_database_engine)) -> AuditRepository:
    """Dependency provider for AuditRepository."""
    return AuditRepository(engine=engine)


def get_registry() -> ModelRegistry:
    """Dependency provider for ModelRegistry."""
    return get_model_registry()


def get_validation_service(
    repository: AuditRepository = Depends(get_audit_repository),
) -> ValidationService:
    """Dependency provider for ValidationService."""
    return ValidationService(repository=repository)


def get_forecasting_service(
    registry: ModelRegistry = Depends(get_registry),
    repository: AuditRepository = Depends(get_audit_repository),
) -> ForecastingService:
    """Dependency provider for ForecastingService."""
    return ForecastingService(registry=registry, repository=repository)


def get_analytics_service(
    repository: AuditRepository = Depends(get_audit_repository),
    engine: Engine = Depends(get_database_engine),
) -> AnalyticsService:
    """Dependency provider for AnalyticsService."""
    return AnalyticsService(repository=repository, engine=engine)
