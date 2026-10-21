"""
SalesPulse AI - API Route Submodules
"""

from salespulse.api.routes.health import router as health_router
from salespulse.api.routes.data import router as data_router
from salespulse.api.routes.forecasts import router as forecasts_router
from salespulse.api.routes.analytics import router as analytics_router

__all__ = [
    "health_router",
    "data_router",
    "forecasts_router",
    "analytics_router",
]
