"""
SalesPulse AI - Canonical Datetime & Timestamp Utilities
Provides deterministic parsing, safe formatting, and timezone-aware normalization.
Guaranteed to never throw AttributeError or unhandled exceptions on invalid or string inputs.
"""

import math
import datetime
from typing import Optional, Any


def parse_datetime_safe(val: Any) -> Optional[datetime.datetime]:
    """Parse ISO date strings, datetime objects, or date objects into timezone-aware UTC datetimes.
    
    Guarantees:
    - Never raises AttributeError or ValueError on invalid inputs.
    - Preserves timezone awareness (converts to UTC).
    - Returns None if val is None, empty string, or unparseable.
    """
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=datetime.timezone.utc)
        return val.astimezone(datetime.timezone.utc)
    if isinstance(val, datetime.date):
        return datetime.datetime(val.year, val.month, val.day, tzinfo=datetime.timezone.utc)
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return None
        # Handle trailing Z
        if val_str.endswith("Z"):
            val_str = val_str[:-1] + "+00:00"
        try:
            dt = datetime.datetime.fromisoformat(val_str)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)
        except Exception:
            for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.datetime.strptime(val_str, pattern)
                    return dt.replace(tzinfo=datetime.timezone.utc)
                except Exception:
                    continue
            return None
    return None


def format_datetime_safe(val: Any, fmt: str = "%Y-%m-%d %H:%M:%S", fallback: str = "-", default: Optional[str] = None) -> str:
    """Safely format a datetime, date, or ISO string.
    
    Guarantees:
    - Never calls .strftime() on an unvalidated string.
    - Returns fallback/default if val is None, empty, or unparseable.
    - Retains UTC timezone consistency when formatting.
    """
    fb = default if default is not None else fallback
    dt = parse_datetime_safe(val)
    if dt is None:
        return fb
    try:
        return dt.strftime(fmt)
    except Exception:
        return fb
