"""
SalesPulse AI - Data Contracts, Schema Definitions & Issue Reporting
Defines the expected schemas for transaction rows, daily aggregates, and feature batches.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import json
import warnings
import datetime
import pandas as pd
import numpy as np

from salespulse.features.temporal import ENGINEERED_FEATURES
from salespulse.data.time_series import MISSING_DATE_POLICY


class GrainType(str, Enum):
    """Supported input data granularities."""
    TRANSACTION = "transaction"
    DAILY_AGGREGATED = "daily_aggregated"
    FEATURE_BATCH = "feature_batch"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Severity of a validation finding."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Canonical Issue Codes
CODE_EMPTY_DATASET = "ERR_EMPTY_DATASET"
CODE_MISSING_COLUMNS = "ERR_MISSING_REQUIRED_COLUMNS"
CODE_MALFORMED_DATES = "ERR_MALFORMED_DATES"
CODE_NON_NUMERIC_SALES = "ERR_NON_NUMERIC_SALES"
CODE_INFINITE_VALUES = "ERR_INFINITE_VALUES"
CODE_ALL_ROWS_INVALID = "ERR_ALL_ROWS_INVALID"
CODE_UNKNOWN_GRAIN = "ERR_UNKNOWN_GRAIN"
CODE_DUPLICATE_DATES = "ERR_DUPLICATE_DATES"

CODE_QUARANTINED_ROWS = "WARN_QUARANTINED_ROWS"
CODE_NULL_VALUES = "WARN_NULL_VALUES"
CODE_DUPLICATES = "WARN_DUPLICATE_RECORDS"
CODE_DUPLICATE_IDENTIFIERS = "WARN_DUPLICATE_IDENTIFIERS"
CODE_DATA_STALENESS = "WARN_DATA_STALENESS"
CODE_NEGATIVE_DAILY_SALES = "WARN_NEGATIVE_DAILY_SALES"
CODE_ANOMALY_DISTRIBUTION = "WARN_ANOMALY_DISTRIBUTION"

CODE_LEGITIMATE_RETURNS = "INFO_LEGITIMATE_RETURNS"
CODE_EXTRA_COLUMNS = "INFO_EXTRA_COLUMNS"
CODE_MISSING_CALENDAR_DATES = "INFO_MISSING_CALENDAR_DATES"


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize dictionaries/lists for strict RFC 8259 JSON compliance.

    Converts float NaN, Infinity, and -Infinity to None (serializes to JSON null).
    Converts NumPy scalar types and timestamps to standard JSON-compatible Python types.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(obj)
    elif pd.isna(obj):
        return None
    return obj


class ValidationIssue:
    """Structured representation of a data quality or schema issue."""

    def __init__(
        self,
        code: str,
        severity: ValidationSeverity,
        message: str,
        affected_count: int = 0,
        sample_rows: Optional[List[int]] = None,
        remediation: str = "",
    ):
        self.code = code
        self.severity = severity
        self.message = message
        self.affected_count = affected_count
        self.sample_rows = sample_rows or []
        self.remediation = remediation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "affected_count": self.affected_count,
            "sample_rows": self.sample_rows[:5],  # Cap sample rows to prevent leaking large datasets
            "remediation": self.remediation,
        }

    def __repr__(self) -> str:
        return f"<ValidationIssue [{self.severity.value}] {self.code}: {self.message} (affected={self.affected_count})>"


class ValidationReport:
    """Comprehensive validation outcome and diagnostics for an ingested dataset."""

    def __init__(
        self,
        grain: GrainType = GrainType.UNKNOWN,
        total_rows_read: int = 0,
    ):
        self.grain = grain
        self.total_rows_read = total_rows_read
        self.valid_rows_count = 0
        self.quarantined_rows_count = 0
        self.issues: List[ValidationIssue] = []
        self.diagnostics: Dict[str, Any] = {}

    @property
    def status(self) -> str:
        """Overall validation verdict: FAILED, WARNING, or PASSED."""
        if any(issue.severity == ValidationSeverity.ERROR for issue in self.issues):
            return "FAILED"
        if any(issue.severity == ValidationSeverity.WARNING for issue in self.issues):
            return "WARNING"
        return "PASSED"

    def add_issue(
        self,
        code: str,
        severity: ValidationSeverity,
        message: str,
        affected_count: int = 0,
        sample_rows: Optional[List[int]] = None,
        remediation: str = "",
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                affected_count=affected_count,
                sample_rows=sample_rows,
                remediation=remediation,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        raw_dict = {
            "verdict": self.status,
            "grain": self.grain.value if hasattr(self.grain, "value") else str(self.grain),
            "total_rows_read": int(self.total_rows_read),
            "valid_rows_count": int(self.valid_rows_count),
            "quarantined_rows_count": int(self.quarantined_rows_count),
            "diagnostics": self.diagnostics,
            "issues": [issue.to_dict() for issue in self.issues],
        }
        return sanitize_for_json(raw_dict)

    def to_json(self, indent: int = 2) -> str:
        sanitized = self.to_dict()
        return json.dumps(sanitized, indent=indent, default=str, allow_nan=False)


def parse_business_calendar_date(val: Any) -> pd.Timestamp:
    """Extract local calendar date (year, month, day) without shifting across midnight boundaries.

    In retail point-of-sale datasets, dates indicate the local business trading day.
    This helper extracts normalized calendar midnight dates while safely ignoring time offsets.
    """
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return pd.Timestamp(year=val.year, month=val.month, day=val.day)
    val_str = str(val).strip()
    if not val_str:
        return pd.NaT
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(val_str, errors="coerce", format="mixed")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)
    except Exception:
        return pd.NaT


def safe_parse_calendar_dates(series: pd.Series) -> pd.Series:
    """Robust date parsing preserving intended business calendar dates.

    Safely handles:
    - Naive date/datetime strings
    - UTC ISO-8601 strings
    - Timezone offset-aware strings (without shifting dates across midnight)
    - Mixed timezone-aware and timezone-naive series
    - Unparseable or malformed date values (returned as NaT without unhandled exceptions)
    """
    if series is None or series.empty:
        return pd.Series([], dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dtype, "tz", None) is not None:
            return series.dt.tz_localize(None).dt.normalize()
        return series.dt.normalize()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if hasattr(parsed, "dt") and not (hasattr(parsed, "dtype") and parsed.dtype == object):
                if getattr(parsed.dtype, "tz", None) is not None:
                    return parsed.dt.tz_localize(None).dt.normalize()
                return parsed.dt.normalize()
        except Exception:
            pass

    # Element-wise fallback for mixed timezone offsets or object-wrapped series
    res = series.apply(parse_business_calendar_date)
    return pd.to_datetime(res, errors="coerce")


def detect_dataset_grain(df: pd.DataFrame) -> Tuple[GrainType, Dict[str, str]]:
    """Determine whether a DataFrame represents transaction-level rows, daily sales, or feature batches.

    Returns:
        Tuple of (GrainType, mapping of standard column roles: {'date': col, 'sales': col})
    """
    if df is None or df.empty:
        return GrainType.UNKNOWN, {}

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # 1. Check if input matches the 13 engineered features batch contract
    if set(ENGINEERED_FEATURES).issubset(set(df.columns)):
        return GrainType.FEATURE_BATCH, {}

    # 2. Check for date and sales columns
    date_col = None
    sales_col = None

    for candidate in ["order date", "order_date", "date", "timestamp", "datetime", "day"]:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break

    for candidate in ["sales", "revenue", "order amount", "amount", "total_sales", "daily_sales"]:
        if candidate in cols_lower:
            sales_col = cols_lower[candidate]
            break

    if not date_col or not sales_col:
        return GrainType.UNKNOWN, {
            "date": date_col or "missing",
            "sales": sales_col or "missing",
        }

    # 3. Check for transaction-level grain vs daily-aggregated grain
    parsed_dates = safe_parse_calendar_dates(df[date_col])
    has_duplicate_dates = bool(parsed_dates.dropna().duplicated().any())

    has_transaction_id = any(
        c in cols_lower for c in ["order id", "order_id", "row id", "row_id", "transaction id", "customer id"]
    )

    if has_duplicate_dates or has_transaction_id:
        return GrainType.TRANSACTION, {"date": date_col, "sales": sales_col}

    return GrainType.DAILY_AGGREGATED, {"date": date_col, "sales": sales_col}


def validate_dataset_schema(
    df: pd.DataFrame,
    expected_grain: Optional[GrainType] = None,
    max_staleness_days: Optional[int] = None,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[ValidationReport, pd.DataFrame, pd.DataFrame]:
    """Validate a DataFrame against SalesPulse data contracts.

    Identifies invalid rows, separates clean rows from quarantined rows,
    and returns a structured ValidationReport.

    Args:
        df: Input DataFrame.
        expected_grain: Optional enforced granularity. If None, auto-detected.
        max_staleness_days: Optional freshness threshold in days.
        reference_date: Reference date for freshness (defaults to now).

    Returns:
        Tuple of (ValidationReport, valid_df, quarantined_df)
    """
    if df is None or df.empty:
        report = ValidationReport(grain=GrainType.UNKNOWN, total_rows_read=0)
        report.add_issue(
            code=CODE_EMPTY_DATASET,
            severity=ValidationSeverity.ERROR,
            message="The provided dataset contains 0 rows or is empty.",
            remediation="Provide a non-empty CSV or Excel file containing valid sales transaction or daily records.",
        )
        return report, pd.DataFrame(), pd.DataFrame()

    total_rows = len(df)
    detected_grain, col_map = detect_dataset_grain(df)
    active_grain = expected_grain if expected_grain is not None else detected_grain

    report = ValidationReport(grain=active_grain, total_rows_read=total_rows)

    # ------------------------------------------------------------------------
    # CASE 1: Feature Batch Grain
    # ------------------------------------------------------------------------
    if active_grain == GrainType.FEATURE_BATCH:
        missing_features = [f for f in ENGINEERED_FEATURES if f not in df.columns]
        if missing_features:
            report.add_issue(
                code=CODE_MISSING_COLUMNS,
                severity=ValidationSeverity.ERROR,
                message=f"Missing engineered feature column(s): {missing_features}",
                remediation=f"Ensure the feature batch contains all 13 engineered features: {ENGINEERED_FEATURES}",
            )
            return report, pd.DataFrame(), df.copy()

        # Check numeric validity & NaN/Inf in features
        invalid_mask = pd.Series(False, index=df.index)
        for feat in ENGINEERED_FEATURES:
            numeric_col = pd.to_numeric(df[feat], errors="coerce")
            is_nan = numeric_col.isna()
            is_inf = np.isinf(numeric_col.fillna(0))
            bad_feat = is_nan | is_inf
            if bad_feat.any():
                invalid_mask = invalid_mask | bad_feat
                bad_indices = df.index[bad_feat].tolist()
                report.add_issue(
                    code=CODE_NON_NUMERIC_SALES if is_nan.any() else CODE_INFINITE_VALUES,
                    severity=ValidationSeverity.ERROR,
                    message=f"Feature '{feat}' contains {int(bad_feat.sum())} non-numeric, NaN, or infinite values.",
                    affected_count=int(bad_feat.sum()),
                    sample_rows=bad_indices[:5],
                    remediation=f"Clean feature '{feat}' by imputing or removing non-finite values.",
                )

        valid_df = df[~invalid_mask].copy()
        quarantined_df = df[invalid_mask].copy()
        report.valid_rows_count = len(valid_df)
        report.quarantined_rows_count = len(quarantined_df)

        if report.quarantined_rows_count > 0 and report.valid_rows_count > 0:
            report.add_issue(
                code=CODE_QUARANTINED_ROWS,
                severity=ValidationSeverity.WARNING,
                message=f"{report.quarantined_rows_count} feature rows were quarantined due to invalid values.",
                affected_count=report.quarantined_rows_count,
                sample_rows=quarantined_df.index[:5].tolist(),
                remediation="Review quarantined feature rows for sensor or pipeline calculation issues.",
            )
        elif report.valid_rows_count == 0:
            report.add_issue(
                code=CODE_ALL_ROWS_INVALID,
                severity=ValidationSeverity.ERROR,
                message="All rows in the feature batch are invalid.",
                remediation="Ensure input features match expected numeric scales and types.",
            )

        return report, valid_df, quarantined_df

    # ------------------------------------------------------------------------
    # CASE 2: Unknown Grain
    # ------------------------------------------------------------------------
    if active_grain == GrainType.UNKNOWN:
        report.add_issue(
            code=CODE_UNKNOWN_GRAIN,
            severity=ValidationSeverity.ERROR,
            message="Dataset schema does not match Transaction, Daily Aggregated, or Feature Batch contracts.",
            remediation="Provide a dataset with recognized date ('Order Date', 'date') and sales ('Sales', 'sales') columns, or 13 engineered features.",
        )
        return report, pd.DataFrame(), df.copy()

    # ------------------------------------------------------------------------
    # CASE 3: Transaction or Daily Aggregated Grain
    # ------------------------------------------------------------------------
    date_col = col_map.get("date")
    sales_col = col_map.get("sales")

    if not date_col or date_col not in df.columns or not sales_col or sales_col not in df.columns:
        report.add_issue(
            code=CODE_MISSING_COLUMNS,
            severity=ValidationSeverity.ERROR,
            message=f"Missing essential contract columns. Found: {list(df.columns)}",
            remediation="Include at least one valid date column ('Order Date', 'date') and one monetary sales column ('Sales', 'sales').",
        )
        return report, pd.DataFrame(), df.copy()

    # Identify extra columns (non-breaking, informational)
    known_cols = {date_col, sales_col}
    extra_cols = [c for c in df.columns if c not in known_cols]
    if extra_cols:
        report.add_issue(
            code=CODE_EXTRA_COLUMNS,
            severity=ValidationSeverity.INFO,
            message=f"Found {len(extra_cols)} additional metadata columns: {extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}",
            affected_count=len(extra_cols),
            remediation="Additional columns are preserved during ingestion and bypassed during daily aggregation.",
        )

    # 1. Evaluate Date column validity using robust business calendar parsing
    parsed_dates = safe_parse_calendar_dates(df[date_col])
    bad_dates_mask = parsed_dates.isna()
    bad_dates_count = int(bad_dates_mask.sum())
    if bad_dates_count > 0:
        sample_bad_date_rows = df.index[bad_dates_mask].tolist()
        report.add_issue(
            code=CODE_MALFORMED_DATES,
            severity=ValidationSeverity.ERROR,
            message=f"Found {bad_dates_count} rows with missing, unparseable, or malformed dates in '{date_col}'.",
            affected_count=bad_dates_count,
            sample_rows=sample_bad_date_rows[:5],
            remediation="Ensure all dates use standard formats (e.g. YYYY-MM-DD, MM/DD/YYYY) without null values.",
        )

    # 2. Evaluate Sales column numeric validity
    numeric_sales = pd.to_numeric(df[sales_col], errors="coerce")
    nan_sales_mask = numeric_sales.isna()
    inf_sales_mask = np.isinf(numeric_sales.fillna(0))
    bad_sales_mask = nan_sales_mask | inf_sales_mask
    bad_sales_count = int(bad_sales_mask.sum())

    if int(nan_sales_mask.sum()) > 0:
        sample_nan_rows = df.index[nan_sales_mask].tolist()
        report.add_issue(
            code=CODE_NON_NUMERIC_SALES,
            severity=ValidationSeverity.ERROR,
            message=f"Found {int(nan_sales_mask.sum())} rows with missing or non-numeric sales values in '{sales_col}'.",
            affected_count=int(nan_sales_mask.sum()),
            sample_rows=sample_nan_rows[:5],
            remediation=f"Ensure '{sales_col}' contains strictly numeric values without currency symbols or text strings.",
        )

    if int(inf_sales_mask.sum()) > 0:
        sample_inf_rows = df.index[inf_sales_mask].tolist()
        report.add_issue(
            code=CODE_INFINITE_VALUES,
            severity=ValidationSeverity.ERROR,
            message=f"Found {int(inf_sales_mask.sum())} rows with infinite (inf/-inf) values in '{sales_col}'.",
            affected_count=int(inf_sales_mask.sum()),
            sample_rows=sample_inf_rows[:5],
            remediation=f"Replace infinite values in '{sales_col}' with valid finite numbers.",
        )

    # 3. Legitimate Negative Transactions vs Negative Daily Sales
    negative_mask = (~bad_sales_mask) & (numeric_sales < 0)
    negative_count = int(negative_mask.sum())
    if negative_count > 0:
        sample_neg_rows = df.index[negative_mask].tolist()
        if active_grain == GrainType.TRANSACTION:
            report.add_issue(
                code=CODE_LEGITIMATE_RETURNS,
                severity=ValidationSeverity.INFO,
                message=f"Detected {negative_count} transaction rows with negative sales (legitimate returns/refunds/adjustments).",
                affected_count=negative_count,
                sample_rows=sample_neg_rows[:5],
                remediation="Legitimate returns will be factored into net daily sales during aggregation.",
            )
        else:
            report.add_issue(
                code=CODE_NEGATIVE_DAILY_SALES,
                severity=ValidationSeverity.WARNING,
                message=f"Detected {negative_count} daily records with negative sales total (net returns exceeding revenue).",
                affected_count=negative_count,
                sample_rows=sample_neg_rows[:5],
                remediation="Review days with net negative sales to verify whether substantial merchandise returns occurred.",
            )

    # 4. Identifier-Aware Duplicate Detection (Fix F-04)
    cols_lower_map = {str(c).strip().lower(): c for c in df.columns}

    # Detect unique line-item primary keys vs order header IDs
    line_id_col = None
    for candidate in ["row id", "row_id", "line id", "line_id", "transaction id", "transaction_id"]:
        if candidate in cols_lower_map:
            line_id_col = cols_lower_map[candidate]
            break

    order_id_col = None
    for candidate in ["order id", "order_id"]:
        if candidate in cols_lower_map:
            order_id_col = cols_lower_map[candidate]
            break

    prod_id_col = None
    for candidate in ["product id", "product_id", "item id", "item_id", "sku"]:
        if candidate in cols_lower_map:
            prod_id_col = cols_lower_map[candidate]
            break

    if line_id_col and line_id_col in df.columns:
        line_dups = df[line_id_col].duplicated()
        line_dup_count = int(line_dups.sum())
        if line_dup_count > 0:
            sample_line_dup_rows = df.index[line_dups].tolist()
            report.add_issue(
                code=CODE_DUPLICATE_IDENTIFIERS,
                severity=ValidationSeverity.WARNING,
                message=f"Detected {line_dup_count} duplicate primary line-item identifiers in '{line_id_col}'. Review for unintended record duplication.",
                affected_count=line_dup_count,
                sample_rows=sample_line_dup_rows[:5],
                remediation=f"Verify unique keys in '{line_id_col}' are not accidentally repeated.",
            )
    elif order_id_col and prod_id_col:
        order_prod_dups = df.duplicated(subset=[order_id_col, prod_id_col])
        op_dup_count = int(order_prod_dups.sum())
        if op_dup_count > 0:
            sample_op_rows = df.index[order_prod_dups].tolist()
            report.add_issue(
                code=CODE_DUPLICATE_IDENTIFIERS,
                severity=ValidationSeverity.WARNING,
                message=f"Detected {op_dup_count} duplicate line-item combinations for ({order_id_col}, {prod_id_col}).",
                affected_count=op_dup_count,
                sample_rows=sample_op_rows[:5],
                remediation="Check whether repeated items in the same order are distinct purchases or accidental duplicates.",
            )

    # Exact full-row duplicate detection
    exact_duplicates = df.duplicated()
    exact_dup_count = int(exact_duplicates.sum())
    if exact_dup_count > 0:
        sample_dup_rows = df.index[exact_duplicates].tolist()
        if active_grain == GrainType.TRANSACTION:
            report.add_issue(
                code=CODE_DUPLICATES,
                severity=ValidationSeverity.WARNING,
                message=f"Detected {exact_dup_count} exact duplicate rows. In retail POS logs, identical items may be distinct transactions unless a unique line-item ID is duplicated.",
                affected_count=exact_dup_count,
                sample_rows=sample_dup_rows[:5],
                remediation="Verify if identical rows represent separate purchases or accidental re-submissions.",
            )
        else:
            report.add_issue(
                code=CODE_DUPLICATES,
                severity=ValidationSeverity.WARNING,
                message=f"Detected {exact_dup_count} exact duplicate rows in daily series.",
                affected_count=exact_dup_count,
                sample_rows=sample_dup_rows[:5],
                remediation="Deduplicate daily records before forecasting.",
            )

    # 5. Duplicate Daily Dates Blocking Check (Fix F-02)
    # In daily-aggregated grain, duplicate calendar dates are a FATAL contract error that blocks ingestion
    if active_grain == GrainType.DAILY_AGGREGATED:
        date_duplicates = parsed_dates.dropna().duplicated()
        date_dup_count = int(date_duplicates.sum())
        if date_dup_count > 0:
            sample_date_dup_rows = df.index[parsed_dates.dropna().index[date_duplicates]].tolist()
            report.add_issue(
                code=CODE_DUPLICATE_DATES,
                severity=ValidationSeverity.ERROR,
                message=f"Found {date_dup_count} duplicate calendar dates in daily-aggregated series. Daily series requires strictly unique calendar dates to prevent revenue double-counting.",
                affected_count=date_dup_count,
                sample_rows=sample_date_dup_rows[:5],
                remediation="Aggregate transactions by calendar date so that each date appears at most once, or re-ingest as Transaction grain.",
            )

    # 6. Determine Quarantine Mask (fatal schema errors)
    fatal_mask = bad_dates_mask | bad_sales_mask
    valid_df = df[~fatal_mask].copy()
    quarantined_df = df[fatal_mask].copy()

    report.valid_rows_count = len(valid_df)
    report.quarantined_rows_count = len(quarantined_df)

    if report.quarantined_rows_count > 0 and report.valid_rows_count > 0:
        report.add_issue(
            code=CODE_QUARANTINED_ROWS,
            severity=ValidationSeverity.WARNING,
            message=f"{report.quarantined_rows_count} rows were quarantined due to fatal schema errors (malformed date or non-numeric/infinite sales).",
            affected_count=report.quarantined_rows_count,
            sample_rows=quarantined_df.index[:5].tolist(),
            remediation="Quarantined records are safely isolated so valid data can proceed to aggregation.",
        )
    elif report.valid_rows_count == 0:
        report.add_issue(
            code=CODE_ALL_ROWS_INVALID,
            severity=ValidationSeverity.ERROR,
            message="No valid rows remain after quarantine.",
            remediation="Check data formatting and source exports to supply valid records.",
        )

    # 7. Freshness & Missing Calendar Dates Check (on valid subset)
    if not valid_df.empty:
        valid_dates = safe_parse_calendar_dates(valid_df[date_col])
        min_date = valid_dates.min()
        max_date = valid_dates.max()

        # Freshness Check
        if max_staleness_days is not None and max_staleness_days > 0 and pd.notna(max_date):
            ref_ts = pd.Timestamp(reference_date).normalize() if reference_date else pd.Timestamp.now().normalize()
            staleness_days = (ref_ts - max_date).days
            if staleness_days > max_staleness_days:
                report.add_issue(
                    code=CODE_DATA_STALENESS,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Data staleness alert: Latest observation date is {max_date.date()} "
                        f"({staleness_days} days prior to reference date {ref_ts.date()}), "
                        f"exceeding threshold of {max_staleness_days} days. (Configurable threshold check, not live monitoring)."
                    ),
                    affected_count=staleness_days,
                    remediation=f"Ingest recent sales data up to the current planning cycle (within {max_staleness_days} days).",
                )

        # Missing calendar dates check
        if pd.notna(min_date) and pd.notna(max_date):
            calendar_span = (max_date - min_date).days + 1
            unique_dates_count = valid_dates.nunique()
            missing_calendar_days = max(0, calendar_span - unique_dates_count)
            if missing_calendar_days > 0:
                report.add_issue(
                    code=CODE_MISSING_CALENDAR_DATES,
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"Calendar coverage has {missing_calendar_days} missing days across a {calendar_span}-day span. "
                        f"Under the documented {MISSING_DATE_POLICY[:17]}, absent dates will be zero-filled ($0.00)."
                    ),
                    affected_count=missing_calendar_days,
                    remediation="Verify whether missing calendar dates represent zero-sales store closures or gaps in export.",
                )

    return report, valid_df, quarantined_df
