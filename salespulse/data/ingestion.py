"""
SalesPulse AI - Safe Data Ingestion Module
Handles multi-format file ingestion (CSV/XLSX), defensive validation, row quarantine,
and structured diagnostic generation without confidential data leakage.
"""

import os
import io
from typing import Optional, Union, BinaryIO, Tuple, Dict, Any
import pandas as pd
import numpy as np

from salespulse.data.contracts import (
    GrainType,
    ValidationReport,
    ValidationSeverity,
    validate_dataset_schema,
    detect_dataset_grain,
    CODE_EMPTY_DATASET,
)
from salespulse.data.quality import compute_data_quality_diagnostics
from salespulse.data.time_series import aggregate_to_daily_sales


class IngestionResult:
    """Outcome of safe tabular file ingestion."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        quarantined_df: pd.DataFrame,
        report: ValidationReport,
        file_name: str = "dataset",
    ):
        self.raw_df = raw_df
        self.valid_df = valid_df
        self.quarantined_df = quarantined_df
        self.report = report
        self.file_name = file_name

    @property
    def is_usable(self) -> bool:
        """Indicates if the dataset passes validation and contains valid records."""
        return self.report.status != "FAILED" and len(self.valid_df) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "is_usable": self.is_usable,
            "report": self.report.to_dict(),
            "summary": {
                "raw_rows": len(self.raw_df),
                "valid_rows": len(self.valid_df),
                "quarantined_rows": len(self.quarantined_df),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return self.report.to_json(indent=indent)


def read_tabular_file(
    file_or_path: Union[str, BinaryIO, io.BytesIO, bytes],
    file_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Safely parse CSV or Excel data with multi-encoding fallback.

    Args:
        file_or_path: Filepath or file-like buffer.
        file_name: Explicit filename or name attribute.

    Returns:
        Tuple of (pd.DataFrame, detected filename)

    Raises:
        ValueError: If file is empty, unsupported format, or cannot be parsed.
    """
    fname = file_name
    if fname is None:
        if isinstance(file_or_path, str):
            fname = os.path.basename(file_or_path)
        elif hasattr(file_or_path, "name"):
            fname = getattr(file_or_path, "name")
        else:
            fname = "uploaded_file.csv"

    fname_lower = fname.lower()

    # Reject unsupported file extensions
    supported_extensions = (".csv", ".xlsx", ".xls")
    if not any(fname_lower.endswith(ext) for ext in supported_extensions):
        raise ValueError(
            f"Unsupported file format '{fname}'. SalesPulse AI supports CSV (.csv) "
            f"and Excel (.xlsx, .xls) files."
        )

    # Convert bytes to BytesIO if necessary
    if isinstance(file_or_path, bytes):
        buffer: Union[str, BinaryIO, io.BytesIO] = io.BytesIO(file_or_path)
    else:
        buffer = file_or_path

    # Read Excel
    if fname_lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(buffer)
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file '{fname}': {exc}")
        if df.empty:
            raise ValueError(f"Excel file '{fname}' contains no rows.")
        return df, fname

    # Read CSV with encoding fallback (UTF-8 -> latin1 -> cp1252)
    encodings_to_try = ["utf-8", "latin1", "cp1252"]
    last_err: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            if hasattr(buffer, "seek"):
                buffer.seek(0)
            df = pd.read_csv(buffer, encoding=enc)
            if df.empty:
                raise ValueError(f"CSV file '{fname}' is empty.")
            return df, fname
        except (UnicodeDecodeError, UnicodeError) as u_err:
            last_err = u_err
            continue
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file '{fname}' is empty or contains no columns.")
        except Exception as parse_err:
            last_err = parse_err
            break

    raise ValueError(f"Failed to parse CSV file '{fname}': {last_err}")


def safe_ingest_file(
    file_or_path: Union[str, BinaryIO, io.BytesIO, bytes],
    file_name: Optional[str] = None,
    expected_grain: Optional[GrainType] = None,
    max_staleness_days: Optional[int] = 30,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
) -> IngestionResult:
    """End-to-end safe ingestion pipeline with contract validation and diagnostics.

    Args:
        file_or_path: Filepath or file-like buffer.
        file_name: Optional explicit filename.
        expected_grain: Granularity to enforce (or None to auto-detect).
        max_staleness_days: Configurable freshness threshold in days.
        reference_date: Reference date for freshness evaluation.

    Returns:
        IngestionResult containing raw_df, valid_df, quarantined_df, and ValidationReport.
    """
    resolved_fname = file_name or (file_or_path if isinstance(file_or_path, str) else "dataset.csv")

    try:
        raw_df, actual_name = read_tabular_file(file_or_path, file_name=file_name)
    except Exception as exc:
        report = ValidationReport(grain=GrainType.UNKNOWN, total_rows_read=0)
        report.add_issue(
            code=CODE_EMPTY_DATASET,
            severity=ValidationSeverity.ERROR,
            message=str(exc),
            remediation="Upload a valid, non-empty CSV or Excel file.",
        )
        return IngestionResult(
            raw_df=pd.DataFrame(),
            valid_df=pd.DataFrame(),
            quarantined_df=pd.DataFrame(),
            report=report,
            file_name=str(resolved_fname),
        )

    # Run contract validation
    report, valid_df, quarantined_df = validate_dataset_schema(
        df=raw_df,
        expected_grain=expected_grain,
        max_staleness_days=max_staleness_days,
        reference_date=reference_date,
    )

    # Compute diagnostics on the valid dataset
    detected_grain, col_map = detect_dataset_grain(valid_df if not valid_df.empty else raw_df)
    date_col = col_map.get("date")
    sales_col = col_map.get("sales")

    diagnostics = compute_data_quality_diagnostics(
        df=valid_df if not valid_df.empty else raw_df,
        grain=report.grain,
        date_col=date_col,
        sales_col=sales_col,
        max_staleness_days=max_staleness_days,
        reference_date=reference_date,
    )
    report.diagnostics = diagnostics

    return IngestionResult(
        raw_df=raw_df,
        valid_df=valid_df,
        quarantined_df=quarantined_df,
        report=report,
        file_name=actual_name,
    )


def safe_ingest_and_aggregate_to_daily(
    file_or_path: Union[str, BinaryIO, io.BytesIO, bytes],
    file_name: Optional[str] = None,
    max_staleness_days: Optional[int] = 30,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
    fill_missing_dates: bool = True,
) -> Tuple[Optional[pd.DataFrame], IngestionResult]:
    """Ingest a file, validate its schema, and aggregate valid records into a daily sales series.

    Returns:
        Tuple of (daily_sales_df or None if invalid, IngestionResult)
    """
    result = safe_ingest_file(
        file_or_path=file_or_path,
        file_name=file_name,
        max_staleness_days=max_staleness_days,
        reference_date=reference_date,
    )

    if not result.is_usable or result.valid_df.empty:
        return None, result

    # Find date and sales columns
    _, col_map = detect_dataset_grain(result.valid_df)
    date_col = col_map.get("date")
    sales_col = col_map.get("sales")

    if not date_col or not sales_col:
        return None, result

    daily_df = aggregate_to_daily_sales(
        df=result.valid_df,
        date_col=date_col,
        sales_col=sales_col,
        fill_missing_dates=fill_missing_dates,
    )

    return daily_df, result
