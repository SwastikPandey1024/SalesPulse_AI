"""
SalesPulse AI - Shared Validation Domain Service
Provides:
- Encapsulation of safe file ingestion and tabular contract validation.
- File SHA-256 fingerprinting and strict JSON audit report generation.
- Privacy-safe persistence via AuditRepository (never stores raw customer data).
"""

import os
import hashlib
from typing import Optional, Union, BinaryIO, Dict, Any, Tuple

from salespulse.data.contracts import ValidationReport, GrainType
from salespulse.data.ingestion import safe_ingest_file, IngestionResult
from salespulse.storage.repository import AuditRepository
from salespulse.storage.models import DatasetAuditLog


class ValidationService:
    """Enterprise domain service for file validation, contract enforcement, and audit persistence."""

    def __init__(self, repository: Optional[AuditRepository] = None):
        self.repository = repository or AuditRepository()

    def validate_file(
        self,
        file_source: Union[bytes, BinaryIO],
        file_name: str,
        expected_grain: Optional[GrainType] = None,
        persist_audit: bool = True,
    ) -> IngestionResult:
        """Safely ingest and validate tabular file with optional persistent audit logging."""
        # 1. Read bytes for SHA-256 computation
        if isinstance(file_source, bytes):
            file_bytes = file_source
        elif hasattr(file_source, "read"):
            pos = file_source.tell() if hasattr(file_source, "tell") else 0
            file_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(pos)
        else:
            raise TypeError(f"file_source must be bytes or file-like binary buffer, got {type(file_source)}")

        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        sanitized_filename = os.path.basename(file_name)

        # 2. Execute existing safe ingestion pipeline
        try:
            result: IngestionResult = safe_ingest_file(
                file_or_path=file_bytes,
                file_name=sanitized_filename,
                expected_grain=expected_grain,
            )
        except Exception as exc:
            # Defensive fallback: build failure report and persist audit log
            report = ValidationReport(grain=GrainType.UNKNOWN, total_rows_read=0)
            from salespulse.data.contracts import ValidationSeverity, CODE_EMPTY_DATASET
            report.add_issue(
                code=CODE_EMPTY_DATASET,
                severity=ValidationSeverity.ERROR,
                message=f"Ingestion pipeline error: {str(exc)}",
                remediation="Ensure the file is a well-formed CSV or Excel tabular document.",
            )
            import pandas as pd
            result = IngestionResult(
                raw_df=pd.DataFrame(),
                valid_df=pd.DataFrame(),
                quarantined_df=pd.DataFrame(),
                report=report,
                file_name=sanitized_filename,
            )

        # 3. Optional persistent audit record (Privacy safe: no raw data stored)
        if persist_audit:
            report: ValidationReport = result.report
            err_cnt = len([i for i in report.issues if i.severity.value == "ERROR"])
            warn_cnt = len([i for i in report.issues if i.severity.value == "WARNING"])
            is_usable = (report.status != "FAILED" and report.valid_rows_count > 0)
            self.repository.record_validation(
                file_sha256=file_sha256,
                file_name=sanitized_filename,
                detected_grain=report.grain.value if hasattr(report.grain, "value") else str(report.grain),
                row_count=report.total_rows_read,
                validation_status=report.status,
                is_usable=is_usable,
                error_count=err_cnt,
                warning_count=warn_cnt,
                report=report,
            )

        return result

    def ingest_and_aggregate_daily(
        self,
        file_source: Union[bytes, BinaryIO],
        file_name: str,
        sales_scope: str = "TOTAL_STORE",
        persist_actuals: bool = True,
        persist_audit: bool = True,
    ) -> Tuple[Optional[Any], IngestionResult]:
        """End-to-end ingestion pipeline: validate, aggregate to daily sales, and idempotently persist actuals.
        
        Returns:
            Tuple[Optional[pd.DataFrame], IngestionResult]: (daily_sales_df or None if invalid, result)
        """
        import pandas as pd
        from salespulse.data.contracts import detect_dataset_grain
        from salespulse.data.time_series import aggregate_to_daily_sales

        result = self.validate_file(
            file_source=file_source,
            file_name=file_name,
            persist_audit=persist_audit,
        )

        if not result.is_usable or result.valid_df.empty:
            return None, result

        _, col_map = detect_dataset_grain(result.valid_df)
        date_col = col_map.get("date")
        sales_col = col_map.get("sales")

        if not date_col or not sales_col:
            return None, result

        daily_df = aggregate_to_daily_sales(
            df=result.valid_df,
            date_col=date_col,
            sales_col=sales_col,
            fill_missing_dates=True,
        )

        if persist_actuals and self.repository:
            # Extract file SHA-256 from audit record or compute
            file_sha = getattr(result.report, "file_sha256", None)
            records = [
                {"calendar_date": row["date"], "daily_sales": row["sales"]}
                for _, row in daily_df.iterrows()
            ]
            self.repository.ingest_actual_sales_batch(
                records=records,
                sales_scope=sales_scope,
                source_file_sha256=file_sha,
            )

        return daily_df, result
