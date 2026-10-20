"""
SalesPulse AI - Shared Validation Domain Service
Provides:
- Encapsulation of safe file ingestion and tabular contract validation.
- File SHA-256 fingerprinting and strict JSON audit report generation.
- Privacy-safe persistence via AuditRepository (never stores raw customer data).
"""

import os
import hashlib
from typing import Optional, Union, BinaryIO, Dict, Any

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
        result: IngestionResult = safe_ingest_file(
            file_or_path=file_bytes,
            file_name=sanitized_filename,
            expected_grain=expected_grain,
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
