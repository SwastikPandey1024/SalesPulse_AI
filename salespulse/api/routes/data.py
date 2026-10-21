"""
SalesPulse AI - Data Ingestion & Quality Validation Routes
Provides:
- POST /api/v1/data/validate: Validates tabular CSV/XLSX file uploads or JSON record batches.
- Enforces 10MB upload file size limit.
- Sanitizes file names to prevent local path disclosure or traversal.
- Conforms to existing ValidationReport contracts and audit logging.
"""

import os
from typing import Optional, List, Dict, Any
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Body
from pydantic import BaseModel, Field

from salespulse.api.schemas import DataValidationResponse, ValidationIssueSchema
from salespulse.api.dependencies import get_validation_service
from salespulse.services.validation_service import ValidationService

router = APIRouter(prefix="/api/v1/data", tags=["Data Quality & Validation"])

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


class JSONDataValidationRequest(BaseModel):
    """Payload for direct JSON tabular records validation."""
    source_name: Optional[str] = Field("api_payload.json", max_length=100)
    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="List of row dictionaries (up to 50,000 rows)",
    )
    persist_audit: bool = Field(True, description="Whether to store validation summary in audit logs")


@router.post(
    "/validate",
    response_model=DataValidationResponse,
    summary="Validate Tabular Dataset",
    description="Validates a CSV/XLSX file upload or JSON tabular record set against data quality contracts.",
)
async def validate_dataset(
    file: Optional[UploadFile] = File(None, description="Tabular dataset file (CSV or XLSX, <= 10MB)"),
    persist_audit: bool = Form(True, description="Whether to record validation in persistent audit storage"),
    validation_service: ValidationService = Depends(get_validation_service),
) -> DataValidationResponse:
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A file upload is required for this endpoint. Send as multipart form data.",
        )

    # 1. Sanitize file name (basename only, reject null bytes and path traversal)
    raw_filename = os.path.basename(file.filename or "").replace("\x00", "").strip()
    sanitized_filename = raw_filename.replace("/", "").replace("\\", "")
    if not sanitized_filename or sanitized_filename in (".", ".."):
        sanitized_filename = "uploaded_file.csv"

    # Validate file extension
    sanitized_lower = sanitized_filename.lower()
    if not (sanitized_lower.endswith(".csv") or sanitized_lower.endswith(".xlsx") or sanitized_lower.endswith(".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{sanitized_filename}'. Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.",
        )

    # 2. Enforce upload size limit via bounded streaming read (prevents OOM DoS)
    CHUNK_SIZE = 64 * 1024  # 64 KB
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    content = b"".join(chunks)

    # 3. Execute validation via shared service
    try:
        result = validation_service.validate_file(
            file_source=content,
            file_name=sanitized_filename,
            persist_audit=persist_audit,
        )
        report = result.report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse and validate file: {str(e)}",
        )

    rep_dict = report.to_dict()
    issues_out = [
        ValidationIssueSchema(
            code=iss.get("code", "UNKNOWN"),
            severity=iss.get("severity", "INFO"),
            message=iss.get("message", ""),
            affected_count=int(iss.get("affected_count", 0)),
            sample_rows=iss.get("sample_rows"),
            remediation=iss.get("remediation", ""),
        )
        for iss in rep_dict.get("issues", [])
    ]

    return DataValidationResponse(
        verdict=rep_dict.get("verdict", "FAILED"),
        grain=rep_dict.get("grain", "unknown"),
        total_rows_read=int(rep_dict.get("total_rows_read", 0)),
        valid_rows_count=int(rep_dict.get("valid_rows_count", 0)),
        quarantined_rows_count=int(rep_dict.get("quarantined_rows_count", 0)),
        diagnostics=rep_dict.get("diagnostics", {}),
        issues=issues_out,
    )


@router.post(
    "/validate-json",
    response_model=DataValidationResponse,
    summary="Validate JSON Tabular Records",
    description="Validates a JSON array of tabular rows against data quality contracts.",
)
def validate_json_records(
    payload: JSONDataValidationRequest,
    validation_service: ValidationService = Depends(get_validation_service),
) -> DataValidationResponse:
    try:
        df = pd.DataFrame(payload.records)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        result = validation_service.validate_file(
            file_source=csv_bytes,
            file_name="api_payload.csv",
            persist_audit=payload.persist_audit,
        )
        report = result.report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tabular record format: {str(e)}",
        )

    rep_dict = report.to_dict()

    issues_out = [
        ValidationIssueSchema(
            code=iss.get("code", "UNKNOWN"),
            severity=iss.get("severity", "INFO"),
            message=iss.get("message", ""),
            affected_count=int(iss.get("affected_count", 0)),
            sample_rows=iss.get("sample_rows"),
            remediation=iss.get("remediation", ""),
        )
        for iss in rep_dict.get("issues", [])
    ]

    return DataValidationResponse(
        verdict=rep_dict.get("verdict", "FAILED"),
        grain=rep_dict.get("grain", "unknown"),
        total_rows_read=int(rep_dict.get("total_rows_read", 0)),
        valid_rows_count=int(rep_dict.get("valid_rows_count", 0)),
        quarantined_rows_count=int(rep_dict.get("quarantined_rows_count", 0)),
        diagnostics=rep_dict.get("diagnostics", {}),
        issues=issues_out,
    )
