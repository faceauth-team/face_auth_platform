"""Pydantic response models mirroring JSON shapes."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EnrollAcceptedResponse(BaseModel):
    enrollment_id: str
    status: str
    frames_received: int
    estimated_completion_seconds: int


class EnrollStatusResponse(BaseModel):
    enrollment_id: str
    status: str
    frames_processed: int
    frames_accepted: int
    templates_generated: int
    liveness_passed: bool
    employee_id: str
    failure_reason: Optional[str] = None
    progress_percent: Optional[int] = None


class IdentifyResponse(BaseModel):
    result: str  # "match" | "no_match" | "ambiguous" | "rejected"
    employee_id: Optional[str] = None
    confidence: Optional[float] = None
    liveness: Optional[str] = None  # "pass" | "fail"
    auth_token: Optional[str] = None
    expires_in: Optional[int] = None
    reason: Optional[str] = None


class AuditLogEntry(BaseModel):
    timestamp: str
    employee_id: Optional[str]
    result: str
    confidence: Optional[float]
    application_id: Optional[str]
    liveness_result: Optional[str]
    request_id: Optional[str] = None


class AuditLogResponse(BaseModel):
    logs: list[AuditLogEntry]


class EmployeeSummary(BaseModel):
    employee_id: str
    full_name: str
    department: Optional[str] = None
    email: Optional[str] = None
    status: str
    template_count: int
    enrolled_at: Optional[str] = None


class EmployeeListResponse(BaseModel):
    count: int
    employees: list[EmployeeSummary]


class EmployeeUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    status: Optional[str] = None  # active | suspended | offboarded


class AuthorizeResponse(BaseModel):
    authorized: bool
    employee_id: Optional[str] = None
    confidence: Optional[float] = None
    liveness: Optional[str] = None
    auth_token: Optional[str] = None   # scoped, single-use step-up token
    expires_in: Optional[int] = None
    patient_id: Optional[str] = None
    field_name: Optional[str] = None
    reason: Optional[str] = None


class EMRWriteRequest(BaseModel):
    patient_id: str
    field_name: str
    value: str


class EMREntryResponse(BaseModel):
    entry_id: str
    patient_id: str
    field_name: str
    value: str
    author_employee_id: str
    auth_method: Optional[str] = None
    created_at: str


class EMRWriteReceipt(BaseModel):
    """Signed receipt returned instead of echoing PHI back. Contains a
    hash of the written value so the EMR can verify the write landed."""
    entry_id: str
    patient_id: str
    field_name: str
    value_hash: str
    author_employee_id: str
    auth_method: Optional[str] = None
    created_at: str
    entry_hash: str


class EMREntryListResponse(BaseModel):
    entries: list[EMREntryResponse]
