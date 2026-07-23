"""
POST /enroll, GET /enroll/{enrollment_id}/status,
DELETE /employees/{employee_id}/templates 
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from app.api.schemas import (
    EnrollAcceptedResponse,
    EnrollStatusResponse,
    EmployeeListResponse,
    EmployeeSummary,
    EmployeeUpdateRequest,
)
from app.api.utils import decode_frames
from app.auth.dependencies import require_admin_token
from app.core import config
from app.db.database import get_session
from app.db.models import EnrollmentJob, FaceTemplate, Employee, AuthAuditLog, ConsentRecord
from app.ml.vector_index import get_index

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/enroll", status_code=status.HTTP_202_ACCEPTED, response_model=EnrollAcceptedResponse)
async def enroll(
    request: Request,
    background_tasks: BackgroundTasks,
    employee_id: str = Form(...),
    consent_token: str = Form(...),
    full_name: str = Form(default=""),
    department: str = Form(default=""),
    email: str = Form(default=""),
    frames: list[UploadFile] = None,
    _admin=Depends(require_admin_token),
):
    if not frames:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no frames provided")
    if len(frames) > config.ENROLL_MAX_FRAMES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"too many frames ({len(frames)}), max is {config.ENROLL_MAX_FRAMES}",
        )

    decoded = await decode_frames(frames, max_bytes=config.MAX_ENROLL_UPLOAD_SIZE_BYTES)

    with get_session() as session:
        job = EnrollmentJob(
            employee_id=employee_id,
            status="processing",
            frames_received=len(decoded),
            frames_accepted=0,
            templates_generated=0,
            liveness_passed=False,
        )
        session.add(job)
        session.flush()
        enrollment_id = job.enrollment_id

    from app.workers.enrollment_job import process_enrollment

    background_tasks.add_task(
        process_enrollment,
        enrollment_id=enrollment_id,
        employee_id=employee_id,
        full_name=full_name,
        department=department,
        email=email,
        frames_bgr=decoded,
        consent_token=consent_token,
        client_ip=request.client.host if request.client else "",
    )

    return EnrollAcceptedResponse(
        enrollment_id=enrollment_id,
        status="processing",
        frames_received=len(decoded),
        estimated_completion_seconds=max(5, len(decoded) // 20),
    )


@router.get("/enroll/{enrollment_id}/status", response_model=EnrollStatusResponse)
async def enrollment_status(enrollment_id: str, _admin=Depends(require_admin_token)):
    with get_session() as session:
        job = session.get(EnrollmentJob, enrollment_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "enrollment_id not found")

        from app.core.progress import get_progress
        prog = get_progress(enrollment_id)
        if job.status in ("completed", "failed"):
            pct = 100
        elif prog and prog["total"]:
            # Reserve the top 5% for the finalizing phase (index build + DB
            # write) so the bar climbs 0->95 during analysis, holds, then
            # jumps to 100 — never showing 100 then resetting.
            pct = min(95, int(prog["processed"] / prog["total"] * 95))
        else:
            pct = 95  # analysis done, finalizing

        return EnrollStatusResponse(
            enrollment_id=job.enrollment_id,
            status=job.status,
            frames_processed=job.frames_received or 0,
            frames_accepted=job.frames_accepted or 0,
            templates_generated=job.templates_generated or 0,
            liveness_passed=bool(job.liveness_passed),
            employee_id=job.employee_id,
            failure_reason=job.failure_reason,
            progress_percent=pct,
        )


@router.get("/employees", response_model=EmployeeListResponse)
async def list_employees(_admin=Depends(require_admin_token)):
    index = get_index()
    with get_session() as session:
        employees = session.query(Employee).order_by(Employee.created_at.desc()).all()
        summaries = []
        for e in employees:
            active_templates = (
                session.query(FaceTemplate)
                .filter_by(employee_id=e.employee_id, is_active=True)
                .count()
            )
            summaries.append(
                EmployeeSummary(
                    employee_id=e.employee_id,
                    full_name=e.full_name,
                    department=e.department,
                    email=e.email,
                    status=e.status or "active",
                    template_count=active_templates,
                    enrolled_at=e.created_at.isoformat() if e.created_at else None,
                )
            )
    return EmployeeListResponse(count=len(summaries), employees=summaries)


@router.patch("/employees/{employee_id}", response_model=EmployeeSummary)
async def update_employee(
    employee_id: str,
    payload: EmployeeUpdateRequest,
    _admin=Depends(require_admin_token),
):
    if payload.status is not None and payload.status not in ("active", "suspended", "offboarded"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "status must be active, suspended or offboarded")
    with get_session() as session:
        employee = session.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "employee not found")
        if payload.full_name is not None:
            employee.full_name = payload.full_name
        if payload.department is not None:
            employee.department = payload.department
        if payload.email is not None:
            employee.email = payload.email
        if payload.status is not None:
            employee.status = payload.status
        session.add(
            AuthAuditLog(
                employee_id=employee_id,
                application_id="admin-api",
                result="employee_updated",
                confidence=None,
                liveness_result=None,
            )
        )
        active_templates = (
            session.query(FaceTemplate).filter_by(employee_id=employee_id, is_active=True).count()
        )
        return EmployeeSummary(
            employee_id=employee.employee_id,
            full_name=employee.full_name,
            department=employee.department,
            email=employee.email,
            status=employee.status or "active",
            template_count=active_templates,
            enrolled_at=employee.created_at.isoformat() if employee.created_at else None,
        )


@router.delete("/employees/{employee_id}/templates", status_code=status.HTTP_204_NO_CONTENT)
async def delete_employee_templates(employee_id: str, _admin=Depends(require_admin_token)):
    with get_session() as session:
        employee = session.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "employee not found")

        templates = session.query(FaceTemplate).filter_by(employee_id=employee_id, is_active=True).all()
        for t in templates:
            t.is_active = False

        session.add(
            AuthAuditLog(
                employee_id=employee_id,
                application_id="admin-api",
                result="templates_deleted",
                confidence=None,
                liveness_result=None,
            )
        )

    get_index().remove_employee(employee_id)
    get_index().refit_discriminant_and_rebuild()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/employees/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_employee(employee_id: str, _admin=Depends(require_admin_token)):
    """Full right-to-erasure: remove the clinician record and all biometric
    data (templates, enrollment jobs, consent records) + their vectors in the
    index. EMR clinical entries and the audit trail are intentionally retained
    (they reference employee_id as plain strings, not FKs) for compliance."""
    with get_session() as session:
        employee = session.get(Employee, employee_id)
        if employee is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "employee not found")

        session.query(FaceTemplate).filter_by(employee_id=employee_id).delete()
        session.query(EnrollmentJob).filter_by(employee_id=employee_id).delete()
        session.query(ConsentRecord).filter_by(employee_id=employee_id).delete()
        session.delete(employee)

        session.add(
            AuthAuditLog(
                employee_id=employee_id,
                application_id="admin-api",
                result="employee_deleted",
                confidence=None,
                liveness_result=None,
            )
        )

    get_index().remove_employee(employee_id)
    get_index().refit_discriminant_and_rebuild()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
