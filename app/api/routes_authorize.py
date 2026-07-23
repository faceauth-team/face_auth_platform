"""
Step-up authorization — the EMR-plugin entry point.

A clinician is already logged into the EMR (email/SSO). When they try to
change a gated field, the EMR calls POST /authorize with: who they claim to
be (employee_id, resolved from the EMR identity), the patient + field being
changed, and a short face-capture burst. We:

  1. enforce a per-clinician lockout (repeated failures -> temporary block),
  2. 1:1 *verify* the live face is really that clinician (not 1:N — we know
     who is claimed),
  3. on success, issue a single-use token scoped to exactly this
     patient+field, which the subsequent /emr/entries call must present.

There is deliberately NO PIN/password fallback: the whole point is a factor
that cannot be delegated to a junior. A failure denies the action (audited).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status

from app.api.schemas import AuthorizeResponse
from app.api.utils import decode_frames
from app.auth.dependencies import require_client_token
from app.auth.tokens import issue_stepup_token
from app.core import config
from app.core.concurrency import acquire_inference_slot, release_inference_slot
from app.db.database import get_session
from app.db.models import AuthAuditLog, Employee
from app.ml.matcher import verify_identity

router = APIRouter()


def _is_locked_out(session, employee_id: str) -> bool:
    """True if this clinician has had >= MAX_FAILED_ATTEMPTS failed step-ups
    within the lockout window (spec §9.5: velocity/rate limiting)."""
    since = datetime.now(timezone.utc) - timedelta(seconds=config.LOCKOUT_WINDOW_SECONDS)
    fails = (
        session.query(AuthAuditLog)
        .filter(
            AuthAuditLog.employee_id == employee_id,
            AuthAuditLog.result.in_(("rejected", "no_match")),
            AuthAuditLog.created_at >= since,
        )
        .count()
    )
    return fails >= config.MAX_FAILED_ATTEMPTS


@router.post("/authorize", response_model=AuthorizeResponse)
async def authorize(
    request: Request,
    patient_id: str = Form(...),
    field_name: str = Form(...),
    employee_id: str = Form(default=""),
    email: str = Form(default=""),
    frames: list[UploadFile] = None,
    client_app_id: str = Depends(require_client_token),
):
    """Identity is supplied by the EMR session. The EMR identifies a clinician
    by their login `email`; we resolve that to the enrolled `employee_id` via
    the stored mapping (the identity-binding lookup). `employee_id` may also be
    passed directly. Exactly one of the two is required."""
    if not frames or not (config.IDENTIFY_MIN_FRAMES <= len(frames) <= config.IDENTIFY_MAX_FRAMES):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"expected {config.IDENTIFY_MIN_FRAMES}-{config.IDENTIFY_MAX_FRAMES} frames",
        )
    if not employee_id and not email:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "employee_id or email is required")

    client_ip = request.client.host if request.client else ""
    device_info = request.headers.get("user-agent", "")

    with get_session() as session:
        if not employee_id:
            emp = session.query(Employee).filter(Employee.email == email).first()
            if emp is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no enrolled clinician for this email")
            employee_id = emp.employee_id
        elif session.get(Employee, employee_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "clinician not enrolled")
        if _is_locked_out(session, employee_id):
            _log(session, client_app_id, employee_id, "rejected", None, client_ip, device_info)
            raise HTTPException(
                status.HTTP_423_LOCKED,
                f"too many failed attempts; locked for up to {config.LOCKOUT_WINDOW_SECONDS}s",
            )

    decoded = await decode_frames(frames)
    sem = await acquire_inference_slot()
    try:
        result = verify_identity(employee_id, decoded)
    finally:
        release_inference_slot(sem)

    with get_session() as session:
        if result.decision == "MATCH":
            _log(session, client_app_id, employee_id, "match", result.confidence, client_ip, device_info, "pass")
            token, ttl = issue_stepup_token(employee_id, client_app_id, patient_id, field_name)
            return AuthorizeResponse(
                authorized=True, employee_id=employee_id, confidence=round(result.confidence, 3),
                liveness="pass", auth_token=token, expires_in=ttl,
                patient_id=patient_id, field_name=field_name,
            )

        if result.decision == "LIVENESS_FAILED":
            _log(session, client_app_id, employee_id, "rejected", None, client_ip, device_info, "fail")
            return AuthorizeResponse(authorized=False, liveness="fail", reason="liveness_check_failed")

        _log(session, client_app_id, employee_id, "no_match", result.confidence, client_ip, device_info, "pass")
        reason = {
            "NO_FACE_DETECTED": "no_face_detected",
            "QUALITY_FAILED": "capture_quality_too_low",
        }.get(result.decision, "face_does_not_match_clinician")
        return AuthorizeResponse(
            authorized=False, confidence=round(result.confidence, 3) if result.confidence else 0.0,
            liveness="pass", reason=reason,
        )


def _log(session, app_id, employee_id, result, confidence, ip, device, liveness=None):
    session.add(AuthAuditLog(
        employee_id=employee_id, application_id=app_id, result=result,
        confidence=confidence, liveness_result=liveness, ip_address=ip, device_info=device,
    ))
