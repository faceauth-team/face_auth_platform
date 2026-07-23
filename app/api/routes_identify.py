"""POST /identify """
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status

from app.api.schemas import IdentifyResponse
from app.api.utils import decode_frames
from app.auth.dependencies import require_client_token
from app.auth.tokens import issue_token
from app.core import config
from app.core.concurrency import acquire_inference_slot, release_inference_slot
from app.db.database import get_session
from app.db.models import AuthAuditLog
from app.ml.matcher import identify_from_frames

router = APIRouter()

_DECISION_TO_RESULT = {
    "MATCH": "match",
    "AMBIGUOUS": "ambiguous",
    "NO_MATCH": "no_match",
}


@router.post("/identify", response_model=IdentifyResponse)
async def identify(
    request: Request,
    application_id: str = Form(...),
    frames: list[UploadFile] = None,
    client_app_id: str = Depends(require_client_token),
):
    if not frames or not (config.IDENTIFY_MIN_FRAMES <= len(frames) <= config.IDENTIFY_MAX_FRAMES):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"expected {config.IDENTIFY_MIN_FRAMES}-{config.IDENTIFY_MAX_FRAMES} frames, got {len(frames) if frames else 0}",
        )
    # The client token already establishes which application is calling;
    # the body's application_id must match it (defense against a
    # compromised/confused client claiming to be a different app).
    if application_id != client_app_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "application_id does not match authenticated client")

    decoded = await decode_frames(frames)
    sem = await acquire_inference_slot()
    try:
        result = identify_from_frames(decoded)
    finally:
        release_inference_slot(sem)
    client_ip = request.client.host if request.client else ""
    device_info = request.headers.get("user-agent", "")

    if result.decision in ("LIVENESS_FAILED", "NO_FACE_DETECTED", "QUALITY_FAILED"):
        _log_attempt(client_app_id, None, None, "rejected", "fail" if result.decision == "LIVENESS_FAILED" else None, client_ip, device_info)
        reason = {
            "LIVENESS_FAILED": "liveness_check_failed",
            "NO_FACE_DETECTED": "no_face_detected",
            "QUALITY_FAILED": "capture_quality_too_low",
        }[result.decision]
        return IdentifyResponse(result="rejected", reason=reason, liveness="fail" if result.decision == "LIVENESS_FAILED" else None)

    api_result = _DECISION_TO_RESULT[result.decision]
    _log_attempt(client_app_id, result.employee_id, result.confidence, api_result, "pass", client_ip, device_info)

    if result.decision == "MATCH":
        token, expires_in = issue_token(result.employee_id, client_app_id)
        return IdentifyResponse(
            result="match",
            employee_id=result.employee_id,
            confidence=round(result.confidence, 3),
            liveness="pass",
            auth_token=token,
            expires_in=expires_in,
        )

    # AMBIGUOUS or NO_MATCH (spec Section 8: ambiguous -> "request liveness
    # re-check or fallback (PIN/badge)" — surfaced as a distinct result so
    # the calling app's UI can offer that fallback rather than silently
    # treating it the same as a confident no_match).
    return IdentifyResponse(
        result=api_result,
        confidence=round(result.confidence, 3),
        liveness="pass",
        reason="step_up_required" if result.decision == "AMBIGUOUS" else None,
    )


def _log_attempt(application_id, employee_id, confidence, result, liveness_result, ip, device_info):
    with get_session() as session:
        session.add(
            AuthAuditLog(
                employee_id=employee_id,
                application_id=application_id,
                result=result,
                confidence=confidence,
                liveness_result=liveness_result,
                ip_address=ip,
                device_info=device_info,
            )
        )
