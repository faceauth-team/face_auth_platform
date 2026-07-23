"""GET /audit-logs — Append-only (WORM): no update or
delete endpoints exposed. The audit log is the immutable record."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.schemas import AuditLogEntry, AuditLogResponse
from app.auth.dependencies import require_admin_token
from app.db.database import get_session
from app.db.models import AuthAuditLog

router = APIRouter()


@router.get("/audit-logs", response_model=AuditLogResponse)
async def audit_logs(
    employee_id: Optional[str] = Query(default=None),
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
    _admin=Depends(require_admin_token),
):
    with get_session() as session:
        q = session.query(AuthAuditLog)
        if employee_id:
            q = q.filter(AuthAuditLog.employee_id == employee_id)
        if from_:
            q = q.filter(AuthAuditLog.created_at >= datetime.fromisoformat(from_))
        if to:
            q = q.filter(AuthAuditLog.created_at <= datetime.fromisoformat(to))
        rows = q.order_by(AuthAuditLog.created_at.desc()).limit(limit).all()

        return AuditLogResponse(
            logs=[
                AuditLogEntry(
                    timestamp=row.created_at.isoformat() if row.created_at else "",
                    employee_id=row.employee_id,
                    result=row.result,
                    confidence=row.confidence,
                    application_id=row.application_id,
                    liveness_result=row.liveness_result,
                    request_id=row.request_id,
                )
                for row in rows
            ]
        )
