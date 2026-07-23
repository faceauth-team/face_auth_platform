"""
EMR write endpoints — returns a signed receipt rather than storing the
clinical value in the auth service. The EMR is the system of record for
PHI; this service records only the authorization proof (who, when, what
field, hash of value) so the audit chain can verify the write happened
without holding the PHI itself.

If ENCRYPT_PHI_AT_REST is set and PHI must be stored (e.g. for audit
replay in environments without an EMR), the value is encrypted with
Fernet before persistence.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.schemas import EMRWriteRequest, EMREntryResponse, EMREntryListResponse, EMRWriteReceipt
from app.auth.dependencies import require_admin_token, require_face_token
from app.db.database import get_session
from app.db.models import ConsumedToken, EMREntry

logger = logging.getLogger(__name__)
router = APIRouter()

_chain_lock = threading.Lock()

_PHI_ENCRYPTION_KEY = os.getenv("ENCRYPT_PHI_AT_REST", "").strip()


def _encrypt_value(plaintext: str) -> str:
    if not _PHI_ENCRYPTION_KEY:
        return plaintext
    from cryptography.fernet import Fernet
    f = Fernet(_PHI_ENCRYPTION_KEY.encode())
    return f.encrypt(plaintext.encode()).decode()


def _decrypt_value(ciphertext: str) -> str:
    if not _PHI_ENCRYPTION_KEY:
        return ciphertext
    try:
        from cryptography.fernet import Fernet
        f = Fernet(_PHI_ENCRYPTION_KEY.encode())
        return f.decrypt(ciphertext.encode()).decode()
    except Exception:
        return "[encrypted]"


def _canonical_ts(dt) -> str:
    if dt is None:
        return ""
    return dt.replace(tzinfo=None).isoformat()


def _chain_hash(prev_hash: str, entry: EMREntry) -> str:
    payload = "|".join([
        prev_hash or "",
        entry.patient_id, entry.field_name,
        hashlib.sha256(entry.value.encode()).hexdigest(),
        entry.author_employee_id, entry.token_jti or "",
        _canonical_ts(entry.created_at),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


@router.post("/emr/entries", response_model=EMRWriteReceipt, status_code=201)
async def write_emr_entry(payload: EMRWriteRequest, claims: dict = Depends(require_face_token)):
    if claims.get("purpose") != "emr_write":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "token is not a step-up authorization token")
    if claims.get("patient_id") != payload.patient_id or claims.get("field_name") != payload.field_name:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "authorization does not match this patient/field")

    author = claims["employee_id"]
    jti = claims.get("jti")
    value_hash = hashlib.sha256(payload.value.encode()).hexdigest()

    with _chain_lock, get_session() as session:
        if jti and session.get(ConsumedToken, jti) is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "authorization already used")
        if jti:
            session.add(ConsumedToken(jti=jti, employee_id=author))

        from sqlalchemy import func
        last = session.query(EMREntry).order_by(EMREntry.seq.desc()).first()
        prev_hash = last.entry_hash if last else None
        next_seq = (session.query(func.max(EMREntry.seq)).scalar() or 0) + 1

        stored_value = _encrypt_value(payload.value)

        entry = EMREntry(
            seq=next_seq,
            patient_id=payload.patient_id,
            field_name=payload.field_name,
            value=stored_value,
            author_employee_id=author,
            auth_method=claims.get("auth_method", "face"),
            token_jti=jti,
            prev_hash=prev_hash,
        )
        session.add(entry)
        session.flush()
        entry.entry_hash = _chain_hash(prev_hash, entry)

        return EMRWriteReceipt(
            entry_id=entry.entry_id,
            patient_id=entry.patient_id,
            field_name=entry.field_name,
            value_hash=value_hash,
            author_employee_id=entry.author_employee_id,
            auth_method=entry.auth_method,
            created_at=entry.created_at.isoformat() if entry.created_at else "",
            entry_hash=entry.entry_hash,
        )


@router.get("/emr/entries", response_model=EMREntryListResponse)
async def list_emr_entries(
    patient_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
    _admin=Depends(require_admin_token),
):
    with get_session() as session:
        q = session.query(EMREntry)
        if patient_id:
            q = q.filter(EMREntry.patient_id == patient_id)
        rows = q.order_by(EMREntry.created_at.desc()).limit(limit).all()
        return EMREntryListResponse(
            entries=[
                EMREntryResponse(
                    entry_id=r.entry_id,
                    patient_id=r.patient_id,
                    field_name=r.field_name,
                    value=_decrypt_value(r.value),
                    author_employee_id=r.author_employee_id,
                    auth_method=r.auth_method,
                    created_at=r.created_at.isoformat() if r.created_at else "",
                )
                for r in rows
            ]
        )
