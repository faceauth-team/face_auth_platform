"""
SQLAlchemy models — a direct mapping of spec Section 12's PostgreSQL
schema. Runs against SQLite by default (prototype) or Postgres (set
DATABASE_URL) with no model changes; UUID/TIMESTAMPTZ/BIGSERIAL are
expressed via SQLAlchemy's portable types so both backends work.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(String(20), primary_key=True)
    full_name = Column(String(255), nullable=False)
    department = Column(String(100))
    email = Column(String(255))
    status = Column(String(20), default="active")  # active, suspended, offboarded
    created_at = Column(DateTime(timezone=True), default=_now)


class FaceTemplate(Base):
    __tablename__ = "face_templates"

    template_id = Column(String(36), primary_key=True, default=_uuid)
    employee_id = Column(String(20), ForeignKey("employees.employee_id"))
    vector_db_id = Column(String(64), nullable=False)  # reference into FaceIndex (app/ml/vector_index.py)
    template_version = Column(Integer, nullable=False, default=1)
    quality_score = Column(Float)
    pose_bucket = Column(String(20))  # 'frontal','left','right','up','down'
    created_at = Column(DateTime(timezone=True), default=_now)
    is_active = Column(Boolean, default=True)


class EnrollmentJob(Base):
    __tablename__ = "enrollment_jobs"

    enrollment_id = Column(String(36), primary_key=True, default=_uuid)
    employee_id = Column(String(20), ForeignKey("employees.employee_id"))
    status = Column(String(20))  # processing, completed, failed
    frames_received = Column(Integer)
    frames_accepted = Column(Integer)
    templates_generated = Column(Integer)
    liveness_passed = Column(Boolean)
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True))
    failure_reason = Column(Text)  # not in spec's literal DDL, but referenced by status="failed" handling


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    consent_id = Column(String(36), primary_key=True, default=_uuid)
    employee_id = Column(String(20), ForeignKey("employees.employee_id"))
    consent_text_version = Column(String(20))
    signed_at = Column(DateTime(timezone=True), default=_now)
    ip_address = Column(String(45))
    revoked_at = Column(DateTime(timezone=True))


class EMREntry(Base):
    """A clinical value written into a record, authorized by a face-issued
    JWT. Captures who wrote it (clinician employee_id from the verified
    token), into which patient/field, and when — an immutable signed entry."""
    __tablename__ = "emr_entries"

    entry_id = Column(String(36), primary_key=True, default=_uuid)
    # Monotonic sequence assigned under the chain lock at write time — gives a
    # reliable total order for the hash chain (timestamps can collide at
    # microsecond resolution under concurrent writes; UUIDs aren't ordered).
    seq = Column(Integer, index=True)
    patient_id = Column(String(64), nullable=False)
    field_name = Column(String(100), nullable=False)  # e.g. "plan_of_management"
    value = Column(Text, nullable=False)
    author_employee_id = Column(String(20), nullable=False)  # from the verified JWT
    auth_method = Column(String(20))  # "face"
    token_jti = Column(String(64))    # JWT id, links the write to a specific auth event
    created_at = Column(DateTime(timezone=True), default=_now)
    # Tamper-evident hash chain: each entry's hash covers its content + the
    # previous entry's hash, so any retroactive edit/deletion breaks the chain.
    prev_hash = Column(String(64))
    entry_hash = Column(String(64))


class ConsumedToken(Base):
    """Single-use step-up token ledger. A token's jti is recorded here the
    first time it authorizes a write; a second attempt with the same jti is
    rejected — so one face approval authorizes exactly one action."""
    __tablename__ = "consumed_tokens"

    jti = Column(String(64), primary_key=True)
    employee_id = Column(String(20))
    consumed_at = Column(DateTime(timezone=True), default=_now)


class AuthAuditLog(Base):
    __tablename__ = "auth_audit_log"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(20))
    application_id = Column(String(50))
    result = Column(String(20))  # match, no_match, rejected
    confidence = Column(Float)
    liveness_result = Column(String(10))
    ip_address = Column(String(45))
    device_info = Column(Text)
    request_id = Column(String(36))
    created_at = Column(DateTime(timezone=True), default=_now)


class AdminUser(Base):
    """Per-user admin accounts with role-based access (replaces single
    shared ADMIN_TOKEN). Passwords stored as bcrypt hashes."""
    __tablename__ = "admin_users"

    admin_id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(100), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="admin")  # admin, super_admin
    employee_id = Column(String(20), ForeignKey("employees.employee_id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_now)
