"""
Token issuance/verification — RS256 with auto-generated keys (dev) or
externally-provided keys via JWT_PRIVATE_KEY_PATH / JWT_PUBLIC_KEY_PATH
(production). JWKS endpoint exposed at /v1/.well-known/jwks.json.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from app.core import config

logger = logging.getLogger(__name__)

_private_key = None
_public_key = None
_kid = None


def _load_keys():
    global _private_key, _public_key, _kid
    if _private_key is not None:
        return

    priv_path = Path(config.JWT_PRIVATE_KEY_PATH) if config.JWT_PRIVATE_KEY_PATH else None
    pub_path = Path(config.JWT_PUBLIC_KEY_PATH) if config.JWT_PUBLIC_KEY_PATH else None

    if priv_path and priv_path.exists():
        _private_key = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
        _public_key = _private_key.public_key()
        _kid = config.JWT_KID or "prod-1"
        logger.info("Loaded RS256 keys from disk kid=%s", _kid)
    elif config.JWT_SECRET != "CHANGE_ME_IN_PRODUCTION":
        # Legacy HS256 mode — only if explicitly set (no more hardcoded dev default)
        _private_key = config.JWT_SECRET
        _public_key = config.JWT_SECRET
        _kid = "hs256-legacy"
        logger.warning("Using HS256 JWT — set JWT_PRIVATE_KEY_PATH for RS256 in production")
    else:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        _private_key = key
        _public_key = key.public_key()
        _kid = f"auto-{uuid.uuid4().hex[:8]}"
        logger.warning("Auto-generated RS256 keys (ephemeral, dev only) kid=%s", _kid)


def _algorithm() -> str:
    _load_keys()
    if isinstance(_private_key, str):
        return "HS256"
    return "RS256"


def get_jwks() -> dict:
    """Return a JWKS document for the current public key."""
    _load_keys()
    if isinstance(_public_key, str):
        return {"keys": []}
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    import base64

    pub_numbers = _public_key.public_numbers()

    def _b64url(n: int, length: int) -> str:
        return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()

    n_bytes = (pub_numbers.n.bit_length() + 7) // 8
    return {
        "keys": [{
            "kty": "RSA",
            "use": "sig",
            "kid": _kid,
            "alg": "RS256",
            "n": _b64url(pub_numbers.n, n_bytes),
            "e": _b64url(pub_numbers.e, 3),
        }]
    }


def issue_token(employee_id: str, application_id: str, auth_method: str = "face") -> tuple[str, int]:
    _load_keys()
    now = int(time.time())
    claims = {
        "sub": employee_id,
        "employee_id": employee_id,
        "auth_method": auth_method,
        "aud": application_id,
        "iat": now,
        "exp": now + config.JWT_TTL_SECONDS,
        "jti": str(uuid.uuid4()),
        "iss": config.JWT_ISSUER,
    }
    headers = {"kid": _kid} if _kid else {}
    token = jwt.encode(claims, _private_key, algorithm=_algorithm(), headers=headers)
    return token, config.JWT_TTL_SECONDS


def issue_stepup_token(employee_id: str, application_id: str, patient_id: str, field_name: str) -> tuple[str, int]:
    _load_keys()
    now = int(time.time())
    claims = {
        "sub": employee_id,
        "employee_id": employee_id,
        "auth_method": "face",
        "purpose": "emr_write",
        "patient_id": patient_id,
        "field_name": field_name,
        "aud": application_id,
        "iat": now,
        "exp": now + config.STEPUP_TOKEN_TTL_SECONDS,
        "jti": str(uuid.uuid4()),
        "iss": config.JWT_ISSUER,
    }
    headers = {"kid": _kid} if _kid else {}
    token = jwt.encode(claims, _private_key, algorithm=_algorithm(), headers=headers)
    return token, config.STEPUP_TOKEN_TTL_SECONDS


def verify_token(token: str, audience: str | None = None) -> dict:
    _load_keys()
    options = {"verify_aud": audience is not None}
    return jwt.decode(
        token,
        _public_key,
        algorithms=[_algorithm()],
        audience=audience,
        options=options,
    )
