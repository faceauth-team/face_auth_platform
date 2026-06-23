"""
Central configuration for the Face Authentication Platform.

All values are overridable via environment variables. No hardcoded dev
secrets — unset required values fail loudly at startup.
"""
import os
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load a local .env (if present) before reading any env vars below. Real
# deployments inject env vars directly; this only helps local development.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
DATA_DIR = Path(os.getenv("FACE_AUTH_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR/'face_auth.db'}")

# --- Vector index ----------------------------------------------------------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")
VECTOR_INDEX_PATH = Path(os.getenv("VECTOR_INDEX_PATH", DATA_DIR / "vector.index"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# --- Object storage --------------------------------------------------------
RAW_FRAME_STORE = os.getenv("RAW_FRAME_STORE", "local")
RAW_FRAME_DIR = Path(os.getenv("RAW_FRAME_DIR", DATA_DIR / "raw_frames"))
RAW_FRAME_RETENTION_DAYS = int(os.getenv("RAW_FRAME_RETENTION_DAYS", "0"))

# --- Enrollment capture targets (Spec FR-1, Section 7) ---------------------
ENROLL_MIN_FRAMES = int(os.getenv("ENROLL_MIN_FRAMES", "300"))
ENROLL_MAX_FRAMES = int(os.getenv("ENROLL_MAX_FRAMES", "500"))
ENROLL_TEMPLATES_MIN = int(os.getenv("ENROLL_TEMPLATES_MIN", "10"))
# Cap how many of the captured frames are actually run through the heavy
# detect->quality->embed path. The full burst (~300) is overkill for ~20
# templates and is slow on CPU; subsampling evenly preserves pose variety.
ENROLL_MAX_PROCESS_FRAMES = int(os.getenv("ENROLL_MAX_PROCESS_FRAMES", "120"))
ENROLL_TEMPLATES_MAX = int(os.getenv("ENROLL_TEMPLATES_MAX", "20"))

# --- Quality filtering thresholds (Section 7.1) -----------------------------
QUALITY_MIN_BLUR_VAR = float(os.getenv("QUALITY_MIN_BLUR_VAR", "60.0"))
QUALITY_MIN_BRIGHTNESS = float(os.getenv("QUALITY_MIN_BRIGHTNESS", "40.0"))
QUALITY_MAX_BRIGHTNESS = float(os.getenv("QUALITY_MAX_BRIGHTNESS", "230.0"))
QUALITY_MAX_POSE_DEG = float(os.getenv("QUALITY_MAX_POSE_DEG", "35.0"))
QUALITY_MIN_FACE_FRACTION = float(os.getenv("QUALITY_MIN_FACE_FRACTION", "0.08"))

# --- Identification thresholds (Section 8) -------
THRESHOLD_HIGH_ARCFACE = float(os.getenv("THRESHOLD_HIGH_ARCFACE", "0.62"))
THRESHOLD_LOW_ARCFACE = float(os.getenv("THRESHOLD_LOW_ARCFACE", "0.45"))
THRESHOLD_HIGH_CLASSICAL = float(os.getenv("THRESHOLD_HIGH_CLASSICAL", "0.80"))
THRESHOLD_LOW_CLASSICAL = float(os.getenv("THRESHOLD_LOW_CLASSICAL", "0.55"))
VERIFY_THRESHOLD_ARCFACE = float(os.getenv("VERIFY_THRESHOLD_ARCFACE", "0.50"))
# Cap frames scored by the anti-spoof model per verify (latency vs robustness).
LIVENESS_MAX_SCORED_FRAMES = int(os.getenv("LIVENESS_MAX_SCORED_FRAMES", "3"))
VERIFY_THRESHOLD_CLASSICAL = float(os.getenv("VERIFY_THRESHOLD_CLASSICAL", "0.80"))
IDENTIFY_TOP_K = int(os.getenv("IDENTIFY_TOP_K", "5"))
IDENTIFY_MIN_FRAMES = int(os.getenv("IDENTIFY_MIN_FRAMES", "3"))
IDENTIFY_MAX_FRAMES = int(os.getenv("IDENTIFY_MAX_FRAMES", "5"))

# --- Liveness (Section 9) ---------------------------------------------------
LIVENESS_REQUIRE_BLINK_OR_MOTION = os.getenv("LIVENESS_REQUIRE_BLINK_OR_MOTION", "true").lower() == "true"
LIVENESS_MIN_SCORE = float(os.getenv("LIVENESS_MIN_SCORE", "0.5"))
LIVENESS_MIN_REAL_SCORE = float(os.getenv("LIVENESS_MIN_REAL_SCORE", "0.55"))
# When true (production default), liveness fails closed if the trained model
# is unavailable — no silent fallback to the weaker heuristic.
LIVENESS_FAIL_CLOSED = os.getenv("LIVENESS_FAIL_CLOSED", "true").lower() == "true"

# --- Auth / tokens (Section 13) ---------------------------------------------
# RS256 is the default. Provide PEM key paths for production; auto-generates
# ephemeral keys for dev if neither JWT_SECRET nor key paths are set.
JWT_PRIVATE_KEY_PATH = os.getenv("JWT_PRIVATE_KEY_PATH", "")
JWT_PUBLIC_KEY_PATH = os.getenv("JWT_PUBLIC_KEY_PATH", "")
JWT_KID = os.getenv("JWT_KID", "")
JWT_ISSUER = os.getenv("JWT_ISSUER", "face-auth-platform")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "600"))
STEPUP_TOKEN_TTL_SECONDS = int(os.getenv("STEPUP_TOKEN_TTL_SECONDS", "60"))

# Warn loudly if running with unsafe defaults in a non-test context
if JWT_SECRET == "CHANGE_ME_IN_PRODUCTION" and not JWT_PRIVATE_KEY_PATH and "pytest" not in sys.modules:
    warnings.warn(
        "Neither JWT_PRIVATE_KEY_PATH nor JWT_SECRET is set — using auto-generated "
        "ephemeral RS256 keys. Tokens will not survive a restart.",
        stacklevel=1,
    )

# --- Enrollment integrity ---------------------------------------------------
ENROLL_REQUIRE_LIVENESS = os.getenv("ENROLL_REQUIRE_LIVENESS", "true").lower() == "true"

# --- Biometric template encryption at rest (spec §13: AES-256) --------------
TEMPLATE_ENCRYPTION_KEY = os.getenv("TEMPLATE_ENCRYPTION_KEY", "").strip()

# --- Rate limiting / lockout (Section 9.5) ----------------------------------
MAX_FAILED_ATTEMPTS = int(os.getenv("MAX_FAILED_ATTEMPTS", "5"))
LOCKOUT_WINDOW_SECONDS = int(os.getenv("LOCKOUT_WINDOW_SECONDS", "300"))

# --- API access tokens (Section 10/13) --------------------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
_default_client_tokens = "hr-portal:dev-hrportal-client-token-change-me"
APP_CLIENT_TOKENS = dict(
    pair.split(":", 1) for pair in os.getenv("APP_CLIENT_TOKENS", _default_client_tokens).split(",") if ":" in pair
)

# --- CORS -------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

# --- Concurrency / upload limits -------------------------------------------
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10MB (identify/authorize: ~5 frames)
# Enrollment sends a guided ~300-frame burst, so it needs a much larger ceiling.
MAX_ENROLL_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_ENROLL_UPLOAD_SIZE_BYTES", str(150 * 1024 * 1024)))  # 150MB

# --- Audit retention --------------------------------------------------------
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "2555"))  # ~7 years (HIPAA)
CONSUMED_TOKEN_RETENTION_HOURS = int(os.getenv("CONSUMED_TOKEN_RETENTION_HOURS", "24"))
