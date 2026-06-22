# Face Authentication Platform — Working Prototype

This is a running implementation of the architecture in
`Face_Auth_Enterprise_Spec.pdf`: 1:N identification, 1:1 step-up
authorization, REST API, RS256 JWT issuance, audit logging,
right-to-erasure, and hash-chained EMR receipts, all on an open-source
stack. It is **not a mockup** — enrollment, identification, and matching
work end-to-end against a real FastAPI server with the production ML
models active (ArcFace embeddings + Silent-Face anti-spoofing). The one
remaining step before gating live clinical actions is field calibration
of the match thresholds (spec Section 17).

## Models: ArcFace + Silent-Face are installed and active

This build runs the production ML path. The pretrained weights are present
under `models/` and the code activates them automatically — confirm via
`GET /healthz`:

- **Face embedding — ArcFace** (`models/buffalo_l/w600k_r50.onnx`, ~166MB).
  `app/ml/embedder.py:get_embedder()` returns `ArcFaceEmbedder` (512-d
  universal embedding, `requires_discriminant=False`) whenever the ONNX
  file exists at `ARCFACE_MODEL_PATH`.
- **Anti-spoof — Silent-Face MiniFASNet** (`models/anti_spoof/*.pth`).
  `app/ml/liveness.py` uses the trained model as the primary liveness
  decision and **fails closed** when `LIVENESS_FAIL_CLOSED=true`
  (production default).

What runs as an **automatic offline fallback** (used only when a weight
file is missing):

- **Face detection + 478-pt landmarks**: Google MediaPipe (Apache-2.0),
  weights bundled in the pip wheel. This is the active detector;
  RetinaFace remains the documented swap.
- **Embedding fallback**: a classical LBP+HOG descriptor + PCA/LDA
  discriminant ("Fisherfaces"), used *only* if the ArcFace weights are
  absent. Not validated to any FAR/FRR target — a stand-in, not a
  substitute for ArcFace.
- **Liveness fallback**: blink/motion + moiré + reflection heuristics,
  used *only* when the trained model is unavailable **and**
  `LIVENESS_FAIL_CLOSED=false`.
- **Vector search**: FAISS in-process (stand-in for Milvus/Qdrant).
- **Token issuance**: RS256 JWT (auto-generated keys in dev; mount PEM
  keys for production). Stands in for Keycloak.

Everything else — the API contract, the database schema, the
enrollment/identify/authorize/EMR workflows, audit logging, and
right-to-erasure — is a direct implementation of the spec.

> **Still required before going live:** field calibration. Tune
> `THRESHOLD_HIGH_ARCFACE` / `THRESHOLD_LOW_ARCFACE` against real pilot
> data per spec Section 17 before gating clinical actions; embeddings from
> ArcFace and the classical fallback are not comparable, so re-enroll if
> you ever switch paths. To re-fetch the ArcFace pack on a clean machine:
> `pip install insightface` then download `buffalo_l`, or pull
> `w600k_r50.onnx` from the InsightFace model zoo into `models/buffalo_l/`
> (see `models/README.md`).

## Architecture

```
app/
  core/config.py        all tunables, env-var overridable
  ml/
    detector.py          MediaPipe face detection + 478-pt landmarks + 112x112 alignment
    quality.py            blur/brightness/pose/occlusion filtering (spec 7.1)
    liveness.py            blink/motion + moire + reflection checks (spec Section 9)
    embedder.py             ClassicalFeatureExtractor + ArcFaceEmbedder (swap point)
    discriminant.py          PCA+LDA projector for the classical path
    templates.py               KMeans clustering of frames -> 10-20 templates (spec 7.1)
    vector_index.py             FAISS index, stand-in for Milvus/Qdrant
    enrollment.py                 ties detection->quality->embed->cluster together
    matcher.py                     ties detection->liveness->embed->search->threshold together
  db/
    models.py            SQLAlchemy models = spec Section 12 schema exactly
    database.py            engine/session (SQLite prototype / Postgres production)
  auth/
    tokens.py             JWT issuance, stand-in for Keycloak
    dependencies.py         admin_token / app_client_token FastAPI dependencies
  api/
    main.py               FastAPI app, mounted at /v1
    routes_enroll.py         POST /enroll, GET /enroll/{id}/status, DELETE /employees/{id}/templates
    routes_identify.py         POST /identify (1:N)
    routes_authorize.py          POST /authorize (1:1 step-up, issues scoped token)
    routes_emr.py                  POST /emr/entries (hash-chained write, consumes token)
    routes_audit.py                  GET /audit-logs
  workers/
    enrollment_job.py     background enrollment processing (BackgroundTasks stand-in for Kafka/Celery)
    retention.py            reaps stale consumed tokens / old audit rows per retention policy

sdk/
  python/face_auth_sdk/  Python client (spec 11.2)
  js/face-auth-sdk.js    JS client + drop-in <FaceAuthButton> (spec 11.1, 11.3)
  demo/kiosk-demo.html   browser demo exercising both SDKs against a live webcam

db/schema.sql            literal Postgres DDL (matches app/db/models.py)
infra/docker-compose.yml api + Postgres (fully wired) + Qdrant/Keycloak/MinIO (reference topology, not yet wired into app code)
Dockerfile                container build for the API service
tests/test_pipeline.py   pytest smoke tests
```

## Running it

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

Open `sdk/demo/kiosk-demo.html` in a browser (serve it over `http://`,
not `file://`, so camera permissions work — e.g.
`python3 -m http.server` from the `sdk/` directory) to enroll and
identify against a real webcam. Default dev tokens are
`dev-admin-token-change-me` and `dev-hrportal-client-token-change-me` —
**change these before running anywhere but localhost** (see
`.env.example`).

### With Docker Compose (API + Postgres)

```bash
cd infra && docker compose up
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Two tests need a real face photo (mediapipe's detector won't fire on a
synthetic shape) — point `FACE_AUTH_TEST_IMAGE` at one locally; they
skip automatically otherwise. Don't commit a real photo into this repo
as a checked-in fixture.

## API quick reference (spec Section 10)

All endpoints under `/v1`. Admin endpoints take `Authorization: Bearer
<ADMIN_TOKEN>`; `/identify` and `/audit-logs` take a per-application
client token (see `APP_CLIENT_TOKENS`).

| Method | Path | Purpose |
|---|---|---|
| POST | `/enroll` | multipart: `employee_id`, `consent_token`, `full_name`, `frames[]` |
| GET | `/enroll/{enrollment_id}/status` | poll enrollment progress |
| POST | `/identify` | multipart: `application_id`, `frames[]` (3-5) — 1:N |
| POST | `/authorize` | step-up 1:1 verify: `employee_id`/`email`, `patient_id`, `field_name`, `frames[]`; issues a scoped single-use token |
| POST | `/emr/entries` | write a hash-chained EMR entry, consuming a step-up token |
| DELETE | `/employees/{employee_id}/templates` | right-to-erasure |
| GET | `/audit-logs?employee_id=&from=&to=` | audit trail query |
| GET | `/healthz` | embedder/index/liveness status |
| GET | `/v1/.well-known/jwks.json` | public keys for offline token verification |

`/identify` returns one extra result value beyond the spec's literal
`match`/`no_match`/`rejected` examples: **`ambiguous`**, for scores
between the low/high thresholds — spec Section 8 step 5 describes this
band ("request liveness re-check or fallback") but the worked API
examples only show the other three, so this is a documented, reasoned
extension rather than an invented behavior.

## What's deliberately out of scope here

This prototype implements spec Phases 2-3 (core ML pipeline, APIs,
SDKs) and the data/compliance scaffolding (Phase 5's schema and
audit-trail mechanics). It does **not** implement: Kubernetes manifests
(Phase 1/7), Kafka/Celery (BackgroundTasks stands in), Vault-managed
secrets, Prometheus/Grafana/OpenSearch observability, or Keycloak
itself (local JWT stands in). All of those are genuine infrastructure
deployment work, not Python logic — `infra/docker-compose.yml` gives
you the real service topology to build that out against.
