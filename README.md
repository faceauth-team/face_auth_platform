# Face Authentication Platform

A face-based authentication service for clinical / EMR workflows. The goal
is to let a hospital **enroll** a clinician once and then **verify their
identity by face** before sensitive clinical actions are committed — a
biometric step-up that is faster than passwords and harder to share or
spoof.

> 🚧 **Status: in active development.** This repository is being built out
> incrementally. The sections below describe what the platform is and what
> we are building toward, not a finished product.

## The problem we're solving

In an EMR, critical record changes (diagnosis, prescription, plan of
management) need to be reliably attributed to the clinician who made them.
Passwords and shared logins are weak: they get shared, reused, and
shoulder-surfed. We want a **fast, hard-to-repudiate identity check** at the
moment of action, with biometric data handled responsibly under DPDP /
GDPR.

## What we're building

A small, self-hostable platform with two clearly separated surfaces:

- **IT Enrollment Console** — operated by IT/admin. Captures a clinician's
  face (guided multi-pose capture) and registers their details.
- **Clinician Login / Step-Up** — used at the point of care. When a
  clinician edits a gated field, a live face check confirms it's really
  them before the change is committed.

Behind those, a backend service that provides:

- **Enrollment** — turn a short capture into reusable face templates.
- **1:1 verification (step-up)** — confirm a known clinician's identity.
- **1:N identification** — find who a face belongs to.
- **Liveness / anti-spoof** — reject photos, screens, and masks.
- **Token issuance** — short-lived, single-use authorization tokens.
- **Audit trail & right-to-erasure** — every action logged; biometric data
  deletable on request.

## Planned tech stack

| Area | Direction |
|---|---|
| API | Python + FastAPI |
| Face embeddings | ArcFace (InsightFace) |
| Anti-spoofing | Silent-Face / MiniFASNet |
| Vector search | FAISS (→ Milvus/Qdrant later) |
| Auth tokens | RS256 JWT (→ Keycloak later) |
| Storage | SQLite for dev → PostgreSQL |
| Frontend | Lightweight HTML/JS consoles |

> Model weights are large pretrained files and are **not** stored in this
> repo. They're fetched per-machine from their public sources — see
> `models/README.md` once that's added.

## Roadmap (high level)

- [ ] Project scaffolding & API skeleton
- [ ] Enrollment pipeline (capture → templates)
- [ ] 1:1 step-up verification + liveness
- [ ] 1:N identification
- [ ] Token issuance & EMR write flow
- [ ] Audit logging & data erasure
- [ ] IT and clinician UIs
- [ ] Threshold calibration on pilot data
- [ ] Deployment / infra hardening

## Getting started

Setup instructions will be added here as the codebase lands. Local config
is provided via a `.env` file (a no-secrets `.env.example` template will be
committed); real secrets, keys, and model weights stay out of the repo by
design.

## Team

Built by a three-person team. Contributions go through feature branches and
pull requests into `main`.
